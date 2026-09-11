"""
Extract ID-only internal features for every Dyck-1 Transformer at every checkpoint.

Outputs heldout/features/cp{k}.npz with, per model (row order = ids):
  generic  : task-agnostic attention statistics, pooled (max/mean/min) over heads within each layer
  hand     : the paper's hand-crafted head scores (neg / nonneg / sign / ambi), max over heads per layer
  headraw  : per-head raw stats (n_models, 3 layers, 4 heads, n_stats), NaN where head doesn't exist
  id_probs : P(balanced) on the 1000 ID test strings
  ood_probs: P(balanced) on the 1000 OOD test strings
"""
import os, sys, math, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, torch as t
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.model import get_transformer
from utils.data import SimpleTokenizer

t.set_num_threads(1)
MAX_LEN = 40
tok = SimpleTokenizer("()")
HP = pd.read_csv(f"{ROOT}/data/transformer_head_properties.csv")
ID_DF = pd.read_csv(f"{ROOT}/data/model_preds/indist_data_preds.csv", usecols=["string", "balanced", "matched"])
OOD_DF = pd.read_csv(f"{ROOT}/data/model_preds/ood_data_preds.csv", usecols=["string", "balanced", "matched"])

def prep(strings):
    toks = tok.tokenize(strings, max_len=MAX_LEN)            # (N, 42)
    L = np.array([len(s) for s in strings])
    N, T = toks.shape
    depth = np.zeros((N, T), dtype=np.int32)                 # depth after each char at char positions 1..L
    is_open = np.zeros((N, T), dtype=bool); is_close = np.zeros((N, T), dtype=bool)
    for i, s in enumerate(strings):
        d = 0
        for j, c in enumerate(s):
            d += 1 if c == "(" else -1
            depth[i, j + 1] = d
            (is_open if c == "(" else is_close)[i, j + 1] = True
    pos = np.arange(T)[None, :]
    char_mask = (pos >= 1) & (pos <= L[:, None])             # bracket positions
    key_mask = pos <= (L[:, None] + 1)                       # BOS..EOS
    return dict(toks=toks, L=L, depth=depth, is_open=is_open, is_close=is_close,
                char_mask=char_mask, key_mask=key_mask, N=N, T=T)

ID = prep(ID_DF.string.tolist()); OOD = prep(OOD_DF.string.tolist())

GENERIC_STATS = ["ent", "a_bos", "a_self", "a_open", "a_close", "relpos", "maxw", "a_first", "a_last",
                 "std_ent", "allq_ent", "allq_self", "allq_prev", "allq_bos"] + [f"prof{i}" for i in range(10)]
HAND_STATS = ["neg", "nonneg", "sign", "ambi"]

def head_stats(A, D):
    """A: (N, T, T) attention for one head over the batch. Returns (generic[24], hand[4])."""
    N, T = D["N"], D["T"]
    L = D["L"]; eos = L + 1
    a = A[np.arange(N), eos]                                   # (N, T) EOS query row
    km = D["key_mask"]; cm = D["char_mask"]
    a = np.where(km, a, 0.0)
    eps = 1e-12
    ent = -(a * np.log(a + eps)).sum(1) / np.log(eos + 1)      # normalized entropy
    a_bos = a[:, 0]
    a_self = a[np.arange(N), eos]
    a_open = (a * D["is_open"]).sum(1); a_close = (a * D["is_close"]).sum(1)
    rel = np.arange(T)[None, :] / eos[:, None]
    relpos = (a * rel * km).sum(1)
    maxw = a.max(1)
    a_first = a[:, 1]
    a_last = a[np.arange(N), L]
    # attention mass over relative-position deciles of bracket positions
    relc = np.clip(((np.arange(T)[None, :] - 1) / np.maximum(L[:, None], 1) * 10).astype(int), 0, 9)
    prof = np.zeros((N, 10))
    for b in range(10):
        prof[:, b] = (a * cm * (relc == b)).sum(1)
    # all-query stats (queries 1..eos, causal keys)
    qm = cm.copy(); qm[np.arange(N), eos] = True              # query positions 1..L+1
    Aq = A * qm[:, :, None]
    ent_all = -(Aq * np.log(Aq + eps)).sum(2)                  # (N, T) per query
    nq = qm.sum(1)
    q_idx = np.arange(T)[None, :]
    norm = np.log(q_idx + 1.0)                                 # each query j attends to j+1 keys
    allq_ent = ((ent_all / np.maximum(norm, eps)) * qm).sum(1) / nq
    diag = A[:, np.arange(T), np.arange(T)]
    allq_self = (diag * qm).sum(1) / nq
    prev = np.zeros((N, T)); prev[:, 1:] = A[:, np.arange(1, T), np.arange(0, T - 1)]
    allq_prev = (prev * qm).sum(1) / nq
    allq_bos = (A[:, :, 0] * qm).sum(1) / nq
    generic = np.array([ent.mean(), a_bos.mean(), a_self.mean(), a_open.mean(), a_close.mean(), relpos.mean(),
                        maxw.mean(), a_first.mean(), a_last.mean(), ent.std(), allq_ent.mean(), allq_self.mean(),
                        allq_prev.mean(), allq_bos.mean()] + prof.mean(0).tolist())
    # hand-crafted (paper) scores on bracket positions only
    dep = D["depth"]
    negm = cm & (dep < 0); nnm = cm & (dep >= 0)
    has_both = negm.any(1) & nnm.any(1)
    big = 1e9
    min_neg = np.where(negm, a, big).min(1); max_neg = np.where(negm, a, -big).max(1)
    min_nn = np.where(nnm, a, big).min(1); max_nn = np.where(nnm, a, -big).max(1)
    neg_pref = min_neg > max_nn
    nn_pref = min_nn > max_neg
    final_neg = dep[np.arange(N), L] < 0
    sign_pref = np.where(final_neg, neg_pref, nn_pref)
    hb = has_both
    hand = np.array([neg_pref[hb].mean(), nn_pref[hb].mean(), sign_pref[hb].mean(), 0.0]) if hb.any() else np.zeros(4)
    hand[3] = hand[0] + hand[1]
    return generic, hand

def run_model(row, cp, D, return_attn=True):
    model = get_transformer(n_layer=int(row.n_layer), n_head=int(row.n_head), n_embd=int(row.n_embd),
                            embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
    sd = t.load(f"{ROOT}/data/model_weights/run_{row.id}/run_{row.id}_checkpoint_{cp}.pt", map_location="cpu")
    model.load_state_dict(sd); model.eval()
    with t.no_grad():
        logits, _ = model(D["toks"])
    N = D["N"]; eos = t.as_tensor(D["L"] + 1)
    lg = logits[t.arange(N), eos]                                # (N, vocab)
    probs = t.softmax(lg[:, :2], dim=-1)[:, 1].numpy()           # P(balanced) uses logits 0/1 like the repo
    attn = [blk.attn.last_attn_weights.numpy() for blk in model.transformer.h] if return_attn else None  # each (N, nh, T, T)
    return probs, attn

def process(args):
    idx, cp = args
    row = HP.iloc[idx]
    id_probs, attn = run_model(row, cp, ID)
    ood_probs, _ = run_model(row, cp, OOD, return_attn=False)
    nL, nH = int(row.n_layer), int(row.n_head)
    headraw = np.full((3, 4, len(GENERIC_STATS)), np.nan); handraw = np.full((3, 4, 4), np.nan)
    for l in range(nL):
        for h in range(nH):
            g, hd = head_stats(attn[l][:, h], ID)
            headraw[l, h] = g; handraw[l, h] = hd
    return idx, id_probs.astype(np.float32), ood_probs.astype(np.float32), headraw, handraw

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cps", default="5"); ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    for cp in [int(c) for c in a.cps.split(",")]:
        out = f"{ROOT}/heldout/features/cp{cp}.npz"
        if os.path.exists(out): print("exists", out); continue
        t0 = time.time()
        with Pool(a.procs) as pool:
            res = pool.map(process, [(i, cp) for i in range(len(HP))], chunksize=4)
        res.sort(key=lambda r: r[0])
        n = len(HP)
        id_probs = np.stack([r[1] for r in res]); ood_probs = np.stack([r[2] for r in res])
        headraw = np.stack([r[3] for r in res]); handraw = np.stack([r[4] for r in res])
        # pooled generic features: per layer, max/mean/min over existing heads; zeros for missing layers
        gen, gnames = [], []
        for l in range(3):
            for pool_name, fn in [("max", np.nanmax), ("mean", np.nanmean), ("min", np.nanmin)]:
                with np.errstate(all="ignore"):
                    v = fn(headraw[:, l], axis=1)                          # (n, n_stats)
                v = np.nan_to_num(v, nan=0.0)
                gen.append(v); gnames += [f"L{l+1}_{pool_name}_{s}" for s in GENERIC_STATS]
        generic = np.concatenate(gen, axis=1)
        hand, hnames = [], []
        for l in range(3):
            with np.errstate(all="ignore"):
                v = np.nan_to_num(np.nanmax(handraw[:, l], axis=1), nan=0.0)
            hand.append(v); hnames += [f"L{l+1}_max_{s}" for s in HAND_STATS]
        hand = np.concatenate(hand, axis=1)
        np.savez_compressed(out, ids=HP.id.values.astype(str), generic=generic, generic_names=np.array(gnames),
                            hand=hand, hand_names=np.array(hnames), headraw=headraw, handraw=handraw,
                            headraw_names=np.array(GENERIC_STATS), handraw_names=np.array(HAND_STATS),
                            id_probs=id_probs, ood_probs=ood_probs)
        print(f"cp{cp}: saved {out} in {time.time()-t0:.1f}s; generic {generic.shape}, hand {hand.shape}", flush=True)

if __name__ == "__main__":
    main()
