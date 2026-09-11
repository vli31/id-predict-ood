"""
Extract ID-only internal features from the question-formation (QF) models (hier_gen TransformerLM, 6L/8H/512d).
ID inputs: the 'quest' examples of question.val (the paper's ID set for head detection, n=406).
Per model -> generic attention statistics of the 'quest' query position (pooled over heads per layer),
hand-crafted matrix-aux-detector scores (recomputed to validate against the released CSV), RDM of the quest-position
final-layer representation, ID accuracy, and (optional) OOD P(hier) on a subsample for validation.
"""
import os, sys, json, time, argparse, re
import numpy as np, pandas as pd, torch
from collections import defaultdict
HIER = os.environ.get("HIER_GEN", os.path.expanduser("~/hier_gen")); sys.path.insert(0, HIER); os.chdir(HIER)   # clone of github.com/sunnytqin/hier_gen
import types as _types
_tm = _types.ModuleType("transformers"); _gb = _types.ModuleType("transformers.gated_bert_utilities"); _gb.ConcreteGate = object
_tm.gated_bert_utilities = _gb; sys.modules.setdefault("transformers", _tm); sys.modules.setdefault("transformers.gated_bert_utilities", _gb)
from vocabulary import WordVocabulary
from models.transformer_lm import TransformerLM
from layers import Transformer
from layers.transformer import multi_head_attention as mha

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); QF = os.environ.get("QF_MODELS", os.path.expanduser("~/qf_models"))   # <run>/args.json and <run>/checkpoint_300000.pth per model
DATA = f"{ROOT}/question_formation_data/question_formation"
OUT = f"{ROOT}/heldout/features/qf_cp300000.npz"

# ---------- vocab exactly as build_datasets_lm(splits=[train,val,test]) ----------
def read_split(split):
    return [l.strip().replace("\t", " ") for l in open(f"{DATA}/question.{split}")]
sents = {s: read_split(s) for s in ["train", "val", "test"]}
vocab = WordVocabulary(sents["train"] + sents["val"] + sents["test"], split_punctuation=False)
V = len(vocab); EOS = V; SOS = V + 1
# ---------- POS map ----------
t2t = defaultdict(list); tok2tag = {}
for line in open(f"{HIER}/cfgs/tag_token_map.txt"):
    line = line.strip()
    if not line: continue
    ty, tk = line.split("\t"); t2t[ty].append(tk); tok2tag.setdefault(tk, ty)
AUXS = set(t2t["aux_s"]) | set(t2t["aux_p"]); RELS = set(t2t["rel"])
def coarse(tag):
    if tag.startswith("aux"): return "aux"
    if tag.startswith("n_"): return "n"
    if tag.startswith("v_"): return "v"
    if tag in ("det", "rel", "prep"): return tag
    return "other"
CATS = ["det", "n", "rel", "aux", "v", "punct", "other"]

def parse(sent):
    prefix = sent.split(" quest ", 1)[0] + " quest"; words = prefix.split()
    aux_idxs = [i for i, w in enumerate(words) if w in AUXS]; rel_idxs = [i for i, w in enumerate(words) if w in RELS]
    first_aux = aux_idxs[0]
    if rel_idxs:
        before = [i for i in aux_idxs if i < rel_idxs[0]]; matrix = before[0] if before else aux_idxs[-1]
    else: matrix = first_aux
    cats = ["sos"] + [("punct" if w in (".", "?") else ("quest" if w == "quest" else coarse(tok2tag.get(w, "other")))) for w in words]
    return dict(prefix=prefix, words=["<s>"] + words, quest_i=len(words), matrix_aux_i=matrix + 1, first_aux_i=first_aux + 1,
                matrix_aux_word=words[matrix], first_aux_word=words[first_aux], cats=cats)
id_meta = pd.DataFrame([parse(s) for s in sents["val"] if " quest " in s])
ood_meta = pd.DataFrame([parse(s) for s in sents["test"] if " quest " in s])
print("ID quest examples:", len(id_meta), " OOD:", len(ood_meta), " vocab:", V)

def batchify(meta):
    toks = [[SOS] + vocab(p) for p in meta.prefix]; L = np.array([len(t) for t in toks]); T = L.max()
    X = np.zeros((len(toks), T), dtype=np.int64)
    for i, t in enumerate(toks): X[i, :len(t)] = t
    return torch.tensor(X), torch.tensor(L)

# ---------- attention capture ----------
CAP = []
_orig = mha.MultiHeadAttentionBase._masked_softmax
def _patched(self, logits, mask):
    out = _orig(self, logits, mask)
    bb, td, ts = out.shape
    CAP.append(out.detach().view(bb // self.n_heads, self.n_heads, td, ts).cpu())
    return out
mha.MultiHeadAttentionBase._masked_softmax = _patched

def load_model(run):
    args = json.load(open(f"{QF}/{run}/args.json"))
    m = TransformerLM(V, args["vec_dim"], args["n_heads"], num_encoder_layers=args["encoder_n_layers"], pos_scale=args["pos_scale"],
                      transformer=Transformer, dropout=args["dropout"], tied_embedding=args["tied_embedding"],
                      embedding_init="xavier", scale_mode="opennmt")
    sd = torch.load(f"{QF}/{run}/checkpoint_300000.pth", map_location="cpu")["model_state_dict"]
    m.load_state_dict({k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}); m.eval()
    return m, args

def forward(model, meta, device, bs=64, want_attn=True, want_rep=True):
    X, L = batchify(meta); logits_q = []; attn = []; reps = []
    rep_store = {}
    h = model.output_map.register_forward_pre_hook(lambda mod, inp: rep_store.__setitem__("x", inp[0].detach().cpu()))
    with torch.no_grad():
        for st in range(0, len(X), bs):
            xb, lb = X[st:st + bs].to(device), L[st:st + bs].to(device); CAP.clear()
            res = model(xb, lb); lg = res.data if hasattr(res, "data") else res
            q = torch.tensor(meta.quest_i.values[st:st + bs], device=device)
            logits_q.append(lg[torch.arange(len(xb), device=device), q].float().cpu())
            if want_attn: attn.append([a.clone() for a in CAP])          # list over layers of (b, H, T, T)
            if want_rep: reps.append(rep_store["x"][torch.arange(len(xb)), q.cpu()])
    h.remove()
    return torch.cat(logits_q), attn, (torch.cat(reps) if want_rep else None), X.shape[1]

GEN = ["ent", "a_sos", "a_self", "relpos", "maxw", "std_ent", "allq_ent", "allq_self", "allq_prev", "allq_sos"] + [f"prof{i}" for i in range(10)] + [f"mass_{c}" for c in CATS]

def head_stats(A, meta, T):
    """A: (N, T, T) attention of one head (rows = queries). Generic + hand stats over ID examples."""
    N = len(meta); q = meta.quest_i.values; L = q + 1                    # real tokens 0..q
    a = A[np.arange(N), q]; pos = np.arange(T)[None, :]; km = pos < L[:, None]; a = np.where(km, a, 0.0); eps = 1e-12
    ent = -(a * np.log(a + eps)).sum(1) / np.log(L)
    a_sos = a[:, 0]; a_self = a[np.arange(N), q]; rel = pos / np.maximum(q[:, None], 1); relpos = (a * rel * km).sum(1); maxw = a.max(1)
    relc = np.clip((pos / np.maximum(q[:, None], 1) * 10).astype(int), 0, 9); prof = np.stack([(a * km * (relc == b)).sum(1) for b in range(10)], 1)
    cats = np.full((N, T), "pad", dtype=object)
    for i, c in enumerate(meta.cats): cats[i, :len(c)] = c
    mass = np.stack([(a * (cats == c)).sum(1) for c in CATS], 1)
    qm = (pos >= 1) & (pos <= q[:, None]); nq = qm.sum(1)
    Aq = A * qm[:, :, None]; ent_all = -(Aq * np.log(Aq + eps)).sum(2); norm = np.log(pos + 1.0)
    allq_ent = ((ent_all / np.maximum(norm, eps)) * qm).sum(1) / nq
    diag = A[:, np.arange(T), np.arange(T)]; allq_self = (diag * qm).sum(1) / nq
    prev = np.zeros((N, T)); prev[:, 1:] = A[:, np.arange(1, T), np.arange(0, T - 1)]; allq_prev = (prev * qm).sum(1) / nq
    allq_sos = (A[:, :, 0] * qm).sum(1) / nq
    generic = np.concatenate([[ent.mean(), a_sos.mean(), a_self.mean(), relpos.mean(), maxw.mean(), ent.std(), allq_ent.mean(), allq_self.mean(), allq_prev.mean(), allq_sos.mean()], prof.mean(0), mass.mean(0)])
    am = a[np.arange(N), meta.matrix_aux_i.values]; af = a[np.arange(N), meta.first_aux_i.values]
    hand = np.array([(am >= 0.20).mean(), am.mean(), (af >= 0.20).mean(), af.mean()])   # prop_met, mean_score (matrix aux); same for first aux
    return generic, hand
HAND = ["matrix_prop_met", "matrix_mean", "first_prop_met", "first_mean"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=None); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n_ood_check", type=int, default=300)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mem = pd.read_csv(f"{ROOT}/question_formation_data/qf_matrix_aux_detector_raw_quest_membership_300000.csv")
    runs = [m.replace("__checkpoint_300000", "") for m in mem.model_checkpoint]
    if a.runs: runs = [r for r in runs if any(s in r for s in a.runs.split(","))]
    heads_csv = pd.read_csv(f"{ROOT}/question_formation_data/qf_matrix_aux_detector_raw_quest_heads_300000.csv")
    phier = pd.read_csv(f"{ROOT}/question_formation_data/qf_p_hier_by_model_checkpoint_300000.csv", nrows=a.n_ood_check)
    rdm_idx = np.random.default_rng(0).choice(len(id_meta), min(300, len(id_meta)), replace=False)
    done = {}
    part = OUT.replace(".npz", "_partial.npz")
    if os.path.exists(part):
        z = np.load(part, allow_pickle=True); done = {r: {k: z[k][i] for k in ["generic", "hand", "headraw", "handraw", "rdm", "id_acc", "ood_check"]} for i, r in enumerate(z["runs"])}
        print("resuming, have", len(done))
    hier_id = np.array([vocab(w)[0] for w in ood_meta.hier_aux_word]) if "hier_aux_word" in ood_meta else None
    for run in runs:
        if run in done or not os.path.exists(f"{QF}/{run}/checkpoint_300000.pth"): continue
        t0 = time.time(); model, args = load_model(run); model.to(device)
        Lyr, H = args["encoder_n_layers"], args["n_heads"]
        lq, attn, rep, T = forward(model, id_meta, device)
        # ID accuracy: argmax == matrix aux
        tgt = np.array([vocab(w)[0] for w in id_meta.matrix_aux_word]); id_acc = float((lq.argmax(-1).numpy() == tgt).mean())
        # stack attention per layer over batches (pad T differs per batch -> pad to global T)
        A_l = []
        for l in range(Lyr):
            parts = []
            for ab in attn:
                x = ab[l].numpy(); pad = T - x.shape[-1]
                parts.append(np.pad(x, ((0, 0), (0, 0), (0, pad), (0, pad))))
            A_l.append(np.concatenate(parts, 0))                     # (N, H, T, T)
        headraw = np.zeros((Lyr, H, len(GEN))); handraw = np.zeros((Lyr, H, len(HAND)))
        for l in range(Lyr):
            for h in range(H):
                headraw[l, h], handraw[l, h] = head_stats(A_l[l][:, h], id_meta, T)
        # validate hand feature vs CSV
        if a.validate:
            sub = heads_csv[heads_csv.model_checkpoint == run + "__checkpoint_300000"]
            ref = sub.pivot(index="layer", columns="head", values="prop_met").values
            print(f"  hand prop_met max|diff| vs CSV: {np.abs(ref - handraw[:, :, 0]).max():.4f}; mean_score diff {np.abs(sub.pivot(index='layer', columns='head', values='mean_score').values - handraw[:, :, 1]).max():.4f}")
        # OOD check on a subsample: P(hier aux) among {hier, linear} candidates
        col = run + "__checkpoint_300000"; ood_check = np.nan
        if a.n_ood_check > 0 and col in phier.columns:
            sub = ood_meta.iloc[:a.n_ood_check].copy()
            lo, _, _, _ = forward(model, sub, device, want_attn=False, want_rep=False)
            p = torch.softmax(lo, -1).numpy()
            hier_ids = np.array([vocab(w)[0] for w in phier.hier_aux_word]); lin_ids = np.array([vocab(w)[0] for w in phier.linear_aux_word])
            ph = p[np.arange(len(sub)), hier_ids]; pl = p[np.arange(len(sub)), lin_ids]
            ref = phier[col].values
            ood_check = dict(raw=float(np.abs(ph - ref).max()), norm=float(np.abs(ph / (ph + pl) - ref).max()), argmax=float(np.abs((lo.argmax(-1).numpy() == hier_ids).astype(float) - ref).max()))
            if a.validate: print("  OOD p_hier check vs CSV (max abs diff): ", ood_check)
        X = rep[rdm_idx].numpy(); X = X - X.mean(1, keepdims=True); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        iu = np.triu_indices(len(rdm_idx), 1); rdm = (1 - (X @ X.T)[iu]).astype(np.float16)
        gen = np.concatenate([np.concatenate([fn(headraw[l], 0) for fn in (np.max, np.mean, np.min)]) for l in range(Lyr)])
        hand = np.concatenate([np.max(handraw[l], 0) for l in range(Lyr)])
        done[run] = dict(generic=gen, hand=hand, headraw=headraw, handraw=handraw, rdm=rdm, id_acc=id_acc, ood_check=json.dumps(ood_check))
        print(f"{run}: id_acc={id_acc:.4f} {time.time()-t0:.1f}s  ({len(done)}/{len(runs)})", flush=True)
        del model
        if len(done) % 5 == 0 or len(done) == len(runs):
            rs = list(done); np.savez_compressed(part, runs=np.array(rs), **{k: np.stack([done[r][k] for r in rs]) for k in ["generic", "hand", "headraw", "handraw", "rdm"]},
                                id_acc=np.array([done[r]["id_acc"] for r in rs]), ood_check=np.array([done[r]["ood_check"] for r in rs]))
    rs = list(done)
    gnames = [f"L{l+1}_{p}_{s}" for l in range(6) for p in ("max", "mean", "min") for s in GEN]; hnames = [f"L{l+1}_max_{s}" for l in range(6) for s in HAND]
    np.savez_compressed(OUT, runs=np.array(rs), generic_names=np.array(gnames), hand_names=np.array(hnames), headraw_names=np.array(GEN), handraw_names=np.array(HAND),
                        **{k: np.stack([done[r][k] for r in rs]) for k in ["generic", "hand", "headraw", "handraw", "rdm"]},
                        id_acc=np.array([done[r]["id_acc"] for r in rs]), ood_check=np.array([done[r]["ood_check"] for r in rs]), rdm_idx=rdm_idx)
    print("saved", OUT, len(rs))

if __name__ == "__main__":
    main()
