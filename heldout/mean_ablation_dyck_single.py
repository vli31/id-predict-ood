"""
One-head-at-a-time mean ablation on the Dyck-1 population (checkpoint 5): for each head, replace only that head's attention pattern
with its dataset-mean pattern (mean post-softmax attention over the 1000 OOD test sequences), keep every other head intact, and
measure the change in OOD accuracy. Compared with the released one-at-a-time uniform ablation (cp5_l{l}_h{h}_ood) and split by the
head's own OOD type (sign-matching / violation-detecting score >= 0.8).
Output: heldout/results/mean_ablation_dyck_single.csv (one row per head) and a printed summary.
"""
import os, sys, math, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, torch as t, torch.nn.functional as F
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from utils.model import get_transformer
from utils.data import SimpleTokenizer
from utils.minGPT.model import CausalSelfAttention
t.set_num_threads(1)
HP = pd.read_csv(f"{ROOT}/data/transformer_head_properties.csv")
OOD = pd.read_csv(f"{ROOT}/data/model_preds/ood_data_preds.csv", usecols=["string", "balanced"])
tok = SimpleTokenizer("()"); TOKS = tok.tokenize(OOD.string.tolist(), max_len=40); L = np.array([len(s) for s in OOD.string]); Y = OOD.balanced.values.astype(int)

def attn_forward(self, x):
    """CausalSelfAttention.forward with optional capture of the post-softmax attention and per-head replacement by the mean pattern."""
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2); q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2); v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")); att = F.softmax(att, dim=-1)
    if getattr(self, "capture", None) is not None: self.capture.append(att.detach().sum(0))            # sum over batch, (nh, T, T)
    heads = getattr(self, "replace_heads", None)
    if heads:                                                                                          # replace only the listed heads
        att = att.clone()
        for h in heads: att[:, h] = self.mean_att[h, :T, :T].unsqueeze(0).expand(B, -1, -1)
    y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y))
CausalSelfAttention.forward = attn_forward

def ood_acc(model):
    with t.no_grad(): logits, _ = model(TOKS)
    pred = (logits[t.arange(len(L)), t.as_tensor(L + 1), 1] > logits[t.arange(len(L)), t.as_tensor(L + 1), 0]).numpy().astype(int)
    return float((pred == Y).mean())

def process(idx):
    row = HP.iloc[idx]; nl, nh = int(row.n_layer), int(row.n_head)
    m = get_transformer(n_layer=nl, n_head=nh, n_embd=int(row.n_embd), embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
    m.load_state_dict(t.load(f"{ROOT}/data/model_weights/run_{row.id}/run_{row.id}_checkpoint_5.pt", map_location="cpu")); m.eval()
    attns = [blk.attn for blk in m.transformer.h]
    for a in attns: a.capture = []
    base = ood_acc(m)                                                                                  # baseline + capture mean patterns
    for a in attns: a.mean_att = a.capture[0] / len(L); a.capture = None
    out = []
    for l in range(nl):
        for h in range(nh):
            attns[l].replace_heads = [h]; abl = ood_acc(m); attns[l].replace_heads = None             # one head at a time
            out.append(dict(id=row.id, n_layer=nl, n_head=nh, wd=row.wd, layer=l + 1, head=h + 1, ood_acc=base, ood_acc_single_mean=abl, delta_mean_single=abl - base,
                            delta_uniform_single=row[f"cp5_l{l+1}_h{h+1}_ood"] - row.cp5_ood_acc, csv_ood_acc=row.cp5_ood_acc,
                            sign_score=row[f"cp5_sign_head_l{l+1}_h{h+1}_ood"], viol_score=row[f"cp5_neg_head_l{l+1}_h{h+1}_ood"]))
    return out

if __name__ == "__main__":
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool: rows = [r for rs in pool.map(process, range(len(HP)), chunksize=2) for r in rs]
    df = pd.DataFrame(rows)
    df["head_type"] = np.select([(df.sign_score >= 0.8) & (df.viol_score >= 0.8), df.sign_score >= 0.8, df.viol_score >= 0.8], ["both", "sign-matching", "violation-detecting"], "neither")
    os.makedirs(f"{ROOT}/heldout/results", exist_ok=True); df.to_csv(f"{ROOT}/heldout/results/mean_ablation_dyck_single.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s; {len(df)} heads; baseline vs CSV max|diff| = {np.abs(df.ood_acc - df.csv_ood_acc).max():.4f}")
    from scipy.stats import spearmanr, wilcoxon
    for name, d in [("all heads", df), ("heads in 2- and 3-layer models", df[df.n_layer >= 2])]:
        print(f"\n{name} (n={len(d)}): Spearman(delta_mean_single, delta_uniform_single) = {spearmanr(d.delta_mean_single, d.delta_uniform_single).correlation:.3f}")
        g = d.groupby("head_type").agg(n=("delta_mean_single", "size"), mean_delta_meanabl=("delta_mean_single", "mean"), median_delta_meanabl=("delta_mean_single", "median"),
                                       frac_improved=("delta_mean_single", lambda x: (x > 0.01).mean()), frac_damaged=("delta_mean_single", lambda x: (x < -0.01).mean()), mean_delta_uniform=("delta_uniform_single", "mean"))
        print(g.round(3).to_string())
        for k in ["sign-matching", "violation-detecting"]:
            x = d[d.head_type == k].delta_mean_single
            if len(x) > 5: print(f"  {k}: Wilcoxon p = {wilcoxon(x).pvalue:.3g} (mean {x.mean():+.3f})")
    # paper Fig. 17 style: per model, the head whose mean ablation changes OOD accuracy most (in absolute value), grouped by that head's type
    ml = df[df.n_layer >= 2]; top = ml.loc[ml.groupby("id").delta_mean_single.apply(lambda s: s.abs().idxmax())]
    print("\nper-model most-affecting head (2- and 3-layer models), by its type:")
    print(top.groupby("head_type").agg(n=("delta_mean_single", "size"), mean_delta=("delta_mean_single", "mean"), frac_improved=("delta_mean_single", lambda x: (x > 0.01).mean()), frac_damaged=("delta_mean_single", lambda x: (x < -0.01).mean())).round(3).to_string())
