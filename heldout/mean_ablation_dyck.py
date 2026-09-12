"""
Mean ablation on the Dyck-1 population (checkpoint 5): replace every attention head's pattern with its dataset-mean pattern
(mean post-softmax attention over the 1000 OOD test sequences), all heads at once, and measure the change in OOD accuracy.
Compared with the uniform ablation released in data/transformer_head_properties.csv (cp5_full_ablate_ood), split by OOD head type
as in Fig. 5 (sign-matching / violation-detecting, score >= 0.8 on OOD data).
Output: heldout/results/mean_ablation_dyck.csv and a printed summary.
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
    """CausalSelfAttention.forward with optional capture / replacement of the post-softmax attention (mean ablation)."""
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2); q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2); v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf")); att = F.softmax(att, dim=-1)
    if getattr(self, "capture", None) is not None: self.capture.append(att.detach().sum(0))            # sum over batch, (nh, T, T)
    if getattr(self, "replace", None) is not None: att = self.replace[:, :T, :T].unsqueeze(0).expand(B, -1, -1, -1)   # dataset-mean pattern
    y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y))
CausalSelfAttention.forward = attn_forward

def ood_acc(model):
    with t.no_grad(): logits, _ = model(TOKS)
    pred = (logits[t.arange(len(L)), t.as_tensor(L + 1), 1] > logits[t.arange(len(L)), t.as_tensor(L + 1), 0]).numpy().astype(int)
    return float((pred == Y).mean())

def process(idx):
    row = HP.iloc[idx]
    m = get_transformer(n_layer=int(row.n_layer), n_head=int(row.n_head), n_embd=int(row.n_embd), embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0)
    m.load_state_dict(t.load(f"{ROOT}/data/model_weights/run_{row.id}/run_{row.id}_checkpoint_5.pt", map_location="cpu")); m.eval()
    attns = [blk.attn for blk in m.transformer.h]
    for a in attns: a.capture = []
    base = ood_acc(m)                                                                                  # pass 1: baseline + capture mean patterns
    for a in attns: a.replace = (a.capture[0] / len(L)); a.capture = None                              # dataset-mean pattern per head
    ablated = ood_acc(m)                                                                               # pass 2: all heads replaced by their mean pattern
    return dict(id=row.id, n_layer=int(row.n_layer), n_head=int(row.n_head), wd=row.wd, ood_acc=base, ood_acc_mean_ablated=ablated, delta_mean=ablated - base,
                delta_uniform=row.cp5_full_ablate_ood - row.cp5_ood_acc, csv_ood_acc=row.cp5_ood_acc)

def has_type(row, kind):   # OOD head type as in the paper's Fig. 5: any head with score >= 0.8 on OOD data
    cols = [c for c in HP.columns if c.startswith(f"cp5_{kind}_head_l") and c.endswith("_ood")]
    return bool(row[cols].max() >= 0.8)

if __name__ == "__main__":
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool: rows = pool.map(process, range(len(HP)), chunksize=4)
    df = pd.DataFrame(rows).set_index("id").loc[HP.id].reset_index()
    df["sign"] = [has_type(r, "sign") for _, r in HP.iterrows()]; df["viol"] = [has_type(r, "neg") for _, r in HP.iterrows()]
    df["head_type"] = np.select([df.sign & df.viol, df.sign, df.viol], ["both", "sign-matching", "violation-detecting"], "neither")
    os.makedirs(f"{ROOT}/heldout/results", exist_ok=True); df.to_csv(f"{ROOT}/heldout/results/mean_ablation_dyck.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s; baseline vs CSV max|diff| = {np.abs(df.ood_acc - df.csv_ood_acc).max():.4f}")
    from scipy.stats import spearmanr, wilcoxon
    ml = df[df.n_layer >= 2]
    print(f"2- and 3-layer models (n={len(ml)}): Spearman(delta_mean, delta_uniform) = {spearmanr(ml.delta_mean, ml.delta_uniform).correlation:.3f}")
    g = ml.groupby("head_type").agg(n=("delta_mean", "size"), mean_delta_meanabl=("delta_mean", "mean"), median_delta_meanabl=("delta_mean", "median"), frac_improved=("delta_mean", lambda x: (x > 0.01).mean()), frac_damaged=("delta_mean", lambda x: (x < -0.01).mean()),
                                     mean_delta_uniform=("delta_uniform", "mean"))
    print(g.round(3).to_string())
    for k in ["sign-matching", "violation-detecting"]:
        d = ml[ml.head_type == k].delta_mean
        if len(d) > 5: print(f"{k}: Wilcoxon p = {wilcoxon(d).pvalue:.3g} (mean {d.mean():+.3f})")
