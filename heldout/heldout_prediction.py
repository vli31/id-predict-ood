"""
Held-out prediction of OOD rule choice from ID-only information (Appendix "Held-out evaluation", Figure "heldout_prediction").

Target: does a held-out model follow the NESTED rule OOD (OOD accuracy >= 0.5)?  Predictors see ID data only:
  (i)   ID hierarchical-head rule: any head, any layer, tracks violations on >= 80% of relevant ID sequences (score = max over heads);
  (ii)  nesting-agnostic attention statistics: 24 statistics per head of the EOS attention on the 1000 ID test strings that never
        reference nesting (entropy, attention to BOS/EOS/first/last token, centre of mass, largest weight, 10-bin positional profile,
        entropy variability, all-query entropy/self/previous/BOS), pooled max/mean/min over all heads and over the final layer's heads,
        random forest;
  (iii) baseline: logistic regression on hyperparameters alone (depth, width, weight decay).
Evaluation: 50 random splits (150 train / 120 test) and leave-one-(depth, weight decay)-setting-out (six of nine settings contain
both classes and can be scored).  Question formation: same rule and statistics, 5-fold CV x 20 over the 79 models.

Inputs : heldout/features/cp5.npz (from extract_features.py), heldout/features/qf_cp300000.npz (from extract_qf_features.py, optional),
         data/transformer_head_properties.csv, question_formation_data/qf_matrix_aux_detector_raw_quest_membership_300000.csv
Outputs: heldout/results/heldout_auroc.csv, heldout/results/per_setting.csv, heldout/figures/heldout_prediction.pdf
"""
import os, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = f"{ROOT}/heldout"

# ---------------------------------------------------------------- Dyck-1 ----------------------------------------------------------------
hp = pd.read_csv(f"{ROOT}/data/transformer_head_properties.csv")
F = np.load(f"{HERE}/features/cp5.npz", allow_pickle=True); assert list(F["ids"]) == hp.id.tolist()
acc = 1 - (F["ood_probs"] >= 0.5).mean(1); assert np.allclose(acc, hp.cp5_ood_acc.values, atol=1e-3)
y = (acc >= 0.5).astype(int); n = len(y)
hand_names = list(F["hand_names"]); hand = F["hand"].astype(float)
rule_score = hand[:, [i for i, nm in enumerate(hand_names) if nm.endswith("ambi")]].max(1)          # (i) max ID hierarchical-head score
headraw = F["headraw"].astype(float); n_stats = headraw.shape[-1]                                   # (n, 3 layers, 4 heads, 24), NaN = head absent
with np.errstate(all="ignore"):
    allheads = np.concatenate([np.nan_to_num(fn(headraw.reshape(n, -1, n_stats), axis=1)) for fn in (np.nanmax, np.nanmean, np.nanmin)], 1)
    lastlayer = np.stack([headraw[i, int(hp.n_layer[i]) - 1] for i in range(n)])
    lastpool = np.concatenate([np.nan_to_num(fn(lastlayer, axis=1)) for fn in (np.nanmax, np.nanmean, np.nanmin)], 1)
generic = np.concatenate([allheads, lastpool], 1)                                                    # (ii) 144 numbers per model
hyper = pd.get_dummies(hp[["n_layer", "n_head", "wd"]].astype(str)).values.astype(float)            # (iii)
cells = (hp.n_layer.astype(str) + "_wd" + hp.wd.astype(str)).values

def predict(name, tr, te):
    if name == "rule": return rule_score[te]
    if name == "generic": return RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1).fit(generic[tr], y[tr]).predict_proba(generic[te])[:, 1]
    if name == "hyper": return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(hyper[tr], y[tr]).predict_proba(hyper[te])[:, 1]

PRED = {"rule": "ID hierarchical-head rule", "generic": "Nesting-agnostic attention statistics (random forest)", "hyper": "Hyperparameters alone (logistic regression)"}
rows, per_cell = [], []
splits = list(StratifiedShuffleSplit(n_splits=50, train_size=150, test_size=120, random_state=0).split(np.zeros(n), y))
for name in PRED:
    au = [roc_auc_score(y[te], predict(name, tr, te)) for tr, te in splits]
    rows.append(dict(setting="Dyck-1", scheme="random 150/120 splits (x50)", predictor=PRED[name], auroc_mean=np.mean(au), auroc_sd=np.std(au, ddof=1), n_folds=len(au)))
for tr, te in LeaveOneGroupOut().split(np.zeros(n), y, cells):
    cell = cells[te][0]
    if len(set(y[te])) < 2: per_cell.append(dict(cell=cell, n=len(te), scorable=False)); continue
    per_cell.append(dict(cell=cell, n=len(te), scorable=True, **{name: roc_auc_score(y[te], predict(name, tr, te)) for name in PRED}))
pc = pd.DataFrame(per_cell); sc = pc[pc.scorable]
for name in PRED:
    rows.append(dict(setting="Dyck-1", scheme="leave one (depth, weight decay) setting out", predictor=PRED[name], auroc_mean=sc[name].mean(), auroc_sd=sc[name].std(ddof=1), n_folds=len(sc)))
d = sc["rule"] - sc["hyper"]; wil = wilcoxon(d).pvalue
print(f"Dyck-1: {int(sc.shape[0])} of {len(pc)} held-out settings scorable; rule > hyperparameters in {(d > 0).sum()}/{len(d)} (mean diff {d.mean():+.3f}, Wilcoxon p={wil:.3f})")

# ------------------------------------------------------------ question formation --------------------------------------------------------
qf_path = f"{HERE}/features/qf_cp300000.npz"
if os.path.exists(qf_path):
    Q = np.load(qf_path, allow_pickle=True)
    mem = pd.read_csv(f"{ROOT}/question_formation_data/qf_matrix_aux_detector_raw_quest_membership_300000.csv").set_index("model_checkpoint")
    mem = mem.loc[[r + "__checkpoint_300000" for r in Q["runs"]]]
    yq = (mem.ood_accuracy.values >= 0.5).astype(int); nq = len(yq)
    qhand_names = list(Q["hand_names"]); qhand = Q["hand"].astype(float)
    q_rule = qhand[:, [i for i, nm in enumerate(qhand_names) if nm.endswith("matrix_prop_met") and int(nm[1]) <= 3]].max(1)   # first three layers, as in the paper
    qg_names = list(Q["generic_names"]); qgen = Q["generic"].astype(float)[:, [i for i, nm in enumerate(qg_names) if "mass_" not in nm]]  # no part-of-speech information
    rows.append(dict(setting="question formation", scheme="fixed rule, all 79 models", predictor="Main Auxiliary-Detecting head (first three layers)", auroc_mean=roc_auc_score(yq, q_rule), auroc_sd=np.nan, n_folds=1))
    au = []
    for rep in range(20):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(qgen, yq):
            au.append(roc_auc_score(yq[te], RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1).fit(qgen[tr], yq[tr]).predict_proba(qgen[te])[:, 1]))
    rows.append(dict(setting="question formation", scheme="5-fold CV (x20)", predictor="Attention statistics at the 'quest' position, no part-of-speech information (random forest)", auroc_mean=np.mean(au), auroc_sd=np.std(au, ddof=1), n_folds=len(au)))
else:
    print("no QF features found (run extract_qf_features.py); skipping question formation")

res = pd.DataFrame(rows); os.makedirs(f"{HERE}/results", exist_ok=True)
res.to_csv(f"{HERE}/results/heldout_auroc.csv", index=False); pc.to_csv(f"{HERE}/results/per_setting.csv", index=False)
pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))

# ------------------------------------------------------------------ figure --------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("whitegrid"); plt.rcParams["font.family"] = "serif"; plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]; plt.rcParams["figure.dpi"] = 300
items = [("Training\nConfiguration", "hyper"), ("Nesting-Agnostic\nAttention Stats.", "generic"), ("ID Hierarchical\nHead Rule", "rule")]
schemes = [("Random Splits", "random 150/120 splits (x50)", "#b6e6e6"), ("Unseen Hyperparameters", "leave one (depth, weight decay) setting out", "#1bb5b8")]
dy = res[res.setting == "Dyck-1"].set_index(["scheme", "predictor"])
fig, ax = plt.subplots(figsize=(7, 5.2)); h = 0.36
for i, (lab, key) in enumerate(items):
    for j, (sname, scheme, col) in enumerate(schemes):
        r = dy.loc[(scheme, PRED[key])]; yy = i + (j - 0.5) * h
        ax.barh(yy, r.auroc_mean, height=h, color=col, edgecolor="black", linewidth=1.5, label=sname if i == 0 else None, zorder=2)
        ax.errorbar(r.auroc_mean, yy, xerr=r.auroc_sd, fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)
        ax.text(min(r.auroc_mean + r.auroc_sd + 0.015, 1.02), yy, f"{r.auroc_mean:.2f}", va="center", ha="left", fontsize=17)
ax.axvline(0.5, color="black", ls=":", lw=1.5, zorder=1)
ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=20)
ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=20)
ax.set_xlabel("AUROC (Predicting Nested OOD)", fontsize=22); ax.set_ylabel("Predictor Input", fontsize=22); ax.set_ylim(-0.6, len(items) + 0.55); ax.grid(False)
for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(1.5); s.set_edgecolor("0.6")
ax.tick_params(axis="both", width=1.5, length=5)
leg = ax.legend(title="Held-Out Models", fontsize=15, title_fontsize=15, loc="upper right", frameon=True, handlelength=1.4, borderpad=0.6)
leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("black"); leg.get_frame().set_linewidth(1)
plt.tight_layout(); os.makedirs(f"{HERE}/figures", exist_ok=True); plt.savefig(f"{HERE}/figures/heldout_prediction.pdf"); print("saved heldout/figures/heldout_prediction.pdf")
