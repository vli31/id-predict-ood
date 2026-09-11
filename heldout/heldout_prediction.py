"""
Held-out prediction of OOD rule choice from ID-only information (Appendix "Held-out evaluation", Figure "heldout_prediction").

Target: does a held-out model lie in the NESTED region of the paper's Fig. 2a?  Clusters are k-means (k=4) on each model's OOD output
probabilities, the same vectors as the t-SNE; the NESTED region is everything outside the EQUAL-COUNT cluster (lowest mean OOD accuracy)
and the FIRST-SYMBOL cluster (highest agreement with the first-symbol heuristic).  Predictors see ID data only:
  (i)   ID hierarchical-head rule: any head, any layer, tracks violations on >= 80% of relevant ID sequences (score = max over heads);
  (ii)  five nesting-agnostic attention statistics, each summarized by its maximum, mean and minimum over all heads, measured on the 1000 ID
        test strings: centre of mass of the EOS attention along the string; EOS attention mass on the last tenth of the string; largest
        EOS attention weight; EOS attention to itself; self-attention averaged over all positions.  Logistic regression (and random forest).
  (iii) baseline: logistic regression on hyperparameters alone (depth, width, weight decay).
Evaluation: 50 random splits (150 train / 120 test) and leave-one-(depth, weight decay)-setting-out (six of nine settings contain
both classes and can be scored).  Question formation: target = the HIERARCHICAL region of Fig. 2b, i.e. outside the LINEAR cluster (k-means, k=3, on P(hierarchical) over the OOD
questions); same rule and the same five
statistics at the 'quest' position, 5-fold CV x 20 over the 79 models.

Inputs : heldout/features/cp5.npz (from extract_features.py), heldout/features/qf_cp300000.npz (from extract_qf_features.py, optional),
         data/transformer_head_properties.csv, question_formation_data/qf_matrix_aux_detector_raw_quest_membership_300000.csv,
         question_formation_data/qf_p_hier_by_model_checkpoint_300000.csv
Outputs: heldout/results/heldout_auroc.csv, heldout/results/per_setting.csv, heldout/figures/heldout_prediction.pdf
"""
import os, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); HERE = f"{ROOT}/heldout"

# ---------------------------------------------------------------- Dyck-1 ----------------------------------------------------------------
hp = pd.read_csv(f"{ROOT}/data/transformer_head_properties.csv")
F = np.load(f"{HERE}/features/cp5.npz", allow_pickle=True); assert list(F["ids"]) == hp.id.tolist()
acc = 1 - (F["ood_probs"] >= 0.5).mean(1); assert np.allclose(acc, hp.cp5_ood_acc.values, atol=1e-3)
ood_strings = pd.read_csv(f"{ROOT}/data/model_preds/ood_data_preds.csv", usecols=["string"]).string
first_symbol = (((F["ood_probs"] >= 0.5) == (~ood_strings.str.startswith(")").values)[None, :]).mean(1))   # agreement with the FIRST-SYMBOL heuristic
lab = KMeans(4, n_init=50, random_state=0).fit_predict(F["ood_probs"].astype(float))                  # clusters in OOD-output space (paper Fig. 2a)
eq_count = min(range(4), key=lambda c: acc[lab == c].mean()); first_sym = max(range(4), key=lambda c: first_symbol[lab == c].mean())
y = (~np.isin(lab, [eq_count, first_sym])).astype(int); n = len(y)                                  # NESTED region = outside the EQUAL-COUNT and FIRST-SYMBOL clusters
print(f"Dyck-1 clusters: {[int((lab == c).sum()) for c in range(4)]} models, mean OOD acc {[round(float(acc[lab == c].mean()), 2) for c in range(4)]}; EQUAL-COUNT={eq_count}, FIRST-SYMBOL={first_sym}; NESTED region n={y.sum()}")
hand_names = list(F["hand_names"]); hand = F["hand"].astype(float)
rule_score = hand[:, [i for i, nm in enumerate(hand_names) if nm.endswith("ambi")]].max(1)          # (i) max ID hierarchical-head score
headraw = F["headraw"].astype(float); stats = [str(x) for x in F["headraw_names"]]                # (n, 3 layers, 4 heads, 24), NaN = head absent
FIVE = ["relpos", "prof8", "maxw", "a_self", "allq_self"]                                              # (ii) the five statistics (see docstring)
POOLS = ["all_max", "all_mean", "all_min"]                                                              # summary over heads: maximum, mean and minimum over all heads (15 numbers per model)
def pool(raw, n_layer, names, sel, pools):
    N = raw.shape[0]; idx = [names.index(x) for x in sel]
    with np.errstate(all="ignore"):
        last = np.stack([raw[i, int(n_layer[i]) - 1] for i in range(N)])[:, :, idx]; allh = raw.reshape(N, -1, raw.shape[-1])[:, :, idx]
        parts = {"last_max": np.nanmax(last, 1), "last_mean": np.nanmean(last, 1), "last_min": np.nanmin(last, 1), "all_max": np.nanmax(allh, 1), "all_mean": np.nanmean(allh, 1), "all_min": np.nanmin(allh, 1)}
    return np.nan_to_num(np.concatenate([parts[q] for q in pools], 1))
generic = pool(headraw, hp.n_layer.values, stats, FIVE, POOLS)                                          # (n, 5 x len(POOLS))
hyper = pd.get_dummies(hp[["n_layer", "n_head", "wd"]].astype(str)).values.astype(float)            # (iii)
cells = (hp.n_layer.astype(str) + "_wd" + hp.wd.astype(str)).values

def predict(name, tr, te):
    if name == "rule": return rule_score[te]
    if name == "generic": return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(generic[tr], y[tr]).predict_proba(generic[te])[:, 1]
    if name == "generic_rf": return RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1).fit(generic[tr], y[tr]).predict_proba(generic[te])[:, 1]
    if name == "hyper": return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(hyper[tr], y[tr]).predict_proba(hyper[te])[:, 1]

PRED = {"rule": "ID hierarchical-head rule", "generic": "Five attention statistics (logistic regression)", "generic_rf": "Five attention statistics (random forest)", "hyper": "Hyperparameters alone (logistic regression)"}
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
    ph = pd.read_csv(f"{ROOT}/question_formation_data/qf_p_hier_by_model_checkpoint_300000.csv")
    Pq = ph[[r + "__checkpoint_300000" for r in Q["runs"]]].values.T                                       # P(hierarchical) on the 10000 OOD questions
    labq = KMeans(3, n_init=50, random_state=0).fit_predict(Pq); linear = int(np.argmin([Pq[labq == c].mean() for c in range(3)]))
    yq = (labq != linear).astype(int); nq = len(yq)                                                     # HIERARCHICAL region of Fig. 2b = outside the LINEAR cluster
    print(f"QF clusters: {[int((labq == c).sum()) for c in range(3)]}, mean P(hier) {[round(float(Pq[labq == c].mean()), 2) for c in range(3)]}; LINEAR={linear}; hierarchical region n={yq.sum()}")
    qhand_names = list(Q["hand_names"]); qhand = Q["hand"].astype(float)
    q_rule = qhand[:, [i for i, nm in enumerate(qhand_names) if nm.endswith("matrix_prop_met") and int(nm[1]) <= 3]].max(1)   # first three layers, as in the paper
    qstats = [str(x) for x in Q["headraw_names"]]; qraw = Q["headraw"].astype(float)                          # (79, 6 layers, 8 heads, stats)
    qgen = pool(qraw, np.full(nq, qraw.shape[1]), qstats, [{"a_bos": "a_sos"}.get(x, x) for x in FIVE], POOLS)   # same five statistics at the 'quest' position
    rows.append(dict(setting="question formation", scheme="fixed rule, all 79 models", predictor="Main Auxiliary-Detecting head (first three layers)", auroc_mean=roc_auc_score(yq, q_rule), auroc_sd=np.nan, n_folds=1))
    for label, mk in [("Five attention statistics (logistic regression)", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))), ("Five attention statistics (random forest)", lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=1))]:
        au = [roc_auc_score(yq[te], mk().fit(qgen[tr], yq[tr]).predict_proba(qgen[te])[:, 1]) for rep in range(20) for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(qgen, yq)]
        rows.append(dict(setting="question formation", scheme="5-fold CV (x20)", predictor=label, auroc_mean=np.mean(au), auroc_sd=np.std(au, ddof=1), n_folds=len(au)))
else:
    print("no QF features found (run extract_qf_features.py); skipping question formation")

res = pd.DataFrame(rows); os.makedirs(f"{HERE}/results", exist_ok=True)
res.to_csv(f"{HERE}/results/heldout_auroc.csv", index=False); pc.to_csv(f"{HERE}/results/per_setting.csv", index=False)
pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))

# ------------------------------------------------------------------ figure --------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("whitegrid"); plt.rcParams["font.family"] = "serif"; plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]; plt.rcParams["figure.dpi"] = 300
items = [("Training\nConfiguration", "hyper"), ("Five Attention\nStatistics", "generic_rf"), ("ID Hierarchical\nHead Rule", "rule")]
schemes = [("Random Splits", "random 150/120 splits (x50)", "#b6e6e6"), ("Unseen Hyperparameters", "leave one (depth, weight decay) setting out", "#1bb5b8")]
dy = res[res.setting == "Dyck-1"].set_index(["scheme", "predictor"])
fig, ax = plt.subplots(figsize=(7, 5.2)); h = 0.36
for i, (lab, key) in enumerate(items):
    for j, (sname, scheme, col) in enumerate(schemes):
        r = dy.loc[(scheme, PRED[key])]; yy = i + (j - 0.5) * h
        ax.barh(yy, r.auroc_mean, height=h, color=col, edgecolor="black", linewidth=1.5, label=sname if i == 0 else None, zorder=2)
        hi = min(r.auroc_sd, 1.0 - r.auroc_mean)                                                          # AUROC cannot exceed 1
        ax.errorbar(r.auroc_mean, yy, xerr=[[r.auroc_sd], [hi]], fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)
        ax.text(r.auroc_mean + hi + 0.015, yy, f"{r.auroc_mean:.2f}", va="center", ha="left", fontsize=17)
ax.axvline(0.5, color="black", ls=":", lw=1.5, zorder=1)
ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=20)
ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=20)
ax.set_xlabel("AUROC (Predicting Nested Cluster)", fontsize=22); ax.set_ylabel("Predictor Input", fontsize=22); ax.set_ylim(-0.6, len(items) + 0.55); ax.grid(False)
for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(1.5); s.set_edgecolor("0.6")
ax.tick_params(axis="both", width=1.5, length=5)
leg = ax.legend(title="Held-Out Models", fontsize=15, title_fontsize=15, loc="upper right", frameon=True, handlelength=1.4, borderpad=0.6)
leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("black"); leg.get_frame().set_linewidth(1)
plt.tight_layout(); os.makedirs(f"{HERE}/figures", exist_ok=True); plt.savefig(f"{HERE}/figures/heldout_prediction.pdf"); print("saved heldout/figures/heldout_prediction.pdf")
