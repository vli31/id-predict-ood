"""QF Fig 3 panel: Subject-Binding head (single predictor) — 73-model 300k pop."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams['figure.dpi'] = 300

ROOT = "/n/home07/vrli/hier_gen/question_formation/qf_analysis"
sm = pd.read_csv(f"{ROOT}/qf_ood_model_summary.csv")
attn = pd.read_csv(f"{ROOT}/qf_id_attention_decl_form.csv")

# Subject-Binding: period -> matrix_subj at layer 1, threshold > 0.3
T2_LAYER, T2_THRESH = 1, 0.3


def has_sb(mn):
    sub = attn[(attn.model_name == mn) & (attn.layer == T2_LAYER)]
    return bool((sub["att_period_to_matrix_subj"] > T2_THRESH).any())


sm["has_sb"] = sm.model_name.apply(has_sb)
sm["Head"] = sm.has_sb.map({True: "Subject-Binding", False: "Other"})

a = sm[sm.has_sb].ood_acc
b = sm[~sm.has_sb].ood_acc
u, p = mannwhitneyu(a, b, alternative="greater")
print(f"n_present = {len(a)} (mean {a.mean():.3f}), n_absent = {len(b)} (mean {b.mean():.3f})")
print(f"one-sided Mann-Whitney p = {p:.4f}")

color_map = {"Subject-Binding": "#2072b8", "Other": "#C4C4C4"}
sm["Head"] = pd.Categorical(sm["Head"], categories=["Subject-Binding", "Other"], ordered=True)
data = sm[["Head", "ood_acc"]].sort_values("Head")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x="ood_acc", y="Head", data=data, orient="h",
            hue="Head", palette=color_map, legend=False, linewidth=1.5,
            width=0.5, fliersize=0, ax=ax)
sns.stripplot(x="ood_acc", y="Head", data=data, orient="h",
              color="black", size=4.5, alpha=0.55, jitter=0.15, ax=ax)
ax.set_ylabel("ID Hierarchical Head", fontsize=24)
ax.set_xlabel("OOD Accuracy", fontsize=28)
ax.set_xticks([0, 0.5, 1])
ax.set_xlim(-0.02, 1.02)
ax.tick_params(axis="x", width=1.5, length=5)
ax.tick_params(axis="both", which="major", labelsize=22)
ax.grid(False)
ax.spines["bottom"].set_visible(True)
ax.xaxis.set_ticks_position("bottom")
plt.margins(x=0)
plt.tight_layout()

out = "/n/home07/vrli/emnlp_submission/images/qf_split_by_head_type.pdf"
plt.savefig(out, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print("Saved", out)
sm.to_csv("/n/home07/vrli/emnlp_submission/qf_300k_sb_categorization.csv", index=False)
