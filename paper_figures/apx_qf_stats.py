"""
Verify QF Part (2) and Part (3) results on the 73-model 300k population.

Part (2): Do Matrix-Aux and Subject-Binding ID heads predict higher OOD hier acc?
Part (3): Does ablating those heads reduce OOD hier (causal alignment with correlational)?
"""
import os
import numpy as np
import pandas as pd

ROOT = "/n/home07/vrli/hier_gen/question_formation/qf_analysis"

import sys
CKPT = sys.argv[1] if len(sys.argv) > 1 else "300k"
suffix = "_600k" if CKPT == "600k" else ""
sm = pd.read_csv(f"{ROOT}/qf_ood_model_summary{suffix}.csv")
attn = pd.read_csv(f"{ROOT}/qf_id_attention_decl_form{suffix}.csv")
abl = pd.read_csv(f"{ROOT}/qf_per_head_ablation.csv")  # all available

# Layer for each head type
T1_LAYER, T2_LAYER = 4, 3
T1_THRESH, T2_THRESH = 0.5, 0.3  # matches run_two_type_ablation.py


def has_matrix_aux(model_name):
    """Returns True if model has a Matrix-Aux head."""
    sub = attn[(attn.model_name == model_name) & (attn.layer == T1_LAYER)]
    return (sub["att_period_to_matrix_aux"] > T1_THRESH).any()


def has_subj_binding(model_name):
    """Returns True if model has a Subject-Binding (period->matrix_subj) head."""
    sub = attn[(attn.model_name == model_name) & (attn.layer == T2_LAYER)]
    return (sub["att_period_to_matrix_subj"] > T2_THRESH).any()


def matrix_aux_heads(model_name):
    sub = attn[(attn.model_name == model_name) & (attn.layer == T1_LAYER)]
    return sorted(sub.loc[sub["att_period_to_matrix_aux"] > T1_THRESH, "head"].tolist())


def subj_binding_heads(model_name):
    sub = attn[(attn.model_name == model_name) & (attn.layer == T2_LAYER)]
    return sorted(sub.loc[sub["att_period_to_matrix_subj"] > T2_THRESH, "head"].tolist())


sm["has_ma"] = sm.model_name.apply(has_matrix_aux)
sm["has_sb"] = sm.model_name.apply(has_subj_binding)


def category(r):
    if r.has_ma and r.has_sb:
        return "Both"
    if r.has_ma:
        return "Matrix-Aux"
    if r.has_sb:
        return "Subject-Binding"
    return "Neither"


sm["cat"] = sm.apply(category, axis=1)

print("=" * 60)
print(f"QF 300k population: n = {len(sm)} models")
print("=" * 60)
print("\n--- Part (2): ID head presence predicts OOD hier accuracy ---")
print(sm.groupby("cat").ood_acc.agg(["count", "mean", "median", "std"]).round(3))

# Mann-Whitney test: has_ma vs not
from scipy.stats import mannwhitneyu
a = sm[sm.has_ma].ood_acc
b = sm[~sm.has_ma].ood_acc
u, p = mannwhitneyu(a, b, alternative="greater")
print(f"\nMatrix-Aux present vs absent: U={u}, p_one-sided={p:.4e}")
print(f"  has Matrix-Aux: n={len(a)}, mean={a.mean():.3f}")
print(f"  no  Matrix-Aux: n={len(b)}, mean={b.mean():.3f}")

a = sm[sm.has_sb].ood_acc
b = sm[~sm.has_sb].ood_acc
u, p = mannwhitneyu(a, b, alternative="greater")
print(f"\nSubject-Binding present vs absent: U={u}, p_one-sided={p:.4e}")
print(f"  has Subject-Binding: n={len(a)}, mean={a.mean():.3f}")
print(f"  no  Subject-Binding: n={len(b)}, mean={b.mean():.3f}")

# Either head type
sm["has_either"] = sm.has_ma | sm.has_sb
a = sm[sm.has_either].ood_acc
b = sm[~sm.has_either].ood_acc
u, p = mannwhitneyu(a, b, alternative="greater")
print(f"\nEither type present vs neither: U={u}, p_one-sided={p:.4e}")
print(f"  either: n={len(a)}, mean={a.mean():.3f}")
print(f"  neither: n={len(b)}, mean={b.mean():.3f}")

print("\n--- Part (3): Ablation aligned with correlational signal? ---")
# qf_two_type_ablation.csv has set-ablation deltas for T1 and T2 heads
ta = pd.read_csv(f"{ROOT}/qf_two_type_ablation.csv")
ta = ta.merge(sm[["model_name", "has_ma", "has_sb", "cat"]], on="model_name", how="left")

ma_models = ta[ta.has_ma & (ta.n_t1 > 0)]
sb_models = ta[ta.has_sb & (ta.n_t2 > 0)]

print("\nMatrix-Aux (T1) ablation delta on hier OOD acc:")
print(f"  models tested: {len(ma_models)}")
print(f"  mean delta: {ma_models.delta_t1.mean():+.4f}")
print(f"  median delta: {ma_models.delta_t1.median():+.4f}")
print(f"  # models where ablation reduces OOD by > 0.02: {(ma_models.delta_t1 < -0.02).sum()}")
print(f"  # models where ablation increases OOD by > 0.02: {(ma_models.delta_t1 > 0.02).sum()}")
print(f"  # models with |delta| <= 0.02: {(ma_models.delta_t1.abs() <= 0.02).sum()}")

print("\nSubject-Binding (T2) ablation delta on hier OOD acc:")
print(f"  models tested: {len(sb_models)}")
print(f"  mean delta: {sb_models.delta_t2.mean():+.4f}")
print(f"  median delta: {sb_models.delta_t2.median():+.4f}")
print(f"  # models where ablation reduces OOD by > 0.02: {(sb_models.delta_t2 < -0.02).sum()}")
print(f"  # models where ablation increases OOD by > 0.02: {(sb_models.delta_t2 > 0.02).sum()}")
print(f"  # models with |delta| <= 0.02: {(sb_models.delta_t2.abs() <= 0.02).sum()}")

# Sign test
from scipy.stats import wilcoxon
if len(ma_models) >= 5:
    stat, p = wilcoxon(ma_models.delta_t1, alternative="less")
    print(f"\nWilcoxon (Matrix-Aux delta<0): stat={stat}, p={p:.4f}")
if len(sb_models) >= 5:
    stat, p = wilcoxon(sb_models.delta_t2, alternative="less")
    print(f"Wilcoxon (Subject-Binding delta<0): stat={stat}, p={p:.4f}")

# Save for reuse
sm.to_csv("/n/home07/vrli/emnlp_submission/qf_300k_categorization.csv", index=False)
print("\nSaved categorization to qf_300k_categorization.csv")
