# Paper figures

Scripts that produce every figure in the paper. Each script writes its PDF
(and a PNG copy) to the paper's `images/` directory; run from the repo root
unless otherwise noted.

## Figure 1 (TSNE clusters)

| Panel | Source | What it shows |
|---|---|---|
| 1a — Dyck-1 | (existing) `images/v2tsne_plot_label.pdf` | t-SNE of 270 Dyck-1 models' OOD output probabilities. Three hulls: `Nested`, `Equal-Count`, `First-Symbol`. Built by the original analysis pipeline; included as a static PDF. |
| 1b — Question formation | [`fig1b_qf_tsne.py`](fig1b_qf_tsne.py) | t-SNE of 73 QF models' OOD output probabilities at the 300k checkpoint, perplexity 20. Two hulls: `Linear` (n=22) and `Hierarchical` (n=51, three-cluster KMeans solution merging the Intermediate sub-cluster). |

## Figure 2 (head-type boxplots)

| Panel | Source | What it shows |
|---|---|---|
| 2a — Dyck-1 | [`fig2a_dyck_split.py`](fig2a_dyck_split.py) | OOD accuracy of 2- and 3-layer Dyck models split by the presence of `Sign-Matching` / `Violation-Detecting` / `Both` / `Neither` heads (classification by **OOD** attention patterns, matching Fig 3's coloring). |
| 2b — Question formation | [`fig2b_qf_split.py`](fig2b_qf_split.py) | OOD hierarchical accuracy split by presence of a Subject-Binding head (period $\to$ matrix subject, layer 1, threshold $0.3$). |

## Figure 3 (Dyck ablation scatter)

[`fig3_dyck_ablation_scatter.py`](fig3_dyck_ablation_scatter.py) — OOD accuracy before vs. after uniform attention ablation for 2- and 3-layer Dyck models, colored by OOD head type. Points below the diagonal = ablation hurt OOD; points above = ablation helped.

## Appendix figures

| Script | Output |
|---|---|
| [`apx_attention_examples.py`](apx_attention_examples.py) | `images/attention_activations_violation.pdf`: example attention rows for a Sign-Matching vs. Violation-Detecting head on a Dyck-1 input. |
| [`qf_ablation_types.py`](qf_ablation_types.py) | `qf_subjbind_ablation_types.csv`: zero / mean / uniform-attention ablation deltas for the 20 QF models that carry a Subject-Binding head. Used in Appendix Table 2 (`apx:qf-ablation`). Requires the QF model checkpoints in `hier_gen/`. |
| [`apx_qf_stats.py`](apx_qf_stats.py) | Verification script for the appendix numbers (head population sizes, Mann-Whitney $p$ values, cluster x head crosstab). |

## How the numbers in the paper were verified

- Dyck precision/recall of the ID-hierarchical-head detector (Appendix
  `apx:classifier`): computed in `apx_qf_stats.py` (reuses the Dyck head
  CSV in `data/transformer_head_properties.csv`).
- QF $n=73$ population, Mann-Whitney $p=0.037$ for Subject-Binding: see
  `apx_qf_stats.py`.
- QF zero / mean / uniform ablation deltas: see `qf_ablation_types.py`.

## Reproducing figures

```bash
# All paths assume CWD = repo root (id-predict-ood/).
python paper_figures/fig1b_qf_tsne.py
python paper_figures/fig2a_dyck_split.py
python paper_figures/fig2b_qf_split.py
python paper_figures/fig3_dyck_ablation_scatter.py
python paper_figures/apx_attention_examples.py
python paper_figures/apx_qf_stats.py
```

Each script prints the summary numbers to stdout and writes PDFs to
`/n/home07/vrli/emnlp_submission/images/`.
