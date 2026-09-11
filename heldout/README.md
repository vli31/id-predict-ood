# Held-out prediction of OOD rule choice

Code for the "Held-out evaluation" appendix and its figure: can ID-only information predict whether a model the predictor has never seen follows the NESTED rule OOD?

| Script | Does | Needs |
|---|---|---|
| `extract_features.py` | For each of the 270 Dyck-1 Transformers at checkpoint 5: recomputes the ID hierarchical-head scores (matches `data/transformer_head_properties.csv`), 24 nesting-agnostic attention statistics per head on the 1000 ID test strings, and the ID/OOD prediction vectors. Writes `features/cp5.npz`. | `data/model_weights`, `data/model_preds` (in this repo). CPU is fine (a few minutes). |
| `extract_qf_features.py` | Same for the 79 question-formation models at the 300K checkpoint, from the `quest` position. Writes `features/qf_cp300000.npz`. | A clone of [hier_gen](https://github.com/sunnytqin/hier_gen) (`HIER_GEN`, default `~/hier_gen`) and the checkpoints (`QF_MODELS`, default `~/qf_models/<run>/{args.json,checkpoint_300000.pth}`). |
| `heldout_prediction.py` | Fits the three predictors (hierarchical-head rule; nesting-agnostic statistics with a random forest; hyperparameters alone with logistic regression) on some models and scores them on held-out models: 50 random 150/120 splits, and leaving out one (depth, weight decay) setting at a time. Question formation: 5-fold CV over the 79 models. Writes `results/heldout_auroc.csv`, `results/per_setting.csv`, `figures/heldout_prediction.pdf`. | The two feature files. |

```bash
pip install torch pandas numpy scikit-learn scipy matplotlib seaborn
python heldout/extract_features.py            # -> heldout/features/cp5.npz
python heldout/extract_qf_features.py         # optional, question formation
python heldout/heldout_prediction.py          # -> results/, figures/
```

`results/heldout_auroc.csv` holds the numbers quoted in the paper (AUROC, mean and SD over splits; 0.5 = chance).
