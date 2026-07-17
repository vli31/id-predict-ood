"""Try three ablation types for QF Subject-Binding heads:
   - zero ablation: set head output to 0
   - mean ablation: replace head output with its mean across an ID batch
   - uniform attention: force attention weights to uniform

Subject-Binding head = period -> matrix_subj > 0.3 at layer 1.
We use the 73-model 300k population matching qf_id_attention_decl_form.csv.
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

QF_ROOT = "/n/netscratch/barak_lab/Everyone/sqin/variation/qf/hier90_linear10"
DATA_DIR = "/n/home07/vrli/hier_gen/data_utils"
DATA_NAME = "question_formation_data"
OUT = "/n/home07/vrli/hier_gen/question_formation/qf_analysis"

sys.path.insert(0, "/n/home07/vrli/hier_gen")
from models.transformer_lm import TransformerLM
from data_utils.lm_dataset_helpers import read_lm_data
from util import test_continuations
from vocabulary import WordVocabulary

LAYER = 1
THRESH = 0.3
N_HEADS = 8
AUXES = ["doesn't", "does", "do", "don't"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelWrapper:
    def __init__(self, m): self.model = m; self.encoder_sos = m.encoder_sos
    def __call__(self, *a, **k): return self.model(*a, **k)
    def __getattr__(self, n): return getattr(self.model, n)


def load_model(cp):
    raw = torch.load(cp, map_location=device, weights_only=False)
    sd = raw["model_state_dict"]
    args = json.load(open(os.path.join(os.path.dirname(cp), "args.json")))
    vs = sd["model.input_embedding.weight"].shape[0]
    ss = sd["model.input_embedding.weight"].shape[1]
    model = TransformerLM(
        n_input_tokens=vs - 2, state_size=ss,
        tied_embedding=args.get("tied_embedding", True),
        num_encoder_layers=args["encoder_n_layers"],
        nheads=args.get("n_heads", 8),
        ff_multiplier=args.get("ff_multiplier", 4))
    model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()})
    return ModelWrapper(model.to(device).eval())


class AblationHook:
    """Supports mode='zero', 'mean', 'uniform' (attention)."""

    def __init__(self, model, layer_idx, heads, mode="zero", head_mean=None):
        self.model = model
        self.layer_idx = layer_idx
        self.heads = list(heads)
        self.mode = mode
        self.head_dim = model.state_size // N_HEADS
        self.head_mean = head_mean  # (n_heads, head_dim) tensor

    def __enter__(self):
        layer = self.model.trafo.encoder.layers[self.layer_idx]
        sa = layer.self_attn
        ablated = self.heads
        mode = self.mode
        head_mean = self.head_mean
        n_heads, head_dim = N_HEADS, self.head_dim

        def patched(curr_state, attend_to, mask, need_weights=False):
            k, v = sa.transform_data(attend_to, sa.data_to_kv, 2)
            (q,) = sa.transform_data(curr_state, sa.data_to_q, 1)
            n_batch = curr_state.shape[0]
            seq_len = curr_state.shape[1]
            logits = torch.bmm(q, k.transpose(1, 2))
            if mode == "uniform":
                # Setting logits to zero before the masked softmax yields a
                # uniform distribution over allowed (unmasked) positions.
                logits_for_softmax = torch.zeros_like(logits)
            else:
                logits_for_softmax = logits * sa.scale
            scores = sa._masked_softmax(logits_for_softmax, mask)
            scores = sa.dropout(scores)
            attn_out = torch.bmm(scores, v)
            attn_out = attn_out.view(n_batch, n_heads, seq_len, head_dim)
            if mode == "zero":
                for h in ablated:
                    attn_out[:, h] = 0
            elif mode == "mean":
                # Replace ablated heads' outputs with the per-head mean over
                # ID data (broadcast across batch and sequence positions).
                if head_mean is not None:
                    for h in ablated:
                        attn_out[:, h] = head_mean[h]
            # mode == 'uniform': we already changed the attention; do not modify
            # head outputs further.
            attn_out = (attn_out.permute(0, 2, 1, 3).contiguous()
                        .view(n_batch, seq_len, -1))
            data = sa.multi_head_merge(attn_out)
            if need_weights:
                w = scores.view(n_batch, n_heads, seq_len, seq_len)
                return data, w.mean(1)
            return data

        sa._orig_fwd = sa.forward
        sa.forward = patched
        self._sa = sa
        return self

    def __exit__(self, *a):
        self._sa.forward = self._sa._orig_fwd
        del self._sa._orig_fwd


def compute_head_mean_outputs(model, layer_idx, prefixes, tokenizer, in_vocab):
    """Hook the layer during a normal test_continuations call to capture head
    outputs, then average across batch and sequence positions to get a single
    (n_heads, head_dim) tensor representing the dataset-mean head output."""
    layer = model.trafo.encoder.layers[layer_idx]
    sa = layer.self_attn
    head_dim = model.state_size // N_HEADS

    captured = {"sum": None, "count": 0}

    def patched(curr_state, attend_to, mask, need_weights=False):
        k, v = sa.transform_data(attend_to, sa.data_to_kv, 2)
        (q,) = sa.transform_data(curr_state, sa.data_to_q, 1)
        n_batch = curr_state.shape[0]
        seq_len = curr_state.shape[1]
        logits = torch.bmm(q, k.transpose(1, 2))
        scores = sa._masked_softmax(logits * sa.scale, mask)
        scores = sa.dropout(scores)
        attn_out = torch.bmm(scores, v)
        attn_out_h = attn_out.view(n_batch, N_HEADS, seq_len, head_dim)
        # accumulate sum over batch and seq dims
        s = attn_out_h.detach().sum(dim=(0, 2))  # (n_heads, head_dim)
        if captured["sum"] is None:
            captured["sum"] = s
        else:
            captured["sum"] = captured["sum"] + s
        captured["count"] += n_batch * seq_len
        attn_out = (attn_out_h.permute(0, 2, 1, 3).contiguous()
                    .view(n_batch, seq_len, -1))
        data = sa.multi_head_merge(attn_out)
        if need_weights:
            w = scores.view(n_batch, N_HEADS, seq_len, seq_len)
            return data, w.mean(1)
        return data

    sa._orig_fwd = sa.forward
    sa.forward = patched
    try:
        with torch.no_grad():
            _ = test_continuations(tokenizer, model, prefixes, 0)
        head_mean = captured["sum"] / max(captured["count"], 1)
        return head_mean
    finally:
        sa.forward = sa._orig_fwd
        del sa._orig_fwd


def main():
    sm = pd.read_csv(os.path.join(OUT, "qf_ood_model_summary.csv"))  # 73 models, 300k
    attn = pd.read_csv(os.path.join(OUT, "qf_id_attention_decl_form.csv"))

    def sb_heads(mn):
        sub = attn[(attn.model_name == mn) & (attn.layer == LAYER)]
        return sorted(sub.loc[sub["att_period_to_matrix_subj"] > THRESH, "head"].tolist())

    has_sb_models = [m for m in sm.model_name if sb_heads(m)]
    print(f"Models with Subject-Binding head at L{LAYER}: {len(has_sb_models)}")

    in_sents, _ = read_lm_data(["train", "val", "test"], data_name=DATA_NAME, data_dir=DATA_DIR)
    in_vocab = WordVocabulary(in_sents, split_punctuation=False)

    meta = pd.read_csv(os.path.join(OUT, "qf_ood_example_meta.csv"))
    matrix_auxs = meta["matrix_aux"].tolist()
    prefixes = []
    for full_input in meta["full_input"]:
        ws = full_input.split()
        idx = ws.index("quest")
        prefixes.append(" ".join(ws[:idx + 1]))
    aux_ids = [in_vocab[w] for w in AUXES]

    # ID prefixes for computing the per-head mean. We use declarative prefixes.
    id_prefixes = prefixes[:200]  # subsample for speed

    rows = []
    for mn in tqdm(has_sb_models, desc="ablation_types"):
        ckpt_dir = os.path.join(QF_ROOT, mn)
        c300 = os.path.join(ckpt_dir, "checkpoint_300000.pth")
        if not os.path.exists(c300):
            continue

        model = load_model(c300)
        sb = sb_heads(mn)

        def tokenizer(s, _m=model):
            return [_m.encoder_sos] + in_vocab(s)

        def eval_ood(_m=model):
            probs = test_continuations(tokenizer, _m, prefixes, 0).cpu().numpy()
            sub = probs[:, aux_ids]
            pred_idx = sub.argmax(axis=1)
            pred_words = [AUXES[i] for i in pred_idx]
            return float(np.mean([pw == m for pw, m in zip(pred_words, matrix_auxs)]))

        baseline = eval_ood()

        # Mean head output computed on ID prefixes
        head_mean = compute_head_mean_outputs(model, LAYER, id_prefixes, tokenizer, in_vocab)

        deltas = {}
        for mode in ("zero", "mean", "uniform"):
            with AblationHook(model, LAYER, sb, mode=mode, head_mean=head_mean):
                ood = eval_ood()
            deltas[mode] = ood - baseline

        rows.append({
            "model_name": mn,
            "sb_heads": str(sb),
            "ood_baseline": baseline,
            "delta_zero": deltas["zero"],
            "delta_mean": deltas["mean"],
            "delta_uniform": deltas["uniform"],
        })
        print(f"  {mn}: base={baseline:.3f} zero={deltas['zero']:+.3f} "
              f"mean={deltas['mean']:+.3f} uniform={deltas['uniform']:+.3f}")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "qf_subjbind_ablation_types.csv"), index=False)
    print(f"\nSaved {len(df)} rows.")
    print(df[["delta_zero", "delta_mean", "delta_uniform"]].describe().round(4))


if __name__ == "__main__":
    main()
