import pandas as pd
import os
from utils.data import BracketsDataset, SimpleTokenizer
from torch.utils.data import DataLoader
from run_model_utils import run_model_on_datapoint, get_models_from, INDIST_PATH, OOD_PATH, TRAIN_PATH
import pickle

tokenizer = SimpleTokenizer(alphabet='()')

INDIST_PATH = "indist_data_preds.csv"

def record_attn(df, checkpoint, ds="train"):
    assert ds in ["train", "indist", "ood"]
    if ds == "train":
        data = pd.read_csv(TRAIN_PATH)
    elif ds == "indist":
        data = pd.read_csv(INDIST_PATH)
    elif ds == "ood":
        data = pd.read_csv(OOD_PATH)

    models = get_models_from(df)

    for run_id in models.keys():
        path = f"attn_wts_saved/{run_id}_cp{checkpoint}_{ds}.pkl"
        if os.path.exists(path):
            print(f"Attention weights already saved for run_id: {run_id}")
            continue
        model = models[run_id]
        dl = DataLoader(BracketsDataset(data), batch_size=1, shuffle=False)
        
        all_attention_weights = {}
        
        for i, (input_strings, inputs, _) in enumerate(dl):
            tokens = inputs[0]
            attention_weights = run_model_on_datapoint(model, tokens)
            
            # Store attention weights and input string for each datapoint
            all_attention_weights[f'datapoint_{i}'] = {
                'attention_weights': attention_weights,
                'input_string': input_strings[0]
            }

        # Save all attention weights to a pickle file
        with open(path, 'wb') as f:
            pickle.dump(all_attention_weights, f)

    print(f"Attention weights saved to {path}")


transformer_df_all = pd.read_csv(f"data/transformers_sweep_data_cutoffs_vecs.csv")
incl_lr = [1e-4] # 3e-4
transformer_df = transformer_df_all[transformer_df_all["lr"].isin(incl_lr)] 
record_attn(transformer_df, 5, "indist")

def read_attn(run_id, id_or_ood, checkpoint):
    cp_name = '' if checkpoint == 5 else f'cp{checkpoint}_'
    
    # Construct the file path
    file_name = f"{run_id}_{cp_name}{id_or_ood}.pkl"
    path = os.path.join("attn_wts_saved", file_name)
    
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"No attention weights file found for run_id: {run_id}")
    
    # Read the pickle file
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def get_tokens(input_as_string):
    tokens = tokenizer.tokenize(input_as_string)[0]
    return tokens

def get_non_pad_indices(tokens):
    # Create a mask for non-pad tokens
    mask = tokens != 1  # Assuming 1 is the pad token
    non_pad_indices = mask.nonzero().squeeze()
    return non_pad_indices

def get_attn_by_head_and_index(attn_weights, layer, head, token_idx, tokens):
    non_pad_indices = get_non_pad_indices(tokens)
    layer_head_key = f"layer_{layer}_head_{head}"
    if layer_head_key not in attn_weights:
        raise ValueError(f"Layer {layer} head {head} not found in attention weights")
    attn = attn_weights[layer_head_key]

    attn_np = attn.cpu().numpy()
    attn_np = attn_np[non_pad_indices][:, non_pad_indices]
    
    final_attn = attn_np[token_idx]
    
    return final_attn

def classify_attn_head(runs_df_filename, layers=[1, 2, 3], token_idx=-1, checkpoint=5, indist_or_ood="ood"):
    runs_df = pd.read_csv(runs_df_filename)
    prefer_vals = {'sign': {}}

    def prefers_same_depth_sign(string, attns):
        depths = []
        depth = 0
        for char in string:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            depths.append(depth)
        neg_depth_attns = [attns[i] for i in range(len(string)) if depths[i] < 0]
        nonneg_depth_attns = [attns[i] for i in range(len(string)) if depths[i] >= 0]
        if len(neg_depth_attns) == 0 or len(nonneg_depth_attns) == 0:
            return None
        if depths[-1] < 0: # ends with neg depth
            return min(neg_depth_attns) > max(nonneg_depth_attns)
        else: # ends with nonneg depth
            return max(neg_depth_attns) < min(nonneg_depth_attns)


    for i, row in runs_df.iterrows():
        
        run_id = row['id']
        data = read_attn(run_id, indist_or_ood, checkpoint=checkpoint)
        assert len(data) == 1000, f"Data length is {len(data)} for run_id: {run_id}"

        for key in prefer_vals.keys():
            prefer_vals[key][run_id] = [[[] for _ in range(4)] for _ in layers]

        for layer in [x for x in layers if x <= row['n_layer']]:

            for datapt in range(len(data)):
                # get attention weights for datapoint
                attention_weights = data[f'datapoint_{datapt}']['attention_weights']
                string = data[f'datapoint_{datapt}']['input_string']
                
                # get token indices exluding padding tokens
                non_pad_indices = get_non_pad_indices(string)

                layer_idx = layer - 1
                for head_idx in range(row['n_head']):
                    head = head_idx + 1
                    layer_head_key = f"layer_{layer}_head_{head}"
                    if layer_head_key not in attention_weights:
                        continue
                    # get attention row for final token (or, token at index token_idx)
                    # for linear regression experiment omit the *second* indexing into non_pad_indices, probably?
                    # e.g. attns = attention_weights[layer_head_key][non_pad_indices][token_idx]
                    # check dimensions - should end up with a 1d tensor
                    attns = get_attn_by_head_and_index(attention_weights, layer, head, token_idx, non_pad_indices)
                    # exclude BOS and EOS tokens when classifying head
                    # should not be used for linear regression experiment
                    attns = attns[1:len(string)+1]

                    prefers = {}
                    prefers['sign'] = prefers_same_depth_sign(string, attns)
                    # prefers['open'], prefers['close'] = prefers_open_or_close(string, attns)
                    # prefers['neg'], prefers['nonneg'] = prefers_nonneg_or_neg_depth(string, attns)
                    # prefers['first'] = prefers_first_char_to_other_instances(string, attns)

                    for ht in prefer_vals.keys():
                        if prefers[ht] is not None:
                            prefer_vals[ht][run_id][layer_idx][head_idx].append(prefers[ht])


        for ht in prefer_vals.keys():
            for head_idx in range(row['n_head']):
                for layer_idx in range(row['n_layer']):
                    val_ls = prefer_vals[ht][run_id][layer_idx][head_idx]
                    prefer_vals[ht][run_id][layer_idx][head_idx] = float(sum(val_ls)/len(val_ls)) if len(val_ls) else 0.0
    
    for ht in prefer_vals.keys():
        if True: 
            for layer_idx in range(3):
                for head_idx in range(4):
                    ls = []
                    for run_id in runs_df['id']:
                        ls.append(prefer_vals[ht][run_id][layer_idx][head_idx])
                    runs_df[f"cp{checkpoint}_{ht}_head_l{layer_idx+1}_h{head_idx+1}_{indist_or_ood}"] = [prefer_vals[ht][run_id][layer_idx][head_idx] for run_id in runs_df['id']]
    # save csv
    runs_df.to_csv("all_sweeps_df.csv", index=False)
    return prefer_vals



def most_impactful_head(ablated_row, layer, checkpoint, by_ood=True):
    ood_or_indist = "ood" if by_ood else "indist"
    original_acc = ablated_row[f'cp{checkpoint}_{ood_or_indist}_acc']
    most_impactful_head = 1
    max_diff = 0
    for head in range(1, 5):
        key = f"l{layer}_h{head}_{ood_or_indist}"
        key = f"cp{checkpoint}_{key}"
        if key in ablated_row:
            if ablated_row[key] != -1:
                diff = ablated_row[key] - original_acc
                if abs(diff) > abs(max_diff):
                    max_diff = diff
                    most_impactful_head = head
    key = f"layer_{layer}_head_{most_impactful_head}"
    return key, max_diff



def head_score(row, head_type, layer, head_num, checkpoint, indist_or_ood='indist'):
    x = row[f"cp{checkpoint}_{head_type}_head_l{layer}_h{head_num}_{indist_or_ood}"]
    if x == "[]":
        return 0
    return float(x)

def max_head_score(row, head_type, layer, checkpoint, indist_or_ood='indist'):
    return max([head_score(row, head_type, layer, head_num, checkpoint, indist_or_ood) for head_num in [1,2,3,4]])

def has_head_type(row, head_type, layer, checkpoint, threshold=0.5, indist_or_ood='indist'):
    return max_head_score(row, head_type, layer, checkpoint, indist_or_ood) >= threshold

def has_head_combo(row, head_combo, layer, checkpoint, threshold=0.5, indist_or_ood='indist'):
    # e.g. a head which is both an open head and a first-symbol head
    for head_num in [1,2,3,4]:
        score_1 = head_score(row, head_combo[0], layer, head_num, checkpoint, indist_or_ood=indist_or_ood)
        score_2 = head_score(row, head_combo[1], layer, head_num, checkpoint, indist_or_ood=indist_or_ood)
        if score_1 >= threshold and score_2 >= threshold:
            return True
    return False