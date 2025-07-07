import pandas as pd 
from collections import defaultdict 

import sys
sys.path.append('..')
import os
import torch as t
from ast import literal_eval
import pandas as pd
from torch.utils.data import DataLoader
from utils.data import BracketsDataset
from run_model_utils import run_model_on_datapoint 
from utils.model import get_transformer
from tqdm import tqdm

def get_ood_heuristic(): 
    # 0.554
    df = pd.read_csv(OOD_PATH)
    fst_char_closed = df['string'].str.startswith(')').sum()
    print(f"{fst_char_closed / len(df)} of the OOD strings start with ) -->\n" + 
          f"{fst_char_closed / len(df)} OOD accuracy indicates heuristic phase")

INDIST_PATH = "in_dist_test_binomial(40,0.5).csv"
OOD_PATH = "test_binomial(40,0.5).csv"
TRAIN_PATH = "train_binomial(40,0.5).csv"

MODEL_BASE_PATH = "models/sweep_pdxsbevz"

def get_models_modified(df_or_row, checkpoint, ablate_heads=None):
    """Load model weights for each model in df['id']"""
    path_of = lambda run_id : MODEL_BASE_PATH + f"run_{run_id}/run_{run_id}_checkpoint_{checkpoint}.pt"
    models = {}
    if isinstance(df_or_row, pd.DataFrame):
        df = df_or_row
    else:
        df = pd.DataFrame([df_or_row])

    for _, row in df.iterrows():
        model_path = path_of(row['id'])
        model = get_transformer(n_layer=row['n_layer'], n_head=row['n_head'], n_embd=row['n_embd'],
                                embd_pdrop=row['embd_pdrop'], attn_pdrop=row['attn_pdrop'], resid_pdrop=row['resid_pdrop'],
                                ablate_heads=ablate_heads)
        model.load_state_dict(t.load(model_path, map_location=t.device('cuda' if t.cuda.is_available() else 'cpu')))
        models[row['id']] = model

    return models


def record_pred(df, checkpoint, ds, save_dir):
    # Get pred from model {id} at {checkpoint} on {ds}
    assert ds in ["train", "indist", "ood"]
    if ds == "train":
        data = pd.read_csv(TRAIN_PATH)
    elif ds == "indist":
        data = pd.read_csv(INDIST_PATH)
    elif ds == "ood":
        data = pd.read_csv(OOD_PATH)
    dl = DataLoader(BracketsDataset(data), batch_size=1, shuffle=False)

    models = get_models_modified(df, checkpoint)
    
    for run_id in tqdm(models.keys()):
        model = models[run_id]
        dl = DataLoader(BracketsDataset(data), batch_size=1, shuffle=False)

        attn_pred = {}

        for i, (input_strings, inputs, _) in enumerate(dl):
            tokens = inputs[0]
            attention_weights, prob_pos = run_model_on_datapoint(model, tokens, return_probs=True)
            
            # Store input string, pred, attention weights for each datapoint
            attn_pred[f'datapoint_{i}'] = {
                'string': input_strings[0],
                'pred': prob_pos,
                **attention_weights,
            }
        
        pd.DataFrame(attn_pred).transpose().to_csv(save_dir + f"{run_id}_{ds}.csv", index=False)

SAVE_BASE_DIR = f"./data"

def get_heuristic_data():
    # save *all* checkpoints of runs that may have gone through the heuristic phase

    # set up results df, datapoints corresponding to checkpoints
    df = pd.read_csv("data/transformers_sweep_data_cutoffs_vecs.csv")
    df = df[df["lr"] == 0.0001] 
    df['ood_test_acc_vector'] = df['ood_test_acc_vector'].astype(str).apply(literal_eval)
    df['indist_test_acc_vector'] = df['indist_test_acc_vector'].astype(str).apply(literal_eval)
    df['datapoints_seen_vector'] = df['datapoints_seen_vector'].astype(str).apply(literal_eval)

    all_runs = []
    for checkpoint in range(1,6):
        SAVE_DIR = f"{SAVE_BASE_DIR}/cp{checkpoint}/"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        corr_dp = checkpoint * 200000 
        results = df.copy()
        results[f'ood_test_acc_cp{checkpoint}'] = results.apply(
            lambda row: row['ood_test_acc_vector'][row['datapoints_seen_vector'].index(corr_dp)], axis=1)
        
        results[f'indist_test_acc_cp{checkpoint}'] = results.apply(
            lambda row: row['indist_test_acc_vector'][row['datapoints_seen_vector'].index(corr_dp)], axis=1)

        # heuristic phase: ood_test_acc between 0.5 and 0.6, indist_test_acc > 0.99
        results = results[(results[f'ood_test_acc_cp{checkpoint}'].between(0.5, 0.6, inclusive="both") &
                          (results[f'indist_test_acc_cp{checkpoint}'] > 0.99))]
        
        # results.to_csv(SAVE_DIR + "_heuristic_runs.csv", index=False)

        all_runs.extend(results["id"].tolist())

    all_runs = list(set(all_runs)) # 47
    all_runs_possible_heuristic = df[df["id"].isin(all_runs)]
    for checkpoint in range(1,6):
        SAVE_DIR = f"{SAVE_BASE_DIR}/cp{checkpoint}/"
        record_pred(all_runs_possible_heuristic, checkpoint, ds="ood", save_dir=SAVE_DIR) 
    
def get_all_indist_data():
    df = pd.read_csv("data/transformers_sweep_data_cutoffs_vecs.csv")
    df = df[df["lr"] == 0.0001] 
    for checkpoint in range(1,6):
        SAVE_DIR = f"{SAVE_BASE_DIR}/cp{checkpoint}/indist/"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        record_pred(df, checkpoint, ds="indist", save_dir=SAVE_DIR) 

# if __name__ == "__main__":
#     # get_heuristic_data()
#     get_all_indist_data()

def find_heuristic(cutoff=0.9):
    heuristic_data = defaultdict(dict)
    
    for cp in range(1, 6): 
        df_cp = pd.read_csv(f"{SAVE_BASE_DIR}/cp{cp}/_heuristic_runs.csv")

        for _, row in df_cp.iterrows():
            id = row["id"]
            df_id = pd.read_csv(f"{SAVE_BASE_DIR}/cp{cp}/{id}_ood.csv") 

            close_df = df_id[df_id['string'].str.startswith(')')] 
            open_df = df_id[df_id['string'].str.startswith('(')]
            
            # Calculate percentages for classification criteria
            close_false_percentage = len(close_df[close_df['pred'] < 0.5]) / len(close_df)
            open_true_percentage = len(open_df[open_df['pred'] > 0.5]) / len(open_df)
            if row["n_layer"] != 1:
                print(f"cfp: {close_false_percentage:.2f}, otp: {open_true_percentage:.2f}, n_layer: {row["n_layer"]}")

            if (close_false_percentage >= cutoff) and (open_true_percentage >= cutoff):
                if heuristic_data[id] == {}:
                    for col in ["lr", "wd", "rdm_seed", "shuffle_seed", "n_layer", "n_head"]:
                        heuristic_data[id][col] = row[col]

                heuristic_data[id][f"cp{cp}_ood_test_acc"] = row[f"ood_test_acc_cp{cp}"]
                heuristic_data[id][f"cp{cp}_indist_test_acc"] = row[f"indist_test_acc_cp{cp}"]
                heuristic_data[id][f"cp{cp}_close_false"] = close_false_percentage
                heuristic_data[id][f"cp{cp}_open_true"] = open_true_percentage
    
    # pd.DataFrame.from_dict(heuristic_data, orient="index").to_csv("heuristic_candidates.csv")

find_heuristic()