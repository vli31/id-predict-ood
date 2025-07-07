import torch as t
import pandas as pd
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
from utils.data import BracketsDataset
from utils.model import get_transformer
import math
from typing import Dict, List, Tuple, Optional

LOCAL = "id-predict-ood/" # working directory
INDIST_PATH = "indist_data_preds.csv"
OOD_PATH = "ood_data_preds.csv"
TRAIN_PATH = None

N_CLASSES = 2

def get_datapoint(data_df, idx):
   # dataloader with just one row/batch
   single_row_df = data_df.iloc[[idx]]
   dl = DataLoader(BracketsDataset(single_row_df), batch_size=1, shuffle=False)
   
   input_strings, inputs, targets = next(iter(dl))
   
   return inputs[0]

def run_model_on_datapoint(model, tokens, premask=False, return_probs=False):
   """Run model on single datapoint and return attention weights and optionally prediction probability"""
   model.eval()
   device = 'cuda' if t.cuda.is_available() else 'cpu'
   
   with t.no_grad():
       tokens = tokens.to(device)
       
       # model outputs for single batch
       outputs = model(tokens.unsqueeze(0))[0]
       
       seq_len = t.sum(tokens != 1)
       last_idx = (seq_len - 1).unsqueeze(0).repeat(1, N_CLASSES)
       last_idx = last_idx.unsqueeze(1).to(device)
       
       # gwt predictions for final tokens
       preds = outputs.gather(1, last_idx)[:, 0, :].cpu()
       
       # prob of positive class
       logits = preds[0]
       prob_pos = math.exp(logits[1]) / (math.exp(logits[0]) + math.exp(logits[1]))
       
       # get attention weights (masked or unmasked)
       attn_weights = (model.get_attention_weights_premask() if premask 
                      else model.get_attention_weights())

   return (attn_weights, prob_pos) if return_probs else attn_weights



def get_models_from(df_or_row, ablate_heads=None, checkpoint=5):
    """Load model weights for each model in df['id']"""
    path_of = lambda run_id : LOCAL + f"model_weights/run_{run_id}/run_{run_id}_checkpoint_{checkpoint}.pt"
    models = {}
    if isinstance(df_or_row, pd.DataFrame):
        df = df_or_row
    else:
        df = pd.DataFrame([df_or_row])
    for i, row in df.iterrows():
        model_path = path_of(row['id'])
        model = get_transformer(n_layer=row['n_layer'], n_head=row['n_head'], n_embd=row['n_embd'],
                                embd_pdrop=row['embd_pdrop'], attn_pdrop=row['attn_pdrop'], resid_pdrop=row['resid_pdrop'],
                                ablate_heads=ablate_heads)
        model.load_state_dict(t.load(model_path, map_location=t.device('cuda' if t.cuda.is_available() else 'cpu')))
        models[row['id']] = model
    return models



def get_run_ids_satisfying(df, cond_key_vals, ood_acc_min, ood_acc_max):
    conditions = lambda row: all([row[key] == val for key, val in cond_key_vals])
    bool_ls = []
    for i, row in df.iterrows():
        if conditions(row):
            bool_ls += [row['cp5_ood_acc'] >= ood_acc_min and row['cp5_ood_acc'] <= ood_acc_max]
        else:
            bool_ls += [False]
    sub_df = df[bool_ls]
    return sub_df

def models_not_already_recorded(runs_df, data_ls, checkpoint, run_id_suffix, return_df=False):
    mask = []
    for i, row in runs_df.iterrows():
        for data in data_ls:
            if row['id'] + ("" if checkpoint == 5 else f"_cp{checkpoint}_") + run_id_suffix not in data.columns:
                mask += [True]
                break
        else:
            mask += [False]
    if return_df:
        return runs_df[mask]
    return get_models_from(runs_df[mask], checkpoint=checkpoint)


def process_single_batch(model: t.nn.Module, batch: Tuple[List[str], t.Tensor, t.Tensor],
                         n_classes: int, device) -> Tuple[List[str], List[float]]:
    """Process a single batch and return input strings and predictions."""
    input_strings, inputs, _ = batch
    inputs = inputs.to(device)
    
    outputs = model(inputs)[0]
    seq_lens = t.tensor([len(s) + 2 for s in input_strings])
    last_indices = (seq_lens - 1).unsqueeze(1).repeat(1, n_classes)
    last_indices = last_indices.to(device)
    
    preds = outputs.gather(1, last_indices.unsqueeze(1))[:, 0, :].cpu()
    
    # Calculate prediction probabilities
    pred_probs = [
        math.exp(preds[i, 1]) / (math.exp(preds[i, 1]) + math.exp(preds[i, 0]))
        for i in range(preds.size(0))
    ]
    
    return input_strings, pred_probs


def get_single_model_preds(model, dataloader, n_classes=N_CLASSES):
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        model = model.eval().to(device)
        
        all_preds = []
        
        with t.no_grad():
            for batch in dataloader:
                input_strs, pred_probs = process_single_batch(
                    model, batch, n_classes, device
                )
                all_preds.extend(pred_probs)
        return all_preds

def get_preds(
    models: Dict[str, t.nn.Module],
    dataset_path: str,
    checkpoint: int,
    n_classes: int = N_CLASSES,
    batch_size: int = 32,
    save: bool = True
) -> pd.DataFrame:
    """
    Get predictions for multiple models on a dataset.
    
    Args:
        models: Dictionary mapping run_ids to models
        dataset_path: Path to dataset CSV
        checkpoint: Checkpoint number to use in column names
        n_classes: Number of output classes
        batch_size: Batch size for DataLoader
        save: Whether to save results to CSV
    
    Returns:
        DataFrame with model predictions
    """
    df = pd.read_csv(dataset_path)
    dataloader = DataLoader(BracketsDataset(df), batch_size=batch_size, shuffle=False)
    
    for run_id, model in models.items():
        key = f"cp{checkpoint}_{run_id}"
        if key in df.columns:
            print(f"Already have predictions for {key}")
            continue
        
        df[key] = get_single_model_preds(model, dataloader, n_classes)
        
    if save:
        df.to_csv(dataset_path, index=False)
    
    return df

def compute_acc_from_preds(preds, df, label='balanced'):
    correct = 0
    total = 0
    for i, row in df.iterrows():
        if preds[i] >= 0.5 and row[label] == 1:
            correct += 1
        elif preds[i] < 0.5 and row[label] == 0:
            correct += 1
        total += 1
    return correct / total

def get_acc(model, dl):
    model.eval()
    with t.no_grad():
        if t.cuda.is_available():
            model = model.cuda()
        correct = 0
        total = 0
        for i, (input_strings, inputs, targets) in enumerate(dl):
            if t.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)[0]
            seq_lens = t.tensor([len(s) + 2 for s in input_strings])  
            last_indices = (seq_lens - 1).unsqueeze(1).repeat(1, N_CLASSES)
            if t.cuda.is_available():
                last_indices = last_indices.cuda()
            preds = outputs.gather(1, last_indices.unsqueeze(1))[:, 0, :]
            _, predicted = t.max(preds, 1)
            total += targets.size(0) 
            correct += (predicted == targets).sum().item()
    return correct / total