import pandas as pd
import torch as t
from model import get_transformer



def get_models_from(df_or_row, ablate_heads=None, checkpoint=5):
    """Load model weights for each model in df['id']"""
    path_of = lambda run_id : f"model_weights/run_{run_id}/run_{run_id}_checkpoint_{checkpoint}.pt"
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


