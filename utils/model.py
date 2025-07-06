"""
Get models of different types
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from minGPT.model import GPT # variation of minGPT supporting attention visualization

MAX_LEN = 40
VOCAB = "()"
N_CLASSES = 2

# Transformer model with default parameters
def get_transformer(n_layer:int=2, n_head:int=2, n_embd:int=64,
                    embd_pdrop:float=0.0, resid_pdrop: float=0.0, 
                    attn_pdrop:float=0.0, vocab_size:int=len(VOCAB)+3, 
                    block_size:int=MAX_LEN+2, tr_model_type:str|None=None,
                    ablate_heads=None) -> GPT:
    config = GPT.get_default_config()
    
    setattr(config, 'n_layer', n_layer)
    setattr(config, 'n_head', n_head)
    setattr(config, 'n_embd', n_embd)

    setattr(config, 'embd_pdrop', embd_pdrop)
    setattr(config, 'resid_pdrop', resid_pdrop)
    setattr(config, 'attn_pdrop', attn_pdrop)

    setattr(config, 'vocab_size', vocab_size)
    setattr(config, 'block_size', block_size)

    setattr(config, 'model_type', tr_model_type)
    
    if ablate_heads is not None:
        model = GPT(config, ablate_heads=ablate_heads)
    else:
        model = GPT(config)
    
    return model #, config 

# ------------------------------------------------------------------------------
# LSTM with default parameters
class LSTMClassifier(nn.Module):
    
    def __init__(self, batch_size:int, num_layers:int, dropout_rate:float, 
                 embedding_dim:int, hidden_dim:int, vocab_size:int, label_size:int):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(dropout_rate)

    def init_hidden(self, batch_size):
        # Adjust function to account for cases where batch_size != self.batch_size (last batch)
        # the first is the hidden h
        # the second is the cell c
        return (autograd.Variable(t.zeros(self.num_layers, batch_size, self.hidden_dim).cuda()),
                autograd.Variable(t.zeros(self.num_layers, batch_size, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = self.dropout(lstm_out) 
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs

def get_LSTM(batch_size:int, num_layers:int=2, dropout_rate:float=0.0, 
             embedding_dim:int=64, hidden_dim:int=64, 
             vocab_size:int=len(VOCAB)+3, label_size:int=N_CLASSES) -> LSTMClassifier:
    
    # config = locals()
    model = LSTMClassifier(batch_size, num_layers, dropout_rate, 
                           embedding_dim, hidden_dim, vocab_size, label_size)
    
    return model #, config

