# twdlstm model_transfo v0.8

import math

# Positional Encoding (same for all tokens, independent of targets)
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, seq_len): # max_len=5000
        super().__init__()
        pe = torch.zeros(seq_len, d_model) # max_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # max_len
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0) # batch first
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Encoder-only transformer model
class Model(torch.nn.Module):
    def __init__(self, i_dim, seq_len, model_dim, num_layers, nhead, feedfwd_dim, o_dim):
        super().__init__()
        self.input_proj = torch.nn.Linear(i_dim, model_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=feedfwd_dim, # def=2048
            dropout=0.0, # def=0.1
            batch_first=False
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(model_dim, seq_len)
        self.output_proj = torch.nn.Linear(model_dim, o_dim)
        self.actout = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return self.actout(self.output_proj(x))

# i_dim = nb covariates
# seq_len = nb time steps in covariates = batch_len
# o_dim = o_size = 1
# model_dim = d_model = latent dimension, must be even. good def = 512
# num_layers = nb layer in encoder (N=6 in paper). good def for now = 1
# nhead = nb heads in multihead attention (h=8 in paper). good def for now = 1
# feedfwd_dim = fully connected feedforward (inner layer) dimension, good def = 64

# model = Model(i_dim, seq_len, model_dim, num_layers, nhead, feedfwd_dim, o_dim) # instantiate
# model.train() # print(model)
