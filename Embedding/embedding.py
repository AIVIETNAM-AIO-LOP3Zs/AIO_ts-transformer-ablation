import torch 
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, drop_out, max_length = 1000):
        super(PositionalEncoding, self).__init__()
        self.drop_out = nn.Dropout(p = drop_out)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0,max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, a):
        a += self.pe[:, : a.size(1)].requires_grad_(False)
        return self.drop_out(a)
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocal_size, d_model, drop_out, max_length = 1000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocal_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model = d_model, drop_out=drop_out, max_length=max_length)
        self.d_model = d_model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_token = self.token_embedding(x) * math.sqrt(self.d_model)
        out_final = self.positional_encoding(out_token)
        return out_final