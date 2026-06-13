import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model                          # size
        self.n_heads = n_heads                          # number of head
        self.d_head = d_model // n_heads                # size of head
        self.output_attention = output_attention        

        self.W_q = nn.Linear(d_model, d_model)          
        self.W_k = nn.Linear(d_model, d_model)          
        self.W_v = nn.Linear(d_model, d_model)          
        self.fc_out = nn.Linear(d_model, d_model)       

        self.dropout = nn.Dropout(dropout)              

    def split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)     
        x = x.transpose(1, 2)                           
        return x

    def combine_heads(self, x):
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()              
        x = x.view(B, T, H * Dh)                        
        return x

    def forward(self, queries, keys=None, values=None, attn_mask=None, is_causal=False):
        # Self attention
        if keys is None:
            keys = queries                              
        if values is None:
            values = keys                               

        Q = self.W_q(queries)                           
        K = self.W_k(keys)                              
        V = self.W_v(values)                            

        Q = self.split_heads(Q)                        
        K = self.split_heads(K)                         
        V = self.split_heads(V)                         

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )                                               

        out = self.combine_heads(attn_output)           
        out = self.fc_out(out)                         
        out = self.dropout(out)                         

        # Return matrix attention weight
        if self.output_attention:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  

            if is_causal:
                T_q, T_k = scores.size(-2), scores.size(-1)
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=scores.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask

            attn = torch.softmax(scores, dim=-1)        
            return out, attn

        return out