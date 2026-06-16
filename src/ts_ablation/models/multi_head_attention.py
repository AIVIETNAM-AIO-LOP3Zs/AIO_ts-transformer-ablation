import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        # Input: d_model, n_heads, dropout, output_attention
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model                  # kích thước embedding
        self.n_heads = n_heads                  # số lượng head
        self.d_head = d_model // n_heads        
        self.output_attention = output_attention

        self.W_q = nn.Linear(d_model, d_model)  # query
        self.W_k = nn.Linear(d_model, d_model)  # key
        self.W_v = nn.Linear(d_model, d_model)  # value
        self.fc_out = nn.Linear(d_model, d_model)  

        self.dropout = nn.Dropout(dropout)
        # queries -> Q: (B, T_q, d_model)
        # keys    -> K: (B, T_k, d_model)
        # values  -> V: (B, T_k, d_model)

    def split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        x = x.transpose(1, 2)
        return x
        # Input:  x (B, T, d_model)
        # Q: (B, H, T_q, d_head)
        # K: (B, H, T_k, d_head)
        # V: (B, H, T_k, d_head)

    def combine_heads(self, x):
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, T, H * Dh)
        return x
        # Input: (B, H, T, d_head)
        # merge: (B, T_q, d_model)

    def _build_attn_bias(self, attn_mask, is_causal, T_q, T_k, device, dtype):
        """Build a single additive attention bias that merges an optional
        ``attn_mask`` (bool: True=keep, or float: additive) with optional causal
        masking. Returns a tensor broadcastable to (B, H, T_q, T_k), or ``None``.

        Merging both into one bias lets us avoid SDPA's restriction that
        ``attn_mask`` and ``is_causal=True`` cannot be supplied together.
        """
        bias = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bias = torch.zeros_like(attn_mask, dtype=dtype).masked_fill(~attn_mask, float("-inf"))
            else:
                bias = attn_mask.to(dtype)
        if is_causal:
            causal = torch.triu(
                torch.ones(T_q, T_k, device=device, dtype=torch.bool), diagonal=1
            )
            causal_bias = torch.zeros(T_q, T_k, device=device, dtype=dtype).masked_fill(
                causal, float("-inf")
            )
            bias = causal_bias if bias is None else bias + causal_bias
        return bias

    def forward(self, queries, keys=None, values=None, attn_mask=None, is_causal=False):
        # Input:
        #   queries: (B, T_q, d_model)
        #   keys   : (B, T_k, d_model) hoặc None (self attention)
        #   values : (B, T_k, d_model) hoặc None
        # Output:
        #   out    : (B, T_q, d_model)

        if keys is None:
            keys = queries
        if values is None:
            values = keys

        Q = self.split_heads(self.W_q(queries))
        K = self.split_heads(self.W_k(keys))
        V = self.split_heads(self.W_v(values))

        dropout_p = self.dropout.p if self.training else 0.0

        if self.output_attention:
            # Manual path: compute the attention weights ONCE and derive the
            # output from them (no second, redundant attention computation).
            #   Q: (B, H, T_q, d_head), K: (B, H, T_k, d_head)
            #   scores / attn: (B, H, T_q, T_k)
            T_q, T_k = Q.size(-2), K.size(-2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
            bias = self._build_attn_bias(attn_mask, is_causal, T_q, T_k, Q.device, scores.dtype)
            if bias is not None:
                scores = scores + bias
            attn = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(F.dropout(attn, p=dropout_p, training=self.training), V)
        else:
            # Fused path. Keep SDPA's fast causal kernel when there is no extra
            # mask; otherwise merge causal + mask into one bias so we never pass
            # both attn_mask and is_causal=True (which SDPA rejects).
            if attn_mask is None:
                attn_output = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
                )
            else:
                bias = self._build_attn_bias(
                    attn_mask, is_causal, Q.size(-2), K.size(-2), Q.device, Q.dtype
                )
                attn_output = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=bias, dropout_p=dropout_p, is_causal=False
                )

        out = self.combine_heads(attn_output)
        out = self.fc_out(out)
        # NOTE: no output dropout here — residual dropout is owned by the
        # enclosing Encoder/Decoder layer, avoiding double dropout.

        if self.output_attention:
            # Output:
            #   out : (B, T_q, d_model)
            #   attn: (B, H, T_q, T_k)
            return out, attn

        # Output: (B, T_q, d_model)
        return out

def main():
    # Realistic ETT-style shapes: encoder lookback window seq_len=96
    batch_size = 32
    seq_len = 96
    d_model = 512
    n_heads = 8

    model = MultiHeadAttention(d_model, n_heads, dropout=0.1, output_attention=True)
    x = torch.randn(batch_size, seq_len, d_model)

    print("SELF ATTENTION")
    output, attn_weights = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected: ({batch_size}, {n_heads}, {seq_len}, {seq_len})")

    print("CROSS ATTENTION")
    seq_len_kv = 72  # decoder query attends to encoder memory of a different length
    keys = torch.randn(batch_size, seq_len_kv, d_model)
    values = torch.randn(batch_size, seq_len_kv, d_model)

    output_cross, attn_cross = model(x, keys, values)
    print(f"Queries shape: {x.shape}")
    print(f"Keys/Values shape: {keys.shape}")
    print(f"Output shape: {output_cross.shape}")
    print(f"Attention weights shape: {attn_cross.shape}")
    print(f"Expected: ({batch_size}, {n_heads}, {seq_len}, {seq_len_kv})")

    print("CASUAL ATTENTION")
    output_causal, attn_causal = model(x, is_causal=True)
    print(f"Output shape: {output_causal.shape}")
    print(f"Attention weights shape: {attn_causal.shape}")

    is_lower_triangular = True
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            if attn_causal[0, 0, i, j] > 1e-6:  # Các phần tử trên đường chéo chính phải gần 0
                is_lower_triangular = False
                break

if __name__ == "__main__":
    main()