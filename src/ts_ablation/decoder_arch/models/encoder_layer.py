import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer (Pre-LN architecture).

    Each layer contains two sub-layers with residual connections:
        1. Multi-Head Self-Attention
        2. Position-wise Feed-Forward Network

    Pre-LN formulation (LayerNorm *before* each sub-layer):
        x' = x + Dropout(Attention(LayerNorm(x)))
        out = x' + Dropout(FFN(LayerNorm(x')))

    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        d_ff: Dimension of the inner feed-forward layer.
        dropout: Dropout probability.
        activation: Activation for FFN, either ``'relu'`` or ``'gelu'``.
        output_attention: If ``True``, also return attention weights.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 activation='gelu', output_attention=False):
        super(EncoderLayer, self).__init__()

        self.output_attention = output_attention

        # Sub-layer 1: Multi-Head Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            output_attention=output_attention,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Sub-layer 2: Position-wise Feed-Forward Network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: Tensor of shape ``(B, T, d_model)``.
            attn_mask: Optional attention mask.

        Returns:
            Tensor of shape ``(B, T, d_model)``.
            If ``output_attention=True``, also returns attention weights.
        """
        # ── Sub-layer 1: Attention + Residual ──
        residual = x
        x = self.norm1(x)

        if self.output_attention:
            x, attn_weights = self.attention(x, attn_mask=attn_mask)
        else:
            x = self.attention(x, attn_mask=attn_mask)
            attn_weights = None

        x = residual + self.dropout1(x)

        # ── Sub-layer 2: FFN + Residual ──
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)

        if self.output_attention:
            return x, attn_weights
        return x


def main():
    # Realistic ETT-style encoder window
    batch_size = 32
    seq_len = 96          # encoder lookback window
    d_model = 512
    n_heads = 8
    d_ff = 2048

    x = torch.randn(batch_size, seq_len, d_model)
    print("EncoderLayer — input shape:", tuple(x.shape))

    # Case 1: standard forward, no attention weights
    layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.1, activation='gelu')
    out = layer(x)
    print(f"[no-attn] output shape: {tuple(out.shape)}  expected: ({batch_size}, {seq_len}, {d_model})")
    assert out.shape == (batch_size, seq_len, d_model)

    # Case 2: with attention weights returned
    layer_attn = EncoderLayer(d_model, n_heads, d_ff, dropout=0.1,
                              activation='gelu', output_attention=True)
    out_a, attn = layer_attn(x)
    print(f"[attn] output shape: {tuple(out_a.shape)}, attn shape: {tuple(attn.shape)}")
    print(f"       expected attn: ({batch_size}, {n_heads}, {seq_len}, {seq_len})")
    assert out_a.shape == (batch_size, seq_len, d_model)
    assert attn.shape == (batch_size, n_heads, seq_len, seq_len)

    # Case 3: with a key-padding mask (last 6 timesteps are padding -> not attended)
    # SDPA boolean mask: True = participate. Shape (B, 1, 1, T_k) broadcasts over heads/queries.
    pad_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    pad_mask[:, :, :, -6:] = False
    out_m = layer(x, attn_mask=pad_mask)
    print(f"[masked] output shape: {tuple(out_m.shape)}")
    assert out_m.shape == (batch_size, seq_len, d_model)

    # Case 4: regression for Bug 1 — attn_mask must be honored even when
    # output_attention=True. Masked vs unmasked must DIFFER.
    layer_attn.eval()
    with torch.no_grad():
        o_unmasked, _ = layer_attn(x)
        o_masked, _ = layer_attn(x, attn_mask=pad_mask)
    diff = (o_unmasked - o_masked).abs().max().item()
    print(f"[attn+mask] max |unmasked - masked| = {diff:.3e}  (must be > 0)")
    assert diff > 1e-6, "attn_mask ignored when output_attention=True (Bug 1)!"

    print("EncoderLayer tests passed.")


if __name__ == "__main__":
    main()
