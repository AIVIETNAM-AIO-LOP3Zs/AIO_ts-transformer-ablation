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
            x, attn_weights = self.attention(x)
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
