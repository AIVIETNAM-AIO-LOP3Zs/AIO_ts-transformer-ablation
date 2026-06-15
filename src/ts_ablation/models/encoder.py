import torch.nn as nn

from .encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """Transformer Encoder — a stack of N identical EncoderLayers.

    Applies a sequence of encoder layers followed by a final LayerNorm
    (required for the Pre-LN architecture to normalize the last layer's output).

    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        d_ff: Dimension of the inner feed-forward layer.
        n_layers: Number of stacked encoder layers.
        dropout: Dropout probability.
        activation: Activation for FFN, either ``'relu'`` or ``'gelu'``.
        output_attention: If ``True``, collect and return attention weights
            from every layer.
    """

    def __init__(self, d_model, n_heads, d_ff, n_layers=2,
                 dropout=0.1, activation='gelu', output_attention=False):
        super(Encoder, self).__init__()

        self.output_attention = output_attention

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                output_attention=output_attention,
            )
            for _ in range(n_layers)
        ])

        # Final LayerNorm (Pre-LN architecture requires this)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: Tensor of shape ``(B, T, d_model)``.
            attn_mask: Optional attention mask passed to every layer.

        Returns:
            Tensor of shape ``(B, T, d_model)``.
            If ``output_attention=True``, also returns a list of attention
            weight tensors, one per layer.
        """
        attentions = []

        for layer in self.layers:
            if self.output_attention:
                x, attn = layer(x, attn_mask=attn_mask)
                attentions.append(attn)
            else:
                x = layer(x, attn_mask=attn_mask)

        # Final normalization
        x = self.norm(x)

        if self.output_attention:
            return x, attentions
        return x
