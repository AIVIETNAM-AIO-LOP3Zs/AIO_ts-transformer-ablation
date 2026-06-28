import torch
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


def main():
    # Realistic ETT-style encoder stack (e_layers=2 as in ModelConfig defaults)
    batch_size = 32
    seq_len = 96          # encoder lookback window
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 2

    x = torch.randn(batch_size, seq_len, d_model)
    print("Encoder — input shape:", tuple(x.shape))

    # Case 1: stack forward, no attention weights
    encoder = Encoder(d_model, n_heads, d_ff, n_layers=n_layers,
                      dropout=0.1, activation='gelu')
    out = encoder(x)
    print(f"[no-attn] output shape: {tuple(out.shape)}  expected: ({batch_size}, {seq_len}, {d_model})")
    assert out.shape == (batch_size, seq_len, d_model)

    # Case 2: collect attention weights from every layer
    encoder_attn = Encoder(d_model, n_heads, d_ff, n_layers=n_layers,
                           dropout=0.1, activation='gelu', output_attention=True)
    out_a, attentions = encoder_attn(x)
    print(f"[attn] output shape: {tuple(out_a.shape)}, num attention maps: {len(attentions)}  expected: {n_layers}")
    assert out_a.shape == (batch_size, seq_len, d_model)
    assert len(attentions) == n_layers
    for i, a in enumerate(attentions):
        assert a.shape == (batch_size, n_heads, seq_len, seq_len), f"layer {i} attn shape {tuple(a.shape)}"
    print(f"       each attn map shape: {tuple(attentions[0].shape)}")

    print("Encoder tests passed.")


if __name__ == "__main__":
    main()
