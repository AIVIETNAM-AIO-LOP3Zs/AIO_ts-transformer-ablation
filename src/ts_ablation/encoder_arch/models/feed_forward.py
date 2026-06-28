import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Applies two linear transformations with an activation in between:
        FFN(x) = W2 · activation(W1 · x + b1) + b2

    Shape: (B, T, d_model) → (B, T, d_ff) → (B, T, d_model)

    Args:
        d_model: Dimension of the model (input and output size).
        d_ff: Dimension of the inner feed-forward layer.
        dropout: Dropout probability applied after activation.
        activation: Activation function, either ``'relu'`` or ``'gelu'``.
    """

    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'relu' or 'gelu'.")

    def forward(self, x):
        """
        Args:
            x: Tensor of shape ``(B, T, d_model)``.

        Returns:
            Tensor of shape ``(B, T, d_model)``.
        """
        x = self.linear1(x)        # (B, T, d_model) → (B, T, d_ff)
        x = self.activation(x)     # non-linearity
        x = self.dropout(x)        # regularization
        x = self.linear2(x)        # (B, T, d_ff) → (B, T, d_model)
        return x


def main():
    # Realistic ETT-style shapes (e.g. ETTm1 encoder window)
    batch_size = 32
    seq_len = 96          # encoder lookback window
    d_model = 512
    d_ff = 2048

    x = torch.randn(batch_size, seq_len, d_model)
    print("FeedForward — input shape:", tuple(x.shape))

    # Case 1: GELU activation (default) preserves shape
    ffn_gelu = FeedForward(d_model, d_ff, dropout=0.1, activation='gelu')
    out_gelu = ffn_gelu(x)
    print(f"[gelu] output shape: {tuple(out_gelu.shape)}  expected: ({batch_size}, {seq_len}, {d_model})")
    assert out_gelu.shape == (batch_size, seq_len, d_model)

    # Case 2: ReLU activation
    ffn_relu = FeedForward(d_model, d_ff, dropout=0.1, activation='relu')
    out_relu = ffn_relu(x)
    print(f"[relu] output shape: {tuple(out_relu.shape)}")
    assert out_relu.shape == (batch_size, seq_len, d_model)

    # Case 3: unsupported activation must raise ValueError
    try:
        FeedForward(d_model, d_ff, activation='swish')
        print("[invalid] ERROR: no exception raised")
    except ValueError as e:
        print(f"[invalid] correctly raised ValueError: {e}")

    print("FeedForward tests passed.")


if __name__ == "__main__":
    main()
