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
