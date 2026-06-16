import torch
import torch.nn as nn

# Import Attention and FFN classes implemented by teammates
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer (Pre-LN architecture).
    Contains three sequential sub-layers:
        1. Masked Self-Attention (is_causal=True to block future information)
        2. Cross-Attention (Queries from decoder, Keys and Values from encoder memory)
        3. Position-wise Feed-Forward Network
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 activation='gelu', output_attention=False):
        # Initialize the base class nn.Module
        super(DecoderLayer, self).__init__()

        # Save config flag for outputting attention weights
        self.output_attention = output_attention

        # First normalization layer (applied before Self-Attention)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multi-Head Attention for Self-Attention (Decoder attends to its own sequence)
        self.self_attention = MultiHeadAttention(
            d_model=d_model,       # Dimension of model input/output
            n_heads=n_heads,       # Number of attention heads
            dropout=dropout,       # Dropout rate for regularization
            output_attention=output_attention, # Whether to return attention weights
        )
        # Dropout layer after Self-Attention
        self.dropout1 = nn.Dropout(dropout)

        # Second normalization layer (applied before Cross-Attention)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-Head Attention for Cross-Attention (Decoder attends to Encoder output)
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,       # Dimension of model features
            n_heads=n_heads,       # Number of attention heads
            dropout=dropout,       # Dropout rate
            output_attention=output_attention, # Whether to return attention weights
        )
        # Dropout layer after Cross-Attention
        self.dropout2 = nn.Dropout(dropout)

        # Third normalization layer (applied before FFN)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Position-wise Feed-Forward Network (two linear layers with activation)
        self.ffn = FeedForward(
            d_model=d_model,       # Model input/output dimensions
            d_ff=d_ff,             # Hidden layer dimension
            dropout=dropout,       # Dropout rate
            activation=activation, # Non-linear activation function ('gelu' or 'relu')
        )
        # Dropout layer after FFN
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass of a single Decoder layer.

        Args:
            x: Input tensor to the decoder, shape (B, L_dec, d_model)
               (B: Batch size, L_dec: Decoder sequence length, d_model: Feature dimension)
            cross: Output tensor from the Encoder (Memory), shape (B, L_enc, d_model)
               (L_enc: Encoder sequence length)
            x_mask: Attention mask for Self-Attention (e.g. padding mask)
            cross_mask: Attention mask for Cross-Attention

        Returns:
            Output tensor of shape (B, L_dec, d_model).
            If output_attention=True, also returns self-attention and cross-attention weights.
        """
        
        # ─────────────────────────────────────────────────────────────────
        # 1. SUB-LAYER 1: Masked Self-Attention + Residual Connection (Pre-LN)
        # ─────────────────────────────────────────────────────────────────
        
        # Save input as residual for skip connection
        residual = x
        
        # Apply LayerNorm before sub-layer (Pre-LN)
        x = self.norm1(x)

        # Apply Self-Attention with Q = K = V = x
        # Note: is_causal=True to mask future tokens (causal masking)
        if self.output_attention:
            # If outputting attention weights, returns (output, attention_weights)
            x, self_attn_weights = self.self_attention(
                queries=x, keys=x, values=x, attn_mask=x_mask, is_causal=True
            )
        else:
            # If not, returns only the output tensor
            x = self.self_attention(
                queries=x, keys=x, values=x, attn_mask=x_mask, is_causal=True
            )
            self_attn_weights = None

        # Add residual connection after applying dropout
        # Output shape remains (B, L_dec, d_model)
        x = residual + self.dropout1(x)

        # ─────────────────────────────────────────────────────────────────
        # 2. SUB-LAYER 2: Cross-Attention + Residual Connection (Pre-LN)
        # ─────────────────────────────────────────────────────────────────
        
        # Save input x as residual for the next skip connection
        residual = x
        
        # Apply LayerNorm before Cross-Attention
        x = self.norm2(x)

        # Apply Cross-Attention
        # - Query (Q) is from Decoder (x): shape (B, L_dec, d_model)
        # - Key (K) and Value (V) are from Encoder (cross): shape (B, L_enc, d_model)
        # - is_causal=False since decoder can see all past context from encoder
        if self.output_attention:
            x, cross_attn_weights = self.cross_attention(
                queries=x, keys=cross, values=cross, attn_mask=cross_mask, is_causal=False
            )
        else:
            x = self.cross_attention(
                queries=x, keys=cross, values=cross, attn_mask=cross_mask, is_causal=False
            )
            cross_attn_weights = None

        # Add residual connection after applying dropout
        # Output shape remains (B, L_dec, d_model)
        x = residual + self.dropout2(x)

        # ─────────────────────────────────────────────────────────────────
        # 3. SUB-LAYER 3: FFN + Residual Connection (Pre-LN)
        # ─────────────────────────────────────────────────────────────────
        
        # Save input x as residual
        residual = x
        
        # Apply LayerNorm before FFN
        x = self.norm3(x)
        
        # Pass through the Feed-Forward Network
        x = self.ffn(x)
        
        # Add final residual connection after applying dropout
        x = residual + self.dropout3(x)

        # Return output tensor along with attention weights if configured
        if self.output_attention:
            return x, self_attn_weights, cross_attn_weights
        return x


class Decoder(nn.Module):
    """
    Transformer Decoder — a stack of N identical DecoderLayers.
    Applies final LayerNorm to normalize the last layer's output (required for Pre-LN).
    """

    def __init__(self, d_model, n_heads, d_ff, n_layers=1,
                 dropout=0.1, activation='gelu', output_attention=False):
        # Initialize the base class nn.Module
        super(Decoder, self).__init__()

        # Save config flag for outputting attention weights
        self.output_attention = output_attention

        # Use nn.ModuleList to manage the N stacked DecoderLayer instances
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                output_attention=output_attention,
            )
            # Loop N times to instantiate N independent layers
            for _ in range(n_layers)
        ])

        # Final LayerNorm layer to normalize output after all stacked decoder layers
        # Crucial for stable gradients in Pre-LN architecture
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass through the entire Decoder stack.

        Args:
            x: Decoder inputs of shape (B, L_dec, d_model)
            cross: Encoder outputs of shape (B, L_enc, d_model)
            x_mask: Attention mask for Self-Attention
            cross_mask: Attention mask for Cross-Attention

        Returns:
            Predicted output tensor of shape (B, L_dec, d_model)
            If output_attention=True, also returns lists of attention weights per layer.
        """
        # Lists to store attention weights from each layer
        self_attentions = []
        cross_attentions = []

        # Iterate sequentially through each decoder layer
        for layer in self.layers:
            if self.output_attention:
                # If outputting attention weights, retrieve the output and weights
                x, self_attn, cross_attn = layer(
                    x, cross, x_mask=x_mask, cross_mask=cross_mask
                )
                # Store attention weights for the current layer
                self_attentions.append(self_attn)
                cross_attentions.append(cross_attn)
            else:
                # If not, just pass the output x to the next layer
                x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        # Apply final LayerNorm (Pre-LN requirement)
        x = self.norm(x)

        # Return output and attention lists if configured
        if self.output_attention:
            return x, self_attentions, cross_attentions
        return x


def main():
    # Realistic ETT-style shapes:
    #   encoder window seq_len=96; decoder input = label_len(48) + pred_len(24) = 72
    batch_size = 32
    seq_len_enc = 96
    seq_len_dec = 72
    d_model = 512
    n_heads = 8
    d_ff = 2048

    x_dec = torch.randn(batch_size, seq_len_dec, d_model)   # decoder input
    cross = torch.randn(batch_size, seq_len_enc, d_model)   # encoder memory
    print("Decoder — x_dec:", tuple(x_dec.shape), " cross:", tuple(cross.shape))

    # Case 1: single DecoderLayer with attention weights
    layer = DecoderLayer(d_model, n_heads, d_ff, dropout=0.1,
                         activation='gelu', output_attention=True)
    out, self_attn, cross_attn = layer(x_dec, cross)
    print(f"[layer/attn] out: {tuple(out.shape)}, self_attn: {tuple(self_attn.shape)}, cross_attn: {tuple(cross_attn.shape)}")
    assert out.shape == (batch_size, seq_len_dec, d_model)
    assert self_attn.shape == (batch_size, n_heads, seq_len_dec, seq_len_dec)
    assert cross_attn.shape == (batch_size, n_heads, seq_len_dec, seq_len_enc)

    # Case 2: Decoder stack (d_layers), no attention weights
    decoder = Decoder(d_model, n_heads, d_ff, n_layers=2,
                      dropout=0.1, activation='gelu')
    out_stack = decoder(x_dec, cross)
    print(f"[stack/no-attn] out: {tuple(out_stack.shape)}  expected: ({batch_size}, {seq_len_dec}, {d_model})")
    assert out_stack.shape == (batch_size, seq_len_dec, d_model)

    # Case 3: Decoder stack with attention weights collected per layer
    decoder_a = Decoder(d_model, n_heads, d_ff, n_layers=2,
                        dropout=0.1, activation='gelu', output_attention=True)
    out_a, self_attns, cross_attns = decoder_a(x_dec, cross)
    print(f"[stack/attn] out: {tuple(out_a.shape)}, num self/cross maps: {len(self_attns)}/{len(cross_attns)}")
    assert len(self_attns) == 2 and len(cross_attns) == 2

    # Case 4: causal self-attention combined with a decoder padding mask.
    # Regression for the SDPA "both attn_mask and is_causal" crash — must NOT raise.
    x_mask = torch.ones(batch_size, 1, 1, seq_len_dec, dtype=torch.bool)
    x_mask[:, :, :, -4:] = False        # last 4 decoder steps are padding
    out_masked = layer(x_dec, cross, x_mask=x_mask)[0]
    print(f"[layer/causal+mask] out: {tuple(out_masked.shape)}  (no SDPA crash)")
    assert out_masked.shape == (batch_size, seq_len_dec, d_model)

    # Case 5: causal masking — perturbing future steps must not change past outputs
    decoder_c = Decoder(d_model, n_heads, d_ff, n_layers=2, dropout=0.0).eval()
    out1 = decoder_c(x_dec, cross)
    x_dec2 = x_dec.clone()
    x_dec2[:, 40:, :] += 5.0                       # change only future steps
    out2 = decoder_c(x_dec2, cross)
    leak = torch.max(torch.abs(out1[:, :40, :] - out2[:, :40, :])).item()
    print(f"[causal] max past-output change after future perturbation: {leak:.2e}")
    assert leak < 1e-5, "Future leaked into the past — causal masking broken!"

    print("Decoder tests passed.")


if __name__ == "__main__":
    main()
