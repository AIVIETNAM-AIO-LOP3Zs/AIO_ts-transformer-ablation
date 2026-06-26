import sys
sys.path.insert(0, "src")

import torch
from ts_ablation.models import DecoderLayer, Decoder

def test_decoder():
    batch_size = 32
    seq_len_enc = 96
    seq_len_dec = 72
    d_model = 512
    n_heads = 8
    d_ff = 2048

    print("----------------------------------------")
    print("Testing DecoderLayer...")
    
    # 1. Instantiate DecoderLayer with attention weights output
    decoder_layer = DecoderLayer(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1,
        activation='gelu',
        output_attention=True
    )
    
    # Mock inputs
    x_dec = torch.randn(batch_size, seq_len_dec, d_model)
    x_enc = torch.randn(batch_size, seq_len_enc, d_model)
    
    # Run forward pass
    out_layer, self_attn, cross_attn = decoder_layer(x_dec, x_enc)
    print(f"DecoderLayer Input x shape: {x_dec.shape}")
    print(f"DecoderLayer Input cross shape: {x_enc.shape}")
    print(f"DecoderLayer Output shape: {out_layer.shape}")
    print(f"Self-Attention weight shape: {self_attn.shape}")
    print(f"Cross-Attention weight shape: {cross_attn.shape}")
    
    # Verify dimensions
    assert out_layer.shape == (batch_size, seq_len_dec, d_model)
    assert self_attn.shape == (batch_size, n_heads, seq_len_dec, seq_len_dec)
    assert cross_attn.shape == (batch_size, n_heads, seq_len_dec, seq_len_enc)
    print("DecoderLayer tests passed successfully!")
    print("----------------------------------------")
    
    print("Testing Decoder Stack...")
    
    # 2. Instantiate Decoder Stack (N=3 layers)
    decoder_stack = Decoder(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=3,
        dropout=0.1,
        activation='gelu',
        output_attention=True
    )
    
    # Run forward pass through stack
    out_stack, self_attns, cross_attns = decoder_stack(x_dec, x_enc)
    print(f"Decoder Stack Output shape: {out_stack.shape}")
    print(f"Number of Self-Attention weights: {len(self_attns)}")
    print(f"Number of Cross-Attention weights: {len(cross_attns)}")
    
    # Verify dimensions
    assert out_stack.shape == (batch_size, seq_len_dec, d_model)
    assert len(self_attns) == 3
    assert len(cross_attns) == 3
    print("Decoder Stack tests passed successfully!")
    print("----------------------------------------")

    print("Testing without outputting attention...")
    decoder_no_attn = Decoder(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=2,
        dropout=0.1,
        activation='gelu',
        output_attention=False
    )
    out_no_attn = decoder_no_attn(x_dec, x_enc)
    print(f"Decoder (no attn) Output shape: {out_no_attn.shape}")
    assert out_no_attn.shape == (batch_size, seq_len_dec, d_model)
    print("All standard tests passed successfully!")
    print("----------------------------------------")

    # 4. Test Causal Masking (Leakage check)
    print("Testing Causal Masking (Future leakage check)...")
    decoder_causal = Decoder(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=2,
        dropout=0.0, # set to 0 to avoid random dropout noise during comparison
        activation='gelu',
        output_attention=False
    )
    decoder_causal.eval() # set to evaluation mode

    # Create dummy inputs
    x_dec1 = torch.randn(batch_size, seq_len_dec, d_model)
    x_enc = torch.randn(batch_size, seq_len_enc, d_model)

    # Output 1
    out1 = decoder_causal(x_dec1, x_enc)

    # Create x_dec2 by copying x_dec1 and changing values ONLY from index 40 onwards (future)
    x_dec2 = x_dec1.clone()
    x_dec2[:, 40:, :] = x_dec2[:, 40:, :] + torch.randn(batch_size, seq_len_dec - 40, d_model) # change future steps with noise

    # Output 2
    out2 = decoder_causal(x_dec2, x_enc)

    # The past outputs (timesteps 0 to 39) must remain EXACTLY identical
    # Future changes must NOT affect past predictions
    difference_in_past = torch.max(torch.abs(out1[:, :40, :] - out2[:, :40, :])).item()
    print(f"Max difference in past timesteps (0-39): {difference_in_past}")
    
    assert difference_in_past < 1e-5, "Future data leaked to the past! Causal masking is broken!"
    print("Causal masking test passed! Future changes do not affect past predictions.")
    print("----------------------------------------")

    # 5. Test Ablation: No Self-Attention
    print("Testing Ablation: No Self-Attention (use_self_attention=False)...")
    decoder_no_self_attn = Decoder(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=2,
        dropout=0.1,
        activation='gelu',
        output_attention=True,
        use_self_attention=False
    )
    out_no_self_attn, self_attns_no_sa, cross_attns_no_sa = decoder_no_self_attn(x_dec, x_enc)
    print(f"Decoder (no self-attn) Output shape: {out_no_self_attn.shape}")
    print(f"Self-attentions (should be None): {self_attns_no_sa}")
    assert out_no_self_attn.shape == (batch_size, seq_len_dec, d_model)
    assert all(sa is None for sa in self_attns_no_sa)
    print("Ablation No Self-Attention test passed successfully!")
    print("----------------------------------------")

    # 6. Test Ablation: No Causal Masking (Future leakage should occur)
    print("Testing Ablation: No Causal Masking (is_causal=False, leakage check)...")
    # Output 1 without causal mask
    out_no_mask_1 = decoder_causal(x_dec1, x_enc, is_causal=False)
    # Output 2 without causal mask
    out_no_mask_2 = decoder_causal(x_dec2, x_enc, is_causal=False)
    
    # Due to no mask, the difference in the past MUST be greater than zero (leakage occurs)
    diff_no_mask = torch.max(torch.abs(out_no_mask_1[:, :40, :] - out_no_mask_2[:, :40, :])).item()
    print(f"Max difference in past timesteps (no mask): {diff_no_mask}")
    assert diff_no_mask > 1e-3, "Future changes did NOT affect past predictions! Masking might still be active!"
    print("Ablation No Causal Mask test passed (leakage successfully occurred as expected)!")
    print("----------------------------------------")

if __name__ == "__main__":
    test_decoder()

