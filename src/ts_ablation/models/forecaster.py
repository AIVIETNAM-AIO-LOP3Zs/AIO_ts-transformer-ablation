import torch
import torch.nn as nn

from .embedding import DataEmbedding
from .encoder import Encoder
from .decoder import Decoder


class TransformerForecaster(nn.Module):
    """End-to-end encoder-decoder Transformer for time-series forecasting.

    This is the top-level module that finally joins the previously-isolated
    building blocks into a single execution flow:

        x_enc, x_enc_mark ─► DataEmbedding ─► Encoder ─────────────┐ (memory)
                                                                   ▼
        x_dec, x_dec_mark ─► DataEmbedding ─► Decoder(self-causal, cross) ─►
                                                  Linear projection ─► y_hat

    The decoder input follows the Informer convention: the first ``label_len``
    steps are the known tail of the lookback window and the last ``pred_len``
    steps are zero placeholders to be predicted. The decoder's self-attention
    is causal, so a placeholder step only attends to earlier steps.

    Args:
        enc_in:   number of input features fed to the encoder.
        dec_in:   number of input features fed to the decoder.
        c_out:    number of output (target) features to predict.
        d_model, n_heads, e_layers, d_layers, d_ff, dropout, activation:
                  standard Transformer hyper-parameters.
        pred_len: forecast horizon; only the last ``pred_len`` decoder steps
                  are returned.
        embed_type: 'fixed' (sin/cos calendar embeddings) or 'learned'.
        max_len:  max sequence length supported by the positional embedding.
    """

    def __init__(self, enc_in, dec_in, c_out, d_model=512, n_heads=8,
                 e_layers=2, d_layers=1, d_ff=2048, dropout=0.1,
                 activation='gelu', pred_len=24, embed_type='fixed',
                 max_len=5000):
        super().__init__()
        self.pred_len = pred_len

        # Separate embeddings for encoder and decoder streams (they may have a
        # different number of input features, e.g. in 'MS' mode).
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed_type, max_len, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed_type, max_len, dropout)

        self.encoder = Encoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=e_layers, dropout=dropout, activation=activation,
        )
        self.decoder = Decoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=d_layers, dropout=dropout, activation=activation,
        )

        # Map the decoder's d_model representation back to the target space.
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_enc_mark, x_dec, x_dec_mark,
                enc_mask=None, dec_self_mask=None, dec_cross_mask=None):
        """
        Args:
            x_enc:      (B, seq_len, enc_in)            encoder values
            x_enc_mark: (B, seq_len, 3)                 encoder calendar marks
            x_dec:      (B, label_len + pred_len, dec_in) decoder values
            x_dec_mark: (B, label_len + pred_len, 3)    decoder calendar marks

        Returns:
            (B, pred_len, c_out) forecast for the prediction horizon.
        """
        # Encoder: build memory from the lookback window.
        enc_in = self.enc_embedding(x_enc, x_enc_mark)
        enc_out = self.encoder(enc_in, attn_mask=enc_mask)

        # Decoder: causal self-attention over its own sequence + cross-attention
        # to the encoder memory. (Causality is enforced inside DecoderLayer.)
        dec_in = self.dec_embedding(x_dec, x_dec_mark)
        dec_out = self.decoder(dec_in, enc_out,
                               x_mask=dec_self_mask, cross_mask=dec_cross_mask)

        # Project to target space and keep only the forecast horizon.
        out = self.projection(dec_out)          # (B, label_len + pred_len, c_out)
        return out[:, -self.pred_len:, :]       # (B, pred_len, c_out)


def main():
    # Realistic ETT-style end-to-end smoke test.
    B = 8
    seq_len, label_len, pred_len = 96, 48, 24
    dec_len = label_len + pred_len            # 72
    enc_in = dec_in = c_out = 7               # 'M' multivariate mode
    d_model, n_heads = 64, 4                  # small config (device-friendly)

    x_enc = torch.randn(B, seq_len, enc_in)
    x_dec = torch.randn(B, dec_len, dec_in)
    x_enc_mark = torch.zeros(B, seq_len, 3)
    x_dec_mark = torch.zeros(B, dec_len, 3)

    model = TransformerForecaster(
        enc_in=enc_in, dec_in=dec_in, c_out=c_out,
        d_model=d_model, n_heads=n_heads, e_layers=2, d_layers=1,
        d_ff=128, dropout=0.1, activation='gelu', pred_len=pred_len,
    )
    y_hat = model(x_enc, x_enc_mark, x_dec, x_dec_mark)
    print(f"TransformerForecaster output: {tuple(y_hat.shape)}  expected: ({B}, {pred_len}, {c_out})")
    assert y_hat.shape == (B, pred_len, c_out)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (small config): {n_params:,}")
    print("TransformerForecaster test passed.")


if __name__ == "__main__":
    main()
