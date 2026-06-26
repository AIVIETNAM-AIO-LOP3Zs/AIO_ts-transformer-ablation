from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    # Architecture
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 4          # encoder layers
    d_layers: int = 2          # decoder layers
    d_ff: int = 2048
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"

    # Ablation switches
    attention_type: Literal["full", "prob_sparse", "autocorrelation"] = "full"
    use_positional_encoding: bool = True
    use_decomposition: bool = False    # trend/seasonal decomp (Autoformer-style)
    use_decoder: bool = True           # encoder-only vs encoder-decoder

    # Encoder component ablation switches
    enc_use_attention: bool = True     # toggle self-attention in encoder layers
    enc_use_ffn: bool = True           # toggle FFN in encoder layers
    enc_use_residual: bool = True      # toggle residual/skip connections
    enc_use_layer_norm: bool = True    # toggle LayerNorm before sub-layers


class TrainConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 3               # early stopping
    grad_clip: float = 1.0
    device: str = "cpu"


class DataConfig(BaseModel):
    csv_path: str
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    features: Literal["M", "S", "MS"] = "M"
    target: str = "OT"


class ExperimentConfig(BaseModel):
    name: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig

    def ablation_tag(self) -> str:
        m = self.model
        parts = [
            f"attn={m.attention_type}",
            f"pe={'on' if m.use_positional_encoding else 'off'}",
            f"decomp={'on' if m.use_decomposition else 'off'}",
            f"dec={'on' if m.use_decoder else 'off'}",
            f"L={m.e_layers}",
        ]
        # Append encoder ablation switches only when non-default (off)
        if not m.enc_use_attention:
            parts.append("enc-attn=off")
        if not m.enc_use_ffn:
            parts.append("enc-ffn=off")
        if not m.enc_use_residual:
            parts.append("enc-res=off")
        if not m.enc_use_layer_norm:
            parts.append("enc-ln=off")
        return "_".join(parts)
