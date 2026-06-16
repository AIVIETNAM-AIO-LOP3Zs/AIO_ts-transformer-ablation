# ts-transformer-ablation

An **ablation study on Transformer architectures for time-series forecasting**, built on the
[ETT](https://github.com/zhouhaoyi/ETDataset) (Electricity Transformer Temperature) datasets.

The goal is not a single leaderboard number but to **measure the contribution of individual
components** — attention mechanism, positional encoding, normalization placement, feed-forward
projection — to forecasting accuracy and computational cost. The codebase is organized so each
component is an isolated, independently-testable module that can be swapped or removed.

---

## Architecture

A standard Informer/Autoformer-style **encoder–decoder Transformer** with a Pre-LN layout:

```
x_enc, x_enc_mark ─► DataEmbedding ─► Encoder (EncoderLayer × e_layers) ─┐ (memory)
                                                                          ▼
x_dec, x_dec_mark ─► DataEmbedding ─► Decoder (DecoderLayer × d_layers) ──► Linear ─► ŷ
                                       │  • causal self-attention
                                       │  • cross-attention to encoder memory
```

The decoder input follows the Informer convention: the first `label_len` steps are the known tail
of the lookback window, and the last `pred_len` steps are zero placeholders to be predicted. The
decoder's self-attention is **causal**, so a placeholder step only attends to earlier steps.

### Modules (`src/ts_ablation/models/`)

| Module | Class | Role |
|--------|-------|------|
| `embedding.py` | `DataEmbedding` | value (Conv1d) + positional (sin/cos) + temporal (calendar) embeddings |
| `multi_head_attention.py` | `MultiHeadAttention` | SDPA-based attention; supports causal + padding masks and optional weight output |
| `feed_forward.py` | `FeedForward` | position-wise FFN (`gelu`/`relu`) |
| `encoder_layer.py` / `encoder.py` | `EncoderLayer`, `Encoder` | Pre-LN self-attention + FFN stack |
| `decoder.py` | `DecoderLayer`, `Decoder` | Pre-LN causal self-attn + cross-attn + FFN stack |
| `forecaster.py` | `TransformerForecaster` | end-to-end model joining all of the above |

Each module has a `main()` self-test with realistic ETT shapes — run any of them directly
(e.g. `uv run python -m ts_ablation.models.decoder`).

---

## Project layout

```
.
├── Data/                       # ETTh1/h2/m1/m2 CSVs (date + 7 channels)
├── Dataloader.py               # ETTDataset: windows + calendar marks, train/val/test split
├── train.py                    # small end-to-end training pipeline (see below)
├── test.py                     # cross-module correctness/causality checks
├── REVIEW.md                   # correctness review of the modules (all issues resolved)
├── experiments/                # JSON logs of runs (config + metrics) for ablation tracking
└── src/ts_ablation/
    ├── models/                 # the building blocks above
    └── configs/experiment.py   # ModelConfig / TrainConfig / DataConfig / ExperimentConfig
```

---

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) for all environment and dependency management.

```bash
uv venv
uv sync
```

## Training

`train.py` runs a **small, CPU-friendly** train → validate → test cycle. It is sized for fast
iteration on a laptop rather than peak accuracy (small model, subset of windows, short horizon):

```bash
uv run python train.py                 # default small run on ETTh1
uv run python train.py --smoke         # 1 epoch, ~32 windows (sanity check)
uv run python train.py --epochs 10 --max-train 1500 --csv Data/ETTh2.csv
```

Key flags: `--csv`, `--epochs`, `--batch-size`, `--lr`, `--max-train`/`--max-eval` (window caps),
`--device {cpu,cuda,mps}`. Runs are driven by `ExperimentConfig` and each writes a JSON log
(config + per-epoch metrics) to `experiments/` for ablation tracking.

> **Why small?** ETTh1 is hourly, so the dataset's `[hour, day, weekday]` marks align with the
> sampling rate. Capping the number of windows is the biggest lever for a weak device; the tiny
> model keeps the *architecture* identical to the full one, so conclusions about *which component
> matters* still transfer. See `train.py`'s docstring for the full reasoning.

## Testing

```bash
uv run python test.py        # causality + cross-module integration checks
```

---

## Ablation configuration

`ExperimentConfig` (in `src/ts_ablation/configs/experiment.py`) centralizes every knob and exposes
an `ablation_tag()` used to name experiment logs. Switches include `attention_type`,
`use_positional_encoding`, `use_decomposition`, `use_decoder`, and layer counts — the dimensions
this study sweeps over.
