# ts-transformer-ablation

An **ablation study on Transformer architectures for time-series forecasting**, built on the
[ETT](https://github.com/zhouhaoyi/ETDataset) (Electricity Transformer Temperature) datasets.

The goal is not a single leaderboard number but to **measure the contribution of individual
components** — self-attention, feed-forward network, residual connections, normalization, causal
masking, and the decoder stack itself — to forecasting accuracy and computational cost. Each
component is an isolated, independently-toggleable module so it can be removed and re-measured.

> 📊 **Results at a glance:** see [`ABLATION_SYNTHESIS_REPORT.md`](ABLATION_SYNTHESIS_REPORT.md)
> for the full write-up (tables, figures, reasoning, threats-to-validity), or the per-study docs
> [`ENCODER_ABLATION_RESULTS.md`](ENCODER_ABLATION_RESULTS.md) and
> [`DECODER_ABLATION_RESULTS.md`](DECODER_ABLATION_RESULTS.md).

---

## Two architecture variants

This repo hosts **two independent ablation tracks**, each in its own self-contained model package
to avoid cross-contamination. They share the config, data loader, and training loop, but ship
distinct model code:

| Package | Question it answers | Components toggled |
|---------|---------------------|--------------------|
| `src/ts_ablation/encoder_arch/` | What lets the **encoder** extract features from the lookback window? | self-attention · FFN · residual · LayerNorm |
| `src/ts_ablation/decoder_arch/` | What does the **decoder** need to generate forecasts? | decoder self-attention · causal mask · encoder-only (no-decoder) |

Both are a standard Informer/Autoformer-style **encoder–decoder Transformer** with a Pre-LN layout:

```
x_enc, x_enc_mark ─► DataEmbedding ─► Encoder (EncoderLayer × e_layers) ─┐ (memory)
                                                                          ▼
x_dec, x_dec_mark ─► DataEmbedding ─► Decoder (DecoderLayer × d_layers) ──► Linear ─► ŷ
                                       │  • causal self-attention
                                       │  • cross-attention to encoder memory
```

The decoder input follows the Informer convention: the first `label_len` steps are the known tail
of the lookback window, the last `pred_len` steps are zero placeholders to predict. Decoder
self-attention is **causal**, so a placeholder only attends to earlier steps.

### Headline findings

| Component | Where | Δ MSE when removed | Verdict |
|-----------|-------|:------------------:|---------|
| **Self-Attention** | Encoder | **+28.5%** | 🔴 Core — only cross-timestep mixing mechanism |
| FFN | Encoder | +10.0% | 🟠 Significant — per-position non-linearity |
| Causal Mask | Decoder | +8.4% | 🟠 Prevents future-leakage |
| Residual | Encoder | +5.4% | 🟠 Stabilizes gradient / preserves signal |
| LayerNorm | Encoder | −0.7% | 🟢 Negligible at shallow depth |
| Decoder Self-Attn | Decoder | −3.3% | 🟢 Near-redundant (cross-attn carries it) |
| **No-Decoder → Linear** | Architecture | **−15.1%** | 🔵 Large linear head *beats* the Transformer on small ETTh1 (à la DLinear) |

> ⚠️ The two studies use **different configs** (encoder: `d_model=512, pred_len=96`; decoder:
> `d_model=64, pred_len=24`) — Δ values are only comparable *within* a study. All runs are
> smoke-scale (few epochs / capped windows). See report §3 and §9.

---

## Project layout

```
.
├── Data/                          # ETTh1/h2/m1/m2 CSVs (date + 7 channels, target = OT)
├── Dataloader.py                  # ETTDataset: windows + calendar marks, train/val/test split
├── train.py                       # shared train→val→test loop (+ checkpoint saving)
├── benchmark.py                   # multi-dataset / multi-horizon standard benchmark
├── test.py                        # cross-module correctness / causality checks
├── evaluate_encoder.py            # ► encoder_arch ablation runner (5 variants)
├── evaluate_decoder.py            # ► decoder_arch ablation runner (4 variants)
├── evaluate.py                    # evaluate a trained checkpoint on the test set (+ plots)
├── scripts/make_figures.py        # regenerate all comparison figures (matplotlib + seaborn)
├── experiments/                   # run logs (CSV/JSON) + figures/  (publication-ready PNGs)
├── ABLATION_SYNTHESIS_REPORT.md   # combined analysis — start here
├── ENCODER_ABLATION_RESULTS.md    # encoder study write-up
├── DECODER_ABLATION_RESULTS.md    # decoder study write-up
└── src/ts_ablation/
    ├── encoder_arch/models/       # encoder-ablation model package
    ├── decoder_arch/models/       # decoder-ablation model package
    ├── configs/                   # experiment.py (ablation switches) + standard.py
    └── utils/
```

### Model modules (identical interface in each `*_arch/models/`)

| Module | Class | Role |
|--------|-------|------|
| `embedding.py` | `DataEmbedding` | value (Conv1d) + positional (sin/cos) + temporal (calendar) embeddings |
| `multi_head_attention.py` | `MultiHeadAttention` | SDPA attention; causal + padding masks, optional weight output |
| `feed_forward.py` | `FeedForward` | position-wise FFN (`gelu`/`relu`) |
| `encoder_layer.py` / `encoder.py` | `EncoderLayer`, `Encoder` | Pre-LN self-attention + FFN stack (with ablation toggles) |
| `decoder.py` | `DecoderLayer`, `Decoder` | Pre-LN causal self-attn + cross-attn + FFN stack |
| `forecaster.py` | `TransformerForecaster` | end-to-end model joining all of the above |

---

## Setup

This project uses [**uv**](https://docs.astral.sh/uv/) for **all** environment and dependency
management (never `pip`/`conda`/`venv`).

```bash
uv venv
uv sync
```

---

## Running the ablations

```bash
# Encoder components (self-attention / FFN / residual / LayerNorm)
uv run python evaluate_encoder.py --smoke      # ~30s sanity check
uv run python evaluate_encoder.py              # standard capped run (~10 min)
uv run python evaluate_encoder.py --full       # full data (~1.5–2h CPU)

# Decoder components (self-attention / causal mask / no-decoder)
uv run python evaluate_decoder.py --smoke
uv run python evaluate_decoder.py

# Run on GPU
uv run python evaluate_encoder.py --device cuda
```

Each runner trains every variant under an identical config, prints a ranked importance table, and
writes results to `experiments/{encoder,decoder}_ablation.csv` + `.json`.
Common flags: `--csv`, `--epochs`, `--batch-size`, `--lr`, `--patience`,
`--max-train`/`--max-eval` (window caps), `--device {cpu,cuda,mps}`.

## Training & benchmarking

```bash
uv run python train.py                          # default small run on ETTh1 (decoder_arch)
uv run python train.py --smoke                  # 1 epoch, ~32 windows
uv run python train.py --epochs 10 --max-train 1500 --csv Data/ETTh2.csv

uv run python benchmark.py --quick              # fast end-to-end smoke
uv run python benchmark.py --datasets ETTh1 ETTh2 --horizons 96 192
```

`train.py` saves a checkpoint per run (`<name>_<ablation_tag>.pt`) and a JSON log to
`experiments/`. Evaluate a saved checkpoint:

```bash
uv run python evaluate.py --checkpoint experiments/<name>.pt --plot
uv run python evaluate.py --from-log experiments/<run>.json
```

## Figures

All comparison charts are regenerated from the result files with matplotlib + seaborn:

```bash
uv run python scripts/make_figures.py           # → experiments/figures/*.png
```

Produces: encoder/decoder ablation bars, combined importance ranking, MSE×MAE comparison,
depth-scaling + learning curves, and the ETTh1 dataset overview (used throughout the report).

## Testing

```bash
uv run python test.py        # causality + cross-module integration checks
```

Each model module also has a `main()` self-test with realistic ETT shapes — run any directly,
e.g. `uv run python -m ts_ablation.decoder_arch.models.decoder`.

---

## Ablation configuration

`ExperimentConfig` (in `src/ts_ablation/configs/experiment.py`) centralizes every knob and exposes
`ablation_tag()` used to name experiment logs. Switches span both tracks:

- **Architecture:** `attention_type`, `use_positional_encoding`, `use_decomposition`,
  `use_decoder`, `e_layers`/`d_layers`.
- **Encoder ablation:** `enc_use_attention`, `enc_use_ffn`, `enc_use_residual`, `enc_use_layer_norm`.
- **Decoder ablation:** `dec_use_self_attention`, `dec_use_causal_mask`.

---

## Data

The **ETT** family — hourly (`ETTh1/h2`, ~17.4k rows) and 15-minute (`ETTm1/m2`, ~69.7k rows),
each with 6 load covariates + the target **OT** (oil temperature), spanning ~2 years. Forecasting
is multivariate (`features="M"`); metrics are reported in standardized (scaled) space.
See report §1 for distribution and correlation analysis.
</content>
</invoke>
