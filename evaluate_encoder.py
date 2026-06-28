"""Encoder Historic Feature Extraction — Component Ablation Study.

Measures the contribution of each Encoder sub-component to the model's ability
to extract useful features from the historical lookback window.

The study trains five model variants on ETTh1 (pred_len=96) while keeping the
Decoder and Embedding layers *identical*. Only the Encoder's internal wiring
changes:

    Variant         | Attn | FFN | Residual | LayerNorm
    --------------- | ---- | --- | -------- | --------
    baseline        |  ✓   |  ✓  |    ✓     |    ✓
    no-attention    |  ✗   |  ✓  |    ✓     |    ✓
    no-ffn          |  ✓   |  ✗  |    ✓     |    ✓
    no-residual     |  ✓   |  ✓  |    ✗     |    ✓
    no-layernorm    |  ✓   |  ✓  |    ✓     |    ✗

By comparing ΔMSE = MSE_variant − MSE_baseline, we quantify how much each
component contributes to the Encoder's historic feature extraction capability.
A large positive ΔMSE means the removed component was critical.

Outputs (under ``experiments/``):
  - ``encoder_ablation.json``  — full config + metrics for each variant
  - ``encoder_ablation.csv``   — flat table for spreadsheet/plotting
  - A comparison table printed to stdout

Usage
-----
    # Standard capped run (~10 min on CPU)
    uv run python evaluate_encoder.py

    # Quick smoke test (~30 sec)
    uv run python evaluate_encoder.py --smoke

    # Full data (slow, ~1.5-2h on CPU)
    uv run python evaluate_encoder.py --full

    # GPU
    uv run python evaluate_encoder.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, "src")
from Dataloader import ETTDataset
from ts_ablation.configs.experiment import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from ts_ablation.encoder_arch.models import TransformerForecaster


# ─────────────────────────────────────────────────────────────────────────────
# Ablation variant definitions
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_VARIANTS = {
    "baseline": {
        "enc_use_attention": True,
        "enc_use_ffn": True,
        "enc_use_residual": True,
        "enc_use_layer_norm": True,
    },
    "no-attention": {
        "enc_use_attention": False,
        "enc_use_ffn": True,
        "enc_use_residual": True,
        "enc_use_layer_norm": True,
    },
    "no-ffn": {
        "enc_use_attention": True,
        "enc_use_ffn": False,
        "enc_use_residual": True,
        "enc_use_layer_norm": True,
    },
    "no-residual": {
        "enc_use_attention": True,
        "enc_use_ffn": True,
        "enc_use_residual": False,
        "enc_use_layer_norm": True,
    },
    "no-layernorm": {
        "enc_use_attention": True,
        "enc_use_ffn": True,
        "enc_use_residual": True,
        "enc_use_layer_norm": False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def pick_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(cfg: ExperimentConfig, split: str, max_windows: int | None,
                shuffle: bool) -> DataLoader:
    dataset = ETTDataset(
        csv_path=cfg.data.csv_path,
        seq_len=cfg.data.seq_len,
        label_len=cfg.data.label_len,
        pred_len=cfg.data.pred_len,
        split=split,
        features=cfg.data.features,
        target=cfg.data.target,
    )
    if max_windows is not None and max_windows < len(dataset):
        dataset = Subset(dataset, list(range(max_windows)))
    return DataLoader(dataset, batch_size=cfg.train.batch_size,
                      shuffle=shuffle, drop_last=False)


def infer_dims(loader: DataLoader) -> tuple[int, int, int]:
    batch = next(iter(loader))
    return batch["x_enc"].shape[-1], batch["x_dec"].shape[-1], batch["y"].shape[-1]


def build_model(cfg: ExperimentConfig, enc_in: int, dec_in: int,
                c_out: int) -> TransformerForecaster:
    return TransformerForecaster(
        enc_in=enc_in, dec_in=dec_in, c_out=c_out,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        e_layers=cfg.model.e_layers, d_layers=cfg.model.d_layers,
        d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
        activation=cfg.model.activation, pred_len=cfg.data.pred_len,
        enc_use_attention=cfg.model.enc_use_attention,
        enc_use_ffn=cfg.model.enc_use_ffn,
        enc_use_residual=cfg.model.enc_use_residual,
        enc_use_layer_norm=cfg.model.enc_use_layer_norm,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation loops
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, grad_clip) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for batch in loader:
        x_enc = batch["x_enc"].to(device)
        x_dec = batch["x_dec"].to(device)
        x_enc_mark = batch["x_enc_mark"].to(device)
        x_dec_mark = batch["x_dec_mark"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        y_hat = model(x_enc, x_enc_mark, x_dec, x_dec_mark)
        loss = nn.functional.mse_loss(y_hat, y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs
    return loss_sum / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    for batch in loader:
        x_enc = batch["x_enc"].to(device)
        x_dec = batch["x_dec"].to(device)
        x_enc_mark = batch["x_enc_mark"].to(device)
        x_dec_mark = batch["x_dec_mark"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x_enc, x_enc_mark, x_dec, x_dec_mark)
        bs = y.size(0)
        mse_sum += nn.functional.mse_loss(y_hat, y, reduction="mean").item() * bs
        mae_sum += nn.functional.l1_loss(y_hat, y, reduction="mean").item() * bs
        n += bs
    return {"mse": mse_sum / max(n, 1), "mae": mae_sum / max(n, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Run one ablation variant
# ─────────────────────────────────────────────────────────────────────────────
def run_variant(name: str, cfg: ExperimentConfig, device: torch.device,
                max_train: int | None, max_eval: int | None) -> dict:
    """Train + evaluate a single encoder ablation variant."""
    print(f"\n{'─' * 70}")
    print(f"▶ Variant: {name}")
    print(f"  Encoder: attn={'ON' if cfg.model.enc_use_attention else 'OFF'}  "
          f"ffn={'ON' if cfg.model.enc_use_ffn else 'OFF'}  "
          f"residual={'ON' if cfg.model.enc_use_residual else 'OFF'}  "
          f"layernorm={'ON' if cfg.model.enc_use_layer_norm else 'OFF'}")
    print(f"  Tag: {cfg.ablation_tag()}")

    train_loader = make_loader(cfg, "train", max_train, shuffle=True)
    val_loader = make_loader(cfg, "val", max_eval, shuffle=False)
    test_loader = make_loader(cfg, "test", max_eval, shuffle=False)

    enc_in, dec_in, c_out = infer_dims(train_loader)
    model = build_model(cfg, enc_in, dec_in, c_out).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}  |  train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        ep_t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     cfg.train.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        ep_dur = time.time() - ep_t0
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"    epoch {epoch:2d}/{cfg.train.epochs}  "
              f"train_mse={train_loss:.4f}  "
              f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}  "
              f"({ep_dur:.1f}s)")

        if val_metrics["mse"] < best_val - 1e-6:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.patience:
                print(f"    early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    print(f"  ✅ test_mse={test_metrics['mse']:.4f}  "
          f"test_mae={test_metrics['mae']:.4f}  "
          f"({elapsed:.1f}s)")

    # Free memory between variants
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": name,
        "ablation_tag": cfg.ablation_tag(),
        "config": cfg.model_dump(),
        "n_params": n_params,
        "best_val_mse": best_val,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "epochs_run": len(history),
        "elapsed_sec": elapsed,
        "history": history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_results(rows: list[dict]) -> str:
    """Render comparison table with ΔMSE relative to baseline."""
    baseline_mse = None
    for r in rows:
        if r["variant"] == "baseline":
            baseline_mse = r["test_mse"]
            break

    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("ENCODER ABLATION — HISTORIC FEATURE EXTRACTION ANALYSIS")
    lines.append("=" * 78)
    lines.append("")

    # Table header
    header = (f"{'Variant':<16} {'test_mse':>10} {'test_mae':>10} "
              f"{'Δ MSE':>10} {'Δ MSE %':>10} {'Impact':>10} {'Params':>12}")
    lines.append(header)
    lines.append("─" * 80)

    for r in rows:
        delta = r["test_mse"] - baseline_mse if baseline_mse else 0
        pct = (delta / baseline_mse * 100) if baseline_mse and baseline_mse > 0 else 0
        impact = "BASELINE" if r["variant"] == "baseline" else (
            "🔴 CRITICAL" if pct > 50 else
            "🟠 HIGH" if pct > 20 else
            "🟡 MEDIUM" if pct > 5 else
            "🟢 LOW"
        )
        lines.append(
            f"{r['variant']:<16} {r['test_mse']:>10.4f} {r['test_mae']:>10.4f} "
            f"{delta:>+10.4f} {pct:>+9.1f}% {impact:>10} {r['n_params']:>12,}"
        )

    lines.append("─" * 80)
    lines.append("")

    # Interpretation
    lines.append("📊 INTERPRETATION")
    lines.append("  Δ MSE = MSE_variant − MSE_baseline")
    lines.append("  A large positive Δ MSE means the removed component was")
    lines.append("  critical for extracting useful features from the lookback window.")
    lines.append("")

    # Rank components by importance
    components = [r for r in rows if r["variant"] != "baseline"]
    components.sort(key=lambda r: r["test_mse"], reverse=True)
    lines.append("🏆 COMPONENT IMPORTANCE RANKING (most → least critical):")
    for i, r in enumerate(components, 1):
        delta = r["test_mse"] - baseline_mse if baseline_mse else 0
        name_map = {
            "no-attention": "Multi-Head Self-Attention",
            "no-ffn": "Feed-Forward Network (FFN)",
            "no-residual": "Residual Connections",
            "no-layernorm": "Layer Normalization",
        }
        comp_name = name_map.get(r["variant"], r["variant"])
        lines.append(f"  {i}. {comp_name:<30} (Δ MSE = {delta:+.4f})")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Encoder Historic Feature Extraction — Component Ablation Study"
    )
    parser.add_argument("--csv", default="Data/ETTh1.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=500,
                        help="cap on #training windows (default: 500 for ~10 min)")
    parser.add_argument("--max-eval", type=int, default=200,
                        help="cap on #val/test windows")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="experiments")

    # Presets
    parser.add_argument("--smoke", action="store_true",
                        help="quick smoke test (~30 sec)")
    parser.add_argument("--full", action="store_true",
                        help="full data, no window cap (~1.5-2h on CPU)")

    # Select specific variants
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=list(ABLATION_VARIANTS.keys()),
                        help="run only these variants (default: all)")

    args = parser.parse_args()

    # Presets
    if args.smoke:
        args.epochs = 2
        args.max_train = 100
        args.max_eval = 50
    elif args.full:
        args.max_train = None
        args.max_eval = None

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # Select variants
    variant_names = args.variants or list(ABLATION_VARIANTS.keys())
    # Always run baseline first
    if "baseline" in variant_names and variant_names[0] != "baseline":
        variant_names.remove("baseline")
        variant_names.insert(0, "baseline")

    print("=" * 78)
    print("ENCODER HISTORIC FEATURE EXTRACTION — ABLATION STUDY")
    print(f"  Dataset:  {args.csv}")
    print(f"  Device:   {device}")
    print(f"  Epochs:   {args.epochs}  |  Patience: {args.patience}")
    print(f"  Windows:  train≤{args.max_train or 'ALL'}  eval≤{args.max_eval or 'ALL'}")
    print(f"  Variants: {', '.join(variant_names)}")
    mode = "SMOKE" if args.smoke else "FULL" if args.full else "STANDARD (capped)"
    print(f"  Mode:     {mode}")
    print("=" * 78)

    t0_total = time.time()
    rows = []

    for i, name in enumerate(variant_names):
        switches = ABLATION_VARIANTS[name]
        model_cfg = ModelConfig(
            d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
            dropout=0.05, activation="gelu",
            **switches,
        )
        cfg = ExperimentConfig(
            name=f"enc-ablation-{name}",
            model=model_cfg,
            train=TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                patience=args.patience,
                grad_clip=1.0,
                device=args.device,
            ),
            data=DataConfig(
                csv_path=args.csv,
                seq_len=96, label_len=48, pred_len=96,
                features="M", target="OT",
            ),
        )
        result = run_variant(name, cfg, device, args.max_train, args.max_eval)
        rows.append(result)

    total_elapsed = time.time() - t0_total

    # Print comparison
    summary = render_results(rows)
    print(summary)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "encoder_ablation.json"
    json_path.write_text(json.dumps({
        "study": "encoder_historic_feature_extraction",
        "dataset": args.csv,
        "device": str(device),
        "seed": args.seed,
        "mode": mode,
        "total_elapsed_sec": total_elapsed,
        "results": rows,
    }, indent=2))

    csv_path = out_dir / "encoder_ablation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "test_mse", "test_mae", "delta_mse",
                          "best_val_mse", "n_params", "epochs_run", "elapsed_sec"])
        baseline_mse = rows[0]["test_mse"] if rows else 0
        for r in rows:
            delta = r["test_mse"] - baseline_mse
            writer.writerow([
                r["variant"],
                f"{r['test_mse']:.6f}", f"{r['test_mae']:.6f}",
                f"{delta:+.6f}",
                f"{r['best_val_mse']:.6f}", r["n_params"],
                r["epochs_run"], f"{r['elapsed_sec']:.1f}",
            ])

    print(f"\n💾 Results saved:")
    print(f"   JSON → {json_path}")
    print(f"   CSV  → {csv_path}")
    print("\n✅ Encoder ablation study complete!")


if __name__ == "__main__":
    main()
