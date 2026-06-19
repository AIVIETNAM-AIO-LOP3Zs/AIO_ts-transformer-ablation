"""Evaluate a trained TransformerForecaster checkpoint on the ETT test set.

Load a saved model checkpoint (.pt file) and compute performance metrics
(MSE, MAE, RMSE) on the test split. Optionally plot predictions vs ground truth.

Usage
-----
    # Evaluate a checkpoint (uses same config that was saved during training)
    uv run python evaluate.py --checkpoint experiments/best_model.pt

    # Evaluate with a specific dataset
    uv run python evaluate.py --checkpoint experiments/best_model.pt --csv Data/ETTh1.csv

    # Evaluate + plot first N samples
    uv run python evaluate.py --checkpoint experiments/best_model.pt --plot --n-plot 5

    # Evaluate using a training log JSON (auto-loads config from it)
    uv run python evaluate.py --from-log experiments/small-etth1_attn=full_pe=on_decomp=off_dec=on_L=5.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
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
from ts_ablation.models import TransformerForecaster


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def pick_device(requested: str) -> torch.device:
    """Resolve the requested device, falling back to CPU when unavailable."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_test_loader(cfg: ExperimentConfig, max_windows: int | None) -> DataLoader:
    """Build a DataLoader for the test split."""
    dataset = ETTDataset(
        csv_path=cfg.data.csv_path,
        seq_len=cfg.data.seq_len,
        label_len=cfg.data.label_len,
        pred_len=cfg.data.pred_len,
        split="test",
        features=cfg.data.features,
        target=cfg.data.target,
    )
    if max_windows is not None and max_windows < len(dataset):
        dataset = Subset(dataset, list(range(max_windows)))
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False)


def infer_dims(loader: DataLoader) -> tuple[int, int, int]:
    """Peek at one batch to get enc_in, dec_in, c_out dimensions."""
    batch = next(iter(loader))
    enc_in = batch["x_enc"].shape[-1]
    dec_in = batch["x_dec"].shape[-1]
    c_out = batch["y"].shape[-1]
    return enc_in, dec_in, c_out


def build_model(cfg: ExperimentConfig, enc_in: int, dec_in: int,
                c_out: int) -> TransformerForecaster:
    """Instantiate a TransformerForecaster from an ExperimentConfig."""
    return TransformerForecaster(
        enc_in=enc_in,
        dec_in=dec_in,
        c_out=c_out,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        e_layers=cfg.model.e_layers,
        d_layers=cfg.model.d_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
        pred_len=cfg.data.pred_len,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    """Compute MSE, MAE, RMSE over the entire loader.

    Also collects all predictions and ground truths for optional plotting.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        x_enc = batch["x_enc"].to(device)
        x_dec = batch["x_dec"].to(device)
        x_enc_mark = batch["x_enc_mark"].to(device)
        x_dec_mark = batch["x_dec_mark"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x_enc, x_enc_mark, x_dec, x_dec_mark)

        all_preds.append(y_hat.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)     # (N, pred_len, c_out)
    targets = np.concatenate(all_targets, axis=0)  # (N, pred_len, c_out)

    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(preds),
        "preds": preds,
        "targets": targets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel metrics
# ─────────────────────────────────────────────────────────────────────────────
def per_channel_metrics(preds: np.ndarray, targets: np.ndarray,
                        columns: list[str] | None = None) -> list[dict]:
    """Compute MSE, MAE, RMSE per output channel."""
    n_channels = preds.shape[-1]
    results = []
    for ch in range(n_channels):
        p = preds[:, :, ch]
        t = targets[:, :, ch]
        mse = float(np.mean((p - t) ** 2))
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(mse))
        name = columns[ch] if columns else f"channel_{ch}"
        results.append({"channel": name, "mse": mse, "mae": mae, "rmse": rmse})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_predictions(preds: np.ndarray, targets: np.ndarray,
                     n_samples: int = 5, save_path: str | None = None):
    """Plot predicted vs ground truth for the first n_samples windows.

    Each row is a sample, each column is an output channel.
    """
    import matplotlib.pyplot as plt

    n_samples = min(n_samples, len(preds))
    n_channels = preds.shape[-1]
    # Limit columns to 4 for readability
    n_cols = min(n_channels, 4)

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(4 * n_cols, 3 * n_samples),
                             squeeze=False)
    fig.suptitle("Predictions (orange) vs Ground Truth (blue)", fontsize=14, y=1.02)

    for i in range(n_samples):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.plot(targets[i, :, j], label="Ground Truth", color="#1f77b4",
                    linewidth=1.5)
            ax.plot(preds[i, :, j], label="Prediction", color="#ff7f0e",
                    linewidth=1.5, linestyle="--")
            if i == 0:
                ax.set_title(f"Channel {j}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Sample {i}", fontsize=9)
            ax.tick_params(labelsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved → {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading from training log JSON
# ─────────────────────────────────────────────────────────────────────────────
def load_config_from_log(log_path: str) -> ExperimentConfig:
    """Parse an ExperimentConfig from a training log JSON file."""
    with open(log_path) as f:
        log = json.load(f)
    return ExperimentConfig(**log["config"])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TransformerForecaster on the ETT test set."
    )
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the model checkpoint (.pt file)")

    # Config source (choose one)
    parser.add_argument("--from-log", type=str, default=None,
                        help="Load config from a training log JSON file "
                             "(e.g. experiments/small-etth1_....json)")

    # Data overrides
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to ETT csv (overrides config)")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Cap on #test windows")

    # Model overrides (used when no --from-log is given)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=5)
    parser.add_argument("--d-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)

    # Plotting
    parser.add_argument("--plot", action="store_true",
                        help="Plot predictions vs ground truth")
    parser.add_argument("--n-plot", type=int, default=5,
                        help="Number of samples to plot")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Save plot to file instead of showing")

    # Device
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    # ── Build config ─────────────────────────────────────────────────────
    if args.from_log:
        print(f"📂 Loading config from log: {args.from_log}")
        cfg = load_config_from_log(args.from_log)
        # Override csv path if specified
        if args.csv:
            cfg.data.csv_path = args.csv
    else:
        model_cfg = ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            dropout=0.1,
            activation="gelu",
        )
        train_cfg = TrainConfig(
            batch_size=args.batch_size,
            device=args.device,
        )
        data_cfg = DataConfig(
            csv_path=args.csv or "Data/ETTh1.csv",
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            features="M",
            target="OT",
        )
        cfg = ExperimentConfig(name="eval", model=model_cfg, train=train_cfg, data=data_cfg)

    # Override batch_size and device from CLI
    cfg.train.batch_size = args.batch_size
    cfg.train.device = args.device
    device = pick_device(cfg.train.device)

    # ── Load data ────────────────────────────────────────────────────────
    test_loader = make_test_loader(cfg, args.max_test)
    enc_in, dec_in, c_out = infer_dims(test_loader)

    print("=" * 70)
    print("📊 EVALUATION")
    print(f"   Data:   {cfg.data.csv_path}  (test split)")
    print(f"   Model:  d_model={cfg.model.d_model}  heads={cfg.model.n_heads}  "
          f"e_layers={cfg.model.e_layers}  d_layers={cfg.model.d_layers}  d_ff={cfg.model.d_ff}")
    print(f"   Dims:   enc_in={enc_in}  dec_in={dec_in}  c_out={c_out}")
    print(f"   Device: {device}")
    print(f"   Test windows: {len(test_loader.dataset)}")
    print("=" * 70)

    # ── Build model ──────────────────────────────────────────────────────
    model = build_model(cfg, enc_in, dec_in, c_out).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Params: {n_params:,}")

    # ── Load checkpoint ──────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        print(f"   Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("   ✅ Checkpoint loaded successfully")
    else:
        print("   ⚠️  No checkpoint provided — evaluating with random weights")
        print("      (Use --checkpoint to load a trained model)")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print()
    print("⏳ Running evaluation...")
    t0 = time.time()
    results = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    # ── Print results ────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("📈 OVERALL METRICS")
    print(f"   MSE:       {results['mse']:.6f}")
    print(f"   MAE:       {results['mae']:.6f}")
    print(f"   RMSE:      {results['rmse']:.6f}")
    print(f"   Samples:   {results['n_samples']}")
    print(f"   Time:      {elapsed:.2f}s")
    print("─" * 70)

    # ── Per-channel metrics ──────────────────────────────────────────────
    # Try to get column names from the dataset
    dataset = test_loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    columns = getattr(dataset, "columns", None)

    ch_metrics = per_channel_metrics(results["preds"], results["targets"], columns)
    print()
    print("📊 PER-CHANNEL METRICS")
    print(f"   {'Channel':<15} {'MSE':>10} {'MAE':>10} {'RMSE':>10}")
    print(f"   {'─' * 15} {'─' * 10} {'─' * 10} {'─' * 10}")
    for m in ch_metrics:
        print(f"   {m['channel']:<15} {m['mse']:>10.6f} {m['mae']:>10.6f} {m['rmse']:>10.6f}")
    print("─" * 70)

    # ── Save results to JSON ─────────────────────────────────────────────
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    eval_log = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "from_log": args.from_log,
        "config": cfg.model_dump(),
        "n_params": n_params,
        "test_windows": results["n_samples"],
        "metrics": {
            "mse": results["mse"],
            "mae": results["mae"],
            "rmse": results["rmse"],
        },
        "per_channel": ch_metrics,
        "elapsed_sec": elapsed,
    }
    eval_path = out_dir / f"eval_{cfg.name}_{cfg.ablation_tag()}.json"
    eval_path.write_text(json.dumps(eval_log, indent=2))
    print(f"\n💾 Evaluation log saved → {eval_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    if args.plot:
        save_path = args.save_plot or str(out_dir / f"eval_plot_{cfg.name}.png")
        plot_predictions(
            results["preds"], results["targets"],
            n_samples=args.n_plot, save_path=save_path,
        )

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
