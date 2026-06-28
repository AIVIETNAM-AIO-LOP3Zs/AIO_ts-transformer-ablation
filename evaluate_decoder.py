"""Decoder Components & Architecture — Ablation Study.

This script runs 4 distinct variants of the Decoder architecture to analyze
the importance of Self-Attention, Causal Masking, and the Decoder stack itself:

1. baseline: Standard Encoder-Decoder Transformer
2. no-self-attention: Decoder Self-Attention bypassed (Cross-Attention and FFN only)
3. no-causal-mask: Decoder Self-Attention run WITHOUT causal masking (future leakage allowed)
4. no-decoder: Bypasses Decoder completely; Encoder outputs projected directly to forecast horizon

Usage
-----
    uv run python evaluate_decoder.py --smoke            # Fast sanity check (~30 sec)
    uv run python evaluate_decoder.py                    # Standard capped run (~5-10 min)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project imports
sys.path.insert(0, "src")
from Dataloader import ETTDataset
from ts_ablation.configs.experiment import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from ts_ablation.decoder_arch.models import TransformerForecaster
from train import evaluate, infer_dims, make_loader, pick_device, train_one_epoch

# Define ablation study configuration sweeps
ABLATION_VARIANTS = {
    "baseline": {
        "use_decoder": True,
        "dec_use_self_attention": True,
        "dec_use_causal_mask": True,
    },
    "no-self-attention": {
        "use_decoder": True,
        "dec_use_self_attention": False,
        "dec_use_causal_mask": True,
    },
    "no-causal-mask": {
        "use_decoder": True,
        "dec_use_self_attention": True,
        "dec_use_causal_mask": False,
    },
    "no-decoder": {
        "use_decoder": False,
        "dec_use_self_attention": True,
        "dec_use_causal_mask": True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Runner for a single variant
# ─────────────────────────────────────────────────────────────────────────────
def run_variant(name: str, cfg: ExperimentConfig, device: torch.device,
                max_train: int | None, max_eval: int | None) -> dict:
    """Train and evaluate a single architectural variant."""
    print(f"\n* Running variant: {name}")
    print(f"   Config: {cfg.model.model_dump(include={'use_decoder', 'dec_use_self_attention', 'dec_use_causal_mask'})}")

    # Build data loaders
    train_loader = make_loader(cfg, "train", max_train, shuffle=True)
    val_loader = make_loader(cfg, "val", max_eval, shuffle=False)
    test_loader = make_loader(cfg, "test", max_eval, shuffle=False)

    # Infer input/output dimensions from a mock batch
    enc_in, dec_in, c_out = infer_dims(train_loader)

    # Initialize model with current ablation config
    model = TransformerForecaster(
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
        use_decoder=cfg.model.use_decoder,
        dec_use_self_attention=cfg.model.dec_use_self_attention,
        dec_use_causal_mask=cfg.model.dec_use_causal_mask,
        seq_len=cfg.data.seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Trainable Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    epochs_run = 0
    t0 = time.time()

    # Training Loop with Early Stopping on Val MSE
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg.train.grad_clip
        )
        val_metrics = evaluate(model, val_loader, device)
        epochs_run = epoch

        print(f"   Epoch {epoch:02d}/{cfg.train.epochs:02d} | "
              f"Train MSE: {train_loss:.4f} | "
              f"Val MSE: {val_metrics['mse']:.4f} | Val MAE: {val_metrics['mae']:.4f}")

        if val_metrics["mse"] < best_val - 1e-6:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.patience:
                print(f"   Early stopping triggered at epoch {epoch}")
                break

    # Restore the best validation state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate final metrics on the Test set
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    print(f"   [Done] Test MSE: {test_metrics['mse']:.4f} | Test MAE: {test_metrics['mae']:.4f} ({elapsed:.1f}s)")

    return {
        "variant": name,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "best_val_mse": best_val,
        "n_params": n_params,
        "epochs_run": epochs_run,
        "elapsed_sec": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_results(rows: list[dict]) -> str:
    """Render comparison table with dMSE relative to baseline."""
    baseline_mse = None
    for r in rows:
        if r["variant"] == "baseline":
            baseline_mse = r["test_mse"]
            break

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("DECODER ABLATION STUDY - EMPIRICAL EVALUATION RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Table header
    header = (f"{'Variant':<20} {'test_mse':>10} {'test_mae':>10} "
              f"{'d MSE':>10} {'d MSE %':>10} {'Impact':>10} {'Params':>12}")
    lines.append(header)
    lines.append("-" * 82)

    for r in rows:
        delta = r["test_mse"] - baseline_mse if baseline_mse is not None else 0
        pct = (delta / baseline_mse * 100) if baseline_mse and baseline_mse > 0 else 0
        impact = "BASELINE" if r["variant"] == "baseline" else (
            "CRITICAL" if pct > 30 else
            "HIGH" if pct > 10 else
            "MEDIUM" if pct > 2 else
            "LOW"
        )
        lines.append(
            f"{r['variant']:<20} {r['test_mse']:>10.4f} {r['test_mae']:>10.4f} "
            f"{delta:>+10.4f} {pct:>+9.1f}% {impact:>10} {r['n_params']:>12,}"
        )

    lines.append("-" * 82)
    lines.append("")

    # Interpretation
    lines.append("[Interpretation]")
    lines.append("  d MSE = MSE_variant - MSE_baseline")
    lines.append("  A positive d MSE indicates performance degraded when removing the component.")
    lines.append("  No Causal Masking (leakage) leading to poor generalization is expected.")
    lines.append("")

    # Rank components by importance
    components = [r for r in rows if r["variant"] != "baseline"]
    components.sort(key=lambda r: r["test_mse"], reverse=True)
    lines.append("[Rank] COMPONENT IMPORTANCE RANKING (most -> least critical to Decoder):")
    for i, r in enumerate(components, 1):
        delta = r["test_mse"] - baseline_mse if baseline_mse is not None else 0
        name_map = {
            "no-decoder": "Complete Decoder Stack (Bypassed)",
            "no-causal-mask": "Causal Masking (Future Leakage)",
            "no-self-attention": "Decoder Self-Attention Layer",
        }
        comp_name = name_map.get(r["variant"], r["variant"])
        lines.append(f"  {i}. {comp_name:<35} (d MSE = {delta:+.4f})")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Force stdout encoding to UTF-8 to prevent potential encoding crashes on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        description="Decoder Architecture Components — Ablation Study"
    )
    parser.add_argument("--csv", default="Data/ETTh1.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=500,
                        help="cap on #training windows for CPU-friendly iteration")
    parser.add_argument("--max-eval", type=int, default=200,
                        help="cap on #val/test windows")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="experiments")

    # Presets
    parser.add_argument("--smoke", action="store_true",
                        help="quick smoke test (~30 sec)")
    parser.add_argument("--full", action="store_true",
                        help="full dataset run")

    # Select specific variants
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=list(ABLATION_VARIANTS.keys()))

    args = parser.parse_args()

    # Apply presets
    if args.smoke:
        args.epochs = 2
        args.max_train = 128
        args.max_eval = 64
    elif args.full:
        args.max_train = None
        args.max_eval = None

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # Build evaluation sweep sequence (always run baseline first)
    variant_names = args.variants or list(ABLATION_VARIANTS.keys())
    if "baseline" in variant_names and variant_names[0] != "baseline":
        variant_names.remove("baseline")
        variant_names.insert(0, "baseline")

    print("=" * 80)
    print("DECODER COMPONENTS & ARCHITECTURE - ABLATION STUDY")
    print(f"  Dataset:  {args.csv}")
    print(f"  Device:   {device}")
    print(f"  Epochs:   {args.epochs} | Patience: {args.patience}")
    print(f"  Windows:  train<={args.max_train or 'ALL'} | eval<={args.max_eval or 'ALL'}")
    print(f"  Variants: {', '.join(variant_names)}")
    mode = "SMOKE" if args.smoke else "FULL" if args.full else "STANDARD (capped)"
    print(f"  Mode:     {mode}")
    print("=" * 80)

    t0_total = time.time()
    rows = []

    # Run each architectural variant sequentially
    for name in variant_names:
        switches = ABLATION_VARIANTS[name]
        model_cfg = ModelConfig(
            d_model=64,       # matching small training config for speed
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=128,
            dropout=0.1,
            activation="gelu",
            **switches,
        )
        cfg = ExperimentConfig(
            name=f"dec-ablation-{name}",
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
                seq_len=96,
                label_len=48,
                pred_len=24,
                features="M",
                target="OT",
            ),
        )
        result = run_variant(name, cfg, device, args.max_train, args.max_eval)
        rows.append(result)

    total_elapsed = time.time() - t0_total

    # Render metrics table
    summary_report = render_results(rows)
    print(summary_report)
    print(f"Total ablation study run time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Save JSON log
    json_path = out_dir / "decoder_ablation.json"
    json_path.write_text(json.dumps({
        "study": "decoder_ablation_analysis",
        "dataset": args.csv,
        "device": str(device),
        "seed": args.seed,
        "mode": mode,
        "total_elapsed_sec": total_elapsed,
        "results": rows,
    }, indent=2))

    # Save CSV
    csv_path = out_dir / "decoder_ablation.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "test_mse", "test_mae", "delta_mse",
                         "best_val_mse", "n_params", "epochs_run", "elapsed_sec"])
        baseline_mse = rows[0]["test_mse"] if rows else 0
        for r in rows:
            delta = r["test_mse"] - baseline_mse
            writer.writerow([
                r["variant"],
                f"{r['test_mse']:.6f}",
                f"{r['test_mae']:.6f}",
                f"{delta:+.6f}",
                f"{r['best_val_mse']:.6f}",
                r["n_params"],
                r["epochs_run"],
                f"{r['elapsed_sec']:.1f}",
            ])

    # Save markdown results table for the report
    md_path = Path("DECODER_ABLATION_RESULTS.md")
    report_header = f"""# Decoder Ablation Study Results

This report compiles the experimental findings of the Decoder architecture components on the ETT dataset forecasting task.

## Summary Table
```text
{summary_report}
```
"""
    md_path.write_text(report_header)

    print(f"\n[Saved] Results saved:")
    print(f"   JSON -> {json_path}")
    print(f"   CSV  -> {csv_path}")
    print(f"   MD   -> {md_path}")
    print("\n[Done] Decoder ablation study complete!")


if __name__ == "__main__":
    main()
