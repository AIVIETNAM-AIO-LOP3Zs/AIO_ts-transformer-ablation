"""Genuine ETT forecasting benchmark.

Unlike ``train.py`` (a tiny, CPU-friendly sanity run on a *subset* of windows),
this harness runs the **standard long-term forecasting protocol** so the numbers
are comparable to the published Informer / Autoformer / FEDformer results:

* full data — every train/val/test window, no ``Subset`` cap;
* canonical architecture and training knobs (see ``configs/standard.py``);
* a sweep over datasets × horizons ``pred_len ∈ {96, 192, 336, 720}``;
* MSE/MAE reported in scaled space (the literature convention), with the
  ``StandardScaler`` fit on the *train* split only (fixed in ``Dataloader.py``).

Outputs (under ``experiments/``):
* ``benchmark_<tag>.json`` — full record: per-cell config + metrics + history;
* ``benchmark_<tag>.csv``  — flat results table for spreadsheets/plots;
* a markdown table printed to stdout.

Usage
-----
    uv run python benchmark.py --quick                 # fast end-to-end smoke
    uv run python benchmark.py                         # full sweep, ETTh1 all horizons
    uv run python benchmark.py --datasets ETTh1 ETTh2 --horizons 96 192 336 720
    uv run python benchmark.py --datasets ETTh1 --epochs 10 --device cuda

The full sweep is heavy on CPU (d_model=512). Use ``--device cuda`` when a GPU is
available, or ``--quick`` to validate the pipeline first.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from ts_ablation.configs.standard import (
    ETT_DATASETS,
    STANDARD_HORIZONS,
    standard_config,
)
from ts_ablation.models import TransformerForecaster

# Reuse the data/eval/train building blocks already validated by train.py.
from train import evaluate, infer_dims, make_loader, pick_device, train_one_epoch


def run_cell(cfg, device, max_train=None, max_eval=None, verbose=True) -> dict:
    """Train + evaluate a single (dataset, horizon) benchmark cell.

    Trains with early stopping on val MSE, restores the best checkpoint, and
    reports test MSE/MAE. ``max_train``/``max_eval`` cap the number of windows
    (``None`` = full data); they exist only so ``--quick`` can validate the
    pipeline fast — a genuine benchmark leaves them ``None``.
    """
    train_loader = make_loader(cfg, "train", max_train, shuffle=True)
    val_loader = make_loader(cfg, "val", max_eval, shuffle=False)
    test_loader = make_loader(cfg, "test", max_eval, shuffle=False)

    enc_in, dec_in, c_out = infer_dims(train_loader)
    model = TransformerForecaster(
        enc_in=enc_in, dec_in=dec_in, c_out=c_out,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        e_layers=cfg.model.e_layers, d_layers=cfg.model.d_layers,
        d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
        activation=cfg.model.activation, pred_len=cfg.data.pred_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    epochs_run = 0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     cfg.train.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        epochs_run = epoch
        if verbose:
            print(f"    epoch {epoch:2d}/{cfg.train.epochs}  "
                  f"train_mse={train_loss:.4f}  "
                  f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}")

        if val_metrics["mse"] < best_val - 1e-6:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} "
                          f"(no val improvement for {cfg.train.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    return {
        "n_params": n_params,
        "windows": {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "best_val_mse": best_val,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "epochs_run": epochs_run,
        "elapsed_sec": elapsed,
        "history": history,
    }


def render_markdown(rows: list[dict]) -> str:
    """Render the results rows as a markdown table (MSE/MAE per cell)."""
    header = ("| dataset | pred_len | test_mse | test_mae | best_val_mse | "
              "params | epochs | sec |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['pred_len']} | {r['test_mse']:.4f} | "
            f"{r['test_mae']:.4f} | {r['best_val_mse']:.4f} | {r['n_params']:,} | "
            f"{r['epochs_run']} | {r['elapsed_sec']:.0f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Standard ETT forecasting benchmark")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1"],
                        choices=sorted(ETT_DATASETS),
                        help="ETT datasets to benchmark")
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=list(STANDARD_HORIZONS),
                        help="forecast horizons (pred_len) to sweep")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="standard", help="output filename tag")
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke: ETTh1, pred_len=96, 1 epoch, capped windows")
    args = parser.parse_args()

    if args.quick:
        args.datasets = ["ETTh1"]
        args.horizons = [96]
        args.epochs = 1
        args.tag = "quick"
        max_train, max_eval = 200, 100
    else:
        max_train, max_eval = None, None  # full data

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    print("=" * 78)
    print(f"ETT benchmark{' [QUICK SMOKE]' if args.quick else ''}  |  device={device}")
    print(f"datasets={args.datasets}  horizons={args.horizons}  "
          f"epochs={args.epochs}  seed={args.seed}")
    if args.quick:
        print(f"(smoke caps: max_train={max_train} max_eval={max_eval} — "
              f"NOT a genuine result)")
    print("=" * 78)

    rows = []
    for dataset in args.datasets:
        for pred_len in args.horizons:
            cfg = standard_config(
                dataset, pred_len=pred_len,
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.lr, patience=args.patience, device=args.device,
            )
            print(f"\n▶ {dataset}  pred_len={pred_len}  "
                  f"({cfg.ablation_tag()})")
            result = run_cell(cfg, device, max_train=max_train, max_eval=max_eval)
            print(f"  → test_mse={result['test_mse']:.4f}  "
                  f"test_mae={result['test_mae']:.4f}  "
                  f"params={result['n_params']:,}  "
                  f"({result['elapsed_sec']:.0f}s)")
            rows.append({
                "dataset": dataset,
                "pred_len": pred_len,
                "ablation_tag": cfg.ablation_tag(),
                "config": cfg.model_dump(),
                **result,
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / f"benchmark_{args.tag}.json"
    json_path.write_text(json.dumps({
        "tag": args.tag,
        "quick": args.quick,
        "device": str(device),
        "seed": args.seed,
        "results": rows,
    }, indent=2))

    csv_path = out_dir / f"benchmark_{args.tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "pred_len", "test_mse", "test_mae",
                         "best_val_mse", "n_params", "epochs_run", "elapsed_sec"])
        for r in rows:
            writer.writerow([r["dataset"], r["pred_len"],
                             f"{r['test_mse']:.6f}", f"{r['test_mae']:.6f}",
                             f"{r['best_val_mse']:.6f}", r["n_params"],
                             r["epochs_run"], f"{r['elapsed_sec']:.1f}"])

    print("\n" + "=" * 78)
    print("RESULTS")
    print(render_markdown(rows))
    print("=" * 78)
    print(f"json -> {json_path}")
    print(f"csv  -> {csv_path}")


if __name__ == "__main__":
    main()
