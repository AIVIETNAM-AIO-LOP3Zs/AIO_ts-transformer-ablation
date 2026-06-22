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

Time reporting
--------------
Every epoch prints elapsed time, an ETA for the remaining sweep, and a projected
finish clock-time. The ETA is an *upper bound* (it assumes every cell runs its
full epoch budget; early stopping only makes the real run finish sooner).

Budget guard
------------
``--budget-min N`` caps the whole sweep at N wall-clock minutes. Before starting
each cell the harness estimates its cost from observed epoch times; if running it
would blow the budget, the remaining cells are **skipped and logged** (never
silently truncated). The first cell always runs so you get at least one result.

Usage
-----
    uv run python benchmark.py --quick                 # fast end-to-end smoke
    uv run python benchmark.py --colab-t4              # ~30 min on a Colab T4 GPU
    uv run python benchmark.py                         # full sweep, ETTh1 all horizons
    uv run python benchmark.py --datasets ETTh1 ETTh2 --horizons 96 192 336 720
    uv run python benchmark.py --budget-min 30 --device auto

The full sweep is heavy on CPU (d_model=512). Use ``--device cuda``/``auto`` when
a GPU is available, ``--colab-t4`` for a budget-bounded GPU run, or ``--quick`` to
validate the pipeline first.
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


# ─────────────────────────────────────────────────────────────────────────────
# Time / ETA helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_hms(seconds: float) -> str:
    """Format a duration as H:MM:SS (or M:SS under an hour)."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def cell_weight(label_len: int, pred_len: int) -> int:
    """Cost proxy for one epoch of a cell.

    The decoder sequence length (``label_len + pred_len``) is what differs
    between horizons and dominates the per-epoch cost difference, so we use it to
    weight the ETA — a pred_len=720 epoch counts ~5x a pred_len=96 epoch.
    """
    return label_len + pred_len


class ETAEstimator:
    """Tracks elapsed time and projects the remaining sweep time.

    Works in "cost units" (= cell_weight per epoch). The ETA assumes each cell
    runs its full planned epoch count; when a cell early-stops we reclaim the
    unspent units so later projections stay honest.
    """

    def __init__(self, planned_units: float):
        self.planned_units = float(planned_units)
        self.done_units = 0.0
        self.obs_dur = 0.0      # observed epoch seconds
        self.obs_units = 0.0    # observed cost units
        self.t0 = time.time()

    def epoch_done(self, weight: float, duration: float) -> None:
        self.done_units += weight
        self.obs_dur += duration
        self.obs_units += weight

    def cell_done(self, unspent_units: float) -> None:
        # A cell that early-stopped won't consume its remaining planned epochs.
        self.planned_units = max(self.done_units, self.planned_units - unspent_units)

    def rate(self) -> float | None:
        return self.obs_dur / self.obs_units if self.obs_units > 0 else None

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float | None:
        r = self.rate()
        if r is None:
            return None
        return r * max(0.0, self.planned_units - self.done_units)

    def estimate_units(self, units: float) -> float | None:
        r = self.rate()
        return None if r is None else r * units

    def status(self) -> str:
        rem = self.remaining()
        if rem is None:
            return f"elapsed {fmt_hms(self.elapsed())}  ETA —"
        finish = time.strftime("%H:%M:%S", time.localtime(time.time() + rem))
        return (f"elapsed {fmt_hms(self.elapsed())}  "
                f"ETA ~{fmt_hms(rem)} left  (finish ~{finish})")


def resolve_device(requested: str) -> torch.device:
    """Resolve a device, with ``auto`` preferring CUDA (Colab T4) then CPU.

    Falls back to ``train.pick_device`` for explicit cpu/cuda/mps requests.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return pick_device(requested)


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation for one (dataset, horizon) cell
# ─────────────────────────────────────────────────────────────────────────────
def run_cell(cfg, device, *, eta=None, weight=0, max_train=None, max_eval=None,
             verbose=True) -> dict:
    """Train + evaluate a single (dataset, horizon) benchmark cell.

    Trains with early stopping on val MSE, restores the best checkpoint, and
    reports test MSE/MAE. ``max_train``/``max_eval`` cap the number of windows
    (``None`` = full data); they exist only so ``--quick`` can validate the
    pipeline fast — a genuine benchmark leaves them ``None``. ``eta``/``weight``
    drive the live ETA readout.
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
        ep_t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     cfg.train.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        ep_dur = time.time() - ep_t0
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        epochs_run = epoch
        if eta is not None:
            eta.epoch_done(weight, ep_dur)
        if verbose:
            status = eta.status() if eta is not None else f"epoch {ep_dur:.1f}s"
            print(f"    epoch {epoch:2d}/{cfg.train.epochs}  "
                  f"train_mse={train_loss:.4f}  "
                  f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}  "
                  f"| {status}")

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

    # Reclaim ETA units for epochs we skipped via early stopping.
    if eta is not None:
        eta.cell_done(unspent_units=(cfg.train.epochs - epochs_run) * weight)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    # Free GPU memory between cells so the largest horizon doesn't OOM a T4.
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="standard", help="output filename tag")
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--budget-min", type=float, default=None,
                        help="wall-clock cap (minutes); skip remaining cells if exceeded")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke: ETTh1, pred_len=96, 1 epoch, capped windows")
    parser.add_argument("--colab-t4", action="store_true",
                        help="reduced GPU preset sized for ~30 min on a Colab T4")
    args = parser.parse_args()

    max_train = max_eval = None
    if args.quick:
        args.datasets = ["ETTh1"]
        args.horizons = [96]
        args.epochs = 1
        args.tag = "quick"
        max_train, max_eval = 200, 100
    elif args.colab_t4:
        # Sized to finish within ~30 min on a Colab T4: a single dataset across
        # all four standard horizons, full data, larger GPU batch, fewer epochs.
        # The 30-min budget guard trims the tail if the GPU is slower than assumed.
        args.datasets = ["ETTh1"]
        args.horizons = [96, 192, 336, 720]
        args.epochs = 8
        args.batch_size = 64
        args.patience = 3
        if args.device == "cpu":
            args.device = "auto"
        if args.budget_min is None:
            args.budget_min = 30.0
        args.tag = "colab-t4"

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    budget_sec = args.budget_min * 60 if args.budget_min else None

    # Build the cell plan up front so we can weight the ETA across horizons.
    SEQ_LEN, LABEL_LEN = 96, 48
    cells = [(d, h) for d in args.datasets for h in args.horizons]
    planned_units = sum(args.epochs * cell_weight(LABEL_LEN, h) for _, h in cells)
    eta = ETAEstimator(planned_units)

    print("=" * 78)
    label = (" [QUICK SMOKE]" if args.quick
             else " [COLAB-T4 BUDGET]" if args.colab_t4 else "")
    print(f"ETT benchmark{label}  |  device={device}")
    print(f"datasets={args.datasets}  horizons={args.horizons}  "
          f"epochs={args.epochs}  batch={args.batch_size}  seed={args.seed}")
    if budget_sec:
        print(f"wall-clock budget: {args.budget_min:.0f} min "
              f"(remaining cells skipped if exceeded)")
    if args.quick:
        print(f"(smoke caps: max_train={max_train} max_eval={max_eval} — "
              f"NOT a genuine result)")
    print("=" * 78)

    rows = []
    skipped = []
    for idx, (dataset, pred_len) in enumerate(cells):
        weight = cell_weight(LABEL_LEN, pred_len)

        # Budget guard: once we have a rate estimate, don't start a cell that
        # would push us past the wall-clock budget. The first cell always runs.
        if budget_sec is not None and idx > 0:
            est = eta.estimate_units(args.epochs * weight)
            if est is not None and eta.elapsed() + est > budget_sec:
                remaining = cells[idx:]
                print(f"\n⏱  budget {args.budget_min:.0f} min would be exceeded "
                      f"(elapsed {fmt_hms(eta.elapsed())}, next cell "
                      f"~{fmt_hms(est)}). Skipping {len(remaining)} remaining cell(s):")
                for d, h in remaining:
                    print(f"     - {d} pred_len={h}")
                    skipped.append({"dataset": d, "pred_len": h})
                break

        cfg = standard_config(
            dataset, pred_len=pred_len,
            seq_len=SEQ_LEN, label_len=LABEL_LEN,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, patience=args.patience, device=args.device,
        )
        print(f"\n▶ [{idx + 1}/{len(cells)}] {dataset}  pred_len={pred_len}  "
              f"({cfg.ablation_tag()})  | {eta.status()}")
        result = run_cell(cfg, device, eta=eta, weight=weight,
                          max_train=max_train, max_eval=max_eval)
        print(f"  → test_mse={result['test_mse']:.4f}  "
              f"test_mae={result['test_mae']:.4f}  "
              f"params={result['n_params']:,}  "
              f"(cell {fmt_hms(result['elapsed_sec'])})")
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
        "colab_t4": args.colab_t4,
        "device": str(device),
        "seed": args.seed,
        "total_elapsed_sec": eta.elapsed(),
        "skipped": skipped,
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
    if skipped:
        print(f"\nskipped (budget): "
              + ", ".join(f"{s['dataset']}/pl{s['pred_len']}" for s in skipped))
    print("=" * 78)
    print(f"total time: {fmt_hms(eta.elapsed())}")
    print(f"json -> {json_path}")
    print(f"csv  -> {csv_path}")


if __name__ == "__main__":
    main()
