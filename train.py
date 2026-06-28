"""Training pipeline for the Transformer time-series forecaster (ETT).

Run a *small* end-to-end train/val/test cycle that wires together everything in
the repo for the first time:

    ETTDataset (Dataloader.py) ─► DataLoader ─► TransformerForecaster ─► MSE loss

Why a small run (config reasoning)
----------------------------------
This is an ablation harness meant to run on a laptop, so every knob is sized for
*fast, reliable CPU iteration* rather than leaderboard accuracy:

* Dataset = **ETTh1 (hourly)**. The temporal marks emitted by ETTDataset are
  ``[hour, day-of-month, weekday]``; on the hourly series each row is exactly one
  hour, so the calendar marks line up with the sampling rate (ETTm1 is 15-min, so
  those marks would be aliased). ETTh1 is also the smallest file.
* **Subset the windows** (``--max-train`` / ``--max-eval``). The full ETTh1 train
  split is ~10k windows; we cap it to a few hundred so an epoch is seconds, not
  minutes. This is the single biggest lever for a weak device.
* **Tiny model**: ``d_model=64, n_heads=4, e_layers=2, d_layers=1, d_ff=128``.
  ~50–100k params vs. ~10M for the d_model=512 default — fits in CPU cache and
  trains without a GPU. Architecture is identical to the full model, so ablation
  conclusions about *which component matters* still transfer.
* **Short horizon**: ``seq_len=96, label_len=48, pred_len=24`` — the standard ETT
  short-term setting (1 day lookback → 1 day ahead on hourly data).
* **lr=1e-3** (a notch higher than the 1e-4 default) + a handful of epochs +
  early stopping: small data converges fast and overfits fast, so we stop early.
* **device = cpu by default**. The value embedding uses a *circular-padded*
  Conv1d which is not supported on Apple MPS; CPU (or CUDA when present) avoids
  that. Override with ``--device``.

Everything is driven by the project's own ``ExperimentConfig`` so the run is
tracked the same way a full ablation sweep would be, and a JSON log of the
config + metrics is written to ``experiments/``.

Usage
-----
    uv run python train.py                  # default small run on ETTh1
    uv run python train.py --epochs 5 --max-train 800
    uv run python train.py --smoke          # 1 epoch, ~32 windows (CI/sanity)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from Dataloader import ETTDataset
from ts_ablation.configs.experiment import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from ts_ablation.decoder_arch.models import TransformerForecaster


# ─────────────────────────────────────────────────────────────────────────────
# Config construction — small, device-friendly defaults (see module docstring).
# ─────────────────────────────────────────────────────────────────────────────
def build_config(args) -> ExperimentConfig:
    model = ModelConfig(
        d_model=64,
        n_heads=4,
        e_layers=5,
        d_layers=2,
        d_ff=128,
        dropout=0.1,
        activation="gelu",
    )
    train = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        grad_clip=1.0,
        device=args.device,
    )
    data = DataConfig(
        csv_path=args.csv,
        seq_len=96,
        label_len=48,
        pred_len=24,
        features="M",      # multivariate -> multivariate (predict all 7 channels)
        target="OT",
    )
    return ExperimentConfig(name=args.name, model=model, train=train, data=data)


def pick_device(requested: str) -> torch.device:
    """Resolve the requested device, falling back to CPU when unavailable.

    MPS is intentionally avoided unless explicitly requested: the value
    embedding's circular-padded Conv1d is not implemented on MPS.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(cfg: ExperimentConfig, split: str, max_windows: int | None,
                shuffle: bool) -> DataLoader:
    """Build a DataLoader over a (optionally capped) ETT split."""
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
        # Take a contiguous prefix so windows stay temporally coherent.
        dataset = Subset(dataset, list(range(max_windows)))
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def infer_dims(loader: DataLoader):
    """Peek one batch to wire model input/output dimensions exactly."""
    batch = next(iter(loader))
    enc_in = batch["x_enc"].shape[-1]
    dec_in = batch["x_dec"].shape[-1]
    c_out = batch["y"].shape[-1]
    return enc_in, dec_in, c_out


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Return MSE and MAE over a loader (in scaled space)."""
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


def main():
    parser = argparse.ArgumentParser(description="Small ETT Transformer training pipeline")
    parser.add_argument("--csv", default="Data/ETTh1.csv", help="path to an ETT csv")
    parser.add_argument("--name", default="small-etth1", help="experiment name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=500,
                        help="cap on #training windows (device-friendly)")
    parser.add_argument("--max-eval", type=int, default=200,
                        help="cap on #val/test windows")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny 1-epoch run for a sanity check")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.max_train, args.max_eval, args.batch_size = 1, 32, 32, 8

    torch.manual_seed(args.seed)

    cfg = build_config(args)
    device = pick_device(cfg.train.device)

    print("=" * 70)
    print(f"Experiment: {cfg.name}   |   ablation tag: {cfg.ablation_tag()}")
    print(f"Data: {cfg.data.csv_path}  seq_len={cfg.data.seq_len} "
          f"label_len={cfg.data.label_len} pred_len={cfg.data.pred_len} "
          f"features={cfg.data.features}")
    print(f"Model: d_model={cfg.model.d_model} heads={cfg.model.n_heads} "
          f"e_layers={cfg.model.e_layers} d_layers={cfg.model.d_layers} "
          f"d_ff={cfg.model.d_ff}")
    print(f"Train: epochs={cfg.train.epochs} bs={cfg.train.batch_size} "
          f"lr={cfg.train.learning_rate} device={device} "
          f"max_train={args.max_train} max_eval={args.max_eval}")
    print("=" * 70)

    # Data
    train_loader = make_loader(cfg, "train", args.max_train, shuffle=True)
    val_loader = make_loader(cfg, "val", args.max_eval, shuffle=False)
    test_loader = make_loader(cfg, "test", args.max_eval, shuffle=False)
    print(f"windows -> train: {len(train_loader.dataset)}  "
          f"val: {len(val_loader.dataset)}  test: {len(test_loader.dataset)}")

    # Model (dims inferred from a real batch so feature modes stay correct)
    enc_in, dec_in, c_out = infer_dims(train_loader)
    model = TransformerForecaster(
        enc_in=enc_in, dec_in=dec_in, c_out=c_out,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        e_layers=cfg.model.e_layers, d_layers=cfg.model.d_layers,
        d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
        activation=cfg.model.activation, pred_len=cfg.data.pred_len,
        use_decoder=cfg.model.use_decoder,
        dec_use_self_attention=cfg.model.dec_use_self_attention,
        dec_use_causal_mask=cfg.model.dec_use_causal_mask,
        seq_len=cfg.data.seq_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params:,}  (enc_in={enc_in}, dec_in={dec_in}, c_out={c_out})")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Train with early stopping on val MSE
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     cfg.train.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"epoch {epoch:2d}/{cfg.train.epochs}  "
              f"train_mse={train_loss:.4f}  "
              f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}")

        if val_metrics["mse"] < best_val - 1e-6:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.patience:
                print(f"early stopping at epoch {epoch} (no val improvement for "
                      f"{cfg.train.patience} epochs)")
                break

    # Restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    print("-" * 70)
    print(f"best val_mse={best_val:.4f}  |  "
          f"test_mse={test_metrics['mse']:.4f}  test_mae={test_metrics['mae']:.4f}")
    print(f"total time: {elapsed:.1f}s")

    # Persist an ablation log (config + metrics) for tracking
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"{cfg.name}_{cfg.ablation_tag()}.json"
    log_path.write_text(json.dumps({
        "name": cfg.name,
        "ablation_tag": cfg.ablation_tag(),
        "config": cfg.model_dump(),
        "n_params": n_params,
        "subset": {"max_train": args.max_train, "max_eval": args.max_eval},
        "best_val_mse": best_val,
        "test": test_metrics,
        "history": history,
        "elapsed_sec": elapsed,
    }, indent=2))
    print(f"log written -> {log_path}")

    # Save the best model checkpoint for later evaluation
    ckpt_path = out_dir / f"{cfg.name}_{cfg.ablation_tag()}.pt"
    torch.save(best_state or model.state_dict(), ckpt_path)
    print(f"checkpoint saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
