from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ts_ablation.configs.standard import (
    ETT_DATASETS,
    STANDARD_HORIZONS,
    standard_config,
)
from ts_ablation.models import TransformerForecaster
from train import evaluate, infer_dims, make_loader, pick_device, train_one_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Time / ETA helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def cell_weight(label_len: int, pred_len: int, bridge: str, ffn_variant: str) -> int:
    base = label_len + pred_len
    bridge_mul = 1.0 if bridge == "cross_attn" else 0.9
    ffn_mul = 1.0 if ffn_variant == "standard" else 0.95 if ffn_variant in {"half", "quarter"} else 0.9
    return max(1, int(base * bridge_mul * ffn_mul))


class ETAEstimator:
    def __init__(self, planned_units: float):
        self.planned_units = float(planned_units)
        self.done_units = 0.0
        self.obs_dur = 0.0
        self.obs_units = 0.0
        self.t0 = time.time()

    def epoch_done(self, weight: float, duration: float) -> None:
        self.done_units += weight
        self.obs_dur += duration
        self.obs_units += weight

    def cell_done(self, unspent_units: float) -> None:
        self.planned_units = max(self.done_units, self.planned_units - unspent_units)

    def rate(self):
        return self.obs_dur / self.obs_units if self.obs_units > 0 else None

    def elapsed(self):
        return time.time() - self.t0

    def remaining(self):
        r = self.rate()
        if r is None:
            return None
        return r * max(0.0, self.planned_units - self.done_units)

    def estimate_units(self, units: float):
        r = self.rate()
        return None if r is None else r * units

    def status(self):
        rem = self.remaining()
        if rem is None:
            return f"elapsed {fmt_hms(self.elapsed())}  ETA —"
        finish = time.strftime("%H:%M:%S", time.localtime(time.time() + rem))
        return f"elapsed {fmt_hms(self.elapsed())}  ETA ~{fmt_hms(rem)} left  (finish ~{finish})"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return pick_device(requested)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation modules
# ─────────────────────────────────────────────────────────────────────────────
class AddBridge(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_x, enc_x):
        enc_summary = enc_x.mean(dim=1, keepdim=True)          # (B,1,D)
        fused = dec_x + enc_summary                            # (B,T_dec,D)
        return self.norm(fused), None


class ConcatBridge(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_x, enc_x):
        enc_summary = enc_x.mean(dim=1, keepdim=True).expand(-1, dec_x.size(1), -1)
        fused = torch.cat([dec_x, enc_summary], dim=-1)       # (B,T_dec,2D)
        return self.norm(self.proj(fused)), None


class DotBridge(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = d_model ** -0.5
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_x, enc_x):
        scores = torch.matmul(dec_x, enc_x.transpose(1, 2)) * self.scale   # (B,T_dec,T_enc)
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, enc_x)                                  # (B,T_dec,D)
        return self.norm(ctx), weights


class ConvFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────
class AblationTransformerForecaster(nn.Module):
    """
    Wrapper quanh TransformerForecaster để thay cầu nối cross-attention và FFN.
    Giả định model gốc có encoder/decoder stack kiểu Transformer encoder-decoder.
    Nếu codebase của bạn expose module names khác, chỉ cần sửa inject_* bên dưới.
    """
    def __init__(self, base_model: nn.Module, bridge: str, ffn_variant: str):
        super().__init__()
        self.model = base_model
        self.bridge = bridge
        self.ffn_variant = ffn_variant
        self._inject_ablation()

    def _scaled_ffn_dim(self, d_ff: int) -> int:
        if self.ffn_variant == "standard":
            return d_ff
        if self.ffn_variant == "half":
            return max(1, d_ff // 2)
        if self.ffn_variant == "quarter":
            return max(1, d_ff // 4)
        return d_ff

    def _replace_ffn_module(self, module: nn.Module):
        for name, child in list(module.named_children()):
            lname = name.lower()

            if self.ffn_variant in {"half", "quarter"}:
                if hasattr(child, "conv1") and hasattr(child, "conv2"):
                    in_dim = child.conv1.in_channels
                    old_ff = child.conv1.out_channels
                    new_ff = self._scaled_ffn_dim(old_ff)
                    child.conv1 = nn.Conv1d(in_dim, new_ff, kernel_size=1)
                    child.conv2 = nn.Conv1d(new_ff, in_dim, kernel_size=1)

                elif hasattr(child, "linear1") and hasattr(child, "linear2"):
                    in_dim = child.linear1.in_features
                    old_ff = child.linear1.out_features
                    new_ff = self._scaled_ffn_dim(old_ff)
                    child.linear1 = nn.Linear(in_dim, new_ff)
                    child.linear2 = nn.Linear(new_ff, in_dim)

            elif self.ffn_variant == "conv1d":
                if hasattr(child, "linear1") and hasattr(child, "linear2"):
                    in_dim = child.linear1.in_features
                    ff_dim = child.linear1.out_features
                    dropout = getattr(child, "dropout", 0.1)
                    act_name = "gelu"
                    setattr(module, name, ConvFFN(in_dim, ff_dim, dropout, act_name))
                    continue

            self._replace_ffn_module(child)

    def _replace_cross_attention_module(self, module: nn.Module):
        for name, child in list(module.named_children()):
            lname = name.lower()

            is_cross_attn_like = (
                ("cross" in lname and "attn" in lname) or
                ("cross_attention" in lname) or
                ("enc_attn" in lname)
            )

            if is_cross_attn_like:
                d_model = None
                if hasattr(child, "embed_dim"):
                    d_model = child.embed_dim
                elif hasattr(child, "d_model"):
                    d_model = child.d_model
                elif hasattr(child, "out_proj") and hasattr(child.out_proj, "out_features"):
                    d_model = child.out_proj.out_features

                if d_model is None:
                    continue

                if self.bridge == "concat":
                    setattr(module, name, ConcatBridge(d_model))
                elif self.bridge == "add":
                    setattr(module, name, AddBridge(d_model))
                elif self.bridge == "dot":
                    setattr(module, name, DotBridge(d_model))
                continue

            self._replace_cross_attention_module(child)

    def _inject_ablation(self):
        if self.ffn_variant != "standard":
            self._replace_ffn_module(self.model)

        if self.bridge != "cross_attn":
            self._replace_cross_attention_module(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval one cell
# ─────────────────────────────────────────────────────────────────────────────
def run_cell(cfg, device, *, bridge, ffn_variant, eta=None, weight=0,
             max_train=None, max_eval=None, verbose=True):
    train_loader = make_loader(cfg, "train", max_train, shuffle=True)
    val_loader = make_loader(cfg, "val", max_eval, shuffle=False)
    test_loader = make_loader(cfg, "test", max_eval, shuffle=False)

    enc_in, dec_in, c_out = infer_dims(train_loader)
    base_model = TransformerForecaster(
        enc_in=enc_in, dec_in=dec_in, c_out=c_out,
        d_model=cfg.model.d_model, n_heads=cfg.model.n_heads,
        e_layers=cfg.model.e_layers, d_layers=cfg.model.d_layers,
        d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
        activation=cfg.model.activation, pred_len=cfg.data.pred_len,
    )
    model = AblationTransformerForecaster(base_model, bridge, ffn_variant).to(device)
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.train.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        ep_dur = time.time() - ep_t0
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        epochs_run = epoch

        if eta is not None:
            eta.epoch_done(weight, ep_dur)

        if verbose:
            status = eta.status() if eta is not None else f"epoch {ep_dur:.1f}s"
            print(
                f"    epoch {epoch:2d}/{cfg.train.epochs}  "
                f"train_mse={train_loss:.4f}  "
                f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}  "
                f"| {status}"
            )

        if val_metrics["mse"] < best_val - 1e-6:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} (no val improvement for {cfg.train.patience} epochs)")
                break

    if eta is not None:
        eta.cell_done(unspent_units=(cfg.train.epochs - epochs_run) * weight)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

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
    header = (
        "| dataset | pred_len | bridge | ffn | test_mse | test_mae | "
        "best_val_mse | params | epochs | sec |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['pred_len']} | {r['bridge']} | {r['ffn_variant']} | "
            f"{r['test_mse']:.4f} | {r['test_mae']:.4f} | {r['best_val_mse']:.4f} | "
            f"{r['n_params']:,} | {r['epochs_run']} | {r['elapsed_sec']:.0f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Bridge + FFN ablation benchmark")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1"], choices=sorted(ETT_DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(STANDARD_HORIZONS))
    parser.add_argument("--bridges", nargs="+", default=["cross_attn", "concat", "add", "dot"])
    parser.add_argument("--ffn-variants", nargs="+", default=["standard", "half", "quarter", "conv1d"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="bridge_ffn")
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--budget-min", type=float, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    max_train = max_eval = None
    if args.quick:
        args.datasets = ["ETTh1"]
        args.horizons = [96]
        args.bridges = ["cross_attn", "add"]
        args.ffn_variants = ["standard", "half"]
        args.epochs = 1
        args.tag = "bridge_ffn_quick"
        max_train, max_eval = 200, 100

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    budget_sec = args.budget_min * 60 if args.budget_min else None

    SEQ_LEN, LABEL_LEN = 96, 48
    cells = [
        (d, h, b, f)
        for d in args.datasets
        for h in args.horizons
        for b in args.bridges
        for f in args.ffn_variants
    ]
    planned_units = sum(args.epochs * cell_weight(LABEL_LEN, h, b, f) for d, h, b, f in cells)
    eta = ETAEstimator(planned_units)

    print("=" * 90)
    print(f"Bridge+FFN benchmark  |  device={device}")
    print(
        f"datasets={args.datasets}  horizons={args.horizons}  "
        f"bridges={args.bridges}  ffn={args.ffn_variants}  "
        f"epochs={args.epochs}  batch={args.batch_size}  seed={args.seed}"
    )
    if budget_sec:
        print(f"wall-clock budget: {args.budget_min:.0f} min")
    if args.quick:
        print(f"(smoke caps: max_train={max_train} max_eval={max_eval} — NOT a genuine result)")
    print("=" * 90)

    rows = []
    skipped = []

    for idx, (dataset, pred_len, bridge, ffn_variant) in enumerate(cells):
        weight = cell_weight(LABEL_LEN, pred_len, bridge, ffn_variant)

        if budget_sec is not None and idx > 0:
            est = eta.estimate_units(args.epochs * weight)
            if est is not None and eta.elapsed() + est > budget_sec:
                remaining = cells[idx:]
                print(
                    f"\n⏱ budget {args.budget_min:.0f} min would be exceeded "
                    f"(elapsed {fmt_hms(eta.elapsed())}, next cell ~{fmt_hms(est)}). "
                    f"Skipping {len(remaining)} remaining cell(s):"
                )
                for d, h, b, f in remaining:
                    print(f"     - {d} pred_len={h} bridge={b} ffn={f}")
                    skipped.append({"dataset": d, "pred_len": h, "bridge": b, "ffn_variant": f})
                break

        cfg = standard_config(
            dataset, pred_len=pred_len,
            seq_len=SEQ_LEN, label_len=LABEL_LEN,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, patience=args.patience, device=args.device,
        )

        print(
            f"\n▶ [{idx + 1}/{len(cells)}] {dataset} pred_len={pred_len} "
            f"bridge={bridge} ffn={ffn_variant} | {eta.status()}"
        )

        result = run_cell(
            cfg, device,
            bridge=bridge, ffn_variant=ffn_variant,
            eta=eta, weight=weight,
            max_train=max_train, max_eval=max_eval
        )

        print(
            f"  → test_mse={result['test_mse']:.4f}  "
            f"test_mae={result['test_mae']:.4f}  "
            f"params={result['n_params']:,}  "
            f"(cell {fmt_hms(result['elapsed_sec'])})"
        )

        rows.append({
            "dataset": dataset,
            "pred_len": pred_len,
            "bridge": bridge,
            "ffn_variant": ffn_variant,
            "ablation_tag": f"{cfg.ablation_tag()}__bridge={bridge}__ffn={ffn_variant}",
            "config": cfg.model_dump(),
            **result,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / f"benchmark_{args.tag}.json"
    json_path.write_text(json.dumps({
        "tag": args.tag,
        "device": str(device),
        "seed": args.seed,
        "total_elapsed_sec": eta.elapsed(),
        "skipped": skipped,
        "results": rows,
    }, indent=2))

    csv_path = out_dir / f"benchmark_{args.tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "pred_len", "bridge", "ffn_variant",
            "test_mse", "test_mae", "best_val_mse",
            "n_params", "epochs_run", "elapsed_sec"
        ])
        for r in rows:
            writer.writerow([
                r["dataset"], r["pred_len"], r["bridge"], r["ffn_variant"],
                f"{r['test_mse']:.6f}", f"{r['test_mae']:.6f}",
                f"{r['best_val_mse']:.6f}", r["n_params"],
                r["epochs_run"], f"{r['elapsed_sec']:.1f}"
            ])

    print("\n" + "=" * 90)
    print("RESULTS")
    print(render_markdown(rows))
    if skipped:
        print(
            "\nskipped (budget): " +
            ", ".join(
                f"{s['dataset']}/pl{s['pred_len']}/{s['bridge']}/{s['ffn_variant']}"
                for s in skipped
            )
        )
    print("=" * 90)
    print(f"total time: {fmt_hms(eta.elapsed())}")
    print(f"json -> {json_path}")
    print(f"csv  -> {csv_path}")


if __name__ == "__main__":
    main()