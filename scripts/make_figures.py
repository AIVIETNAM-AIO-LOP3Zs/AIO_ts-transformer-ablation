"""Generate publication-ready comparison figures for the Transformer ablation study.

Reads the experiment result CSV/JSON files under ``experiments/`` and writes
PNG figures to ``experiments/figures/``.

Run:
    uv run python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = EXP / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
DPI = 150

# Consistent colour coding by impact level
LEVEL_COLORS = {
    "HIGH": "#d1495b",      # red
    "MEDIUM": "#edae49",    # amber
    "LOW": "#66a182",       # green-grey
    "NEGLIGIBLE": "#a8c686",
    "BENEFICIAL": "#2e86ab",  # blue (improves on baseline)
    "BASELINE": "#8d99ae",
}


def _bar_labels(ax, fmt="{:+.1f}%", fontsize=11):
    for p in ax.patches:
        w = p.get_width()
        ax.annotate(
            fmt.format(w),
            (w, p.get_y() + p.get_height() / 2),
            ha="left" if w >= 0 else "right",
            va="center",
            xytext=(4 if w >= 0 else -4, 0),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
        )


def fig_encoder(df: pd.DataFrame):
    """Encoder ablation: ΔMSE% horizontal bars, ordered by severity."""
    d = df[df.variant != "baseline"].copy()
    d["pct"] = d.delta_mse / 2.532752 * 100  # baseline MSE
    levels = {"no-attention": "HIGH", "no-ffn": "MEDIUM",
              "no-residual": "MEDIUM", "no-layernorm": "NEGLIGIBLE"}
    d["level"] = d.variant.map(levels)
    d = d.sort_values("pct", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=d, y="variant", x="pct", hue="variant",
                palette=[LEVEL_COLORS[levels[v]] for v in d.variant],
                legend=False, ax=ax)
    _bar_labels(ax)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Δ MSE so với baseline (%)  —  cao hơn = component càng quan trọng")
    ax.set_ylabel("")
    ax.set_title("Encoder Ablation — mức suy giảm khi loại bỏ từng component\n"
                 "(ETTh1, d_model=512, pred_len=96)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_encoder_ablation.png", dpi=DPI)
    plt.close(fig)


def fig_decoder(df: pd.DataFrame):
    """Decoder ablation: ΔMSE% bars (negative = improves)."""
    d = df[df.variant != "baseline"].copy()
    d["pct"] = d.delta_mse / 2.355892 * 100
    levels = {"no-self-attention": "LOW", "no-causal-mask": "MEDIUM",
              "no-decoder": "BENEFICIAL"}
    d["level"] = d.variant.map(levels)
    d = d.sort_values("pct", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=d, y="variant", x="pct", hue="variant",
                palette=[LEVEL_COLORS[levels[v]] for v in d.variant],
                legend=False, ax=ax)
    _bar_labels(ax)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Δ MSE so với baseline (%)  —  âm = TỐT HƠN baseline")
    ax.set_ylabel("")
    ax.set_title("Decoder Ablation — ảnh hưởng khi can thiệp Decoder\n"
                 "(ETTh1, d_model=64, pred_len=24)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_decoder_ablation.png", dpi=DPI)
    plt.close(fig)


def fig_combined(enc: pd.DataFrame, dec: pd.DataFrame):
    """Unified component-importance ranking across both studies."""
    rows = [
        ("Self-Attention (Enc)", 28.5, "HIGH"),
        ("FFN (Enc)", 10.0, "MEDIUM"),
        ("Causal Mask (Dec)", 8.4, "MEDIUM"),
        ("Residual (Enc)", 5.4, "MEDIUM"),
        ("Decoder Self-Attn", -3.3, "LOW"),
        ("LayerNorm (Enc)", -0.7, "NEGLIGIBLE"),
        ("No-Decoder → Linear", -15.1, "BENEFICIAL"),
    ]
    d = pd.DataFrame(rows, columns=["component", "pct", "level"])
    d = d.sort_values("pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=d, y="component", x="pct", hue="component",
                palette=[LEVEL_COLORS[l] for l in d.level],
                legend=False, ax=ax)
    _bar_labels(ax)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(-22, 36)  # headroom so the ±labels don't clip the frame
    ax.set_xlabel("Δ MSE khi loại bỏ (%)  —  |giá trị| càng lớn = càng 'đắt giá'")
    ax.set_ylabel("")
    ax.set_title("Xếp hạng tầm quan trọng thành phần (tổng hợp 2 study)\n"
                 "(Luu y: ΔMSE% đo trong từng study riêng — chỉ so sánh tương đối)",
                 fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEVEL_COLORS[k])
               for k in ["HIGH", "MEDIUM", "LOW", "NEGLIGIBLE", "BENEFICIAL"]]
    ax.legend(handles, ["HIGH", "MEDIUM", "LOW", "NEGLIGIBLE", "BENEFICIAL/Linear"],
              title="Mức ảnh hưởng", loc="upper right", fontsize=9, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_combined_ranking.png", dpi=DPI)
    plt.close(fig)


def fig_mse_mae(enc: pd.DataFrame, dec: pd.DataFrame):
    """Side-by-side grouped MSE & MAE per variant for both studies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, title, base in [
        (axes[0], enc, "Encoder Study", "baseline"),
        (axes[1], dec, "Decoder Study", "baseline"),
    ]:
        m = df.melt(id_vars="variant", value_vars=["test_mse", "test_mae"],
                    var_name="metric", value_name="value")
        sns.barplot(data=m, x="variant", y="value", hue="metric", ax=ax,
                    palette={"test_mse": "#d1495b", "test_mae": "#2e86ab"})
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Error (scaled space)")
        ax.tick_params(axis="x", rotation=30)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.legend(title="", fontsize=9)
    fig.suptitle("Test MSE & MAE theo từng variant (baseline = tham chiếu)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_mse_mae.png", dpi=DPI)
    plt.close(fig)


def fig_depth_and_curves():
    """Depth scaling bars + learning curves from the L=2 / L=5 JSON runs."""
    l2 = json.loads((EXP / "small-etth1_attn=full_pe=on_decomp=off_dec=on_L=2.json").read_text())
    l5 = json.loads((EXP / "small-etth1_attn=full_pe=on_decomp=off_dec=on_L=5.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) depth scaling
    d = pd.DataFrame({
        "config": ["L=2 (120K)", "L=5 (271K)"],
        "Test MSE": [l2["test"]["mse"], l5["test"]["mse"]],
        "Test MAE": [l2["test"]["mae"], l5["test"]["mae"]],
    }).melt(id_vars="config", var_name="metric", value_name="value")
    sns.barplot(data=d, x="config", y="value", hue="metric", ax=axes[0],
                palette={"Test MSE": "#d1495b", "Test MAE": "#2e86ab"})
    axes[0].set_title("Depth Scaling — sâu hơn lại tệ hơn (data nhỏ)", fontsize=13)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Error")

    # (b) learning curves (val MSE per epoch)
    for run, name, c in [(l2, "L=2", "#2e86ab"), (l5, "L=5", "#d1495b")]:
        ep = [h["epoch"] for h in run["history"]]
        mse = [h["mse"] for h in run["history"]]
        axes[1].plot(ep, mse, marker="o", label=name, color=c, lw=2)
    axes[1].set_title("Learning curves (Validation MSE)", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val MSE")
    axes[1].legend(title="#layers")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_depth_curves.png", dpi=DPI)
    plt.close(fig)


def fig_dataset_overview():
    """OT distribution + correlation heatmap for ETTh1 (data section figure)."""
    df = pd.read_csv(ROOT / "Data" / "ETTh1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # OT time series (downsampled for clarity)
    axes[0].plot(df["OT"].values[::24], color="#2e86ab", lw=0.8)
    axes[0].set_title("ETTh1 — Oil Temperature (target), daily-sampled", fontsize=13)
    axes[0].set_xlabel("Day index")
    axes[0].set_ylabel("OT (°C)")

    # correlation heatmap
    corr = df.drop(columns=["date"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Tương quan biến (ETTh1)\ncovariate↔OT yếu → temporal signal", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_dataset_overview.png", dpi=DPI)
    plt.close(fig)


def main():
    enc = pd.read_csv(EXP / "encoder_ablation.csv")
    dec = pd.read_csv(EXP / "decoder_ablation.csv")

    fig_encoder(enc)
    fig_decoder(dec)
    fig_combined(enc, dec)
    fig_mse_mae(enc, dec)
    fig_depth_and_curves()
    fig_dataset_overview()

    figs = sorted(OUT.glob("*.png"))
    print(f"✅ Wrote {len(figs)} figures to {OUT.relative_to(ROOT)}/")
    for f in figs:
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()
