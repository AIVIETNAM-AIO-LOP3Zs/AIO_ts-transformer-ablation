"""Standard ETT benchmark configuration.

This module defines the *canonical reference setting* for the ablation study — the
full-size baseline that every ablated variant is compared against. It deliberately
mirrors the long-term forecasting protocol used by Informer / Autoformer / FEDformer
on the ETT datasets so the numbers our benchmark produces are comparable to the
published literature, unlike the deliberately tiny ``train.py`` laptop run.

Canonical setting
-----------------
* Model:   ``d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
            dropout=0.05, activation='gelu'`` (full attention baseline).
* Window:  ``seq_len=96, label_len=48`` lookback; horizon swept over
            ``pred_len ∈ {96, 192, 336, 720}``.
* Data:    multivariate (``features='M'``), ``StandardScaler`` fit on the train
            split only (see ``Dataloader.ETTDataset``).
* Train:   Adam, ``lr=1e-4``, ``batch_size=32``, ``epochs=10`` with early-stopping
            patience 3; MSE loss; metrics MSE/MAE reported in scaled space.

Usage
-----
    from ts_ablation.configs.standard import standard_config, STANDARD_HORIZONS

    cfg = standard_config("ETTh1", pred_len=96)        # one benchmark point
    for h in STANDARD_HORIZONS:                          # full horizon sweep
        cfg = standard_config("ETTh1", pred_len=h)

Caveat (ETTm1/ETTm2)
--------------------
The ETT *minute* series are sampled every 15 min, but ``ETTDataset`` emits only
``[hour, day-of-month, weekday]`` calendar marks — the sub-hour (minute) component
is not encoded. Hourly series (ETTh1/ETTh2) are unaffected; ETTm benchmarks are
therefore slightly handicapped relative to implementations that add a minute mark.
"""

from __future__ import annotations

from .experiment import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)

# The four standard long-term forecasting horizons for ETT.
STANDARD_HORIZONS: tuple[int, ...] = (96, 192, 336, 720)

# Dataset name -> CSV path. ETTh* are hourly; ETTm* are 15-min (see caveat above).
ETT_DATASETS: dict[str, str] = {
    "ETTh1": "Data/ETTh1.csv",
    "ETTh2": "Data/ETTh2.csv",
    "ETTm1": "Data/ETTm1.csv",
    "ETTm2": "Data/ETTm2.csv",
}


def standard_model_config() -> ModelConfig:
    """The canonical full-attention baseline architecture.

    This is the reference point of the ablation study: every switch is at its
    baseline value (full attention, positional encoding on, no decomposition,
    encoder-decoder). Ablations flip exactly one of these at a time.
    """
    return ModelConfig(
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        activation="gelu",
        attention_type="full",
        use_positional_encoding=True,
        use_decomposition=False,
        use_decoder=True,
    )


def standard_config(
    dataset: str = "ETTh1",
    pred_len: int = 96,
    *,
    seq_len: int = 96,
    label_len: int = 48,
    features: str = "M",
    target: str = "OT",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 3,
    device: str = "cpu",
    model: ModelConfig | None = None,
) -> ExperimentConfig:
    """Build a standard ETT benchmark ``ExperimentConfig``.

    Args:
        dataset:   One of ``ETT_DATASETS`` (e.g. ``"ETTh1"``).
        pred_len:  Forecast horizon; usually one of ``STANDARD_HORIZONS``.
        seq_len, label_len: lookback window sizes (canonical 96 / 48).
        features:  ``"M"`` (multivariate, default), ``"S"``, or ``"MS"``.
        target:    target channel for univariate modes (ETT uses ``"OT"``).
        epochs, batch_size, learning_rate, patience, device: training knobs.
        model:     override the architecture; defaults to ``standard_model_config()``.

    Returns:
        A fully-populated ``ExperimentConfig`` named ``std-<dataset>-pl<pred_len>``.
    """
    if dataset not in ETT_DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(ETT_DATASETS)}"
        )

    return ExperimentConfig(
        name=f"std-{dataset}-pl{pred_len}",
        model=model if model is not None else standard_model_config(),
        train=TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            grad_clip=1.0,
            device=device,
        ),
        data=DataConfig(
            csv_path=ETT_DATASETS[dataset],
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            features=features,  # type: ignore[arg-type]
            target=target,
        ),
    )
