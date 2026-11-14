"""
Frozen (train-once, predict-live) models. Pure, no DB.

The live runner normally retrains on every pass. A frozen model is instead
trained ONCE on a historical window (e.g. the public CoinDesk news dataset +
Binance prices) and saved to disk; the live runner then only predicts. Because
the model file is written before the live data exists, every live prediction is
a genuine out-of-period forward test.

Feature policy for frozen models: raw article COUNT columns (*_news_count) are
dropped by default. The historical dataset aggregates ~80 outlets while the live
feed polls 5 RSS feeds, so counts differ in scale between training and serving;
sentiment SCORES and rolling scores are scale-free and are kept.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from signals.dataset import build_model_frame
from signals.model import TrainedModel, train

COUNT_SUFFIX = "_news_count"


def select_features(feature_names: list[str], drop_counts: bool = True) -> list[str]:
    if not drop_counts:
        return list(feature_names)
    return [f for f in feature_names if not f.endswith(COUNT_SUFFIX)]


@dataclass
class FrozenModel:
    model: TrainedModel
    version: str
    freq: str
    horizon_bars: int
    targets: list[str]
    input_coins: list[str]
    meta: dict = field(default_factory=dict)


def train_frozen(features: pd.DataFrame, *, targets, input_coins, freq: str, horizon_bars: int,
                 model_params: dict, start=None, end=None, drop_counts: bool = True,
                 variant: str = "price+news") -> FrozenModel:
    """Train one pooled model on every labelled target bar in [start, end)."""
    frame, feature_names = build_model_frame(features, targets, input_coins, horizon_bars)
    names = select_features(feature_names, drop_counts)
    if variant == "price_only":
        names = [f for f in names if "_news_" not in f]
    rows = frame[frame["coin"].isin(targets) & frame["label"].notna() & frame["close"].notna()]
    if start is not None:
        rows = rows[rows["bucket"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        rows = rows[rows["bucket"] < pd.Timestamp(end, tz="UTC")]
    if rows.empty:
        raise ValueError("No labelled rows in the training window")
    model = train(rows, names, model_params)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    tag = "pn" if variant == "price+news" else "po"
    version = f"frozen-hgb-{freq}-h{horizon_bars}-{tag}-{stamp}"
    meta = {
        "variant": variant,
        "train_start": str(rows["bucket"].min()),
        "train_end": str(rows["bucket"].max()),
        "n_train": int(len(rows)),
        "rows_per_coin": {k: int(v) for k, v in rows.groupby("coin").size().items()},
        "up_rate": float(rows["label"].mean()),
        "features": names,
        "dropped_count_features": drop_counts,
        "model_params": model_params,
        "created_at": stamp,
    }
    return FrozenModel(model, version, freq, horizon_bars, list(targets), list(input_coins), meta)


def save(fm: FrozenModel, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{fm.version}.joblib"
    joblib.dump(fm, path)
    (out / f"{fm.version}.json").write_text(json.dumps(
        {"version": fm.version, "freq": fm.freq, "horizon_bars": fm.horizon_bars,
         "targets": fm.targets, "input_coins": fm.input_coins, **fm.meta}, indent=2, default=str))
    return path


def load(path: str | Path) -> FrozenModel:
    fm = joblib.load(path)
    if not isinstance(fm, FrozenModel):
        raise TypeError(f"{path} is not a FrozenModel file")
    return fm


def check_compatible(fm: FrozenModel, frame_columns) -> list[str]:
    """Feature names the model needs but the live frame lacks (empty list = OK)."""
    cols = set(frame_columns)
    return [f for f in fm.model.feature_names if f not in cols]
