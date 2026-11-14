"""Frozen model: train once on history, save, reload, predict -- no retraining."""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.config import parse_signal_config
from signals.frozen import load, save, select_features, train_frozen
from signals.run_signals import generate_signals
from tests.test_signals import _CFG, INPUTS, TARGETS, make_features


def test_count_features_are_dropped_scores_kept():
    names = ["ret", "btc_news_score", "btc_news_count", "btc_news_roll"]
    assert select_features(names) == ["ret", "btc_news_score", "btc_news_roll"]
    assert select_features(names, drop_counts=False) == names


def test_frozen_model_learns_saves_reloads_and_predicts(tmp_path):
    hist = make_features(1500, seed=1, planted=True)
    fm = train_frozen(hist, targets=TARGETS, input_coins=INPUTS, freq="1h", horizon_bars=1,
                      model_params=_CFG["model"])
    assert "btc_news_count" not in fm.model.feature_names
    assert fm.meta["n_train"] > 0 and fm.version.startswith("frozen-hgb-1h-h1-pn-")

    fm2 = load(save(fm, tmp_path))
    assert fm2.model.feature_names == fm.model.feature_names

    # "Live" data from a later period: the planted pattern must still be found
    live = make_features(300, seed=7, planted=True)
    live["bucket"] = live["bucket"] + pd.Timedelta(days=700)
    cfg = parse_signal_config(_CFG)
    sig, info = generate_signals(live, cfg, frozen=fm2)
    assert set(sig["coin"]) == set(TARGETS)
    assert (sig["model_version"] == fm.version).all()
    # newest live bar's BTC news sign decides direction under the planted rule
    news = live[live["coin"] == "BTC/USDT"].sort_values("bucket")["btc_news_score"].iloc[-1]
    if news > 0:
        assert (sig["prob_up"] > 0.5).all()
    elif news < 0:
        assert (sig["prob_up"] < 0.5).all()


def test_price_only_variant_has_no_news_features():
    hist = make_features(800, seed=2)
    fm = train_frozen(hist, targets=TARGETS, input_coins=INPUTS, freq="1h", horizon_bars=1,
                      model_params=_CFG["model"], variant="price_only")
    assert not [f for f in fm.model.feature_names if "_news_" in f]
