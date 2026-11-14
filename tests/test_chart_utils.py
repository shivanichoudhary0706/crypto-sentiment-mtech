"""Tests for dashboard.chart_utils (gap breaks, live/replay tagging, change flags)."""
import pandas as pd

from dashboard.chart_utils import missing_bars, prepare_history, with_gap_breaks

H = pd.Timedelta(hours=1)


def rows(hours, actions=None, versions=None):
    t0 = pd.Timestamp("2026-10-24", tz="UTC")
    actions = actions or ["HOLD"] * len(hours)
    versions = versions or ["hgb-1h"] * len(hours)
    return [{"bucket": (t0 + h * H).isoformat(), "close": 50.0 + h, "prob_up": 0.5,
             "action": a, "model_version": v,
             "valid_from": (t0 + (h + 1) * H).isoformat(),
             "created_at": (t0 + (h + 1) * H + pd.Timedelta(minutes=10)).isoformat()}
            for h, a, v in zip(hours, actions, versions)]


def test_empty_rows_give_empty_frame():
    assert prepare_history([]).empty


def test_source_is_tagged_from_model_version():
    df = prepare_history(rows([0, 1], versions=["hgb-1h", "replay:hgb-1h"]))
    assert list(df["source"]) == ["live", "replayed"]


def test_changed_marks_first_bar_and_every_switch():
    df = prepare_history(rows([0, 1, 2, 3], actions=["BUY", "BUY", "SELL", "SELL"]))
    assert list(df["changed"]) == [True, False, True, False]


def test_latency_is_created_minus_valid_from():
    assert prepare_history(rows([0]))["latency_min"].iloc[0] == 10


def test_rows_are_sorted_by_time():
    df = prepare_history(rows([2, 0, 1]))
    assert df["bucket"].is_monotonic_increasing


def test_missing_bars_counts_the_hole():
    assert missing_bars(prepare_history(rows([0, 1, 6, 7])), H) == 4   # hours 2-5 missing
    assert missing_bars(prepare_history(rows([0, 1, 2])), H) == 0


def test_gap_break_inserts_one_nan_row_inside_each_gap():
    df = prepare_history(rows([0, 1, 6, 7, 10]))
    out = with_gap_breaks(df, H)
    assert len(out) == len(df) + 2
    breaks = out[out["close"].isna()]
    assert list(breaks["bucket"].dt.hour) == [5, 9]          # inside each gap
    assert out["bucket"].is_monotonic_increasing


def test_no_gap_returns_frame_unchanged():
    df = prepare_history(rows([0, 1, 2]))
    assert with_gap_breaks(df, H) is df