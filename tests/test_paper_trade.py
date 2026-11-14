"""Unit tests for signals/paper_trade.py (hand-made toy frames, not results)."""
import math

import pandas as pd
import pytest

from signals.paper_trade import CREATED_AT_VALID_FROM, evaluate, paper_pnl, prepare, wilson

BAR = pd.Timedelta("1h")
T0 = pd.Timestamp("2026-11-14 00:00", tz="UTC")          # after the created_at fix
NOW = pd.Timestamp("2026-11-20 00:00", tz="UTC")


def frame(symbol, actions, closes, start=T0, version="frozen-hgb-1h-h1-pn-test", skip=()):
    rows = []
    for i, (a, c) in enumerate(zip(actions, closes)):
        if i in skip:
            continue
        vf = start + i * BAR
        rows.append({"valid_from": vf, "bucket": vf - BAR, "symbol": symbol, "close": c,
                     "prob_up": 0.6 if a == "BUY" else 0.4 if a == "SELL" else 0.5,
                     "action": a, "model_version": version,
                     "created_at": vf + pd.Timedelta(minutes=3)})
    return pd.DataFrame(rows)


def test_wilson_matches_handoff_figure():
    lo, hi = wilson(9, 13)                      # handoff: 9/13, Wilson 42-87 %
    assert round(lo, 2) == 0.42 and round(hi, 2) == 0.87
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_results_right_wrong_hold_pending():
    df = prepare(frame("SOLUSDT", ["BUY", "SELL", "HOLD", "BUY"], [100, 101, 102, 101]), BAR, now=NOW)
    assert df["result"].tolist() == ["RIGHT", "WRONG", "HOLD", "MISSING"]
    df = prepare(frame("SOLUSDT", ["BUY", "BUY"], [100, 99]), BAR, now=T0 + BAR)
    assert df["result"].tolist() == ["WRONG", "PENDING"]   # last bar not closed yet


def test_gap_gives_missing_not_wrong_outcome():
    df = prepare(frame("SOLUSDT", ["BUY", "BUY", "BUY"], [100, 200, 101], skip={1}), BAR, now=NOW)
    assert df["result"].iloc[0] == "MISSING"                 # next row is 2 bars later


def test_non_frozen_rows_excluded():
    sig = pd.concat([frame("SOLUSDT", ["BUY", "BUY"], [1, 2]),
                     frame("SOLUSDT", ["SELL", "SELL"], [1, 2], start=T0 + 2 * BAR,
                           version="hgb-retrain-123")])
    df = prepare(sig, BAR, now=NOW)
    assert len(df) == 2 and set(df["action"]) == {"BUY"}


def test_suspect_written_after_outcome():
    sig = frame("SOLUSDT", ["BUY", "BUY"], [100, 101])
    sig.loc[0, "created_at"] = sig.loc[0, "valid_from"] + 2 * BAR
    df = prepare(sig, BAR, now=NOW)
    assert df["suspect"].tolist() == [True, False]
    out = evaluate(sig, BAR, now=NOW)
    assert out["summary"]["n_scored"] == 0 and out["summary"]["n_suspect"] == 1


def test_latency_only_after_created_at_fix():
    early = frame("SOLUSDT", ["HOLD", "HOLD"], [1, 1], start=CREATED_AT_VALID_FROM - 3 * BAR)
    df = prepare(early, BAR, now=NOW)
    assert df["latency_s"].isna().all() and not df["suspect"].any()


def test_paper_pnl_long_only_with_costs():
    # BUY at 100 -> 110 (long), SELL at 110 -> 120 (flat), HOLD at 120 -> 90 (still flat)
    df = prepare(frame("SOLUSDT", ["BUY", "SELL", "HOLD", "HOLD"], [100, 110, 120, 90]), BAR, now=NOW)
    rows, curve = paper_pnl(df, cost_bps=10)
    c = 0.001
    assert rows[0]["paper_return"] == pytest.approx((1 - c) * 1.10 * (1 - c) - 1)
    assert rows[0]["buy_hold_return"] == pytest.approx(0.90 * (1 - c) - 1)
    assert rows[0]["position_changes"] == 2
    assert len(curve) == 4


def test_hold_first_starts_flat_and_decision_times_grouped():
    sig = pd.concat([frame(s, ["HOLD", "BUY", "BUY"], [100, 100, 105]) for s in
                     ("SOLUSDT", "LTCUSDT", "LINKUSDT")])
    out = evaluate(sig, BAR, now=NOW)
    s = out["summary"]
    assert s["n_scored"] == 3 and s["n_right"] == 3      # 3 coins x one decision time
    assert s["decision_times"] == 1 and s["decision_times_right"] == 1
    for r in out["pnl"]:                                   # flat, then long 100 -> 105
        assert r["paper_return"] == pytest.approx(0.999 * 1.05 - 1)


def test_decimal_prices_from_psycopg2():
    from decimal import Decimal
    sig = frame("SOLUSDT", ["BUY", "HOLD"], [100, 101])
    sig["close"] = sig["close"].map(lambda v: Decimal(str(v)))
    sig["prob_up"] = sig["prob_up"].map(lambda v: Decimal(str(v)))
    assert prepare(sig, BAR, now=NOW)["result"].iloc[0] == "RIGHT"


def test_evaluate_is_json_ready():
    import json
    out = evaluate(frame("SOLUSDT", ["BUY", "HOLD"], [100, 101]), BAR, now=NOW)
    json.dumps(out, allow_nan=False)                      # no Timestamp / NaN left
    assert out["equity"][0]["time"].endswith("+00:00")
