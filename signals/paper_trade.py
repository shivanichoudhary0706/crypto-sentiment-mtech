"""
Live paper-trade evaluation of FROZEN signals. Pure pandas, no DB. Unit-testable.

Same core rules as tools/paper_trade_eval.sql:
  * Only rows whose model_version starts with "frozen-" (pre-freeze retraining rows
    are the "live retraining dry run" and are excluded).
  * Outcome of the signal decided at bar close t (valid_from) is the log return to the
    next bar close of the same symbol: ln(close_{t+1} / close_t). close_{t+1} is the
    `close` stored on the NEXT signal row (valid_from + 1 bar). If that row does not
    exist the outcome is PENDING (bar not closed yet) or MISSING (signal gap).
  * BUY is RIGHT if the next return > 0, SELL is RIGHT if < 0. HOLD is not scored.
  * Paper P&L, per coin: long-only, start flat. BUY -> long, SELL -> flat,
    HOLD -> keep the current position. cost_bps is charged per side on every
    position change. Buy & hold: long from the first decision close, one entry cost.
  * created_at is only meaningful (= first write) for rows first written after the
    store.py fix (CREATED_AT_VALID_FROM). Signal latency and the "written after the
    outcome" suspect check use those rows only.

This is a PRELIMINARY LIVE OPERATIONAL VALIDATION tool. The sample is tiny and the
coins move together, so the hit rate is reported with n, a Wilson 95% interval and
the number of independent decision times -- never on its own.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FROZEN_PREFIX = "frozen-"
# signals/store.py fix: from here on created_at = FIRST write of the row.
CREATED_AT_VALID_FROM = pd.Timestamp("2026-11-13 18:41", tz="UTC")
REQUIRED = ("valid_from", "symbol", "close", "prob_up", "action", "model_version", "created_at")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (Wilson 1927)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def prepare(signals: pd.DataFrame, bar: pd.Timedelta, now=None,
            frozen_only: bool = True) -> pd.DataFrame:
    """One row per signal with next_close, next_ret, result, latency_s, suspect."""
    missing = [c for c in REQUIRED if c not in signals.columns]
    if missing:
        raise ValueError(f"signals frame lacks columns: {missing}")
    df = signals.copy()
    for c in ("bucket", "valid_from", "created_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)
    for c in ("close", "prob_up"):                 # psycopg2 returns NUMERIC as Decimal
        df[c] = df[c].astype(float)
    if frozen_only:
        df = df[df["model_version"].astype(str).str.startswith(FROZEN_PREFIX)]
    df = df.sort_values(["symbol", "valid_from"]).reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    nxt = df.groupby("symbol")[["valid_from", "close"]].shift(-1)
    contiguous = nxt["valid_from"] == df["valid_from"] + bar
    df["next_close"] = nxt["close"].where(contiguous)
    df["next_ret"] = np.log(df["next_close"] / df["close"])
    df["outcome_at"] = df["valid_from"] + bar

    created_ok = df["valid_from"] >= CREATED_AT_VALID_FROM
    df["latency_s"] = (df["created_at"] - df["valid_from"]).dt.total_seconds().where(created_ok)
    df["suspect"] = created_ok & (df["created_at"] >= df["outcome_at"])

    is_call = df["action"].isin(["BUY", "SELL"])
    right = np.where(df["action"] == "BUY", df["next_ret"] > 0, df["next_ret"] < 0)
    unknown = df["next_ret"].isna()
    df["result"] = np.select(
        [~is_call, unknown & (df["outcome_at"] > now), unknown, right],
        ["HOLD", "PENDING", "MISSING", "RIGHT"],
        default="WRONG")
    return df


def _hit_rate(df: pd.DataFrame) -> dict:
    scored = df[df["result"].isin(["RIGHT", "WRONG"]) & ~df["suspect"]]
    k, n = int((scored["result"] == "RIGHT").sum()), int(len(scored))
    lo, hi = wilson(k, n)
    # Coins move together: all calls at one decision time are ~one bet.
    per_time = scored.groupby("valid_from")["result"].agg(
        lambda r: "RIGHT" if (r == "RIGHT").all() else "WRONG" if (r == "WRONG").all() else "MIXED")
    known = df[df["next_ret"].notna() & ~df["suspect"]]
    return {
        "n_signals": int(len(df)),
        "n_buy": int((df["action"] == "BUY").sum()),
        "n_sell": int((df["action"] == "SELL").sum()),
        "n_hold": int((df["action"] == "HOLD").sum()),
        "n_scored": n, "n_right": k,
        "hit_rate": k / n if n else None, "wilson_lo": lo, "wilson_hi": hi,
        "decision_times": int(len(per_time)),
        "decision_times_right": int((per_time == "RIGHT").sum()),
        "decision_times_wrong": int((per_time == "WRONG").sum()),
        "decision_times_mixed": int((per_time == "MIXED").sum()),
        "pct_bars_up": float((known["next_ret"] > 0).mean()) if len(known) else None,
        "n_pending": int((df["result"] == "PENDING").sum()),
        "n_missing": int((df["result"] == "MISSING").sum()),
        "n_suspect": int(df["suspect"].sum()),
    }


def paper_pnl(df: pd.DataFrame, cost_bps: float = 10.0) -> tuple[list[dict], pd.DataFrame]:
    """Long-only paper P&L per coin vs buy & hold. Returns (summary rows, equity curves)."""
    cost = cost_bps / 1e4
    rows, curves = [], []
    for sym, g in df.sort_values("valid_from").groupby("symbol"):
        pos = g["action"].map({"BUY": 1.0, "SELL": 0.0}).ffill().fillna(0.0)   # HOLD keeps
        trade = pos.ne(pos.shift(1).fillna(0.0))
        simple = np.expm1(g["next_ret"].fillna(0.0))                          # unknown bar -> 0
        equity = ((1 + pos * simple) * np.where(trade, 1 - cost, 1.0)).cumprod()
        buy_hold = (1 + simple).cumprod() * (1 - cost)
        rows.append({"symbol": sym, "paper_return": float(equity.iloc[-1] - 1),
                     "buy_hold_return": float(buy_hold.iloc[-1] - 1),
                     "position_changes": int(trade.sum()),
                     "time_in_market": float(pos.mean()),
                     "bars": int(len(g))})
        # equity at the close of bar t+1, i.e. after the bar decided at valid_from
        curves.append(pd.DataFrame({"symbol": sym, "time": g["outcome_at"].tolist(),
                                    "paper": equity.values, "buy_hold": buy_hold.values}))
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return rows, curve


def operations(df: pd.DataFrame, bar: pd.Timedelta) -> dict:
    """Signals expected vs present per symbol, and signal latency (valid rows only)."""
    per = []
    for sym, g in df.groupby("symbol"):
        first, last = g["valid_from"].min(), g["valid_from"].max()
        expected = int((last - first) / bar) + 1
        present = int(g["valid_from"].nunique())
        per.append({"symbol": sym, "first": first, "last": last,
                    "expected": expected, "present": present, "missing": expected - present})
    lat = df["latency_s"].dropna()
    return {"per_symbol": per, "n_latency": int(len(lat)),
            "latency_p50_s": float(lat.median()) if len(lat) else None,
            "latency_p95_s": float(lat.quantile(0.95)) if len(lat) else None,
            "model_versions": sorted(df["model_version"].astype(str).unique().tolist())}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def evaluate(signals: pd.DataFrame, bar: pd.Timedelta, cost_bps: float = 10.0,
             now=None, frozen_only: bool = True) -> dict:
    """Full JSON-ready evaluation: summary, P&L, equity curve, operations, per-signal table."""
    df = prepare(signals, bar, now=now, frozen_only=frozen_only)
    if df.empty:
        return {"summary": None, "pnl": [], "equity": [], "operations": None, "signals": []}
    pnl_rows, curve = paper_pnl(df, cost_bps)
    table = df[["valid_from", "symbol", "action", "prob_up", "close", "next_close",
                "next_ret", "result", "model_version"]].copy()
    table["next_ret_bps"] = table.pop("next_ret") * 1e4
    return _jsonable({
        "summary": {**_hit_rate(df), "cost_bps": cost_bps,
                    "window_start": df["valid_from"].min(), "window_end": df["valid_from"].max()},
        "pnl": pnl_rows,
        "equity": curve.to_dict("records"),
        "operations": operations(df, bar),
        "signals": table.to_dict("records"),
    })
