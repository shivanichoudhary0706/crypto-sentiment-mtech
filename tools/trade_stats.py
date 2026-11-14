"""Trade statistics from walk-forward backtest predictions (READ-ONLY: reads a CSV, writes a CSV).

Usage (from project root, venv active):
    python -m tools.trade_stats
    python -m tools.trade_stats --file reports\\backtest\\1h\\predictions.csv --out reports\\backtest\\1h\\trade_stats.csv

Answers: why does price+news (AUC 0.531) lose less than price-only (AUC 0.535)?
-> compare trades, exposure, holding time and fee drag per variant.
"""
import argparse
from pathlib import Path

import pandas as pd


def holding_runs(position: pd.Series) -> pd.Series:
    """Lengths (in bars) of consecutive in-market runs (position > 0)."""
    in_pos = position > 0
    run_id = (in_pos != in_pos.shift()).cumsum()
    runs = in_pos.groupby(run_id).agg(["first", "size"])
    return runs.loc[runs["first"], "size"]


def compound(r: pd.Series) -> float:
    return ((1 + r).prod() - 1) * 100


def coin_stats(g: pd.DataFrame) -> dict:
    g = g.sort_values("bucket")
    months = (g["bucket"].iloc[-1] - g["bucket"].iloc[0]).days / 30.44
    runs = holding_runs(g["position"])
    gross = g["position"] * g["next_return"]
    return {
        "bars": len(g),
        "buy_pct": round((g["action"] == "BUY").mean() * 100, 1),
        "hold_pct": round((g["action"] == "HOLD").mean() * 100, 1),
        "sell_pct": round((g["action"] == "SELL").mean() * 100, 1),
        "trades": int(g["trade"].sum()),
        "trades_per_month": round(g["trade"].sum() / months, 1),
        "round_trips": int(len(runs)),
        "exposure_pct": round((g["position"] > 0).mean() * 100, 1),
        "avg_hold_bars": round(runs.mean(), 1) if len(runs) else 0,
        "median_hold_bars": float(runs.median()) if len(runs) else 0,
        "fees_total_pct": round(g["fee"].sum() * 100, 2),
        "gross_return_pct": round(compound(gross), 1),
        "net_return_pct": round(compound(g["strategy_return"]), 1),
        "buy_hold_pct": round(compound(g["next_return"]), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=r"reports\backtest\1h\predictions.csv")
    ap.add_argument("--out", default=r"reports\backtest\1h\trade_stats.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.file)
    d["bucket"] = pd.to_datetime(d["bucket"], utc=True)
    d["trade"] = d["trade"].astype(str).str.lower().eq("true")

    # ---- sanity checks -------------------------------------------------
    print("=== Sanity checks ===")
    print("rows:", len(d), "| window:", d["bucket"].min(), "->", d["bucket"].max())
    print("position values:", sorted(d["position"].unique().tolist()))
    print("non-zero fee values:", sorted(d.loc[d["fee"] != 0, "fee"].round(6).unique().tolist())[:10])
    print("n_train values:", sorted(d["n_train"].unique().tolist())[:10])
    recon = d["position"] * d["next_return"] - d["fee"]
    print("max |strategy_return - (position*next_return - fee)|:",
          f"{(d['strategy_return'] - recon).abs().max():.2e}  (should be ~0)")

    # ---- per variant x coin -------------------------------------------
    rows = []
    for (variant, coin), g in d.groupby(["variant", "coin"]):
        rows.append({"variant": variant, "coin": coin, **coin_stats(g)})
    out = pd.DataFrame(rows)

    # ---- equal-weight portfolio per variant ----------------------------
    print("\n=== Equal-weight portfolio (mean of 3 coins per bar, compounded) ===")
    for variant, g in d.groupby("variant"):
        net = g.pivot(index="bucket", columns="coin", values="strategy_return").mean(axis=1)
        gross = (g.assign(gr=g["position"] * g["next_return"])
                   .pivot(index="bucket", columns="coin", values="gr").mean(axis=1))
        bh = g.pivot(index="bucket", columns="coin", values="next_return").mean(axis=1)
        print(f"{variant:12s} net {compound(net):7.1f}% | gross {compound(gross):7.1f}% "
              f"| buy&hold {compound(bh):7.1f}%")

    pd.set_option("display.width", 250)
    print("\n=== Per variant x coin ===")
    print(out.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
