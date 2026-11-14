"""Backtest robustness: cost sensitivity + block-bootstrap test (READ-ONLY: CSV in, CSV out).

Usage (project root, venv active):
    python -m tools.backtest_robustness
    python -m tools.backtest_robustness --block 168 --reps 2000 --seed 42

Inputs : reports\\backtest\\1h\\predictions.csv (walk-forward output)
Outputs: reports\\backtest\\1h\\cost_sensitivity.csv, reports\\backtest\\1h\\bootstrap_pn_vs_po_block<N>.csv
         (one file per block length, so sensitivity runs never overwrite each other)

Notes
- Positions do not depend on the fee, so re-costing is EXACT (no re-run needed).
- Portfolio = equal-weight mean of the 3 coins' per-bar returns, compounded (reproduces -21.0% / -58.8%).
- Bootstrap = circular moving-block bootstrap of the per-bar portfolio return DIFFERENCE
  (price+news minus price_only); block keeps autocorrelation/vol clustering; p-value two-sided,
  from the bootstrap distribution centred on the observed mean.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COSTS_BPS = [0, 5, 7.5, 10, 15, 20]
PN, PO = "price+news", "price_only"


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["bucket"] = pd.to_datetime(d["bucket"], utc=True)
    d["trade"] = d["trade"].astype(str).str.lower().eq("true")
    return d


def portfolio(d: pd.DataFrame, variant: str, cost_bps: float) -> pd.Series:
    g = d[d["variant"] == variant]
    r = g["position"] * g["next_return"] - g["trade"] * cost_bps / 1e4
    return r.groupby(g["bucket"]).mean().sort_index()


def total_pct(r: pd.Series) -> float:
    return ((1 + r).prod() - 1) * 100


def breakeven_bps(d: pd.DataFrame, variant: str) -> float:
    lo, hi = 0.0, 100.0
    if total_pct(portfolio(d, variant, lo)) <= 0:
        return 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if total_pct(portfolio(d, variant, mid)) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def block_bootstrap_means(x: np.ndarray, block: int, reps: int, rng) -> np.ndarray:
    n = len(x)
    nb = int(np.ceil(n / block))
    offs = np.arange(block)
    out = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n, size=nb)
        idx = ((starts[:, None] + offs) % n).ravel()[:n]
        out[i] = x[idx].mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=r"reports\backtest\1h\predictions.csv")
    ap.add_argument("--outdir", default=r"reports\backtest\1h")
    ap.add_argument("--block", type=int, default=168, help="block length in bars (168 = 1 week at 1h)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = load(args.file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- sanity: fee column == trade * 10 bps ---------------------------
    fee_gap = (d["fee"] - d["trade"] * 0.001).abs().max()
    print(f"max |fee - trade*0.001| = {fee_gap:.2e}  (should be ~0 -> re-costing is exact)")
    check = {v: total_pct(portfolio(d, v, 10)) for v in (PN, PO)}
    print(f"reproduce @10 bps: {PN} {check[PN]:.1f}%  {PO} {check[PO]:.1f}%  (expect -21.0 / -58.8)\n")

    # ---- 1. cost sensitivity -------------------------------------------
    bh = d[d["variant"] == PN].groupby("bucket")["next_return"].mean().sort_index()
    rows = []
    for c in COSTS_BPS:
        rows.append({"cost_bps_per_side": c,
                     PN: round(total_pct(portfolio(d, PN, c)), 1),
                     PO: round(total_pct(portfolio(d, PO, c)), 1),
                     "buy_hold": round(total_pct(bh), 1)})
    cost = pd.DataFrame(rows)
    print("=== 1. Cost sensitivity: equal-weight portfolio total return (%) ===")
    print(cost.to_string(index=False))
    be = {v: breakeven_bps(d, v) for v in (PN, PO)}
    print(f"\nBreak-even cost per side: {PN} {be[PN]:.2f} bps | {PO} {be[PO]:.2f} bps")
    cost.to_csv(outdir / "cost_sensitivity.csv", index=False)

    # ---- 2. block bootstrap of pn - po ----------------------------------
    rng = np.random.default_rng(args.seed)
    res = []
    for label, c in (("gross (0 bps)", 0), ("net (10 bps)", 10)):
        a, b = portfolio(d, PN, c), portfolio(d, PO, c)
        a, b = a.align(b, join="inner")
        diff = (a - b).to_numpy()
        obs = diff.mean()
        boots = block_bootstrap_means(diff, args.block, args.reps, rng)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = float(np.mean(np.abs(boots - obs) >= abs(obs)))
        res.append({"comparison": f"{PN} - {PO}", "costs": label, "bars": len(diff),
                    "mean_diff_bps_per_bar": round(obs * 1e4, 3),
                    "ci95_low_bps": round(lo * 1e4, 3), "ci95_high_bps": round(hi * 1e4, 3),
                    "p_two_sided": round(p, 4), "block_bars": args.block, "reps": args.reps,
                    "seed": args.seed})
    boot = pd.DataFrame(res)
    print(f"\n=== 2. Block bootstrap (block {args.block} bars, {args.reps} reps, seed {args.seed}) ===")
    print(boot.drop(columns=["block_bars", "reps", "seed"]).to_string(index=False))
    for r in res:
        excl = r["ci95_low_bps"] > 0 or r["ci95_high_bps"] < 0
        print(f"  {r['costs']}: 95% CI {'EXCLUDES' if excl else 'includes'} 0 -> "
              f"{'nominally significant (uncorrected)' if excl else 'not distinguishable from noise'}")
    boot_file = outdir / f"bootstrap_pn_vs_po_block{args.block}.csv"
    boot.to_csv(boot_file, index=False)
    print(f"\nSaved -> {outdir / 'cost_sensitivity.csv'}, {boot_file}")


if __name__ == "__main__":
    main()
