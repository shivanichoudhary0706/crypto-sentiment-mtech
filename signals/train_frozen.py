"""
Train a frozen signal model on historical features_bar rows and save it to models/.

Run in a window where the HISTORICAL database is selected:
    set TIMESCALEDB_DB=crypto_hist
    python -m signals.train_frozen --freq 1h --start 2024-07-01 --end 2025-02-01
    python -m signals.train_frozen --freq 4h --start 2024-07-01 --end 2025-02-01
    python -m signals.train_frozen --freq 1h --start 2024-07-01 --end 2025-02-01 --variant price_only

Writes models/<version>.joblib (the model) and models/<version>.json (metadata:
window, rows, features, parameters) -- the JSON is what you cite in the thesis.
"""
from __future__ import annotations

import argparse

from config.logging_config import get_logger
from features.config import load_config as load_feature_config
from signals import store
from signals.config import FREQS, load_config
from signals.frozen import save, train_frozen
from streaming.db import get_db_connection

logger = get_logger(__name__)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train and save a frozen signal model.")
    p.add_argument("--freq", choices=FREQS, required=True)
    p.add_argument("--start", required=True, help="Training window start (UTC date)")
    p.add_argument("--end", required=True, help="Training window end, exclusive (UTC date)")
    p.add_argument("--variant", choices=["price+news", "price_only"], default="price+news")
    p.add_argument("--keep-counts", action="store_true", help="Keep *_news_count features (not advised)")
    p.add_argument("--out", default="models")
    args = p.parse_args(argv)

    cfg = load_config(args.freq)
    feature_cfg = load_feature_config()
    conn = get_db_connection()
    try:
        features = store.fetch_features(conn, feature_cfg.table, cfg.targets + cfg.input_coins,
                                        cfg.freq, None)
    finally:
        conn.close()
    fm = train_frozen(features, targets=cfg.targets, input_coins=cfg.input_coins, freq=cfg.freq,
                      horizon_bars=cfg.horizon_bars, model_params=cfg.model, start=args.start,
                      end=args.end, drop_counts=not args.keep_counts, variant=args.variant)
    path = save(fm, args.out)
    m = fm.meta
    logger.info("Frozen model saved: %s", path)
    logger.info("  window %s -> %s | %d rows %s | up-rate %.1f%% | %d features",
                m["train_start"], m["train_end"], m["n_train"], m["rows_per_coin"],
                100 * m["up_rate"], len(m["features"]))
    print(f"\nSaved {path}\nUse it live with:  python -m signals.run_signals --freq {args.freq} --model {path}")


if __name__ == "__main__":
    main()
