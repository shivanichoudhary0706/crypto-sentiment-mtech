"""Import the public CoinDesk news dataset (Hugging Face) into raw_sentiment.

Dataset : maryamfakhari/crypto-news-coindesk-2020-2025 (CC BY-NC 4.0, academic use,
          attribution to CoinDesk Data and the dataset author required).
File    : coindesk-crypto-news-2020-2025.csv  (columns used: published_on, title,
          body, url, source)

Rows are mapped onto the SAME schema the live pipeline writes, so the existing
scorer, feature job, lag analysis and backtest run on them unchanged:
    time        <- published_on (UTC)
    source_type <- 'dataset_coindesk_hf'   (keeps provenance explicit)
    source_name <- source
    title       <- title
    summary     <- first SUMMARY_CHARS of body  (live RSS gives title + short summary;
                   truncating keeps FinBERT's input comparable)
    coin_tags   <- coin_tagger.tag_coins(title + summary)  (same keyword rule as live):
                   BTC/USDT, ETH/USDT, or MARKET (general crypto news)

Spillover design (thesis RQ3): the news streams are BTC, ETH and GENERAL (the cause);
SOL, LTC and LINK are price-only targets (the effect). Articles that name a target
coin are dropped -- otherwise "Solana surges" (no BTC/ETH mention) would be tagged
MARKET and leak SOL's own news into the general stream.
  * full names (solana, litecoin, chainlink) match case-insensitively;
  * tickers (SOL, LTC, LINK) match only in UPPER case, so the ordinary English
    words "link" / "sol" do not cause exclusions.

SAFETY: refuses to write into the live database 'crypto_analytics'.

Usage (cmd, project root, venv active):
    set DB_NAME=crypto_hist
    python -m tools.import_news_dataset --csv data\\coindesk-crypto-news-2020-2025.csv --start 2024-01-01 --end 2025-02-01
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from ingestion.sentiment.coin_tagger import tag_coins

LIVE_DB = "crypto_analytics"
SOURCE_TYPE = "dataset_coindesk_hf"
SUMMARY_CHARS = 400
CHUNK_ROWS = 20_000
USECOLS = ["published_on", "title", "body", "url", "source"]

TARGET_NAMES = re.compile(r"\b(solana|litecoin|chainlink)\b", re.IGNORECASE)
TARGET_TICKERS = re.compile(r"\b(SOL|LTC|LINK)\b")

INSERT_SQL = """
    INSERT INTO raw_sentiment
        (time, source_type, source_name, title, summary, url, coin_tags, ingested_at)
    VALUES %s
    ON CONFLICT (time, url) DO NOTHING
"""


def names_target_coin(text: str) -> bool:
    return bool(TARGET_NAMES.search(text) or TARGET_TICKERS.search(text))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD UTC, inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD UTC, exclusive")
    args = p.parse_args()

    dbname = os.getenv("DB_NAME", LIVE_DB)
    if dbname == LIVE_DB:
        print(f"REFUSING: DB_NAME is '{LIVE_DB}' (the live database). "
              "Run 'set DB_NAME=crypto_hist' first.")
        return 1

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"), port=int(os.getenv("DB_PORT", "5432")),
        dbname=dbname, user=os.getenv("DB_USER", "crypto_user"),
        password=os.getenv("DB_PASSWORD", "crypto_pass"),
    )
    print(f"Importing {args.csv} -> {dbname}.raw_sentiment for [{start.date()}, {end.date()})")

    seen = kept = excluded = 0
    tag_counts: dict[str, int] = {}
    now = datetime.now(timezone.utc)
    try:
        for chunk in pd.read_csv(args.csv, usecols=USECOLS, chunksize=CHUNK_ROWS,
                                 dtype=str, keep_default_na=False):
            seen += len(chunk)
            chunk["time"] = pd.to_datetime(chunk["published_on"], utc=True, errors="coerce")
            chunk = chunk[(chunk["time"] >= start) & (chunk["time"] < end)
                          & (chunk["title"].str.strip() != "") & (chunk["url"].str.strip() != "")]
            if chunk.empty:
                continue
            rows = []
            for r in chunk.itertuples(index=False):
                summary = (r.body or "")[:SUMMARY_CHARS]
                text = f"{r.title} {summary}"
                if names_target_coin(text):
                    excluded += 1
                    continue
                tags = tag_coins(text)            # BTC/USDT, ETH/USDT and/or MARKET
                for t in tags:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
                rows.append((r.time.to_pydatetime(), SOURCE_TYPE, r.source or "unknown",
                             r.title, summary, r.url, tags, now))
            kept += len(rows)
            if rows:
                with conn.cursor() as cur:
                    execute_values(cur, INSERT_SQL, rows, page_size=1000)
                conn.commit()
            print(f"  scanned {seen:>7,} | kept {kept:>6,} | excluded (names SOL/LTC/LINK) {excluded:>5,}")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM raw_sentiment WHERE source_type = %s", (SOURCE_TYPE,))
            in_db = cur.fetchone()[0]
    finally:
        conn.close()

    print(f"\nDone. scanned={seen:,} kept={kept:,} excluded={excluded:,} rows_in_db={in_db:,}")
    print("Coin tags:", ", ".join(f"{k}={v:,}" for k, v in sorted(tag_counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
