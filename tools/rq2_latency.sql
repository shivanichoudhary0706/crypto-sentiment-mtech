\echo === A. Market: candle close -> DB insert (live rows, last 3 days) ===
SELECT symbol, count(*) AS n,
  round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY s)::numeric,2) AS p50_s,
  round(percentile_cont(0.95) WITHIN GROUP (ORDER BY s)::numeric,2) AS p95_s,
  round(max(s)::numeric,2) AS max_s
FROM (SELECT symbol, extract(epoch FROM ingested_at - (time + interval '1 minute')) AS s
      FROM market_data
      WHERE time > now() - interval '3 days'
        AND ingested_at < time + interval '10 minutes') x   -- excludes REST backfill rows
GROUP BY ROLLUP(symbol) ORDER BY symbol;

\echo === A2. Market rows excluded as backfill (last 3 days) ===
SELECT count(*) FROM market_data WHERE time > now() - interval '3 days'
  AND ingested_at >= time + interval '10 minutes';

\echo === B. News: publication -> DB insert (RSS only, since 11 Nov 17:22 UTC) ===
SELECT source_name, count(*) AS n,
  round((percentile_cont(0.5)  WITHIN GROUP (ORDER BY s)/60)::numeric,1) AS p50_min,
  round((percentile_cont(0.95) WITHIN GROUP (ORDER BY s)/60)::numeric,1) AS p95_min,
  sum((s < 0)::int) AS negative_n
FROM (SELECT source_name, extract(epoch FROM ingested_at - time) AS s
      FROM raw_sentiment
      WHERE source_type = 'news_rss' AND time >= '2026-11-11 17:22+00') x
GROUP BY ROLLUP(source_name) ORDER BY source_name;

\echo === C. Scoring: DB insert -> FinBERT score written ===
SELECT count(*) AS n,
  round(percentile_cont(0.5)  WITHIN GROUP (ORDER BY s)::numeric,1) AS p50_s,
  round(percentile_cont(0.95) WITHIN GROUP (ORDER BY s)::numeric,1) AS p95_s,
  round(max(s)::numeric,1) AS max_s
FROM (SELECT DISTINCT ON (r.time, r.url) extract(epoch FROM s.scored_at - r.ingested_at) AS s
      FROM raw_sentiment r JOIN sentiment_scores s ON s.time = r.time AND s.url = r.url
      WHERE r.source_type = 'news_rss' AND r.time >= '2026-11-11 17:22+00'
      ORDER BY r.time, r.url, s.scored_at) x;

\echo === D. Signals: bar close -> signal written ===
SELECT freq, (model_version LIKE 'frozen-%') AS frozen, count(*) AS n,
  round((percentile_cont(0.5)  WITHIN GROUP (ORDER BY s)/60)::numeric,1) AS p50_min,
  round((percentile_cont(0.95) WITHIN GROUP (ORDER BY s)/60)::numeric,1) AS p95_min
FROM (SELECT freq, model_version,
             extract(epoch FROM created_at - (bucket + CASE freq WHEN '1h' THEN interval '1 hour' ELSE interval '4 hours' END)) AS s
      FROM signals) x
GROUP BY 1,2 ORDER BY 1,2;