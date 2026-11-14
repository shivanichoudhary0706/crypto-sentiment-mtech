-- tools/paper_trade_eval.sql  (v2: HOLD scoring, model phases, suspect-row check)
-- Paper-trade evaluation of live signals against real Binance 1m closes (market_data).
-- READ-ONLY on real tables: uses session-scoped TEMP objects that disappear when psql exits.
--
-- Usage (Windows cmd, from D:\crypto-sentiment-analytics, one line):
--   docker exec -i crypto-timescaledb psql -U crypto_user -d crypto_analytics -v ON_ERROR_STOP=1 -v ws="2026-11-11 00:00+00" -v we="2026-11-15 00:00+00" -v all_models=1 < tools\paper_trade_eval.sql > reports\paper_trade_eval.txt
--
-- Variables (defaults in brackets):
--   ws, we           window on signals.valid_from (decision time), UTC, [ws, we)   (required)
--   cost_bps   [10]  cost per side, bps
--   all_models [0]   0 = frozen models only; 1 = also pre-freeze retraining models (reported as phase 'retraining')
--   hold_k     [0.5] HOLD verdict in the detail list: RIGHT if |next-bar move| <= hold_k * sigma
--   include_suspect [0]  1 = also score rows written AFTER their outcome was known (not genuine live)
--
-- Scoring : BUY  RIGHT if next-bar close > close at decision; SELL RIGHT if lower.
--           HOLD RIGHT if the coin did NOT move much over the next bar: |move| <= threshold.
--           Thresholds compared: 2 x cost (round trip), 0.5 sigma, 1.0 sigma, where
--           sigma = std dev of the previous 24 bar returns of that coin at that frequency.
--           Baseline for HOLD = "always HOLD" accuracy = share of ALL bars that were quiet.
-- P&L     : long-only. BUY -> long, SELL -> flat, HOLD -> keep position. Start flat. Cost per side.
-- Timing  : signals.bucket = bar open, valid_from = bar close; market_data.time = 1m candle OPEN,
--           so the price at instant T = close of the candle with time = T - 1 minute.
-- Suspect : created_at > valid_from + horizon (row written/rewritten after the outcome existed,
--           e.g. by maintenance.repair). Excluded from scoring unless include_suspect=1.

\if :{?cost_bps}
\else
  \set cost_bps 10
\endif
\if :{?all_models}
\else
  \set all_models 0
\endif
\if :{?hold_k}
\else
  \set hold_k 0.5
\endif
\if :{?include_suspect}
\else
  \set include_suspect 0
\endif

\echo '== 0. Window and settings =='
SELECT :'ws'::timestamptz                              AS window_start_utc,
       :'we'::timestamptz                              AS window_end_utc,
       (:'ws'::timestamptz AT TIME ZONE 'Asia/Kolkata') AS window_start_ist,
       (:'we'::timestamptz AT TIME ZONE 'Asia/Kolkata') AS window_end_ist,
       :'cost_bps'::float8                             AS cost_bps_per_side,
       :all_models                                     AS all_models,
       :'hold_k'::float8                               AS hold_k_sigma,
       :include_suspect                                AS include_suspect,
       now()                                           AS run_at;

CREATE TEMP TABLE pt AS
WITH s AS (
  SELECT sg.*,
         CASE WHEN sg.model_version LIKE 'frozen-%' THEN 'frozen' ELSE 'retraining' END AS phase,
         CASE sg.freq WHEN '1h' THEN interval '1 hour' WHEN '4h' THEN interval '4 hours' END AS step
  FROM signals sg
  WHERE (:all_models = 1 OR sg.model_version LIKE 'frozen-%')
    AND sg.valid_from >= :'ws'::timestamptz
    AND sg.valid_from <  :'we'::timestamptz
)
SELECT s.phase, s.symbol, s.freq, s.valid_from, s.action, s.prob_up, s.model_version, s.created_at,
       s.close                                     AS sig_close,
       e.close                                     AS entry_mkt,
       x.close                                     AS exit_close,
       coalesce(e.close, s.close)                  AS entry_close,
       x.close / coalesce(e.close, s.close) - 1    AS ret,
       v.sigma,
       (s.created_at > s.valid_from + s.horizon_bars * s.step) AS suspect
FROM s
LEFT JOIN market_data e
       ON e.symbol = s.symbol AND e.time = s.valid_from - interval '1 minute'
LEFT JOIN market_data x
       ON x.symbol = s.symbol AND x.time = s.valid_from + s.horizon_bars * s.step - interval '1 minute'
LEFT JOIN LATERAL (
  SELECT stddev_samp(ln(a.close / b.close)) AS sigma
  FROM generate_series(1, 24) k
  JOIN market_data a ON a.symbol = s.symbol AND a.time = s.valid_from - (k - 1) * s.step - interval '1 minute'
  JOIN market_data b ON b.symbol = s.symbol AND b.time = s.valid_from - k * s.step - interval '1 minute'
) v ON true;

-- rows that get scored
CREATE TEMP VIEW sc AS
SELECT * FROM pt
WHERE exit_close IS NOT NULL
  AND (:include_suspect = 1 OR NOT suspect);

\echo
\echo '== 1. Data check (suspect = written after its outcome was known; excluded from scoring by default) =='
SELECT phase, freq,
       count(*)                                                   AS signals,
       count(*) FILTER (WHERE exit_close IS NULL)                 AS outcome_not_yet_known,
       count(*) FILTER (WHERE suspect)                            AS suspect_rows,
       count(*) FILTER (WHERE entry_mkt IS NULL)                  AS entry_candle_missing,
       count(*) FILTER (WHERE abs(entry_mkt - sig_close) > 1e-9)  AS entry_price_mismatch,
       count(*) FILTER (WHERE sigma IS NULL)                      AS sigma_missing,
       count(DISTINCT model_version)                              AS n_model_versions,
       min(valid_from) AT TIME ZONE 'Asia/Kolkata'                AS first_decision_ist,
       max(valid_from) AT TIME ZONE 'Asia/Kolkata'                AS last_decision_ist
FROM pt
GROUP BY phase, freq
ORDER BY phase DESC, freq;

\echo '== 1b. Missing signals (expected decision times from first signal up to now-10min) =='
WITH lim AS (
  SELECT freq, min(valid_from) AS t0 FROM pt GROUP BY freq
), hrs AS (
  SELECT t
  FROM generate_series(date_trunc('hour', (SELECT min(t0) FROM lim) + interval '59 minutes 59 seconds'),
                       least(:'we'::timestamptz, now() - interval '10 minutes') - interval '1 second',
                       interval '1 hour') t
), expct AS (
  SELECT l.freq, h.t, c.symbol
  FROM hrs h
  JOIN lim l ON h.t >= l.t0
  CROSS JOIN (VALUES ('SOLUSDT'), ('LTCUSDT'), ('LINKUSDT')) c(symbol)
  WHERE l.freq = '1h' OR extract(hour FROM h.t AT TIME ZONE 'UTC')::int % 4 = 0
)
SELECT e.freq,
       count(*)                   AS expected,
       count(p.symbol)            AS present,
       count(*) - count(p.symbol) AS missing
FROM expct e
LEFT JOIN pt p ON p.freq = e.freq AND p.symbol = e.symbol AND p.valid_from = e.t
GROUP BY e.freq
ORDER BY e.freq;

\echo '== 1c. Gaps: decision times with no signal at all (1h), merged into periods (IST) =='
WITH lim AS (
  SELECT min(valid_from) AS t0 FROM pt WHERE freq = '1h'
), hrs AS (
  SELECT t
  FROM generate_series((SELECT t0 FROM lim),
                       least(:'we'::timestamptz, now() - interval '10 minutes'),
                       interval '1 hour') t
), miss AS (
  SELECT t, t - (row_number() OVER (ORDER BY t)) * interval '1 hour' AS grp
  FROM hrs
  WHERE NOT EXISTS (SELECT 1 FROM pt WHERE pt.freq = '1h' AND pt.valid_from = hrs.t)
)
SELECT min(t) AT TIME ZONE 'Asia/Kolkata' AS gap_from_ist,
       max(t) AT TIME ZONE 'Asia/Kolkata' AS gap_to_ist,
       count(*)                            AS missing_hours
FROM miss
GROUP BY grp
ORDER BY 1;

\echo '== 2. Directional hit rate: BUY/SELL only (Wilson 95% CI) =='
WITH a AS (
  SELECT phase, freq, coalesce(symbol, 'ALL') AS symbol,
         count(*) FILTER (WHERE action = 'BUY')                      AS buy,
         count(*) FILTER (WHERE action = 'SELL')                     AS sell,
         count(*) FILTER (WHERE action = 'HOLD')                     AS hold,
         count(*) FILTER (WHERE action <> 'HOLD')                    AS calls,
         count(*) FILTER (WHERE (action = 'BUY'  AND ret > 0)
                             OR (action = 'SELL' AND ret < 0))       AS correct,
         round(100.0 * avg((ret > 0)::int), 1)                       AS pct_bars_up
  FROM sc
  GROUP BY phase, freq, ROLLUP (symbol)
), b AS (
  SELECT *, nullif(calls, 0)::float8 AS n, correct::float8 / nullif(calls, 0) AS p FROM a
)
SELECT phase, freq, symbol, buy, sell, hold, calls, correct,
       round((100 * p)::numeric, 1) AS hit_rate_pct,
       round((100 * (p + 1.9208 / n - 1.96 * sqrt(p * (1 - p) / n + 0.9604 / (n * n)))
              / (1 + 3.8416 / n))::numeric, 1) AS ci95_low_pct,
       round((100 * (p + 1.9208 / n + 1.96 * sqrt(p * (1 - p) / n + 0.9604 / (n * n)))
              / (1 + 3.8416 / n))::numeric, 1) AS ci95_high_pct,
       pct_bars_up
FROM b
ORDER BY phase DESC, freq, (symbol = 'ALL'), symbol;

\echo '== 3. HOLD scoring: HOLD is RIGHT if the coin stayed quiet over the next bar =='
\echo '   hold_hit_pct              = share of HOLDs followed by a quiet bar'
\echo '   quiet_pct_after_buy_sell  = same share after BUY/SELL bars (HOLD should be higher if the model knows when to sit out)'
\echo '   always_hold_acc_pct       = baseline: share of ALL bars that were quiet'
\echo '   overall_acc_pct           = (BUY/SELL direction right + HOLD quiet) / all scored signals'
WITH th(rule, ord) AS (
  VALUES ('2 x cost', 1), ('0.5 sigma', 2), ('1.0 sigma', 3)
), q AS (
  SELECT sc.phase, sc.freq, th.rule, th.ord, sc.action,
         abs(sc.ret) <= CASE th.rule
                          WHEN '2 x cost'  THEN 2 * :'cost_bps'::float8 / 10000
                          WHEN '0.5 sigma' THEN 0.5 * sc.sigma
                          ELSE 1.0 * sc.sigma
                        END AS quiet,
         CASE WHEN sc.action = 'BUY'  THEN sc.ret > 0
              WHEN sc.action = 'SELL' THEN sc.ret < 0 END AS dir_ok
  FROM sc CROSS JOIN th
)
SELECT phase, freq, rule,
       count(*) FILTER (WHERE action = 'HOLD')                                   AS holds,
       count(*) FILTER (WHERE action = 'HOLD' AND quiet)                         AS holds_right,
       round(100.0 * avg(quiet::int) FILTER (WHERE action = 'HOLD'), 1)          AS hold_hit_pct,
       round(100.0 * avg(quiet::int) FILTER (WHERE action <> 'HOLD'), 1)         AS quiet_pct_after_buy_sell,
       round(100.0 * avg(quiet::int), 1)                                         AS always_hold_acc_pct,
       round(100.0 * avg((CASE WHEN action = 'HOLD' THEN quiet ELSE dir_ok END)::int), 1) AS overall_acc_pct,
       count(*)                                                                  AS scored
FROM q
GROUP BY phase, freq, rule, ord
ORDER BY phase DESC, freq, ord;

\echo '== 3b. Typical bar size (sigma, bps) and thresholds =='
SELECT phase, freq,
       round((10000 * avg(sigma))::numeric, 1)          AS avg_sigma_bps,
       round((10000 * 0.5 * avg(sigma))::numeric, 1)    AS half_sigma_bps,
       2 * :'cost_bps'::float8                          AS two_x_cost_bps,
       round((10000 * avg(abs(ret)))::numeric, 1)       AS avg_abs_move_bps
FROM sc
GROUP BY phase, freq
ORDER BY phase DESC, freq;

\echo '== 4. Paper P&L per phase (long-only: BUY=long, SELL=flat, HOLD=keep; start flat) =='
WITH r AS (
  SELECT phase, symbol, freq, valid_from, action, ret,
         count(*) FILTER (WHERE action <> 'HOLD')
           OVER (PARTITION BY phase, symbol, freq ORDER BY valid_from) AS grp
  FROM sc
), p AS (
  SELECT *,
         coalesce(first_value(CASE action WHEN 'BUY' THEN 1 WHEN 'SELL' THEN 0 END)
                    OVER (PARTITION BY phase, symbol, freq, grp ORDER BY valid_from), 0) AS pos
  FROM r
), t AS (
  SELECT *,
         abs(pos - coalesce(lag(pos) OVER (PARTITION BY phase, symbol, freq ORDER BY valid_from), 0)) AS traded
  FROM p
), s AS (
  SELECT *, pos * ret - traded * :'cost_bps'::float8 / 10000 AS strat_ret FROM t
)
SELECT phase, freq, symbol,
       count(*)                                                      AS bars,
       sum(traded)                                                   AS position_changes,
       round(100.0 * avg(pos), 1)                                    AS pct_time_long,
       round((100 * (exp(sum(ln(1 + strat_ret))) - 1))::numeric, 3)  AS strategy_ret_pct,
       round((100 * (exp(sum(ln(1 + ret))) - 1))::numeric, 3)        AS buy_hold_ret_pct
FROM s
GROUP BY phase, freq, symbol
ORDER BY phase DESC, freq, symbol;

\echo '== 5. Model diagnostics: prob_up spread; decision times where coins got the SAME prob_up =='
SELECT phase, freq,
       count(*)                          AS signals,
       round(min(prob_up)::numeric, 4)   AS p_min,
       round(avg(prob_up)::numeric, 4)   AS p_avg,
       round(max(prob_up)::numeric, 4)   AS p_max,
       count(DISTINCT prob_up)           AS distinct_probs
FROM pt
GROUP BY phase, freq
ORDER BY phase DESC, freq;

SELECT phase, freq,
       count(*)                                     AS decision_times,
       count(*) FILTER (WHERE n_distinct = 1)       AS all_coins_identical_prob,
       count(*) FILTER (WHERE n_distinct < n_coins) AS at_least_two_coins_identical
FROM (SELECT phase, freq, valid_from, count(*) AS n_coins, count(DISTINCT prob_up) AS n_distinct
      FROM pt GROUP BY phase, freq, valid_from) b
GROUP BY phase, freq
ORDER BY phase DESC, freq;

\echo '== 6. Signal-by-signal detail (IST). HOLD verdict uses |move| <= hold_k * sigma =='
SELECT to_char(valid_from AT TIME ZONE 'Asia/Kolkata', 'DD Mon HH24:MI')  AS decision_ist,
       phase, freq, symbol, action,
       round(prob_up::numeric, 4)                                          AS prob_up,
       entry_close                                                         AS entry,
       exit_close                                                          AS exit,
       round((ret * 10000)::numeric, 1)                                    AS move_bps,
       round((:'hold_k'::float8 * sigma * 10000)::numeric, 1)              AS hold_thr_bps,
       CASE WHEN exit_close IS NULL THEN 'pending'
            WHEN suspect AND :include_suspect = 0 THEN 'SUSPECT (not scored)'
            WHEN action = 'HOLD' AND sigma IS NULL THEN 'no sigma'
            WHEN action = 'HOLD' AND abs(ret) <= :'hold_k'::float8 * sigma THEN 'RIGHT (quiet)'
            WHEN action = 'HOLD' THEN 'WRONG (moved)'
            WHEN (action = 'BUY' AND ret > 0) OR (action = 'SELL' AND ret < 0) THEN 'RIGHT'
            ELSE 'WRONG'
       END                                                                 AS verdict
FROM pt
ORDER BY freq, valid_from, symbol;
