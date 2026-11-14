set -e
R="psql -U crypto_user -d crypto_replay -v ON_ERROR_STOP=1"
L="psql -U crypto_user -d crypto_analytics -v ON_ERROR_STOP=1"
RANGE=$($R -At -c "SELECT min(time)||'|'||max(time) FROM market_data")
LO=${RANGE%%|*}; HI=${RANGE##*|}
echo "replay range: $LO -> $HI"
$R -c "DROP TABLE IF EXISTS live_copy; CREATE TABLE live_copy AS SELECT symbol,time,open,high,low,close,volume FROM market_data WITH NO DATA;"
$L -c "\copy (SELECT symbol,time,open,high,low,close,volume FROM market_data WHERE time BETWEEN '$LO' AND '$HI') TO STDOUT" | $R -c "\copy live_copy FROM STDIN"
$R <<'SQL'
SELECT 'replay market rows' AS k, count(*) FROM market_data
UNION ALL SELECT 'replay duplicates (symbol,time)', count(*) - count(DISTINCT (symbol,time)) FROM market_data
UNION ALL SELECT 'live rows in same range', count(*) FROM live_copy
UNION ALL SELECT 'in replay, NOT in live', count(*) FROM market_data r WHERE NOT EXISTS (SELECT 1 FROM live_copy l WHERE l.symbol=r.symbol AND l.time=r.time)
UNION ALL SELECT 'in live, NOT in replay', count(*) FROM live_copy l WHERE NOT EXISTS (SELECT 1 FROM market_data r WHERE r.symbol=l.symbol AND r.time=l.time)
UNION ALL SELECT 'price/volume mismatches', count(*) FROM market_data r JOIN live_copy l USING (symbol,time) WHERE (r.open,r.high,r.low,r.close,r.volume) IS DISTINCT FROM (l.open,l.high,l.low,l.close,l.volume)
UNION ALL SELECT 'replay news rows (from 1,628 msgs)', count(*) FROM raw_sentiment;
SQL