#!/usr/bin/env bash
# Live smoke test: record one complete batch of 5m up/down markets (all coins, plus the running 15m/4h
# ones), finalize + backfill them, verify against data/raw, and build a merged dataset with
# data/build.py + data/load.py verify() (features.py needs next-day Binance data, so it is skipped).
#   ./smoke_test.sh [WORKDIR]          (~15 min wall clock, ~20% of one core while recording)
set -euo pipefail
cd "$(dirname "$0")"
W=${1:-/tmp/polyrec_smoke}
PY=${PY:-python}
rm -rf "$W" && mkdir -p "$W"
# start 90 s before the next 5-minute open and stop 10 s after that market closes
DUR=$($PY -c "import time; n=time.time(); o=(int(n)//300+1)*300; o+=300*(o-n<90); print(int(o+310-n))")
echo "recording for ${DUR}s into $W/out"
$PY -m polyrec record --out "$W/out" --duration "$DUR" --pre-open 90 --settle-delay 900 2>&1 | tee "$W/record.log"
# gamma resolves 1-6 min after the close and data-api indexes maker legs a few minutes after the
# taker legs: finalize (resolved + consistent trades only) until nothing is pending, up to ~20 min
sleep 240
for i in $(seq 1 16); do
  R=$($PY -m polyrec finalize --out "$W/out" --require-resolved || true)
  echo "$R" | tr -d '\n '; echo
  echo "$R" | grep -q '"unresolved": 0' && echo "$R" | grep -q '"trades_pending": 0' && break
  sleep 60
done
$PY backfill_winners.py --out "$W/out"
$PY verify_output.py --out "$W/out" --json "$W/verify.json"
$PY merge.py --rec "$W/out" --dest "$W/merged" --lock "$W/split_lock.json" \
    --no-underlying-fetch --allow-missing-underlying --no-features
