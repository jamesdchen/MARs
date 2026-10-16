#!/usr/bin/env bash
# Daily driver: wait for the Starter quota reset (midnight UTC), spend the day's budget, export.
# The key comes from the environment only (POLYORDERBOOKS_API_KEY); nothing here prints it.
set -u
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
BUDGET=${BUDGET:-980}
while true; do
  now=$(date -u +%s); next=$(( (now / 86400 + 1) * 86400 + 180 ))       # 00:03 UTC tomorrow
  if [ -n "${RUN_NOW:-}" ]; then next=$now; unset RUN_NOW; fi
  echo "$(date -u +%FT%TZ) next run at $(date -u -d @$next +%FT%TZ)"
  while [ "$(date -u +%s)" -lt "$next" ]; do sleep 60; done
  $PY backfill_starter.py --budget "$BUDGET" --fine-hours 1 >> raw_pob/daily_run.log 2>&1
  $PY export_pob_daily.py >> raw_pob/daily_run.log 2>&1
  date -u +%FT%TZ > raw_pob/LAST_DAILY_DONE
done
