#!/bin/bash
# Restart polyrec on any exit (systemd Restart=always equivalent), 5 s backoff; also restarts the reward logger.
PM=/home/user/MARs/explorations/deep_hedging_cvar/polymarket
source $PM/../.venv/bin/activate
cd $PM/recorder
n=0
while true; do
  n=$((n+1))
  echo "$(date -u +%FT%TZ) start polyrec run w3e$n" >> $PM/analysis/v3_elections/rec/supervise.log
  python -m polyrec record --config ../analysis/v3_elections/sel/polyrec.toml --out ../analysis/v3_elections/rec --run-id w3e$(date -u +%m%d%H%M%S) >> ../analysis/v3_elections/rec/recorder.log 2>&1
  echo "$(date -u +%FT%TZ) polyrec exited rc=$?" >> $PM/analysis/v3_elections/rec/supervise.log
  sleep 5
done
