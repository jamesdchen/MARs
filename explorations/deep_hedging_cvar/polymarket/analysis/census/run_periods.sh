#!/bin/bash
# Out-of-sample replication (Sep 17 - Oct 7) and pooled (Sep 17 - Oct 14) census runs. w4 = Oct 8-14 (out/).
set -e
cd "$(dirname "$0")"
PY=../../../.venv/bin/python
F=../../onchain/data/all/fills
for w in "w1 2026-09-17 2026-09-23" "w2 2026-09-24 2026-09-30" "w3 2026-10-01 2026-10-07" "w4 2026-10-08 2026-10-14"; do
  set -- $w
  [ -f cache/legs_$1.parquet ] && [ "$REBUILD" != 1 ] || nice -n 5 $PY build_legs.py --fills $F --start $2 --end $3 --tag $1 --threads 4
done
run() {  # tags out days pay
  export CENSUS_TAGS=$1 CENSUS_OUT=$2 CENSUS_DAYS=$3 CENSUS_PAY=$4
  mkdir -p $2
  nice -n 5 $PY census.py > $2/census.log 2>&1
  nice -n 5 $PY top_makers.py > $2/top_makers.log 2>&1
  nice -n 5 $PY extras.py > $2/extras.log 2>&1
  nice -n 5 $PY daily.py > $2/daily.log 2>&1
}
# REWARD payouts for epochs Sep 17..Oct 7 are paid Sep 18 00:00 .. Oct 8 ~06:00 UTC
run w4 out 7 1791504000,1792044000
run w1,w2,w3 out_oos 21 1789689600,1791439200
run w1,w2,w3,w4 out_pooled 28 1789689600,1792044000
