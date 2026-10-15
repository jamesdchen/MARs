# v3 track E: US midterm election recording (DECISIONS.md D11), worker w3

No vendor sells historical L2 books for election markets, so we record them ourselves from 2026-10-15 to Nov 3.
Hourly reward-config snapshots go alongside them. These recordings are the only copy of this data.

## Pieces

| file | role |
|---|---|
| `select_markets.py` | Ranks the open US midterm markets by CLOB `/rewards/markets/current` `total_daily_rate` (all pages), joined to gamma's election tags. Writes `sel/selection.json` (the selection plus every rewarded midterm market) and `sel/polyrec.toml` (exact-slug sources, `window = end_date`). |
| `sel/` | The selection used: **top 300 by rate (rate >= $20/day) = $26.4k/day of the $36.6k/day midterm pool** (2,892 rewarded midterm markets out of 18,851 rewarded markets, taken 2026-10-15 20:22 UTC). `markets_per_conn = 10`, reconcile every 120 s. |
| `supervise.sh` | Runs `recorder/` polyrec on that config and restarts it 5 s after any exit (the systemd `Restart=always` equivalent). Each run gets a timestamped run id. |
| `rewards_logger.py` | Snapshots the reward programs of all 2,892 midterm markets every hour to `rec/rewards/<day>/rewards_<day>T<HH>.parquet`, including `in_program` (whether the market is still in a program), the rates, min size, max spread and the full `rewards_config` JSON. |
| `autopush.sh` | At :07 every hour, commits and pushes the compacted `rec/` parquet (books, reconcile, market_qc, coverage, rewards), `sel/`, `rec/status_last.json` and the scripts. The raw websocket log (`rec/raw/`, about 0.5 GB/day) stays local and is not pushed. |
| `build_elections.py`, `BUILD.md` | Turns the recordings plus fills into env binaries (`data/build` schema) with reward-eligibility features. See BUILD.md for the schema and for the v3 env changes needed (episode length, unresolved settlement, per-market reward parameters). |

`rec/` is git-ignored as a whole; autopush force-adds only the compacted files.

## Recording log (UTC)

- 2026-10-15 20:21:13: first start (run `w3e1`). 276 of 300 markets subscribed.
- 20:23 and 20:24: two restarts while moving the recorder under `supervise.sh`. The second restart reused the run id
  `w3e1`, so its part `_00001` **overwrote the first run's books part for about 20:21:13–20:22:14**. That
  minute of opening books is lost.
- **20:24:09 onwards: continuous** (run id `w3e1015202409`).
- 20:46:00: restart (gap of about 5–10 s) to set `pre_open_s` very large. The recorder subscribes at
  `eventStartTime - pre_open_s`, and 24 selected markets have an event start date in the future (for example
  election day), so they had been left pending. That group included 2 of the top 3 by rate:
  `2026-balance-of-power-d-senate-d-house-949` and `...-r-senate-d-house-444`.
  **Those 24 markets are recorded from 20:46 on.** All 300 have been live since.
- `rec/coverage/` records every gap, and builds apply `merge.py`'s coverage rule.

## Operations

```bash
cat rec/supervise.log; python -m polyrec status --out rec          # health (run from recorder/ with the venv)
python select_markets.py --top 300 --min-rate 20 --markets-per-conn 10   # re-select (changes the recorded set!)
```

The selection is fixed so the recorded set stays stable. New reward programs that start later still appear in
the hourly reward snapshots of the 2,892 watched markets, but they are not recorded unless the selection is redone.
