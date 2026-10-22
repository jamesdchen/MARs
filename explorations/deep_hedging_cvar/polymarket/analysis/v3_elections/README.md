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
- 21:00:06: polyrec hit its 1.5 GB RSS limit (exit 76) after 14 min. The supervisor restarted it in 5 s.
  The per-token reconcile rings (4,096 events × 600 tokens, including full snapshots of deep election books) and
  growth of roughly 10–30 MB/min were the cause. The config was changed to `history_events` 1024, then 256, with
  `max_rss_mb` 6000 (the box has 16 GB). That cost restarts at 21:00:30 and 21:07:24 (about 5 s each).
- Since 21:07:29 RSS is about 590 MB after start-up, then grows about 7 MB/min, so the 6 GB cap is reached after
  about 13 h. That means at most about two 5 s restarts a day, which fits merge.py's coverage rule (max gap 5 s,
  total 15 s per window). Oct 15 is a partial, gappy day. **Oct 16 is the first full recorded day.**
- **2026-10-16 02:03:16 to about 02:04:05: the container restarted** (platform-side), so the recorder had a gap
  of about 50 s. The disk survived. supervise.sh, rewards_logger.py and autopush.sh were restarted by hand at
  02:04, and all 300 markets were live again by 02:05. Under merge.py's rule (total gap 15 s or less) this gap
  breaks a full Oct 16 day window. Windows that don't contain 02:03–02:05 (for example 4 h windows) are
  unaffected. A container restart can't be prevented from inside the session; only the hand restart time can
  be kept short.
- **2026-10-16 05:37:00 to 2026-10-21 19:25:59 UTC: down for about 5.6 days.** The session hit the weekly usage
  limit at about 05:32, and the container was reclaimed. The last hourly push was 05:07Z; the books for 05:07–05:37
  were compacted from leftover parts on restart and pushed afterwards. Under D16 the days Oct 16–21 are ineligible
  (total gap > 30 min/day), so they default to train-only. Oct 22 is the next possible full day.
- **2026-10-21 19:24: market list refreshed** with `select_markets.py --top 300 --min-rate 20` (same rule). The
  midterm reward pool is now $32.0k/day across 2,801 markets, and the selected 300 carry $22.7k/day. 207 markets
  overlap with the Oct 15 selection; 93 are new and 93 were dropped. The previous selection and config are kept as
  `sel/selection_20261015.json` / `sel/polyrec_20261015.toml`. The reconcile/health overrides (ring 256, RSS cap
  6 GB, reconcile 120 s) were re-applied to the new config.
- 2026-10-21 23:05: 21 of the refreshed selection were short-lived "Will Trump say X during his midterm rally in
  South Carolina" mention markets ($1.2k/day in rewards together). They carry midterm tags and ended at 23:05.
  polyrec stopped recording them at their end (`awaiting_finalize`), as designed, and 278 markets are recorded
  from then on. A future refresh should exclude mention markets (slug `will-trump-say-*`), or top up the selection
  when markets end.
- `rec/coverage/` records every gap, and builds apply `merge.py`'s coverage rule.

## Operations

```bash
cat rec/supervise.log; python -m polyrec status --out rec          # health (run from recorder/ with the venv)
python select_markets.py --top 300 --min-rate 20 --markets-per-conn 10   # re-select (changes the recorded set!)
```

The selection is fixed so the recorded set stays stable. New reward programs that start later still appear in
the hourly reward snapshots of the 2,892 watched markets, but they are not recorded unless the selection is redone.
