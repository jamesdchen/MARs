# Track S: PolyOrderBooks backfill on the Starter plan (worker w9b)

This folder holds the historical pre-game sports L2 pulled from PolyOrderBooks with the user's key
(`POLYORDERBOOKS_API_KEY`, environment only). The key was never printed, logged or written to a file;
every log and output was grepped for it before each push.

## Plan limits (measured 2026-10-16 01:11 UTC)

| item | value |
|---|---|
| plan | **starter** |
| rate | 60 requests/min (we run at 45) |
| daily quota | **1,000 requests**, reset at **midnight UTC**. The 429 reads `"Daily request limit exceeded. Limit resets at midnight UTC."` with `Retry-After` set to the seconds until 00:00. `/v1/usage` shows only the per-minute counter. |
| history | **3 days** (`max_history_days=3`). Earlier data returns 403 `"Earliest allowed start_ts is <now − 72 h>"`. |
| granularity | 250ms allowed. 1s is used. |

**A 30-day backfill is not possible on this plan.** Only kickoffs inside the last ~72 h can be
fetched, and each day the window drops one day. Pro is 300/min, 50k/day and 30 days of history; with it,
`../v3_sports/README_backfill.md`'s full plan (about 0.5 M requests) would take ~10 days of quota.

## What PolyOrderBooks has (probe, `probes/`)

- **Leagues present:** ucl, uel, lol, cs2, dota2, val, cfb, nhl, nba (and presumably the other
  soccer leagues). **Absent:** atp, wta and mlb return 0 points for the whole window. They are skipped
  (`--skip-leagues`).
- **Book start:** esports and soccer books start ~5–13 h before kickoff. US sports reach back past
  the 3-day floor.
- **Density:** dense. Every 1s bucket is filled with the last book, so 1 h at 1s is 3,600 points, or
  18–19 pages of 200. Ladders are deep (median 45–70 levels, both sides).
- **Mirror:** token 1 is the exact mirror of token 0 (top-3 match 200/200 on 3 markets), so only
  outcome 0 is fetched.
- **Pagination quirk.** A page scans `limit` buckets from `start_ts`, and `next_cursor` is set only
  when the page is full. A range that starts before the book exists, or that crosses a gap, comes back
  short with no cursor, and the rest of the range is silently missing. For example, 1m over 02:00–12:00
  returned only 04:05–05:19, and 1s over 04:00–05:00 returned 0 points although the book starts at 04:05.
  `backfill_pob.py`'s `token_books` stops at such a page. `backfill_starter.py` patches it to resume
  after the scanned span, and tests it offline against a mock with the same semantics: dense, late
  start, gap and empty ranges all come back exact in ≤ 19 pages per hour.

## Method (`backfill_starter.py`, a wrapper that imports w1's `../v3_sports/backfill_pob.py` unedited)

- **Selection:** gamma moneylines with kickoff inside the history window, volume ≥ $50k, the
  default league list minus atp/wta/mlb.
- **Order:** oldest kickoff day first, because it expires first. Within a day, one market per game
  first (soccer has three Yes/No legs), highest volume first; the other legs come after.
- **Data-start detection:** 2 requests per market (1h scan, then 1m). The books window starts at the
  first book, or at the history floor, so no requests go to empty pre-listing time. A market with no
  data at all is marked MISSING after 1 request.
- **Books:** 1s for the **last 1 h** before kickoff and 1m from the data start up to that hour,
  ending at the scheduled `gameStartTime` (w1's cut rule). Mirror on, no seed. This costs ~25–30
  requests per market, so ~30–35 markets per day of quota.
  - The README's 6 h at 1s costs ~115 requests per market, which is ~8 markets per day.
  - With 1 h, the D15 buckets 1h–30m, 30–5m and 5–0m are all at 1s.
- **Budget:** each run has a request cap (`--budget`). A market is admitted only if its estimated cost
  fits, and the run stops at the first 429.
- **Trades and markets:** data-api and gamma, keyless, over the full 7-day gamma window.

## Outputs

- `raw_pob/` (git-ignored): the recorder layout, as in `backfill_pob.py`. It also holds
  `_state/start/<slug>.json` (data-start detection) and `runs.jsonl` (one line per run).
- `pob_daily/<kickoff day>/` (committed): the compact export written by `export_pob_daily.py`, in the
  same format as w1's `rec_daily/`, grouped by kickoff day:
  - books: outcome 0, top 10 levels, zstd 19;
  - trades, markets;
  - `coverage_<day>.jsonl`: book span, rows, chunks, start detection;
  - `summary.json`.

  No file may exceed 50 MB; the exporter refuses to write a larger one.
- `runs.jsonl`: a copy of the run log, with the markets and requests of each run.

## Validation (2026-10-16, day 2026-10-13, 6 games)

- `build_sports.py --rec pob_daily/2026-10-13 --seg 0 --min-cover 0.05` builds 6 segments for 6 games,
  N = 307,206 rows and 6,246 legs.
- `v3env/adapt3.py <build> --check` reports kind sports, 6 markets, x3 true, settle_mode [1], row_s
  [1.0], build_version sports-2.
- `build_sports.py --verify` fails on `leg volume != ticks volume`, but this is not a data problem:
  - The maximum absolute difference is 0.048 shares, on a row with 1.6 M shares (relative 3e-8).
    This is float32 rounding against the absolute tolerance of 1e-2. Only 2 of 307k rows are affected.
  - w1's own `rec_daily/2026-10-15` fails the same way: 0.031 on 730k, also 2 rows.
  - The tolerance in `build_sports.py` (w1's file) should be relative.

## Coverage so far

| kickoff day | markets | leagues | requests |
|---|---|---|---|
| 2026-10-13 | 6 complete (books from 01:55 UTC, the floor, to kickoff); 14,446 trade legs | cfb 1, lol 2, ucl 3 | ~610 |

Requests used on 2026-10-16: about 1,000 of 1,000.
- About 400 went on the probe and coverage checks.
- 500 + 95 + 13 went on runs 1–3. Run 1 lost ~300 to a pagination bug in the first version of the
  wrapper: an empty final cursor page restarted the scan. The bug was fixed and tested before run 2,
  and the rows from run 1 are correct (de-duplicated).

## Gaps

- **Older history (before now − 72 h) cannot be fetched on Starter.** That covers Sep 16 – Oct 12 and
  most of Oct 13. Worker w10's Hugging Face archives (Houroux Jul–Oct, Joseph3222 Feb–Aug) and w1's
  recorder (from 2026-10-15 20:22 UTC) are the other sources.
- **atp, wta and mlb** are not in PolyOrderBooks.
- **Only ~30–35 markets per day of quota,** out of 100–330 eligible per kickoff day.

## How to resume

```bash
cd explorations/deep_hedging_cvar && python3 -m venv .venv && .venv/bin/pip install numpy pandas pyarrow requests aiohttp orjson
cd polymarket/analysis/v3_sports_backfill
nohup ./run_daily.sh > raw_pob/run_daily.out 2>&1 &     # each day at 00:03 UTC: backfill (budget 980) + export
RUN_NOW=1 ./run_daily.sh                                 # or run once immediately, then daily
python backfill_starter.py --budget 0 --dry-run          # plan only
python export_pob_daily.py                               # raw_pob -> pob_daily
```

Then commit `pob_daily/` and `runs.jsonl`. A run is resumable: done chunks are skipped, and the
detected starts are cached. A fresh container starts with an empty `raw_pob/`, and that is fine,
because its days have rolled out of the window anyway. The exporter merges with the committed day
files (books by slug/token/time, trades and markets by slug, coverage by slug), so a fresh `raw_pob/`
never shrinks a day.
