# Track S backfill: PolyOrderBooks sports L2 → raw recorder layout

`backfill_pob.py` downloads historical pre-game sports L2 order books from PolyOrderBooks. It also collects
the fills (data-api, no key needed) and gamma metadata for the same markets. Everything is written in the
recorder's normalized raw layout, so `build_sports.py --rec raw_pob` reads it the same way it reads
`rec_out`. The script is ready to run as soon as `POLYORDERBOOKS_API_KEY` is set.

```bash
source ../../../.venv/bin/activate              # numpy, pandas, pyarrow, requests, aiohttp+orjson (polyrec imports)
python backfill_pob.py --self-test                                     # offline: mocked API -> parquet -> schema check
python backfill_pob.py --start 2026-09-16 --end 2026-10-14 --dry-run   # selection + request/size estimate (no key)
python backfill_pob.py --start 2026-09-16 --end 2026-10-14 --stage meta  # trades + markets now (no key)
export POLYORDERBOOKS_API_KEY=pob_...                                  # env only; never stored or logged
python backfill_pob.py --start 2026-09-16 --end 2026-10-14 --probe     # plan, coverage sample, density, mirror check
python backfill_pob.py --start 2026-09-16 --end 2026-10-14 --mirror --min-volume 50000   # the backfill (resumable)
python backfill_pob.py --verify                                        # outputs vs data/raw schemas + book invariants
python build_sports.py --rec rec_out --rec raw_pob --out build
```

Every command can be re-run. Finished work is recorded in `raw_pob/_state/` and skipped on the next run.
Ctrl-C or a crash loses at most the chunks that were in flight.

## Outputs (`raw_pob/`, git-ignored)

| Path | Content | Schema |
| --- | --- | --- |
| `books/<day>/books_<day>T<HH>.parquet` | One row per (token, bucket) in which the book changed, rotated hourly by `captured_at` | `data/raw/updown_*.parquet`, column for column (no `reconciled`; add it with `--with-reconciled`) |
| `trades/<day>/trades_<day>T<HH>.parquet` | Every fill leg (taker + makers) in the book window. Hour = kickoff hour | `data/raw/trades.parquet` |
| `markets/<day>/markets_<day>T<HH>.parquet` | One row per market. Hour = kickoff hour | `data/raw/markets.parquet` + `kickoff, league, sports_market_type, event_slug, outcome_0, outcome_1, volume, window_start, window_end, trades_consistent` |
| `_state/books/<slug>/<chunk_start>_<res>.json` | Chunk done marker (`ok` / `empty` / `forbidden`) with rows, points and pages. `MISSING.json` = the API returned 404 | |
| `_state/meta/<slug>.json` | Trades/markets done marker (fill counts, consistency checks) | |
| `_gamma/` | Cached gamma selection windows (only windows more than 2 days old) | |
| `probe_<ts>.json`, `backfill.log` | Probe report and run log. Neither contains the key | |

The files are written by the recorder's own `PartSink`: part files go to `<name>/<day>/_parts/` and are
compacted per hour at the end of a run, or with `--compact`. `polyrec.sinks.read_dataset(raw_pob, "books")`
and plain `pd.read_parquet(file)` both work.

Column conventions (sports versions of the data/raw ones):
- `coin` = league (the slug prefix, e.g. `nfl`, `cs2`) and `contract_length` = gamma `sportsMarketType`.
  The recorder's sports rows leave both null.
- `outcome` = the gamma outcome label, e.g. `Ravens`.
- `winner` = `'yes'` when outcome 0 won and `'no'` when outcome 1 won. It is null if the market is
  unresolved or resolved 50-50 (recorder convention). The markets columns `token_up` / `token_down` are outcome 0 / 1.
- `market_end_at` = gamma `endDate`, as in the recorder. For most leagues this is the kickoff, but for
  atp/wta/mlb it is kickoff + 7 d and for mlbb kickoff + 6 h. **Use `markets.kickoff`** (gamma
  `gameStartTime`) for the kickoff.

## Selection

Gamma `GET /markets?tag_id=100639&sports_market_types=<t>&closed=true&end_date_min/max` is queried in 6-hour
endDate windows over [start − 1 d, end + 8 d]. The +8 d is needed because tennis and MLB `endDate` is
kickoff + 7 d. Markets are then filtered on kickoff (`gameStartTime`, falling back to the event `startTime`,
then `endDate`) ∈ [--start, --end + 1 d). Further filters:
- the league (slug prefix) must be in `--leagues`;
- the market must have two tokens;
- the window must already be over.

Default `--leagues`: nfl, cs2, atp, lol, ucl, epl, lal, uel, bun, sea, cfb, wta, mlb, nhl, nba, dota2, val.
This order is the priority (census volume). Default `--market-types moneyline`. The order is also a priority,
e.g. `moneyline,child_moneyline,totals`. Soccer moneylines are three Yes/No markets per game (home, draw, away).

Fetch order `--order day` (default): oldest kickoff day first, then market type, then league priority. The
plan's history window rolls, so the oldest days expire first. `priority` fetches league by league instead.

Books window per market: [kickoff − `--hours-before` (24 h), kickoff + `--post-min` (0)). The last
`--fine-hours` (6) of the window are fetched at `--resolution` (1s) and the earlier part at `--coarse-resolution` (1m).

Selection for 2026-09-16 .. 10-14 (dry run, 2026-10-15):
- 6,035 moneyline markets: atp 1,848, cs2 1,383, wta 884, cfb 477, lol 297, dota2 249, mlb 201, nhl 168,
  lal 72, val 64, nfl 63, epl 60, sea 60, ucl/uel/bun 54, nba 47.
- 150–330 markets per day.
- Median gamma volume is $54k; 3,108 markets have volume ≥ $50k.

## PolyOrderBooks API (from `pob_openapi.json`, v0.1.0, saved 2026-10-15)

- **Base URL** `https://api.polyorderbooks.com`. All `/v1/*` routes need `X-API-Key: pob_…` (or
  `Authorization: Bearer pob_…`).
  - Without a key, the API returns 401 `{"error":"unauthorized","message":"Authentication required."}`.
  - With a bad key, it returns 401 `"Invalid API key."`.
  - Both were verified keyless. Every error uses the same envelope: `{error, message}`.
- **Plans:**
  - Starter: 60 req/min, 1,000/day, 3 days of history.
  - Pro: 300 req/min, 50,000/day, 30 days of history.
  - The finest resolution is 250ms on both. analysis/sports/out/sources_notes.md lists paid add-ons up to 120 days.
  - A 429 comes with `Retry-After`, according to the official client (`polyorderbooks` 0.2.1, `_http.py`).
  - A 403 means the plan does not allow the data point, granularity or date range.
- **Endpoints:**
  - Usage and discovery: `GET /v1/usage` (plan, limits, `max_history_days`, `granularity_allowed`,
    `data_points.books`), `/v1/tags`, `/v1/series`, `/v1/events`, `/v1/markets` (search, cursor, `category`,
    `include_closed`, sort by `updated_at|start_date|end_date`) and `/v1/markets/{id_or_slug}`.
  - Book history: `/v1/markets/{id_or_slug}/books` (all tokens) and `/v1/tokens/{token_id}/books` (one token).
  - Not used here: `/prices`, `/metrics`, and `POST /v1/ai/backtest`.
  - `category=crypto|sports` pins one archive; we always send `sports`.
- **Books format:** `start_ts` (inclusive), `end_ts` (exclusive), unix seconds or ISO.
  - `resolution` ∈ 250ms, 1s, 1m, 5m, 10m, 15m, 1h, 6h, 1d.
  - `limit` is at most 200 points per page; `metadata.next_cursor` gives the next page.
  - Each point is `{t: ISO bucket time, bids: [[price, size], …] best first, asks: [[price, size], …] best first}`.
  - **Aggregation: the last book in each bucket.** These are full-ladder snapshots, not deltas.
- We use the token endpoint, called with the gamma CLOB token ids. This needs no slug→id mapping, and pages
  are a clean 200 points per token. The market endpoint's `limit` appears to be shared across tokens: its 400
  example reads "Minimum required: 3".

How points become rows:
- `captured_at = t + resolution`. The point is the last book in [t, t+res), so it is the book as of the
  bucket's end. This is causal and matches the recorder's "row at t = book after all events before t". Use
  `--no-shift` to keep `t`.
- Ladders are re-sorted (bids descending, asks ascending), and zero-size levels are dropped.
- `best_bid`, `best_ask` and `crossed` are derived as in data/raw.
- Rows identical to the previous row of the same token are dropped (`--keep-unchanged` keeps them), so a
  dense feed ends up "on change" like the recorder's.
- **Seed:** for the window's first chunk, one extra request (`resolution=1h` over [start − 6 h, start)) takes
  the last book before the window and stamps it at the window start. With a sparse (on-change) feed the first
  row would otherwise come only at the first change. `--no-seed` skips it.
- **`--mirror`:** fetch only the outcome-0 token and derive outcome 1 as its mirror (bid p ↔ ask 1−p). This
  halves the requests. The two tokens share one CLOB book (recorder README, "Up / Down mirror"). `--probe`
  checks this on real data (`mirror_match_top3`).

Robustness:
- A shared rate limiter runs at 0.9 × the plan rpm from `/v1/usage` (`--rpm` overrides). `--workers` (2)
  threads share it.
- 5xx and network errors: exponential backoff, 6 tries; the chunk stays undone and the next run retries it.
- 429: sleeps for `Retry-After`. After about 40 consecutive 429s it stops cleanly, which most likely means
  the daily quota is used up.
- 401: stop.
- 403: the chunk is marked `forbidden` (`--retry-forbidden` retries it). After 20 forbidden chunks with
  nothing fetched, the run stops.
- 404: the market is marked missing (`--retry-missing` retries it).
- Chunks older than the plan's history window are skipped before any request is made.
- `--max-requests N` caps a run.
- Chunks are hour-aligned: 1 h at 1s, 3 h at 1m. Each chunk is written as its own part files and then marked
  done. Part names are deterministic, so a re-fetched chunk overwrites its own parts and never duplicates rows.

### Budget (dense upper bound: one point per bucket per token)

Range 2026-09-16 .. 10-14, all 6,035 markets:

| books config | requests | Pro days (50k/day) |
| --- | ---: | ---: |
| 1s for the last 24 h | 5.2 M | 105 |
| 1s for the last 6 h, 1m before (default) | 1.40 M | 28 |
| … + `--mirror` | 0.70 M | 14 |
| 1s for the last 2 h + `--mirror` | 0.27 M | 5.5 |
| 1m only (`--fine-hours 0`) + `--mirror` | 60 k | 1.2 |
| `--min-volume 50000 --mirror` (3,108 markets, 6 h at 1s) | 0.36 M | 7.2 |

A sparse feed (empty buckets omitted) needs fewer requests. The recorder sees about 0.5 book changes per second
per token on live sports books (analysis/sports/out/live_books.txt), and pre-game books are much quieter.
`--probe` measures this (`points_1s_last10min_page1`). On Starter (1k/day, 3 days of history) only a handful
of games per day fit. Use it to verify, not to backfill.

## Trades and markets (keyless)

Trades come from data-api `GET /trades?market=<conditionId>&takerOnly=true|false&start=<s>&end=<s>`, over the
book window only.
- The `start`/`end` filters are undocumented in data/README.md. They were verified here: `end` before market
  creation returns `[]`.
- With them, the 10,000-offset cap is no problem: a window holding more legs than one query can page
  through is split in halves, recursively.
- On nfl-bal-atl-2026-10-12 and cs2-iowast-dtn1-2026-10-11, windowed and split fetches give exactly the same
  leg multiset as the full-history fetch filtered to the window (464 / 1,116 legs for NFL, 23 / 69 for CS2).
- Legs are turned into rows by `polyrec.trades` (LegSet, `build_trade_rows`, `trade_rows_to_table`,
  `rows_consistency`), the same code the recorder uses.
- HTTP goes through `data/fetch_trades.http_json` (retries, browser UA). Both modules are imported, not edited.
- On-chain extraction (`onchain/extract.py --slugs …` then `to_trades.py`) remains the fallback for any
  market whose markets row has `truncated=True`. It is much heavier: it scans every exchange log per block.

The markets row comes from `polyrec.metadata.market_row`, with the sports columns listed above appended.

## Tested without the key (2026-10-15)

- `--self-test` (offline):
  - Mocked responses are shaped like the OpenAPI `BooksResponseSingleToken` / `UsageResponse` examples.
    They include cursor pagination, a sparse 1s feed, one-sided books, unsorted ladders and a 404 token.
  - They run through the real books stage (chunking, seed, shift, dedup, part files, compaction) and then
    `--verify`. The schema equals `data/raw/updown_5m.parquet`. Bid/ask order, `best_*`, `crossed` and
    `seconds_to_close` all hold.
  - A re-run fetches nothing (resume), a 404 is marked missing, `--mirror` output equals the two-token
    fetch, and `probe()` runs against the mocked usage, tags and markets.
- `--dry-run` against live gamma for 3 days and for 29 days (numbers above).
- `--stage meta` on 3 real markets (nfl, cs2, atp), then `--verify`: trades == data/raw/trades.parquet
  schema, and markets == data/raw/markets.parquet + the sports columns. `rows_consistency` ok on all three.
- A fake key gets a 401 that is handled; the key appears nowhere under the out dir.

## Open questions (need the key: run `--probe` first)

1. **Sports coverage and depth of history.** Which leagues and days exist, and whether gamma CLOB token ids
   resolve (`/v1/tokens/{id}/books`) for sports. The probe also checks `tokens_match` and whether `market_id`
   equals the gamma id.
2. **Sparse or dense buckets.** Are empty buckets omitted? The answer sets the real request count. Is `t` the
   bucket start? If `t` turns out to be the bucket end, run with `--no-shift`.
3. **Mirror.** Does PolyOrderBooks store each token's own book, and is token 1 exactly the mirror of token 0?
   If `mirror_match_top3` is ≈ 100%, use `--mirror`.
4. **Ladder depth.** Full depth or truncated? The probe reports `levels_med`.
5. **Plan.** `max_history_days` and the daily quota decide the feasible range. Pro has 30 days, so start
   within days of getting the key, because the oldest days roll off.
6. **`1s` vs `250ms`.** 250ms is allowed but costs 4× the requests. The env grid is ≥ 1 s, so 1s is enough.
