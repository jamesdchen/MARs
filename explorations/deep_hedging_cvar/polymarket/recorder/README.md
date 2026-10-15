# polyrec: Polymarket order book recorder

Records Polymarket CLOB L2 order books, the full websocket delta stream, trades, and market metadata.
It runs unattended for days on a small VPS. Its output uses the same schemas as `../data/raw/`. The
main target is the crypto up/down markets (5m / 15m / 4h, all coins). `merge.py` appends new days to
the `../data` pipeline (`build.py`, `load.py`, `features.py`) without modifying those files.

```
gamma /events/keyset --(60 s)--> markets --(open - 2 min)--> websocket shards (2 markets each)
                                                                 | book / price_change / last_trade_price /
                                                                 | tick_size_change / best_bid_ask / market_resolved
REST GET /book (60 s per token) ---- reconcile ---->  in-memory books --(1 Hz, on change)--> books/*.parquet
                                                         |                                   (data/raw schema + reconciled)
                                       every event ------+--> raw/*.jsonl.zst (+ REST snapshots, connects)
data-api /trades (30 s, both takerOnly modes) --> end + 10 min: full sweep + gamma --> trades/, markets/, market_qc/
```

## Outputs (`out_dir`)

| Path | Content | Schema |
| --- | --- | --- |
| `books/<day>/books_<day>T<HH>.parquet` | One row per (token, second) in which the book changed. Rotated hourly by `captured_at`. | `data/raw/updown_*.parquet` column for column (`market_slug … crossed`), plus `reconciled` (bool) |
| `raw/<day>/events_<day>T<HH>_<run>.jsonl.zst` | Every websocket event (one JSON object per line), REST `/book` snapshots (`event_type: rest_book`), and connect / subscribe / disconnect markers (`_connect`, `_subscribe`, …) | see "Raw event log" |
| `trades/<day>/trades_<day>T<HH>.parquet` | Every fill leg of each finished market (taker leg + maker legs) | `data/raw/trades.parquet` |
| `markets/<day>/markets_<day>T<HH>.parquet` | One row per finished market (gamma metadata, resolution, trade counts) | `data/raw/markets.parquet` |
| `reconcile/<day>/reconcile_<day>T<HH>.parquet` | One row per REST reconcile check: status `ok` / `mismatch` / `seeded`, level differences, best prices, RTT | `polyrec/schema.py: RECONCILE_SCHEMA` |
| `market_qc/<day>/…` | Per-market recorder counters: rows, reconciles, mismatches, disconnects, first live time | `MARKET_QC_SCHEMA` |
| `coverage/coverage_<day>.jsonl` | Liveness transitions per market (`live` / `down`) plus a 10 s heartbeat. `merge.py` uses it to keep only fully covered markets. | |
| `status.json` | Health snapshot, rewritten atomically every 10 s | see "Monitoring" |

`winner` and `is_winning_outcome` are null when written. `backfill_winners.py` fills them once gamma
reports the market as resolved: `winner = 'yes'` when outcome 0 (Up) won, as in data/raw. In the
markets files, `winner_up`, `outcome_price_up` and `final_price` are filled at finalization, or later
by the backfill.

Parquet files are written as crash-safe **part files** every 60 s (temp file + fsync + rename), so a
hard crash loses at most one minute of rows. Each finished hour is **compacted** into one file. The
compacted file records which parts it absorbed in its metadata, so a crash during compaction never
duplicates rows. To read a dataset including the current hour's parts, use
`polyrec.sinks.read_dataset(out, "books")`. Plain `pd.read_parquet(file)` also works on every file;
it is the same reader `data/raw` uses.

### How a book row is made

* **Snapshots, 1 Hz, on change.** A row stamped with second `t` holds the book after every event
  received before `t`. A frame received at or after `t` first closes tick `t`, then gets applied.
  So rows are causal and `captured_at` is a whole second, as in data/raw. Tokens whose book did not
  change produce no row.
* **Deltas.** `price_change` carries the new aggregate size of a level (0 removes the level). A
  `book` event replaces the book. The exchange sends a `book` after every trade.
* **Stale-level pruning.** Every `price_change` item also carries the exchange's `best_bid` /
  `best_ask` after the change. In live data, every disagreement with our book was a level that a
  trade had just consumed. The exchange's follow-up `book` event arrives 50 ms to 1 s later, and in
  that window the book reads as crossed. polyrec drops our levels beyond the reported best right
  away (`bba_pruned_levels`). If instead we lack the best level, it triggers an early REST reconcile
  (`bba_missing`). Without this fix, roughly 1.5 to 3% of data/raw rows are crossed. With it, our
  crossed rate is about 0.
* **REST reconcile every 60 s per token.** The delta stream has no sequence numbers (see
  data/raw/README.md). Every book is re-read with `GET /book`, staggered so about one request per
  second goes out for 64 tokens. `/book`'s `timestamp` is the time of the book's last change. Deltas
  are absolute and therefore idempotent, so REST plus every stream delta with `ts >= rest ts` must
  equal our book. Those deltas come from a per-token ring of the last 4,096 stream events, compared
  after a 1.5 s grace. The REST snapshot can lag the matching engine, so a buffer started only when
  the request goes out would miss deltas and report false mismatches. A mismatch therefore means the
  stream-maintained book really differs (for example, a dropped delta). The book is replaced, the
  next row has `reconciled = True`, and the difference is logged in `reconcile/` (rate shown in
  `status.json`).
* **Up / Down mirror.** The two outcome tokens share one book: Up bid p is Down ask 1 − p. Each
  tick, the Up book is checked against the mirrored Down book (`mirror_mismatch_rate`).
* **Disconnects.** Books of a dropped connection are marked not live, so no rows are written until
  the resubscribe's initial `book` dump arrives. The `coverage/` log records the gap.

### Raw event log

The log is zstd-compressed JSONL: one websocket event per line, in the exchange's own format
(docs.polymarket.com, market channel), with three changes:

* `"r"` is the local receive time (ms) and `"c"` is the connection id.
* `market` (condition id) is dropped from `price_change` / `best_bid_ask`. It is recoverable from
  the asset id.
* The per-order `hash` inside `price_change` items is dropped by default
  (`output.raw_keep_order_hashes`). Those random hashes made up about 70% of the compressed size.

REST snapshots are logged as `{"event_type": "rest_book", ...}`. That means a replay can
re-reconcile, which is impossible from a bare delta stream.
`polyrec.sinks.iter_raw(path)` decodes a file. It tolerates a torn tail after a hard crash: data is
block-flushed every 10 s.

### Trades and metadata

While a market is recorded, data-api `/trades?market=<conditionId>` is polled every 30 s in both
`takerOnly` modes, newest first, until the previous high-water mark minus 3 min. Ten minutes after
the end comes a full sweep: the same paging and BUY/SELL split as `data/fetch_trades.py`. The legs
are merged as a multiset (max count per identical leg, never a sum), so overlapping polls do not
double count. Polling also covers markets whose history would exceed data-api's 20,000-row offset
cap. The final gamma record (`closed=true`, with a `/events/slug/` fallback) gives the
`markets.parquet` row.

Finalization retries every 2 min for up to 30 min until both of these hold:

* gamma reports the resolution (observed 1 to 6 min after the close);
* the legs pass the checks `features.py` asserts: every taker tx has maker legs that sum to its
  size. data-api indexes the `takerOnly=false` view a few minutes after the taker view.

`tick_size` is the finer of gamma's value and the last `tick_size_change` seen on the stream, which
matches data/raw's post-resolution tick.

## Quick start

```bash
cd explorations/deep_hedging_cvar/polymarket/recorder
source ../../.venv/bin/activate           # or: python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests/                # 29 tests, ~8 s
python -m polyrec record --out ./out                      # all crypto up/down markets, until SIGTERM/Ctrl-C
python -m polyrec record --out ./out --coins btc,eth --lengths 5m --duration 600   # a 10 min sample
cat out/status.json
```

Afterwards (cron / timers, see deploy):

```bash
python -m polyrec finalize --out ./out     # metadata + trades for markets the recorder stopped before finalizing
python backfill_winners.py --out ./out     # fill winner / is_winning_outcome once resolved
python verify_output.py --out ./out        # schema + data/raw README checks
```

## Rebuilding the env dataset with new days: `merge.py`

```bash
python merge.py --rec ./out --dest ./merged --dry-run     # what would be included, and why not
python merge.py --rec ./out [--rec ./out_vps2] --dest ./merged
#   -> merged/raw/{updown_5m,updown_15m,updown_4h,markets,trades,underlying}.parquet
#   -> merged/build/{ticks,markets,feat,book,fills,fill_index}.bin, meta.json, splits.json, feat_meta.json
```

* **Market selection.** A market is merged only if it has resolved metadata, its whole life
  `[open, end]` was live in the recorder (largest gap ≤ 5 s, total ≤ 15 s; a recorder restart
  therefore drops the markets running at that moment), the coin and length are known to `build.py`,
  its trade legs pass the checks `features.py` asserts, and Binance 1 s data exists for
  `[open − 1 h, end]`. Binance data comes from `data/fetch_underlying.py`'s own functions; its daily
  dumps appear the next day, so merge yesterday and earlier. Reasons for exclusion go to
  `merged/merge_selection.csv`.
* **Output.** `merged/raw` starts with the original `data/raw` rows, untouched, followed by the new
  markets in exactly the data/raw schemas (the `reconciled` column is dropped and winners are filled
  from metadata). Book files are streamed one hourly file at a time.
* **Splits** are by day, set in `split_config.json` and implemented in `polyrec/splits.py`:
  * The original days are `frozen` and must equal `build.py`'s `SPLIT_DAYS`: 08-20..22 train, 08-23
    val, 08-24 test.
  * Each new day gets an explicit `days` entry (`"2026-10-20": "val"`, ranges `"a..b"`, `exclude`) or
    the default `auto`: a fixed 8-day cycle anchored at 2026-10-15 (train ×6, val, test). Because 8
    is coprime with 7, val and test rotate through the weekdays.
  * A day's split depends only on its date and the config, so recording more days never moves an
    earlier day.
  * Every assignment used is pinned in `split_lock.json`. Changing a locked day requires
    `--reassign`, and a day that was ever test can never change. The original test day can never
    reach training.
* **Build.** `build.py` runs in a subprocess with its `RAW` / `OUT` / `SPLIT_DAYS` globals pointed
  at `merged/` (the files in `data/` are not modified). Then `load.verify()` runs, then `features.py`
  (it needs scipy: `--python /usr/bin/python3`). Finally the original test block (08-24) of the new
  `ticks.bin` is checked to be byte-identical to `data/build`'s.
* **Memory.** `build.py` / `features.py` keep everything in RAM. `ticks.bin` grows by about 2.1 M
  rows (330 MB) per recorded day, and `build.py` needs a few times that.

## Deploy

### VPS with systemd

```bash
sudo useradd --system --home /opt/polyrec polyrec
sudo mkdir -p /opt/polyrec /var/lib/polyrec /etc/polyrec && sudo chown polyrec: /var/lib/polyrec
sudo cp -r polyrec backfill_winners.py verify_output.py requirements.txt /opt/polyrec/
sudo cp config.example.toml /etc/polyrec/config.toml          # edit out_dir / sources if needed
sudo python3 -m venv /opt/polyrec/.venv && sudo /opt/polyrec/.venv/bin/pip install -r /opt/polyrec/requirements.txt
sudo cp deploy/polyrec.service deploy/polyrec-backfill.{service,timer} /etc/systemd/system/
sudo timedatectl set-ntp true                                 # captured_at uses the local clock
sudo systemctl daemon-reload && sudo systemctl enable --now polyrec polyrec-backfill.timer
journalctl -u polyrec -f                                      # one summary line per minute
```

The unit is `Type=notify` with `WatchdogSec=180`: polyrec sends `READY=1` and then `WATCHDOG=1`
every 10 s. It restarts on any exit (`Restart=always`, 5 s), stops gracefully on SIGTERM (90 s
budget: rows flushed, raw zstd frame closed), and is capped at `MemoryMax=2G` and `CPUQuota=150%`.

### Docker

```bash
docker build -t polyrec .
docker run -d --name polyrec --restart unless-stopped --stop-timeout 90 -v /var/lib/polyrec:/data polyrec
# or: docker compose -f deploy/docker-compose.yml up -d --build    (adds the 15 min backfill loop)
```

The image's `HEALTHCHECK` fails when `status.json` is older than 60 s or no market data has arrived
for 5 min.

### Robustness

| Failure | Handling |
| --- | --- |
| WS drop / 1013 "slow consumer" / TLS / DNS | Per-shard reconnect: exponential backoff 1 s → 60 s with jitter, reset after 60 s up (immediate after 1013), then a resubscribe of the full asset set. The initial `book` dump re-initializes the books. |
| Silent connection | A text `PING` every 10 s. No frame (not even `PONG`) for 30 s aborts the socket. |
| Event loop hang | A watchdog thread dumps stacks and exits (code 70) after 120 s; systemd `WatchdogSec` acts as a second watchdog. |
| No data at all for 10 min | Exit 75, then restart. |
| Memory | Bounded websocket queues and reconcile buffers, a columnar row buffer flushed every 60 s / 50k rows, pruned market registries. RSS > 1.5 GB → exit 76, then restart. |
| Disk | Below 2 GB free the raw log pauses (parquet keeps going). Optional `raw_retention_days`. |
| Gamma / REST / data-api errors | Retries with backoff. Discovery failures only delay subscriptions: markets are listed about 24 h ahead. |
| Restart | Leftover parts are compacted, recently ended markets recorded by the previous run are finalized, and running markets are resubscribed. Coverage gaps are logged, so merge.py skips the affected markets. |
| SIGTERM / SIGINT | Rows flushed, parquet parts written, raw frame closed, `status.json` marked `stopped`. |

### Monitoring: `status.json`

`status.json` reports:

* `ws`: shards, connected, connects / disconnects / `disconnects_unexpected`, `last_frame_age_s`
* `markets`: tracked / subscribed / live / awaiting_finalize / finalized
* `rates_per_s`: frames, bytes, price_change, book, rows, reconcile
* `quality`: `reconcile_checks`, `reconcile_mismatch_rate`, `bba_mismatch_rate`, `bba_pruned_levels`,
  `bba_missing`, `mirror_mismatch_rate`, `rows_crossed_rate`, `parse_errors`
* `process`: `rss_mb`, `cpu_pct_avg`, `loop_lag_ms`
* `disk`, `output`, `discovery`, `last_error`

Run `python -m polyrec status --out DIR` to print it.

## Resource use and disk (measured, see "Smoke test")

Measured on all crypto up/down markets: 8 coins × 5m / 15m / 4h, 24 to 48 markets live including
pre-open, 1 to 2 markets per connection (see the smoke test below).

| | Measured | Per day (estimate) |
| --- | --- | --- |
| Inbound websocket | 2.0 to 2.5k frames/s average, 4.2k/s peak at 5-min boundaries; 1.2 to 1.5 MB/s JSON | ~110 to 130 GB/day inbound traffic. Check the VPS bandwidth cap. |
| CPU | 17 to 20% of one core on average (including start-up); event-loop lag ≤ 5 ms | |
| Memory | RSS ~250 MB steady | |
| Book rows | ~40 rows/s, 48 B/row after hourly compaction (zstd 9) | ~3.5 M rows, **~170 MB** |
| Raw event log | 34 to 43 KB/s compressed (zstd 6, ~26× smaller than the JSON) | **~3 to 4 GB** (plan for 6 on busy days) |
| Trades | ~7 kB per finished market | ~3,100 markets, **~25 MB** |
| Reconcile log, markets, market_qc, coverage | | < 15 MB |
| **Total** | | **~3.5 to 6 GB/day** with the raw log (~110 to 180 GB/month); **~0.2 GB/day** (~6 GB/month) with `raw_enabled = false` |

A 2 vCPU / 2 GB VPS with 200 GB of disk holds about a month including the raw log. Set
`raw_retention_days` (or ship `raw/` to object storage) for longer unattended runs. Book parquet
alone is about 6 GB/month. For comparison, data/raw's 4 sampled days are 43 MB.

`merge.py` output is larger than the recorder output because `ticks.bin` is a dense 1 s grid: it
grows by about 2.1 M rows (330 MB) per fully recorded day, plus feat.bin and the other build files.

## Smoke test (2026-10-15, this container)

Run in this container on 2026-10-15 between 07:53 and 08:21 UTC, under shared CPU (load average
5 to 6 from other jobs), in two runs. Run 1 found the reconcile buffer issue described below; run 2
used the final code.

**Run 2 (final code, 08:02:26 to 08:10:11, 465 s).** All crypto up/down markets, `--pre-open 90`,
timed to cover the 08:05 to 08:10 batch of 5m markets from 90 s before open to the close.

* **Markets.** 56 recorded (8 coins: the 08:05 and 08:00 5m batches, 15m 08:00 / 08:15, 4h 04:00 /
  08:00, plus pre-open rows of the 08:10 batch). Up to 32 live at once (64 tokens) on 16
  connections. 0 unexpected disconnects; 16 connects and 16 closes (shards closed as markets ended).
* **Message rates (averages).**
  * 2,525 frames/s, 1.51 MB/s.
  * Events: price_change 2,280/s (4,560 level updates/s), best_bid_ask 221/s, book 13.2/s,
    last_trade_price 6.5/s.
  * Peak ~4.2k frames/s and 2.5 MB/s around the 5-minute boundary in run 1.
* **Rows and resources.** 19,323 rows (41.5/s). CPU 20.5% of one core on average, RSS 247 MB,
  loop lag ≤ 5 ms.
* **REST reconcile.** 407 checks (391 periodic, 16 early), **2 mismatches = 0.49%**, 0 at top of
  book, 0 errors. Median RTT 148 ms, p95 259 ms. Replaying the raw log offline (stream + logged REST
  snapshots) reproduces exactly these 2, both on deep levels of pre-open markets, so they are real
  stream/REST disagreements and were corrected (`reconciled = True` rows).
* **Stream best bid/ask checks.** 0.10% of 2.1 M items disagreed with our book: 2,936 stale levels
  were pruned, and 30 times we were missing the best level, which triggered an early reconcile.
* **Book quality.**
  * Crossed rows: **0.000%** (data/raw: 1.5 to 3.2%).
  * Up/Down mirror mismatch: 0.05% of market-ticks.
  * Top-of-book mirror agreement at equal timestamps: 100%.
* **Files.**
  * Book parquet: 0.93 MB compacted for 19,323 rows (48 B/row).
  * Raw log: 20.1 MB (1.17 M frames).
  * Reconcile log: 21 kB.
  * Trades / markets: see below.

**Run 1 (07:53:39 to 08:00:11, 392 s).**

* **Rates and resources.** 1,958 frames/s, 1.20 MB/s, 14,820 rows, CPU 17.1%, RSS 266 MB. Raw log
  13.2 MB, books 0.67 MB.
* **Reconcile.** 395 checks with 6 mismatches (1.5%). REST `/book` snapshots can lag the matching
  engine slightly, and the replay buffer only started when the request was sent, so deltas between
  `rest_ts` and the request were missing. The always-on per-token history ring
  (`reconcile.history_events`) fixed this. Replaying run 1's raw log with the ring gives 395 / 395
  `ok`.
* **Finalization.** Finalizing 2.5 min after the close showed that data-api indexes maker legs
  several minutes after taker legs. Finalization now waits 10 min and retries until the legs are
  consistent (`trades_consistent` in market_qc), and gamma resolution took up to 5.5 min.

**Checks.**

* `verify_output.py`: every books file loads with `pd.read_parquet`. Its schema equals
  `data/raw/updown_*.parquet` exactly, plus `reconciled`. trades / markets schemas equal data/raw's.
  0 errors on all row checks: ladder order, aligned lists, sizes > 0, best = first level, `crossed`
  definition, `seconds_to_close`, whole seconds, unique (token, second), two tokens per market,
  winner consistency.
* `merge.py` on run 1 (after `finalize` + `backfill_winners.py`): the fully covered, resolved market
  with consistent trades was appended (bnb 07:55 5m, 332 rows per token from 378 s before close to
  0, winner filled). `data/build.py` ran unmodified and `data/load.py` `verify()` passed (ticks
  1,374,506 × 40, 806 markets). The original 08-24 **test block was byte-identical** to data/build's
  (231,427 rows).
* Partially recorded markets (started mid-life) were correctly excluded (`coverage gap`).
* `features.py` could not run on today's markets: Binance's 1 s dumps are published the next day.

* `merge.py` on run 2: `finalize` 6 min after the close (gamma resolved 5.5 min after the close; all
  trade legs consistent), then merge.
  * **All 8 fully covered markets** (08:05 5m batch, every coin) were appended. 32 partially
    covered or unfinished markets were excluded.
  * `build.py` + `load.verify()` passed: 813 markets, train 439 / val 247 / test 127.
  * The original test block was byte-identical.
  * The split lock pinned 2026-10-15 to train: day 0 of the 8-day cycle.

  In the built `ticks.bin`, the recorded markets compare with data/raw's 5m markets as follows:

  | 5m markets | book_valid | crossed_raw | book_from_down | mean book_age | has trades |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | recorder (8) | 100% | 0.0% | 0.0% | 0.2 s | 7/8 (zec had none) |
  | data/raw (471) | 98.2% | 1.25% | 2.5% | 6.3 s | 93% |

Re-run with `./smoke_test.sh /tmp/polyrec_smoke`. It takes about 15 min: it records one complete 5m
batch, then runs finalize, backfill, verify and merge (build + load.verify).

## Polymarket API facts used (verified 2026-10-15)

* **Websocket.** `wss://ws-subscriptions-clob.polymarket.com/ws/market`. The first frame is
  `{"assets_ids": [...], "type": "market", "custom_feature_enabled": true}` (`initial_dump`
  defaults to true, `level` defaults to 2). Later frames are
  `{"operation": "subscribe"|"unsubscribe", "assets_ids": [...]}`; a dynamic subscribe also returns
  an initial `book` (about 0.3 s). Send the text `PING` every 10 s and the server replies `PONG`. A
  frame is one event or a JSON array of events (the initial dump). The server does not negotiate
  permessage-deflate.
* **Message format.**
  * `price_change` messages carry one item per token of the market: Up BUY p with Down SELL 1 − p,
    same size.
  * `book` bids arrive ascending and asks descending (worst first).
  * In `price_change` items, `best_bid = "0"` / `best_ask = "1"` mean an empty side.
* **Load.** All 32 live up/down markets (64 tokens) produce about 2 to 4k frames/s, 1.4 to 2.4 MB/s
  of JSON. A single connection carrying 16 busy tokens was closed with `1013 slow consumer: send
  buffer full` within 20 s, which is why markets go 2 per connection.
* **REST `/book`.** `timestamp` is the time of the book's last change, not the request time. A closed
  market's book returns 404 `No orderbook exists`. The limits are 1500 req/10 s for `/book`, 500 for
  `/books`, 200 for data-api `/trades`, and 500 for gamma `/events`.
* **Gamma.**
  * `GET /events/keyset?tag_slug=up-or-down&start_time_min=…&start_time_max=…` returns every up/down
    market of all coins. In October 2026 these were btc, eth, sol, xrp, bnb, doge, hype and zec, at
    5m / 15m / 4h.
  * Markets are created about 24 h before `eventStartTime`.
  * `/markets` caps `limit` at 100 and its payloads are about 10 kB per market, so listing all open
    markets is impractical. The events keyset query is about 6 kB per event.
* gamma and data-api reject Python's default User-Agent, so a browser-like one is sent.
