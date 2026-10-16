# v3 track S: free historical pre-game sports L2 (worker w10)

Track S history (D11 / D15) from free Hugging Face archives instead of the paid PolyOrderBooks key.
The source is **`Joseph3222/polymarket-orderbook`** (CC BY 4.0; mirror of pmxt's hourly archive,
https://archive.pmxt.dev, **credit pmxt**). It is the raw Polymarket CLOB WebSocket stream for every
market, 2026-02-22 → 2026-08-10, converted to the recorder / backfill raw layout that
`../v3_sports/build_sports.py --rec` reads.

| file | what |
|---|---|
| `convert.py` | Archive → `raw_hf/` (books / trades / markets / coverage, recorder layout) → `daily/<day>/` compact git export. Stages `select`, `books`, `meta`, `export`, `run`, `status`. Resumable. Option `--markets FILE` (any market list, e.g. elections). |
| `replay.py` | WebSocket book replay: snapshot, absolute level sizes, touch cleaning, anchoring. |
| `hfpq.py` | Partial download of one day's parquet: footer, then only the row groups needed, into a sparse local file. |
| `coverage.py` → `COVERAGE.md` | Day × league table of pre-game moneylines with ≥ 1 h of book before kickoff. The planned table uses metadata only (pre-registration basis). `--measured` adds the replayed-book table. |
| `validate_houroux.py` → `qc/houroux_<day>.json` | Cross-check against an independent capture (Houroux `market_ws` and its REST book checkpoints). |
| `daily/<day>/` | `books_<day>.parquet`, `trades_<day>.parquet`, `markets_<day>.parquet`, `coverage_<day>.jsonl`, `summary.json` (with the day's books-stage QC). |
| `raw_hf/` (git-ignored) | Full-depth books (deleted per day after export unless `--keep-raw`), trades, markets, coverage, `ltp/` (archive trade prints), `_state/` (done markers, carried book state, per-market QC). |

## The two archives (step 1)

| | Joseph3222/polymarket-orderbook | Houroux/polymarket-l2-history |
|---|---|---|
| licence | CC BY 4.0 (credit pmxt) | "other": defers to Polymarket's terms |
| span | 2026-02-22 → 08-10. 06-12..17 absent; 08-10 is 1 h. Footers show whole missing hours only on 02-24, 04-04, 05-22, 05-25, 06-05, 06-11, 06-23 | 2026-07-22 → today (still updated) |
| content | Raw WS events: `book` (full snapshot), `price_change` (**absolute** level size, plus exchange best_bid/ask), `last_trade_price` (with tx hash), `tick_size_change`. Plus a 1-min full-depth rollup (not used) | Raw WS messages (`book`, `price_change` with hashes, `best_bid_ask`, `last_trade_price`, `new_market`) with **local receive ns**, REST `/book` checkpoints every ~15 min, discovery logs, `sports_ws` scores, `rtds` |
| markets | Every market. The v1 era (02-22..04-12) has ~half; v2 (04-13 on) has all | ~40 reward-eligible markets at a time (104k over its life) |
| depth | Full | Full |
| size | 1.18 TB raw (8-14 GB/day parquet, ~100k-row groups sorted by condition_id within each hour) + 192 GB 1-min rollup | 118 GB `market_ws` gz JSONL, 160 GB parquet `analytics/` |
| timestamps | `event_ts_ms` = the exchange timestamp of the message (0 ms offset vs Houroux on 111k matched price changes) | exchange timestamp + local receive time (~45-90 ms later) |
| pre-game sports | ~180-560 moneylines/day at vol ≥ $10k (COVERAGE.md) | **Unusable.** 357 moneylines, mostly entering after kickoff (median first sighting 0.2 h after the scheduled start); only 14 have ≥ 1 h of pre-game book |
| verdict | **Used** | Not used for Track S. It does hold ~1,700 reward-eligible 2026 House / Senate / governor / chamber-control markets from late July (intermittent per market): a Track E source, handed to the elections worker |

`MANIFEST.sha256` lists the files this worker read in full (dataset cards and Houroux validation files). For the
large parquet files, only footers and row groups were read; `manifest_parquet.json` records their HF ETag
(the LFS sha256) and size.

## How the books are rebuilt

- **Selection** (`select`): `backfill_pob.select_markets` (gamma sports "Games" tag, `sportsMarketType` =
  moneyline, closed, two tokens, gamma volume ≥ $10k, all leagues). Book window per market =
  [kickoff − 48 h, kickoff + 5 min), the recorder's window. Kickoff is gamma `gameStartTime`; build_sports
  cuts at it.
- **Download**: per archive day, fetch the footer (~65 MB), then only the row groups whose (hour,
  condition-id range) can hold a target market inside its replay window. That is ~70 % of the row groups and
  ~8 GB/day, downloaded with 8 parallel range requests into a sparse file. The next day downloads while the
  current one is processed.
- **Replay** (`replay.py`): events per token in (`event_ts_ms`, file order). `book` replaces the book;
  `price_change` sets the absolute level size (0 removes it). Only the outcome-0 token is replayed and written;
  build_sports mirrors outcome 1.
- **Anchoring**: v2 sends a full `book` only on (re)subscription, to ~7 % of active tokens per hour. A market's
  book is therefore replayed from its **listing** (`acceptingOrdersTimestamp`, empty book plus every change
  since), at most `--track-days` 7 days before its window. The state is carried from day to day in
  `_state/bookstate/`. A book is *verified* when it descends from a listing or a `book` snapshot with no
  archive hour missing in between. A missing hour unverifies it until the next snapshot, or until 20
  consecutive touch matches.
- **Touch cleaning**: every v2 `price_change` carries the exchange's best bid / ask after the change. At the
  last row of each same-millisecond group, replayed bids above the source best bid and asks below the source
  best ask are dropped (lost removals, or bad ordering inside a millisecond). The replayed touch is then
  compared with the source touch (`pc_match`).
- **Rows**: one row per changed second, `captured_at` = end of that second (s + 1: the causal convention of
  backfill_pob), data/raw updown schema, full depth in `raw_hf`.
- **Coverage log** (recorder format): a heartbeat every 10 s for each archive hour present, and per-slug
  `live`/`down` while the book is verified. build_sports therefore treats a missing archive hour as a recorder
  gap (D16) and unverified spans as not live.
- **Trades / markets** (`meta`): data-api fill legs over the book window plus gamma market rows, written by
  `backfill_pob.py`'s own meta code (keyless). The archive's own trade prints go to `raw_hf/ltp/` (QC only).
- **Export** (`daily/`): markets with gamma volume ≥ **$50k** (`--export-min-volume`; the PolyOrderBooks
  backfill default), top **10** levels, rows dropped when their top-10 ladder equals the previous row's, zstd
  19, plus all trades / markets (vol ≥ $10k) and the coverage log. About 20-25 MB/day; no file over 50 MB
  (enforced). build_sports reads `daily/` as a `--rec` input (coverage is included).

## Quality checks (step 2)

2026-08-06 (first day, nothing carried in) and 2026-08-07 (state carried from 08-06):

| check | 08-06 | 08-07 |
|---|---|---|
| replayed touch = exchange touch, per price_change group | 95.1 % of 17.7M | 95.6 % of 18.5M |
| verified book, share of window time | 76 % | 92 % |
| moneylines in their book window / book rows | 844 / 3.24M | 908 / 3.66M |

- **Trades vs data-api** (on-chain fills via data-api; 08-06): 99.95 % of the archive's prints match a data-api
  tx hash, and 99.7 % of data-api outcome-0 taker legs appear in the archive. The archive time leads block
  time by 2.2 s median (p1-p99 1.1-3.5 s): matching-engine time vs chain settlement. The price agrees on
  97 % of matched legs (multi-level fills account for the rest).
- **vs Houroux live capture** (40 markets present in both, 08-06 12-14 UTC, 259k market-seconds,
  `qc/houroux_2026-08-06.json`): timestamps identical; the touch agrees 96.6 %, touch and size 86 %,
  top-5 levels 62 %.
- **vs Houroux REST `/book` checkpoints** (784 exchange snapshots): touch price 97.7 %, top-5 prices 79 %,
  top-5 prices and sizes 30 %. Houroux's own WS replay scores 46 % on the same check, so level sizes deeper
  than the touch drift in both independent replays: a snapshot is not aligned in time with the stream, and
  the comparison is at 1 s resolution. **The touch is reliable; deep sizes are approximate.**
- **build_sports + adapt3** (step 3): 79 games of 08-06 → `build_sports.py --seg 0 --verify` OK (ticks
  4.29M rows, 9.5k maker legs), `v3env/adapt3.py --check` OK (kind sports, settle_mode 1, row_s 1).
- **No overlap with w1's recorder** (`../v3_sports/rec_daily/`, from 2026-10-15): the archive ends 08-10.
  The Houroux cross-check and the data-api trades stand in for that comparison.

## Known gaps and caveats

- **v1 era** (02-22..04-12): pmxt's v1 archive subscribed to ~half of all markets, so many selected markets
  have no events there. It sends a full book about every 10 min per token, so the markets it does have anchor
  quickly. v1 rows carry no best_bid/ask on price_change, so touch matching is not available there.
- **Absent days** (06-12..17) and thin days (see COVERAGE.md): books are unverified across them until
  re-anchored.
- **Same-millisecond order** is file order (the card warns of ~10-15 % crossed books in naive replays). Touch
  cleaning removes the crossed touch but not deeper mis-ordering.
- **Markets listed > 7 days before their window** start unanchored at ws − 7 d and wait for a `book` or for
  touch recovery.
- **Volume floor**: the git export keeps vol ≥ $50k. `raw_hf` (≥ $10k, full depth) is only kept with
  `--keep-raw` and dies with the container. Re-run `convert.py books/export` for anything else.

## Coverage

See `COVERAGE.md`. The planned table is built from metadata only, for pre-registering the v3 sports split;
no model is run on this data before the split is fixed. The measured table is appended by
`coverage.py --measured` after the ingest. Ingest progress: `python convert.py status`, `daily/*/summary.json`.

## Resume

```bash
source ../../../.venv/bin/activate              # numpy pandas pyarrow requests aiohttp orjson (backfill_pob / polyrec)
python convert.py run --start 2026-02-22 --end 2026-08-10 --dl <scratch dir> --min-free-gb 14
```

Days with a done marker (`raw_hf/_state/books/<day>.json`) are skipped. The carried book state for day d is
`raw_hf/_state/bookstate/<d-1>.json.gz`. In a fresh container `raw_hf/` is gone: start from the first day
not in `daily/` and accept unanchored books on that first day (as on 08-06), or re-run from 7 days earlier
and discard the re-exports. Downloads are untrusted data: keep `--dl` in its own directory; they are parsed
only by pyarrow. Run any ad-hoc Python on them with `-I`. Never commit a `.gitattributes` containing
`filter=lfs`.
