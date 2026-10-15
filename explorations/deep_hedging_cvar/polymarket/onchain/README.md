# On-chain Polymarket fill extractor

Pulls every CTF Exchange V2 fill from Polygon with keyless archive RPCs. The output covers months of trades for
calibrating fill and markout models and the synthetic simulator, well beyond the 4-day L2 window in `../data`.
It hardens the prototype in `../research/` (`rpc.py`, `pull_market.py`, `reconcile.py`, `calldata.py`).

```bash
source ../../.venv/bin/activate          # explorations/deep_hedging_cvar/.venv
# 1. one UTC day of crypto up/down fills (markets resolved via gamma, 3 events, both exchanges)
python extract.py --start 2026-08-21 --end 2026-08-21 --market-set crypto-updown --max-event-start 2026-08-23
# 2. to the exact schema of ../data/raw/trades.parquet + markets.parquet
python to_trades.py --src data/crypto-updown --start 2026-08-21 --end 2026-08-21
# 3. reconcile against ../data/raw/trades.parquet (refuses the val/test days 08-23/24)
python validate.py day --date 2026-08-21 --trades data/crypto-updown/trades_2026-08-21_2026-08-21.parquet

# months: fills only, without order_hash (about 100 MB/day instead of 360 MB/day), resumable
python extract.py --start 2026-05-01 --end 2026-07-31 --events OrderFilled --slim
# one market window, plus the matchOrders calldata (original order size, limit price, creation time)
python extract.py --start 2026-08-21T01:50 --end 2026-08-21T02:15 --slugs bnb-updown-5m-1787277900 --out data/val_bnb --calldata
python validate.py market --slug bnb-updown-5m-1787277900 --out data/val_bnb
```

Re-running any `extract.py` command resumes from the ledger. `data/` and `cache/` are git-ignored;
`validation/*.json` holds the reconciliation results below.

## Files

| File | What |
| --- | --- |
| `extract.py` | Block range or date range → decoded `OrderFilled`, `OrdersMatched` and `FeeCharged` events for both exchanges, written as per-day parquet partitions. Handles multi-RPC failover, adaptive chunk sizing, concurrency caps, retry and backoff, checkpoint/resume, and cross-endpoint recounts. `extract.load(out, table, start, end)` reads the output back. |
| `rpc.py` | Endpoint pool: per-endpoint in-flight cap and request-rate cap, cooldown with exponential backoff, error classification (range/result-limit vs transient), adaptive `eth_getLogs` span, JSON-RPC batches. `BlockClock`: cached block→time map from sparse headers, interpolation search for date→block, linear interpolation. |
| `markets.py` | token_id → market map from gamma, cached in `cache/gamma.sqlite`. `crypto-updown` enumerates the deterministic slugs `{coin}-updown-{5m,15m,4h}-{event_start}` for a window. `lookup_tokens` resolves any token, `filter_slug` filters by a regex. Also a CLI. |
| `calldata.py` | Optional (`extract.py --calldata` or standalone). Decodes `matchOrders` calldata into each order's original size, limit price, creation time, signature type and requested fill and fee. Two modes, `txs` and `blocks`, plus `--sample-frac`. |
| `to_trades.py` | Fills → `../data/raw/trades.parquet` schema (Up/Down frame from the token's gamma outcome) and `markets.parquet` schema. Checks that both schemas match the existing files. |
| `validate.py` | `market`: one market vs data-api `/trades`. `day`: a converted day vs `../data/raw/trades.parquet`. |
| `abi.py`, `common.py` | Pure-python keccak-256 and ABI decoder (the venv has no `eth_abi`/`pycryptodome`; the self-test checks all three topic0 hashes and the matchOrders selector `0x3c2b4399`), plus constants and helpers. |

## Outputs (`data/<set>/`, set = `crypto-updown`, `all`, `slugs`, or the `--tokens` file stem)

| Path | Columns |
| --- | --- |
| `fills/date=D/day.parquet` | `block, block_time, tx_hash, log_index, exchange (ctf/negrisk), order_hash, maker, taker, side (BUY/SELL), token_id (decimal str), maker_amount, taker_amount, fee` (1e-6 units: USDC or shares), plus derived `price` (USDC/share), `size` (shares) and `role` (`taker` when the event's taker is the exchange contract, else `maker`) |
| `matches/date=D/` | OrdersMatched: `block, block_time, tx_hash, log_index, exchange, order_hash` (taker order), `maker` (taker order's maker), `side, token_id, maker_amount, taker_amount, price, size` |
| `fees/date=D/` | FeeCharged: `block, block_time, tx_hash, log_index, exchange, receiver, amount` |
| `markets/date=D.parquet` | the token map used as the filter for day D |
| `orders/date=D/orders.parquet` | (calldata) `role, order_index, maker, signer, token_id, side, orig_size, limit_price, sig_type, order_ts, created_s, age_s, fill_amount, fee_amount, partial, condition_id, tx_from` |
| `_ledger.jsonl`, `_runs.jsonl` | per-segment and per-day records (counts, seconds, endpoints, clock check), run summaries |
| `trades_S_E.parquet`, `markets_S_E.parquet` | `to_trades.py` output in the pipeline's schema |

Fills are sorted by `(block, log_index)`. A partition is a directory, so `pd.read_parquet("data/crypto-updown/fills")`
reads every day, with `date` as a hive column. Partial-day pulls stay as `part-LO-HI.parquet` files; a day is
compacted into `day.parquet` once all of it is done.

## How it works

- **Filtering.** `tokenId` is not indexed, so each `eth_getLogs` call asks for every exchange log of the selected
  events in a block span, about 165–290 logs per block in Aug 2026. A tx is kept if any of its
  OrderFilled/OrdersMatched logs touches a token in the market set. All logs of a kept tx are then decoded,
  including its FeeCharged log and complementary-token maker legs.
- **Market set.** For day D, `crypto-updown` selects markets whose trading lifetime `[createdAt, end + 10 min)`
  intersects D. Markets are created about 23.85 h before event start, so day D includes the next day's
  markets' early fills. `--max-event-start` cuts the set; the Aug 21 pull used `2026-08-23`, so no
  val/test market was touched. Gamma is asked for 40 slugs per call. 2026-08-21 resolved 6,264 candidate
  markets and 12,528 tokens in 160 calls (12.5 s), and the results are cached, so consecutive days cost about
  80 calls each.
- **Segments and resume.** Work runs in `--seg` (1000) block segments on an absolute grid, clipped to UTC day
  boundaries. `--workers` (4) segments are in flight. Each finished segment is written atomically as a part
  file and appended (fsync) to `_ledger.jsonl` with a fingerprint of the filter (token list hash, events,
  `--slim`). Re-runs skip finished segments, and a changed filter re-extracts and overwrites the day with a
  warning.
- **RPC robustness.** Endpoint defaults are in `common.DEFAULT_ENDPOINTS`; override them with
  `--rpc url,conc,span,span_max,rps,max_batch`.
  - Each endpoint has its own in-flight cap and rate cap.
  - On 429, 5xx, timeouts, non-JSON or truncated bodies, a non-list result, logs outside the requested
    range, `removed` logs, or an empty answer for 20 or more blocks, the work goes to another endpoint, and
    the failing one cools down exponentially (up to 120 s).
  - On result-limit or range errors, the span shrinks per endpoint. Tenderly's `Try with this block range
    [a, b]` hint is used directly. The learned cap relaxes by 10% after every 20 successes.
  - `--verify-frac` (5%) re-fetches a random 10-block window of a segment and compares the decoded fill count.
  - Tested by mixing in publicnode, which now refuses archive queries without a key ("Archive requests
    require a personal token"). Its errors were classified as transient and the work failed over, with no
    effect on output.
- **Block time.** `block_time` is the `blockTimestamp` field that all three default endpoints attach to each
  log; it is exact and equal to the data-api `timestamp`. Endpoints without it fall back to `BlockClock`
  interpolation. Each compacted day records the error of interpolating from headers every 2000 blocks.
  On 2026-08-21, over 57,380 blocks: mean |err| 0.25 s, p99 0.5 s, max 0.5 s, 100% exact after rounding to
  the second. Polygon ran at exactly 1.500 s per block.

## Validation (`validation/*.json`)

**1. Prototype reconciliation** (`bnb-updown-5m-1787277900` vs data-api, both `takerOnly` modes): 152/152 fills
match exactly on (wallet, side, token, size, tx), and 152/152 when price is included (max |Δprice| 5e-11).
70/70 taker rows and 82 maker rows. Timestamps: data-api minus block time = 0 s for all 152. Fees follow
`size·0.07·p·(1-p)` USDC per taker fill (median ratio 0.97–1.00; the gap is rounding on 0.02-share fills), and
FeeCharged totals equal the OrderFilled fee totals. Calldata: all 70 txs are `matchOrders`, and the maker order
sequence and fills agree with OrderFilled in 70/70 txs. 67/82 maker fills are partial. Maker order age at fill
has a median of 14.7 s and taker 2.6 s; 4 maker orders carry timestamp 0, and the prototype's 14.7 s excludes
them. `txs` and `blocks` modes give identical output.

**2. Full train day 2026-08-21**, crypto up/down, converted with `to_trades.py` and compared with
`../data/raw/trades.parquet` on fills whose block time falls on that day:

| | data/raw | on-chain | match |
| --- | --- | --- | --- |
| markets with fills that day | 238 | 3,335 (all crypto up/down) | 238/238 present |
| fill legs (238 markets) | 238,247 | 238,247 | **238,247 exact** on (slug, tx, outcome, side, is_taker, size, price): 100% both ways |
| taker legs | 93,500 | 93,500 | 93,500; `aggressor_up` agrees 93,500/93,500 |
| timestamps | | | 238,247/238,247 identical |
| per market: legs, takers, volume, notional | | | 238/238 equal (rel. err < 1e-4, float32) |
| `markets.parquet` metadata (33 columns) | | | identical for all 238 |

Before the dust filter the chain had 9 more legs. These are maker legs of 0.005–0.009 shares that the data-api
omits (data/raw has 0 maker legs below 0.01 of 434k; the chain has 56 of 2.1M that day). `to_trades.py` drops
them by default (`--keep-dust` keeps them); the taker leg keeps its full size either way, as in the data-api.
The day's 3,335 markets carry 3.46M legs, about 15× what data/raw has for that day.

**3. Calldata at scale** (`blocks` mode, 1,500 blocks, 45,040 txs): 28 blocks/s, 844 txs/s, 24% of one core. All
txs are `matchOrders`. The calldata taker fill equals the OrderFilled amount for 45,039/45,040; the exception is
a 1-unit rounding at p = 0.999, because calldata carries the requested fill and the event carries the executed
one. 70% of maker fills are partial. Median order age at fill: maker 4.2 s, taker 3.05 s.

## Throughput (measured 2026-10-15, 4 workers, about 0.8 core)

Full day 2026-08-21: 57,600 blocks, **9.52M logs downloaded** (6.20M kept for crypto up/down, giving 3.46M fills,
1.37M matches and 1.37M fees). **202 s wall, 157 s CPU, 285 blocks/s, 47k logs/s**, 0 failed segments.
That is 8.7 GB of JSON, about 0.7 GB on the wire (gzip 12.6×). Load was spread over quiknode (93 calls, 400-block
spans), drpc (160 calls) and tenderly (237 calls, 3 result-cap errors). `to_trades.py` takes 45 s per day.
Disk per day for the crypto set: fills 194 MB, matches 112 MB, fees 53 MB, trades in pipeline schema 56 MB.
`order_hash` alone is half of the fills bytes; `--slim` drops it.

### Cost to pull N months

Run time scales with log volume. OrderFilled per block around noon UTC was 189 on May 1, 276 on Jun 1,
135 on Jul 1, 115 on Aug 1, 141 on Aug 21, 67 on Sep 15 and 60 on Oct 10. So May–June days cost roughly
1.5–2× Aug 21 and Sep–Oct days roughly 0.5×.

| Pull (crypto up/down, all 3 events) | Wall time (range) | CPU (one core) | Disk: extract / trades schema |
| --- | --- | --- | --- |
| 1 day | 3.4 min (2–7) | 2.6 min | 360 MB / 56 MB |
| 1 month | ≈1.7 h (0.9–3.5) | ≈1.3 h | 11 GB / 1.7 GB |
| 3 months | ≈5 h (3–10) | ≈4 h | 33 GB / 5 GB |
| all V2 history, 2026-04-28 → 10-15 (≈170 d) | ≈10 h (6–15) | ≈7 h | 61 GB / 9.5 GB |

- `--events OrderFilled` downloads about 40% fewer logs (OrderFilled is 57% of exchange logs) and, with
  `--slim`, writes about 100 MB/day (3 GB/month). This is an estimate from log shares, not a measured run.
  About 22 GB of disk was free, so use it for anything past 1 month, or delete the extract after `to_trades.py`.
- The unfiltered `--all` mode costs the same download but writes about 5.4M fills/day (all markets).
- Calldata adds about `min(#blocks, #txs) / 30` seconds per day at the keyless rate limits (the auto mode picks
  the cheaper path). That is about 34 min/day for the full crypto set (17 h/month). `--calldata-sample-frac 0.01`
  brings it to about 7 min/day, which is enough for order-age and partial-fill distributions.

## Caveats

- **Block time is not match time.** `block_time` (= the data-api `timestamp`) is when the match settled on Polygon.
  It comes about 2–4 s after the off-chain CLOB match: `../data/COVERAGE.md` finds book alignment 3–4 s earlier,
  and `sim/fit.py` uses `MATCH_LAG = 3`. Calldata `created_s` timestamps the signed order; the taker order age at
  fill (median 2.6–3 s) bounds sign → match → mine latency. Block timestamps have 1 s resolution with 1.5 s blocks.
- **No cancels, queue or book on chain.** Only fills settle on chain. Resting orders that never fill, cancels,
  queue position and book states are invisible, and maker order lifetimes from calldata only cover orders that
  filled (survivorship bias). Extra days give flow and markout statistics, not new L2 episodes. `data/build.py`
  still needs L2 snapshots, and new book episodes need a websocket recorder.
- **Not plug-in for `sim/fit.py` yet.** The trades schema is exact, but `sim/common.train_trades()` reads
  `../data/raw/trades.parquet` and asserts that slugs are in `splits.json`. Pointing the flow fit at months of
  on-chain trades needs a small loader change in `sim/`, which was out of scope here. Keep val/test markets
  (event start 2026-08-23/24) out with `--max-event-start` or `to_trades --slug-regex`.
- **V2 only.** The V2 exchanges were deployed 2026-03-31 (block 84,902,353) but carried no fills until about
  2026-04-28 12:00 UTC. Earlier history is on the V1 exchange contracts, which have different addresses and an
  older OrderFilled layout and are not handled here. The V1 tape is available as Qin & Yang's Polymarket-v1
  dataset (`TimeSeventeen/Polymarket-v1`). Crypto up/down fills are all on the CTF exchange; the neg-risk
  exchange carries 11–23% of all exchange fills (multi-outcome markets).
- **Data-api conventions reproduced by `to_trades.py`.** Maker legs below 0.01 shares are dropped. The maker leg
  price is the maker's limit price, which on chain equals the fill's USDC/shares. The taker price is the taker's
  average price.
- **Calldata.** Amounts are requested fills; the events carry executed fills, which can differ by 1e-6 rounding.
  A few orders sign timestamp 0 (`created_s` = NaN) or a seconds-valued timestamp, which is recognised.
- **Keyless RPC policies change.** Publicnode now needs a token for archive queries. drpc's free tier rejects
  spans over about 100 blocks and batches over 10. quiknode allows 20 requests/s and counts batch elements.
  tenderly caps results at 20k. The pool degrades to whatever answers, and `--rpc` adds keyed endpoints.
  `blockTimestamp` on logs is a non-standard field; without it times are interpolated, with the error above.
