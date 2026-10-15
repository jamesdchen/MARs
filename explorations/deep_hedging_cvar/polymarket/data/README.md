# Polymarket up/down market-making data

Data pipeline for a C market-making RL env (PufferLib-style, memory-mapped flat arrays).
805 resolved crypto up/down markets (5m / 15m / 4h; btc eth sol xrp bnb doge hype zec), 2026-08-21..24 UTC.

## Rebuild

```bash
cd explorations/deep_hedging_cvar/polymarket/data
python3 fetch_trades.py       # gamma metadata + data-api fills -> raw/markets.parquet, raw/trades.parquet (~5 min)
python3 fetch_underlying.py   # Binance 1s prices            -> raw/underlying.parquet (~1 min, ~200 MB of zips cached)
python3 build.py              # -> build/ticks.bin, markets.bin, meta.json, splits.json (~25 s)
python3 load.py               # integrity checks (no NaN, monotone clock, offsets, test sha256)
python3 coverage.py           # numbers for COVERAGE.md -> build/coverage_stats.json
/usr/bin/python3 features.py  # env inputs: feat.bin, book.bin, fills.bin, fill_index.bin, feat_meta.json
                              # (needs scipy for analysis/fair_value; written atomically; see ../ENV.md)
```

Downloads are cached in `raw/cache/` (git-ignored), so reruns are incremental. `build/` is
git-ignored (220 MB); everything under `raw/*.parquet` is committed (each < 17 MB).

| File | What |
| --- | --- |
| `raw/updown_{5m,15m,4h}.parquet` | input L2 snapshots (PolyOrderBooks, see `raw/README.md`) |
| `raw/markets.parquet` | gamma metadata per market: condition id, token ids, start/end, tick, min size, fees, rewards, resolution source, strike (`price_to_beat`), winner |
| `raw/trades.parquet` | every fill leg: `slug, timestamp (unix s), outcome_idx (0 Up/1 Down), side (+1 BUY/-1 SELL, from that leg's own perspective), price, size, is_taker, aggressor_up (+1 aggressor bought Up, -1 sold Up), tx (32-byte hash)` |
| `raw/underlying.parquet` | `coin, open_time (unix s), close, source` 1s grid, 2026-08-20..24 |

## Polymarket API findings (verified 2026-10-15)

* gamma `/markets?slug=<slug>&closed=true` gives `conditionId`, `clobTokenIds` (order matches `outcomes` = `["Up","Down"]`;
  all 805 match the token ids in the order-book files), `eventStartTime` (= end - contract length),
  `eventMetadata.priceToBeat/finalPrice` (Chainlink), `feeSchedule`, rewards fields. Python's default urllib User-Agent gets HTTP 403; send a normal UA.
* data-api `/trades?market=<conditionId>`: newest first; `limit` silently capped at 10000; `offset` > 10000 → HTTP 400
  (`max historical trades offset of 10000 exceeded`), so one query reaches 20000 rows; `side=BUY|SELL` can split a query
  further. No market here came close (max 4400 taker / ~12k legs).
* `takerOnly=true` (the default) returns **one row per matched transaction = the taker's fill**: `side` is the aggressor
  side on the token the taker traded, `price` is the taker's average price, `size` its total size.
  `takerOnly=false` adds **one row per maker order filled** in that tx, from the maker's perspective, at the maker's
  resting limit price. Verified on all 281,497 txs: exactly one taker leg per tx; sum of maker sizes = taker size;
  taker price = size-weighted mean of maker prices (in the Up frame); every maker leg is on the opposite side of the aggressor.
* **Complementary matching**: Up and Down share one book. Taker BUY Down @ p can match maker BUY Up @ 1-p (mint),
  taker SELL Up can match maker SELL Down @ 1-p (merge), or same-token BUY vs SELL. In the Up frame:
  Down BUY @ p ≡ Up SELL @ 1-p, Down SELL @ p ≡ Up BUY @ 1-p. 48% of taker fills are on the Down token, so ignoring
  it would halve the flow.
* Trade `timestamp` is the Polygon block time of settlement, **~3-4 s after the off-chain CLOB match** (see COVERAGE.md:
  trade prices line up best with the book 3-4 s earlier). 786 taker fills carry timestamps after market end.
* Fees (docs.polymarket.com/trading/fees): taker only, `fee_usdc = C * fee_rate * p * (1-p)` with C shares at price p;
  crypto `fee_rate = 0.07` (`fee_exponent = 1`; exponent applies to `p(1-p)`), makers pay 0 and receive a rebate
  pool of 20% of taker fees (`maker_rebate_rate`). `makerBaseFee/takerBaseFee = 1000` in gamma are legacy fields.
* Tick size is dynamic: books are on a 0.01 grid and switch to 0.001 once the price goes extreme (> ~0.96 / < ~0.04)
  and stay there. Gamma reports the post-resolution value (0.001 for 797/805). `markets.bin.tick_size = 0.01` (base),
  `tick_size_final` = gamma. `orderMinSize = 5` for all. `rewards_max_spread` is in cents (0 / 1.5 / 4.5),
  `rewards_min_size` 0 or 50.
* Resolution: Chainlink `<coin>-usd-twap-60s` data stream (TWAP over the last 60 s vs the price at the start).
  Binance is a proxy; `ul_logp_vs_strike` uses Binance price vs Chainlink `priceToBeat` (basis included).

## Underlying

Binance spot `<COIN>USDT` 1s klines from data.binance.vision (checksums verified). **HYPE is not on Binance spot**
(no HYPEUSDT/HYPEUSDC/HYPEFDUSD), USD-M futures has no 1s klines, and Hyperliquid's `candleSnapshot` only serves
the latest 5000 candles (empty for Aug 2026). HYPE therefore uses Binance **USD-M perp aggTrades** resampled to a
1s last-price grid (flag `UL_PERP_PROXY`, `source = um_perp_aggtrades_1s`). All 805 markets have underlying.

## Build outputs (`build/`)

### Causality conventions (row t = unix second t)

* Grid: every second t in `[open, end]` inclusive, `open = end - contract_len_s`, so `n_rows = contract_len_s + 1`;
  `seconds_to_close = end - t` (last row = 0).
* Book: latest snapshot with `captured_at <= t`, in the **Up frame**. Per captured timestamp the Up snapshot is used
  if uncrossed, else the mirrored Down snapshot (`Up bid = 1 - Down ask`, sizes carried) if uncrossed
  (`book_from_down = 1`), else the timestamp is crossed and the **last uncrossed book is forward-filled**
  (`crossed_raw = 1`, `book_age` keeps growing). Snapshots before open are used for the opening book.
  `book_valid = 0` only if no uncrossed snapshot exists yet (then ladders are empty, `book_age = 0`).
* Trades: fills with timestamp in `(t-1, t]` (integer seconds → `timestamp == t`). Aggressor side in the Up frame from
  the taker leg; prices are the **maker legs' resting prices** mapped to the Up frame (so `buy_max_px` is the deepest
  level the sweep reached); volumes in shares; `trade_count` = distinct taker txs.
* Underlying: close of the Binance kline with `open_time = t-1` (the last kline closed by t).
  `ul_logp_rel_open = log(c_t / c_open)`, `ul_logret_1s = log(c_t / c_{t-1})`,
  `ul_logp_vs_strike = log(c_t / price_to_beat)` (strike is fixed at open; 0 when missing, see flag `HAS_STRIKE`).
* `winner_up` / final price **never** appear in `ticks.bin`; `winner_up` lives only in `markets.bin` for terminal settlement.
* Empty levels: bid price 0, ask price 1, size 0. No-trade seconds: volumes, VWAPs, max/min = 0. No NaN/inf anywhere.

### `ticks.bin` — float32, row-major `[N, F]`, N = 1,374,205, F = 40

| idx | column | idx | column |
| --- | --- | --- | --- |
| 0 | seconds_to_close | 27 | buy_vol (aggressor buys of Up, shares) |
| 1-5 | bid_px_0..4 (best first) | 28 | buy_vwap |
| 6-10 | bid_sz_0..4 | 29 | buy_max_px |
| 11-15 | ask_px_0..4 | 30 | sell_vol (aggressor sells of Up) |
| 16-20 | ask_sz_0..4 | 31 | sell_vwap |
| 21 | bid_depth_total (all levels) | 32 | sell_min_px |
| 22 | ask_depth_total | 33 | trade_count |
| 23 | book_age (s) | 34 | ul_logp_rel_open |
| 24 | book_valid | 35 | ul_logret_1s |
| 25 | crossed_raw | 36 | ul_logp_vs_strike |
| 26 | book_from_down | 37 | const_one |
| | | 38-39 | pad (0) |

### `markets.bin` — 88-byte records, little endian, sorted by `(split, open_ts, slug)`

```c
typedef struct {            // offset
    int64_t row_offset;     //  0  first row in ticks.bin
    int64_t open_ts;        //  8  unix s
    int64_t end_ts;         // 16  unix s
    int32_t n_rows;         // 24  = contract_len_s + 1
    int32_t contract_len_s; // 28  300 / 900 / 14400
    int32_t coin_id;        // 32  btc0 eth1 sol2 xrp3 bnb4 doge5 hype6 zec7
    int32_t winner_up;      // 36  1 if Up won (terminal settlement only, never an observation)
    int32_t split;          // 40  0 train, 1 val, 2 test, 3 purged
    int32_t flags;          // 44  1 HAS_TRADES, 2 HAS_UNDERLYING, 4 UL_PERP_PROXY, 8 HAS_STRIKE
    float   tick_size;      // 48  base tick 0.01 (dynamic, see above)
    float   taker_fee_rate; // 52  0.07: fee = C*rate*p*(1-p)
    float   fee_exponent;   // 56  1
    float   min_order_size; // 60  5 shares
    float   rewards_min_size;   // 64
    float   rewards_max_spread; // 68  cents
    float   log_strike_vs_ul_open; // 72  log(binance close at open / priceToBeat)
    float   maker_rebate_rate;  // 76  0.2
    int32_t market_idx;     // 80  index into meta.json "slugs"
    float   tick_size_final;// 84  gamma orderPriceMinTickSize after resolution
} MarketRec;                // sizeof == 88
```

### Splits (`splits.json`, meta.json `splits`)

By market open date UTC: train 2026-08-21..22, val 08-23, test 08-24. A market whose `[open, end)` touches a day of
another split is purged (split 3); none were (4h contracts are aligned to 00/04/.../20 UTC, so no market spans
midnight). Markets are sorted by split, so each split's rows are contiguous; `meta.json.test_rows` stores the row range
and a sha256 of the test rows (checked by `load.py`). Train-only per-coin mean/std of `ul_logret_1s` are recorded in
`meta.json.normalisation_train_only` (not applied to `ticks.bin`).

### Loading

```python
from load import load, episodes, verify
ticks, markets, meta = load()               # np.memmap [N, F], structured array, dict
for m, rows in episodes(ticks, markets, split="train"):
    ...
```
