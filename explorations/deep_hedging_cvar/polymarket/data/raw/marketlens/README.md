---
license: cc-by-4.0
pretty_name: Polymarket BTC 5m L2 Order Book, One Full Day
size_categories:
  - 10M<n<100M
task_categories:
  - time-series-forecasting
language:
  - en
tags:
  - polymarket
  - prediction-markets
  - order-book
  - limit-order-book
  - market-microstructure
  - event-contracts
  - backtesting
  - bitcoin
  - finance
configs:
  - config_name: markets
    data_files:
      - split: train
        path: markets.parquet
  - config_name: snapshots
    data_files:
      - split: train
        path: snapshots.parquet
  - config_name: deltas
    data_files:
      - split: train
        path: deltas.parquet
  - config_name: trades
    data_files:
      - split: train
        path: trades.parquet
---

# Polymarket BTC 5m L2 Order Book, One Full Day

Every "Bitcoin Up or Down" 5 minute market on Polymarket for 2026-06-15 UTC, with
the full limit order book behind each one: 288 markets, 3,889 book snapshots,
41,591,670 level updates and 680,994 trades, on one millisecond time base.

Polymarket's public API serves the book as it is now. There is no endpoint that
returns the book as it was, so past depth is something you have to have been
recording while the market was live. This is that recording for one day, at full
resolution and with nothing sampled out: every market that opened in the day is
present start to finish, and the 5 minute markets are short enough that a single
day is 288 complete lifecycles rather than a slice through the middle of one.

## Coverage

| | |
|---|---|
| Series | `btc-up-or-down-5m` |
| Day | 2026-06-15 00:00:00Z to 2026-06-16 00:00:00Z (by market open time) |
| Markets | 288, all resolved |
| Snapshots | 3,889 (about 14 per market) |
| Level updates | 41,591,670 (about 144,000 per market) |
| Trades | 680,994 |
| Size on disk | 309 MB, zstd compressed parquet |

Data files carry a 10 minute skirt either side of the day so each market is whole.

## Files

`markets.parquet` is the spine. Join everything else to it on `market_id`.

| Column | Type | Notes |
|---|---|---|
| `market_id` | string | Polymarket condition id (`0x...`) |
| `question` | string | e.g. `Bitcoin Up or Down - June 14, 8:00PM-8:05PM ET` |
| `open_time_ms`, `close_time_ms`, `resolved_at_ms` | int64 | Epoch ms, UTC |
| `outcome_names`, `token_ids` | list[string] | CLOB token id per outcome, same order |
| `winning_outcome`, `winning_outcome_index` | string, int32 | What settled true, `Up` 148 and `Down` 140 across the day |
| `tick_size` | float64 | Minimum price increment, `0.01` throughout |

`snapshots.parquet` is a full book at a point in time, roughly every 60 seconds.

| Column | Type | Notes |
|---|---|---|
| `snapshot_id` | string | Parent key for the deltas that follow it |
| `market_id` | string | |
| `collected_at_ms` | int64 | Epoch ms, UTC |
| `bids`, `asks` | list[struct[price, size]] | Every level, not just the top |
| `best_bid`, `best_ask`, `spread`, `midpoint` | float64 | Precomputed from the book |
| `bid_depth`, `ask_depth` | float64 | Total size on each side |
| `bid_levels`, `ask_levels` | int32 | Level counts |
| `is_reseed` | bool | See "Reconstructing the book" |

`deltas.parquet` is one row per level change between snapshots.

| Column | Type | Notes |
|---|---|---|
| `market_id` | string | |
| `snapshot_id` | string | The snapshot this delta chain hangs off |
| `platform_timestamp_ms` | int64 | Exchange time, epoch ms |
| `collected_at_ms` | int64 | When we received it, epoch ms |
| `price` | float64 | The level that changed |
| `size` | float64 | New size at that level, `0` removes it |
| `side` | string | `BUY` (bid) or `SELL` (ask) |

`trades.parquet` is the tape on the same time base.

| Column | Type | Notes |
|---|---|---|
| `trade_id`, `market_id` | string | |
| `platform_timestamp_ms`, `collected_at_ms` | int64 | Epoch ms, UTC |
| `price`, `size` | float64 | |
| `side` | string | `BUY` or `SELL` |
| `fee_rate_bps` | string | As reported by the platform |

## Reconstructing the book

A delta is an absolute level replacement, not an increment. To get the book at
any instant, take the snapshot, then apply every delta whose `snapshot_id`
matches it, in `platform_timestamp_ms` order, up to your target time: set the
size at `price` on `side`, and drop the level when `size` is `0`.

Snapshots are the anchor and they are authoritative. Replaying one snapshot's
chain reproduces the next snapshot exactly in 160 of 170 non-reseed intervals,
measured across a 20 market sample of this file. The 10 that differ do so in one
direction only: the replayed book keeps a price level that the fresh snapshot no
longer lists, because not every removal arrives as its own `price_change`. No
level is ever missing from the replay and no size ever disagrees. Re-anchor on
each snapshot instead of replaying across many and nothing accumulates.

A reseed is the other break. If the WebSocket dropped and the collector pulled a
fresh book, that row carries `is_reseed = true` and the interval ending on it is
not continuous with what came before. This is common at these speeds: 1,351 of
the 3,889 snapshots in this file carry the flag. Skip those intervals rather
than replaying through them, which is why the snapshots matter as much as the
deltas do.

## Loading

```python
import pandas as pd

base = "hf://datasets/marketlens/polymarket-btc-5m-l2-depth"
markets = pd.read_parquet(f"{base}/markets.parquet")
snapshots = pd.read_parquet(f"{base}/snapshots.parquet")
```

`deltas.parquet` is 41.6M rows, so read it per market rather than whole. The
file is written in market order, one set of row groups per market, so a filter
on `market_id` skips the rest of the file:

```python
one = markets.iloc[0]["market_id"]
deltas = pd.read_parquet(f"{base}/deltas.parquet", filters=[("market_id", "=", one)])
```

That returns the 199,033 rows of the first market in about 3 seconds.

Both files load in the datasets library too:

```python
from datasets import load_dataset

ds = load_dataset("marketlens/polymarket-btc-5m-l2-depth", "snapshots", split="train")
```

## Reading it correctly

- **The book is YES centric.** Each market has two tokens. Events on the NO token
  are folded into one book as `1 - price` with the side flipped, so a single
  price series runs 0 to 1 and reads as the probability of the YES outcome.
  There is no token column because there is one book per market.
- **`side` describes the resting order**, so `BUY` is a bid and `SELL` is an ask.
- **Prices are exact to 4 decimals and sizes to 2**, cast from the exchange's
  original decimals. Round rather than compare floats for equality.
- **Use `platform_timestamp_ms` for anything time-based.** `collected_at_ms` is
  our receive time and carries network delay.
- Coverage of the wider archive starts 2026-02-28 and day level completeness
  varies. This day is complete; see marketlens.trade for the current calendar.

## What is not in here

One day and one series. The same collection runs across the whole Polymarket
catalog, sports, weather, equities, commodities and rates included, where the
book depth is far harder to come by than it is for crypto.

If you want more of it, the API and Python SDK serve the same snapshot and delta
chain, plus trades and candles, with a backtesting engine that replays this book
and models queue position, latency, slippage and settlement delay:
[marketlens.trade](https://marketlens.trade/?utm_source=huggingface&utm_medium=dataset&utm_campaign=polymarket-btc-5m-l2-depth) and
[github.com/marketlenstrade/marketlens-python](https://github.com/marketlenstrade/marketlens-python).
The free tier is enough to pull a market and check the data against this file.

The resolved outcome of every Polymarket market since February 2026 is published
as a companion dataset:
[marketlens/polymarket-resolved-markets](https://huggingface.co/datasets/marketlens/polymarket-resolved-markets).

## Provenance

Collected by Marketlens with a dedicated WebSocket connection per market against
Polymarket's public CLOB and Gamma APIs. Snapshots are full books as served by
the exchange; deltas are its `price_change` events, deduplicated on
`(timestamp, price, side)`. Resolutions come from the platform's
`market_resolved` event, not inferred from a final price. Derived from publicly
available Polymarket data.

## Citation

This dataset is archived on Zenodo with a DOI, so it can be cited and the exact
files stay retrievable at a fixed version:
[10.5281/zenodo.21933084](https://doi.org/10.5281/zenodo.21933084).

```bibtex
@dataset{marketlens_polymarket_btc5m_l2_2026,
  title     = {Polymarket Limit Order Book: 5-Minute Bitcoin Markets, One Complete Day (2026-06-15)},
  author    = {Marketlens},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.21933084},
  url       = {https://doi.org/10.5281/zenodo.21933084}
}
```
