# Coverage report

Build v1, generated from `coverage.py` / `build/meta.json` (rerun those to refresh the numbers).

## Markets

| | 5m | 15m | 4h | all |
| --- | ---: | ---: | ---: | ---: |
| markets | 471 | 265 | 69 | 805 |
| with gamma metadata | 471 | 265 | 69 | 805 |
| with ≥1 taker trade in `[open, end]` (flag HAS_TRADES) | 438 | 244 | 62 | 744 |
| with underlying (HAS_UNDERLYING) | 471 | 265 | 69 | 805 |
| with Chainlink strike `priceToBeat` (HAS_STRIKE) | 463 | 260 | 69 | 792 |

HYPE (91 markets) uses the Binance USD-M perp price, not spot (flag UL_PERP_PROXY). Data-api pagination never hit its
cap (limit 10000, offset ≤ 10000); no market was truncated.

## Trades

281,497 taker transactions (715,905 fill legs incl. makers). 270,323 fall in `[open, end]`, 10,388 before open
(markets list ~24 h early), 786 after end (settlement-lag prints, excluded from ticks).

| taker trades per market | 5m | 15m | 4h |
| --- | ---: | ---: | ---: |
| median, in window | 155 | 133 | 89 |
| mean, in window | 387 | 281 | 196 |
| markets with zero in window | 33 | 21 | 7 |
| fraction of seconds with any taker trade | 27.4% | 14.1% | 1.1% |

Flow is thin: a 4h market sees a median of 89 taker trades in 14,400 s. 37% of taker fills are below the 5-share
minimum order size (20% below 1 share) but they carry only 2% of volume. Aggressor side is balanced (50.7% buy Up);
48% of taker fills are on the Down token and are mirrored into the Up frame. Median 1 maker leg per tx (max 50).

## Can replayed trades fill resting quotes?

For each in-window taker tx we take its worst price (deepest maker leg, Up frame) and compare with the best quote on
the side it hit in the book k seconds earlier (row t-k). "At or through" means a resting quote joining that best
level would have been reached.

| book used | 5m at / through | 15m at / through | 4h at / through | at-or-through (all, quote present) |
| --- | --- | --- | --- | ---: |
| t (same second) | 18.0% / 38.6% | 25.9% / 33.9% | 47.9% / 11.0% | 58.4% |
| t-1 | 21.4% / 37.6% | 28.9% / 33.7% | 50.3% / 11.3% | 60.8% |
| t-3 | 29.2% / 41.7% | 33.3% / 41.9% | 62.4% / 14.3% | 72.8% |
| **t-4** | 21.2% / 51.2% | 26.3% / 51.5% | 63.3% / 19.0% | **74.7%** |
| t-6 | 11.6% / 55.5% | 17.7% / 54.8% | 54.4% / 24.5% | 69.5% |

The remainder prints inside the spread (4-21%) or at/beyond the *own-side* best (17-36% at t-1, 7-24% at t-3/4), i.e.
the book had already moved. Two conclusions:

1. **Trade timestamps lag the book by ~3-4 s.** data-api `timestamp` is the Polygon block time of settlement, not the
   CLOB match time; alignment with the book peaks at k = 3-4 for every contract length (at-best for 4h goes from 48%
   at k=0 to 63% at k=3-4, own-side prints fall from 19% to 7%). `ticks.bin` keeps the raw timestamp (row t holds
   trades with block time in (t-1, t]; causal, but the information is ~3-4 s old). A fill simulator should match a
   trade printed at row t against the agent's quotes as they stood ~3-4 rows earlier.
2. With that lag, **~75% of taker trades reach the prevailing best quote** (71-72% on 5m, 75-78% on 15m, 77-82% on 4h),
   and on 5m/15m about half of them sweep past it. Replayed trades can therefore drive fills for quotes at or inside
   the touch; queue position at the touch is unobserved (no L3), so a pro-rata or "fill after displayed size is
   consumed" rule is needed.

## Book

| per second of the grid | 5m | 15m | 4h |
| --- | ---: | ---: | ---: |
| `book_valid` (an uncrossed book exists) | 98.2% | 99.5% | 99.98% |
| `crossed_raw` (latest raw snapshot crossed → last good book carried) | 1.3% | 1.7% | 0.6% |
| `book_from_down` (Up crossed/missing, mirrored Down used) | 2.5% | 1.6% | 1.0% |
| one-sided (empty bid or ask) | 22.2% | 13.7% | 1.7% |
| book_age median / p90 (s) | 0 / 12 | 1 / 13 | 3 / 31 |

**Up/Down mirror agreement** (447,886 timestamps where both tokens have a snapshot; 700 Up-only, 720 Down-only):
top of book mirrors exactly (`Up bid = 1 - Down ask` and `Up ask = 1 - Down bid`) in 97.1% of them, 98.9% when
neither snapshot is crossed; full top-5 prices and sizes mirror in 94.9%. Disagreements are one token's book being
stale (e.g. Down crossed while Up is clean). Crossed rate: Up 2.3%, Down 2.2%, both 1.4%. Hence the build uses Up
as canonical and falls back to the mirrored Down snapshot only when the Up one is crossed or absent.

## Splits

| split | markets | 5m | 15m | 4h | rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 train (open 08-21..22 UTC) | 431 | 248 | 146 | 37 | 739,031 |
| 1 val (08-23) | 247 | 148 | 79 | 20 | 403,747 |
| 2 test (08-24) | 127 | 75 | 40 | 12 | 231,427 |
| 3 purged | **0** | 0 | 0 | 0 | 0 |

No market was purged: 4h contracts are aligned to 00/04/.../20 UTC and 5m/15m to their own grid, so no `[open, end)`
window crosses a UTC midnight. The test rows (1,142,778 to 1,374,205) have sha256
`40a81d59f1bdbfa375dc42dde500b1bc7a8303f6cc9b3bd364797c30afcb171e` (meta.json `test_rows`).
Note that markets of different lengths on the same coin overlap in time within a split (e.g. a 4h market contains
48 5m windows), and the val/test boundary has no embargo beyond the purge rule.

## Known gaps and caveats

* 61 markets have no taker trade in their window (33 × 5m, 21 × 15m, 7 × 4h); they still have books.
* 13 markets lack `priceToBeat` in gamma (8 × 5m, 5 × 15m): `ul_logp_vs_strike = 0`, `HAS_STRIKE` unset.
* Underlying is Binance, while settlement uses the Chainlink 60 s TWAP stream; HYPE is perp-based.
* Trade timestamps have 1 s resolution and lag the match by ~3-4 s; orders within a second are not ordered.
* Book snapshots are sparse (written on change + 60 s REST reconcile); `book_age` p90 is 12-31 s, and crossed rows
  show that one side can be stale. No L3/queue information.
* The tick size switches from 0.01 to 0.001 at extreme prices during a market; `markets.bin.tick_size` is the base
  0.01 and the switch is not tracked per row.
* Fee formula comes from Polymarket's docs (`C * 0.07 * p * (1-p)`, taker only); the data-api does not report fees, so
  it is not verified per trade.
