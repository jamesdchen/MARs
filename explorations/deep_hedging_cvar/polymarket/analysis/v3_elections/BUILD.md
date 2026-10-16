# build_elections.py: env binaries for track E (election reward farming)

`build_elections.py` turns the recorder's election books (`rec/books`), the hourly reward-config
snapshots (`rec/rewards`, plus `sel/selection.json` as the first snapshot), the coverage log
(`rec/coverage`) and the markets' fills into the env binaries of `data/build/`. `data/load.py` and
the current C env (`mm.h`, `env.MmEnv(data_dir=...)`) read the output unchanged. The script only
reads `rec/` and never writes to `data/`.

```bash
source ../../../.venv/bin/activate            # pandas, pyarrow, scipy (features.py import), orjson (coverage)
python build_elections.py --split-days days.json --trades-source data-api            # -> build/
python build_elections.py --emit-tokens tokens.parquet    # token map for onchain/extract.py --tokens
```

| flag | default | meaning |
|---|---|---|
| `--window-s` | 14400 | Window length (D16: 4 h windows at 00/04/08/12/16/20 UTC). It must divide 86400. Episodes are cut further at gaps, the market end and the cutoff. |
| `--min-episode-s` | 600 | Drops (sub-)episodes shorter than this. |
| `--day-max-gap-s` | 1800 | A day is eligible for train/val/test only if its total gap time is at most this, both recorder-wide and for the episode's market. |
| `--ineligible-day` | train-only | D16: an ineligible day keeps its train assignment and loses val/test (purged), with a warning. `purged` moves the whole day to split 3. |
| `--gaps-only` | off | Writes `GAPS.md` and `gaps_per_day.csv` to `--gaps-dir` (default this directory) from `rec/coverage`, builds nothing, and needs no `--trades-source`. |
| `--split-days` | none | JSON file or inline JSON, `{day: train / val / test / purged / exclude}`. A day that is not listed goes to split 3 (purged). Days marked `exclude` are not built. **No default assigns test.** |
| `--cutoff` | 2026-11-03T00:00Z | Windows must end by this time. Election night is out of scope (D11). |
| `--max-gap` | 15 s | A coverage gap longer than this ends the episode, and a new one starts after the gap (D16). Shorter gaps are forward-filled. |
| `--allow-partial` | off | Keeps episodes whose fills coverage is incomplete, with flag TRADES_INCOMPLETE. Use it only for smoke tests. |
| `--top-n` / `--all` | 20 | Builds only the N markets with the highest `rate_per_day` in `sel/selection.json`. `--top-n 0` or `--all` builds every selected market. |
| `--markets`, `--min-rate` | all | Further market filters, applied before `--top-n`. |
| `--trades-source` | required | `onchain` reads `onchain/extract.py` output (`--trades-path`); this is the census source. `data-api` uses `/trades?market=<condition id>` and is cached. `file` reads a trades.parquet in the `data/raw` schema. `none` builds without fills. |
| `--feat-layout` | split | `split` writes feat.bin as [N, 44], which the current env reads, plus feat_x.bin as [N, X]. `wide` writes feat.bin as [N, 44 + X], which needs MM_G = 88 in the env. |
| `--max-gb` | 20 | Refuses to build anything larger than this. |

## Episodes

- **One episode is one market in one window** (`markets.bin` row). The window is
  `[open, open + window_s]`, giving `window_s + 1` one-second rows, as in `data/build.py`.
  `seconds_to_close` counts down to the end of the window, not to resolution.
- **Episodes are cut at gaps (D16).** Each 4 h window is intersected with the market's coverage
  segments from `rec/coverage` (`polyrec.coverage.live_intervals`, gaps of 15 s or less bridged), and
  further cut at the market's `end_date` and at `--cutoff`.
  - Each resulting piece becomes an episode `[start, end]` with `end - start + 1` rows. Its
    `contract_len_s` is `end - start`, so episode lengths vary.
  - Pieces shorter than `--min-episode-s` are dropped.
  - No row inside a gap longer than 15 s is ever synthesised.
- **Reasons and links.** `markets_meta.parquet` records each episode's `start_reason` and its
  `end_reason`; `split_reason` is the `end_reason`. The values are `window_boundary`, `gap`,
  `market_end`, `cutoff`, `coverage_start` (the market's first recorded segment) and `coverage_end`
  (the recording so far ends there).
  - Two consecutive kept segments of one market separated by a gap are linked through
    `prev_segment_id` and `next_segment_id`.
  - `gap_before_s` is the time from the previous kept segment's end; it includes any dropped short
    segment in between. A v3 env can use the link to carry inventory across the gap, as D16
    requires (the pre-gap mark is that segment's last row).
- **Day eligibility (D16).** A day is eligible for train/val/test only if its total gap seconds are
  at most `--day-max-gap-s` (1800), both recorder-wide and for the episode's own market on that day.
  Otherwise the assigned split becomes purged (3) and a warning is printed.
  - Default `train-only` follows D16: such days may still be used for train, never for val/test.
  - Every build writes `GAPS.md` and `gaps_per_day.csv` next to the binaries (see "Gap report").
- **Fills.** The fills source must be complete over the episode; otherwise the episode needs
  `--allow-partial`.
- **Books.** The script uses `data/build.py`'s own functions (`ladder_topk`, `merged_snapshots`,
  `to_up_frame`, `mirror_stats`):
  - It keeps one snapshot per token and captured second (the last one).
  - At each row it uses the latest uncrossed snapshot at or before t.
  - The No book is mapped into the Yes frame. Outcome 0 is `selection.json tokens[0]` (Yes, or the
    first named outcome), which takes the Up slot.
  - The snapshot is only used inside the coverage segment it was recorded in. Gaps of up to
    `--max-gap` are bridged. Rows outside coverage get `book_valid = 0`.
- **Carry-over.** Snapshots carry across days. The last row and the last uncrossed row per token are
  kept, because a quiet market can go hours without a new row. Rows at the next midnight come from
  the next day's 00h file.
- **Trades.** The `data/build.py` rule applies: block time == t, maker legs' prices, and the
  aggressor taken from the taker leg. `fills.bin` and `fill_index.bin` follow the
  `data/features.py` rule exactly.
- **Order.** Episodes are sorted by (split, open, id), and each split's rows are contiguous. The
  test rows are hashed as in `build.py`. The episode id `"<slug>@<open ISO>"` is used for
  `meta.slugs` and `splits.json`.

## markets.bin (88 B, same struct)

| field | election value |
|---|---|
| open_ts, end_ts, n_rows | the window; n_rows = window_s + 1 |
| contract_len_s | end - start of the episode (the full window, or less when the episode is cut at a gap, the market end or the cutoff). The current env's length bit only knows 300, 900 and 14400. |
| coin_id | -1 (no underlying) |
| winner_up | **-1**: unresolved at build time, never a label |
| split | from `--split-days` |
| flags | `build.py`'s bits plus ELECTION 16, UNRESOLVED 32, PARTIAL 64, IN_PROGRAM 128 and TRADES_INCOMPLETE 256. HAS_UNDERLYING and HAS_STRIKE are always 0. |
| tick_size | 0.01 (`build.py` BASE_TICK) |
| tick_size_final | gamma `orderPriceMinTickSize` at build time |
| taker_fee_rate, fee_exponent, maker_rebate_rate | gamma `feeSchedule`. Politics markets are 0.04, 1 and 0.25. |
| min_order_size | gamma `orderMinSize` (5) |
| rewards_min_size, rewards_max_spread | the reward program in force at the window open. Spread is in cents. |
| log_strike_vs_ul_open | 0 |
| market_idx | the episode index |

## ticks.bin (F = 40, `build.py` columns)

The columns are unchanged. The three `ul_*` columns are 0 because there is no underlying.

## feat.bin (G = 44, `features.py` layout and indices)

The columns that do not apply to elections are filled as follows:

- **No pricer.** `p_model` is the two-sided book mid. When the book is one-sided it is the last
  two-sided mid of the episode, else 0.5. So `mark = p_model`, `gap_c` is about 0, and
  `mark_is_fv = 1` when the book is one-sided.
- **Underlying columns are 0:** `z`, `sdv_bp`, `sigma_1m_bp`, `delta_usd`, `ulz*`, `in_twap`,
  `delta_usd_raw`, `sigma` and `sqrt_v`.
- **Time columns:** `tau_frac` and `log_tau` refer to the window. `len_*` is all zeros for a one-day
  window.
- **Liquidity columns:** `liq_mid`, `liq_comp` and `liq_pi` use the market's own program, which can
  differ by market and by hour:
  - `liq_mid` is the size-cutoff adjusted mid, computed on the **full** recorded ladder with the
    market's `rewards_min_size`. It is -1 when the book is unscorable.
  - `liq_comp` is the competitors' Q_min on the 'hi' bound, using the market's v and min size.
  - `liq_pi` is daily_rate / 1440 on each minute's sample second.

## feat_x.bin (X = 44): the reward features

Each row t uses the reward snapshot taken at or before t (hourly; `selection.json` counts as the
first snapshot) and the book snapshot in use at t.

- **`rw_in_program`, `rw_daily_rate`, `rw_min_size`, `rw_max_spread_c`, `rw_cfg_age_s`:** the
  program in force at t.
- **`rw_adj_mid`, `rw_comp_q_bid`, `rw_comp_q_ask`, `rw_comp_q`, `rw_two_sided_only`:** the market
  without us. These are `reward_terms.adjusted_mid` and `side_scores` computed on the full ladder,
  and they were checked against `reward_terms.py` on 500 of 500 raw snapshots. A whole price level
  is treated as one order for the min-size rule.
- **`rw_sample`, `rw_usd_per_sample`:** the per-minute sample second and its USD pool.
- **`rw_{bid,ask}_px_a{1..6}`:** the price the env would quote for each price head. This
  reproduces `mm_targets` with tick_rule 0 and the other side off. Head 0 means off.
- **`rw_{bid,ask}_score_a{1..6}`:** Polymarket's per-share score `((v - d) / v)^2`, where d is the
  distance in cents from `rw_adj_mid`. It is 0 when d >= v, when there is no quote, or when the book
  is unscorable.
- **`rw_size_ok_{0,1,2}`:** whether size head k (min_order_size x {1, 2, 5}, with size_mult 1) is
  at least `rw_min_size` while the market is in the program. `rw_min_size_mult` is the multiple
  needed.
- **Eligibility.** A quote with price head a and size head k is eligible iff `score_a > 0` and
  `size_ok_k`, and it has reached the minimum order age. When `rw_two_sided_only = 1`, only the
  minimum of the two sides scores.
- **`x_covered`, `x_log_days_to_end`:** whether the row is in recorder coverage, and the time to the
  market's end_date.

## Fills sources

- **`onchain` (preferred: complete, the census source).** Run `--emit-tokens` and then
  `onchain/extract.py --tokens <file> --events OrderFilled --slim`. The script converts the output
  with `onchain/to_trades.convert`, and treats whole UTC days present in the output as complete.
- **`data-api` (fallback).** The script fetches takerOnly=false and takerOnly=true listings and
  classifies them as `data/fetch_trades.py` does.
  - A listing is newest-first and capped at 20,000 rows. When it is truncated, the window must start
    after the oldest row returned.
  - The listing counts as complete up to fetch time minus 120 s.
  - As of 2026-10-15 no selected market was truncated: 354k legs in total over 276 markets, and
    every taker leg has maker legs.
- **Neg-risk conversions are invisible as fills.** This is the same limitation as in the census.

## Side tables (self-describing output)

### `rewards_ts.parquet`: one row per episode and UTC minute

The table covers every minute of every window. A minute is included only if its sample second
falls inside the window, which drops the single-second minute at the window end. Each row holds the
values at that minute's sample second, which is the row where the env's hook pays.

| column | type | meaning |
|---|---|---|
| market_idx | int32 | episode index (`markets.bin` row, `meta.slugs`, `markets_meta.market_idx`) |
| condition_id, slug | str | market |
| minute_ts | int64 | UTC minute start (unix s) |
| sample_ts | int64 | the minute's sample second: one uniform draw per minute from `default_rng([20261015, int(condition_id[2:14], 16), window_open])` |
| in_program | bool | latest reward snapshot taken at or before `sample_ts` lists the market as in the program |
| daily_rate_usd | float | that snapshot's `total_daily_rate` (0 if not in the program) |
| min_size | float | that snapshot's `rewards_min_size` (shares), carried forward from earlier snapshots if missing |
| max_spread_c | float | that snapshot's `rewards_max_spread` v (cents), carried forward the same way |
| snapshot_age_s | float | `sample_ts` minus the snapshot's `taken_at` |
| mid | float | (best bid + best ask) / 2 of the env book, Yes frame; NaN if the book is not two-sided or not valid |
| adjusted_mid | float | size-cutoff adjusted mid (formula below); NaN if unscorable |
| comp_q_bid, comp_q_ask | float | competitors' Q on the Yes bid side and the Yes ask side |
| comp_q_min | float | competitors' Q_min, which the env uses as `liq_comp` |
| two_sided_only | bool | adjusted_mid is outside [0.10, 0.90] |
| usd_per_sample | float | daily_rate_usd / 1440 |
| liq_pi | float | what `feat.bin` `liq_pi` holds at `sample_ts`: usd_per_sample if in_program and scorable, else 0 |
| book_valid | bool | env book valid at `sample_ts` |
| covered | bool | `sample_ts` is inside recorder coverage |
| scorable | bool | book valid and both sides non-empty |

### `markets_meta.parquet`: one row per episode

| column | meaning |
|---|---|
| market_idx | episode index |
| condition_id, slug, question | market |
| episode_id | `"<slug>@<open ISO>"` (= `meta.slugs`, `splits.json` key) |
| day | UTC day of the window open |
| window_start_ts, window_end_ts | unix seconds of the 4 h window the episode belongs to |
| window_start_utc, window_end_utc | UTC timestamps of that window |
| episode_start_ts, episode_end_ts | unix seconds: the episode's actual first and last row (= `markets.bin` open_ts, end_ts) |
| episode_start_utc, episode_end_utc | UTC timestamps of the same |
| start_reason, end_reason | window_boundary / gap / market_end / cutoff / coverage_start / coverage_end |
| split_reason | = end_reason (why the episode stops here) |
| prev_segment_id, next_segment_id | episode_id of the same market's segment across a gap ("" if none) |
| gap_before_s | seconds from the previous linked segment's end (0 if none) |
| split_assigned | the day's split from `--split-days` before the eligibility rule |
| day_eligible | the D16 day rule held (recorder-wide and this market) |
| market_end_ts | the market's `end_date` |
| split, split_name | 0 train, 1 val, 2 test, 3 purged |
| row_offset, n_rows | where the episode sits in the row-major binaries |
| rewards_in_program_at_open, rewards_daily_rate_at_open, rewards_min_size, rewards_max_spread_c | reward program at the window open; `markets.bin` holds the same min size and max spread |
| tick_size, min_order_size, taker_fee_rate, fee_exponent, maker_rebate_rate | from gamma |
| flags | the raw `markets.bin` flags |
| flag_has_trades, flag_has_underlying, flag_election, flag_unresolved, flag_partial, flag_in_program, flag_trades_incomplete | the flags decoded |
| cov_total_gap_s, cov_max_gap_s | coverage gaps inside the window |
| rows_valid, rows_covered, rows_two_sided, rows_in_program, rows_scorable | row counts |
| n_taker_txs | taker transactions in the window |
| n_fill_legs | maker legs written to `fills.bin` |

In `--allow-partial` day windows, the program at the open is "not in program" whenever the open
precedes the first snapshot (selection.json, taken 2026-10-15 20:20:55Z).

### Formulas (`analysis/rewards/reward_terms.py`)

The script recomputes these formulas vectorized on the full recorded ladder of each snapshot. On 500
of 500 raw snapshots it matches `reward_terms.adjusted_mid` and `side_scores`.

- **Score per order:** `S(d, v) = ((v - d) / v)^2` if `d < v`, else 0 (`reward_terms.S`). Here d is
  the order's distance in cents from the adjusted mid, and v is `max_spread_c`.
- **Adjusted mid** (`reward_terms.adjusted_mid`):
  `adjusted_mid = (bid_cut + ask_cut) / 2`. On each side, the cut is the price of the first level
  where the cumulative size from the touch reaches `min_size`, or the last level if the side never
  reaches it. Each snapshot is computed in its own token's frame; a No-book result maps into the
  Yes frame as `1 - mid`, with the bid and ask sides swapped.
- **Side Q** (`reward_terms.side_scores`): `Q_bid = sum over bid levels of S(|adjusted_mid - p| * 100, v) * size * [size >= min_size]`, and likewise for asks. Each price level counts as one order.
- **Competitors' Q_min** (`reward_terms.competitor_q`, 'hi' bound): `min(Q_bid, Q_ask) + |Q_bid - Q_ask| / 3`
  if `0.10 <= adjusted_mid <= 0.90`, else `min(Q_bid, Q_ask)`.
- **Our Q_min** (`reward_terms.q_min`, computed by the env):
  `max(min(q1, q2), max(q1, q2) / 3)` inside the band, else `min(q1, q2)`.
- **Per-sample payout:** `pi = daily_rate / 1440` (one sample per minute, `sample_period_s = 60`).
- **Reward at a sample** (`reward_terms.reward_accrual`): `pi * Q_ours / (Q_ours + comp_q_min)`.

## Gap report (`GAPS.md`, `gaps_per_day.csv`)

Both files are written by every build into its build directory, and by `--gaps-only` into this
directory. Gaps come from `rec/coverage/coverage_<day>.jsonl` and are measured per UTC day.

- **Recorder-wide gaps** are times when no recorder run was up.
  - A run counts from its `start` event to its `stop` event. If the run died, it counts to its last
    event; heartbeats are every 10 s, so a crash gap can be overstated by up to 10 s. For example,
    the 2026-10-16 restart appears as 02:03:16–02:04:16.
  - The day span starts at the first recorder start, and ends at the day's end or the last event.
- **Market gaps** are times a selected market was not live, measured from its first `live` event.
  They include the recorder-wide gaps, plus the re-subscription time after each restart (about
  20–60 s for every market) and WS reconnects.

`gaps_per_day.csv` has one row per (scope, day), plus one row per market-day for `scope = market`:

| column | meaning |
|---|---|
| scope | `recorder` or `market` |
| day | UTC day |
| slug, condition_id | the market ("" for recorder rows) |
| span_start_utc, span_end_utc, span_s | the measured span |
| total_gap_s | sum of all gaps in the span |
| n_gaps | number of gaps |
| n_gaps_gt_max | number of gaps longer than `--max-gap` |
| total_gap_gt_max_s | total seconds of those gaps |
| longest_gap_s, longest_gap_start_utc | the longest gap and its start |
| eligible | total_gap_s <= `--day-max-gap-s` |
| day_complete | the span reaches the end of the day |

`GAPS.md` shows the recorder-wide per-day table, a per-day summary across markets, and a list of
every recorder-wide gap.

## Size

Each row takes 580 B, so one market-day (86,401 rows) is about 50 MB. The default 20 markets come to
about 1.0 GB per recorded day; all 276 recorded markets come to about 14 GB per day. The disk has
27 GB free.

## Smoke test (D16 version, 2026-10-16 about 02:10 UTC; recorded 2026-10-15 20:21 to now, `build_smoke/w4h`)

Run with `--allow-partial --split-days '{"2026-10-15": "train", "2026-10-16": "val"}' --trades-source
data-api` and the defaults (top 20, 4 h windows, 15 s gap split, 600 s minimum):

- **Episodes:** 78, with 394,630 rows and 283 fill legs; the build is 229 MB and `load.verify` passes.
  - By split: 58 train on 10-15, 20 val on 10-16.
  - By (start, end) reason: (gap, gap) 36, (gap, window_boundary) 20, (window_boundary, gap) 20,
    (coverage_start, gap) 2.
  - Lengths: median 7,397 rows, minimum 830. 38 episodes are linked to a previous segment across a
    gap.
  - 96 sub-episodes shorter than 600 s were dropped. These are the 20:21–20:24 run, the
    21:00:36–21:07 run, and the segment after the 02:04 restart, which was still under 10 min.
- **Gaps:** both days are eligible; see `GAPS.md`.
- **Env:** the current env refuses most of these episodes because their lengths vary (no length
  bit). mm3.h is meant to accept any length.

Earlier smoke builds (`w300`, `day`, `w300_wide`) predate D16.

## What P/v3env/mm3.h should read

The v3 env is being built centrally as `P/v3env/mm3.h`. This script does not build an env copy; it
only keeps the binaries compatible.

- **`markets.bin`** (88 B, `MmMarket` unchanged; under D16, episode lengths vary and gaps cut episodes):
  - row_offset, n_rows, open_ts, end_ts and contract_len_s, which can take any value. Episode
    selection must not depend on the 300/900/14400 length bits.
  - split, flags. UNRESOLVED (32) means: settle and mark post-end markouts at the last row's
    `feat[mark]` (or `rw_adj_mid`), never at `winner_up` (which is -1). PARTIAL (64) and
    TRADES_INCOMPLETE (256) mark smoke-only episodes.
  - min_order_size, rewards_min_size, rewards_max_spread (the program at the open, for sizing).
  - taker_fee_rate, fee_exponent, maker_rebate_rate.
- **`book.bin`, `fills.bin`, `fill_index.bin`:** unchanged layouts.
- **`feat.bin` [N, 44]:**
  - Observed columns 0..32 and obs 47/48 (cols 41, 42), as now.
  - `p_model` (33) for the empty-side quote base, and `mark` (34).
  - `liq_mid` (38), `liq_comp` (39), `liq_pi` (40).
  - The underlying slots `z` (4), `sdv_bp` (5), `sigma_1m_bp` (6), `delta_usd` (10) and `ulz*`
    (23–25) are 0 and free to repurpose.
- **`feat_x.bin` [N, 44]** (or `feat.bin` columns 44–87 with `--feat-layout wide`). The per-row
  reward hook must use these columns instead of MM_LIQ_V and MM_LIQ_MIN:
  - `rw_max_spread_c` (3) as v, and `rw_min_size` (2) as the min size.
  - `rw_in_program` (0) and `rw_usd_per_sample` (11); `liq_pi` already contains the product.
  - `rw_two_sided_only` (9) for the band rule.
  - For sizes relative to the min size: `rw_min_size` (2) and `rw_min_size_mult` (39).
  - Candidate reward observations: `rw_bid_score_a1..6` (24–29), `rw_ask_score_a1..6` (30–35),
    `rw_size_ok_0..2` (36–38), `rw_comp_q` (8), `rw_daily_rate` (1) and `x_log_days_to_end` (41).
  - Column indices are listed in `feat_meta.json["extra"]["column_index"]`.
- **For evaluation only:** `rewards_ts.parquet` and `markets_meta.parquet`, never in the step.

## Env changes v3 needs (mm.h was not modified)

1. **Episode length.** `mm_len_bit` and `len_mask` only know 300, 900 and 14400, and the `dt_len`
   index maps any other length to the 4h slot. A one-day window needs a new length bit (or a
   length-agnostic list), plus a dt for it.
2. **Terminal settlement.** `winner_up = -1` because markets are unresolved. The env settles the
   final inventory at `winner` and marks post-end markouts at `winner`. v3 must mark both at the last
   row's `mark` (or `rw_adj_mid`) when flag UNRESOLVED is set.
3. **Liquidity hook.** It hard-codes `MM_LIQ_V = 1.5` and `MM_LIQ_MIN = 50` for our own Q. It
   should read v and min size per row (feat_x `rw_max_spread_c` and `rw_min_size`) or from
   markets.bin. Today our share is scored at 1.5 c against competitors scored at the market's
   3.5–6.5 c.
4. **Order sizes.** `MM_SIZES x min_order_size` gives 5, 10 or 25 shares, but reward min sizes are
   20–200. Size heads relative to `rewards_min_size` are needed, or `size_mult` must be set per
   market.
5. **Features.** The env needs `MM_G = 88` (wide layout), or a second mmap of feat_x.bin, to observe
   or use the reward columns. Observation slots tied to the underlying (z, sigma, ulz*, delta) are 0
   and could be repurposed.
6. **Split guard.** The test guard message names the crypto test day (Aug 24) and should be updated.
   The guard logic itself works unchanged.

## meta.json additions for adapt3 / evaluator3 (D16)

- **`episodes[*]`** now also carries `window_end_ts` (the 4 h window's nominal end), `prev_segment_id`,
  `next_segment_id` and `gap_before_s`, the same values as in `markets_meta.parquet`.
  - Segments are linked when an episode ends at a gap and the next kept episode of the same market starts at one.
  - When a piece between them was dropped for being shorter than `--min-episode-s`, the link spans the dropped
    piece and `gap_before_s` includes it. Example: 20:46→21:07 on Oct 15 gives 456 s, because the
    21:00:30–21:07 piece was under 600 s.
- **`gaps.days`** holds the recorder-wide rows (scope = recorder) of `gaps_per_day.csv`, keyed by UTC day:
  `{n_gaps, total_gap_s, eligible}`.
- Fixed: `--trades-source none` crashed on an empty, object-dtype trades frame.
