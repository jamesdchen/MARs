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
| `--window-s` | 86400 | Episode length. It must divide 86400, and windows are aligned to UTC midnight. The current env only accepts 300, 900 and 14400. |
| `--split-days` | none | JSON file or inline JSON, `{day: train / val / test / purged / exclude}`. A day that is not listed goes to split 3 (purged). Days marked `exclude` are not built. **No default assigns test.** |
| `--cutoff` | 2026-11-03T00:00Z | Windows must end by this time. Election night is out of scope (D11). |
| `--max-gap`, `--max-total-gap` | 5 s, 15 s | The coverage rule from `recorder/merge.py`. |
| `--allow-partial` | off | Keeps windows whose recorder or fills coverage is partial, with flag PARTIAL or TRADES_INCOMPLETE. Use it only for smoke tests. |
| `--markets`, `--top-n`, `--min-rate` | all | Restricts the build to a subset of markets (see the size note below). |
| `--trades-source` | required | `onchain` reads `onchain/extract.py` output (`--trades-path`); this is the census source. `data-api` uses `/trades?market=<condition id>` and is cached. `file` reads a trades.parquet in the `data/raw` schema. `none` builds without fills. |
| `--feat-layout` | split | `split` writes feat.bin as [N, 44], which the current env reads, plus feat_x.bin as [N, X]. `wide` writes feat.bin as [N, 44 + X], which needs MM_G = 88 in the env. |
| `--max-gb` | 20 | Refuses to build anything larger than this. |

## Episodes

- **One episode is one market in one window** (`markets.bin` row). The window is
  `[open, open + window_s]`, giving `window_s + 1` one-second rows, as in `data/build.py`.
  `seconds_to_close` counts down to the end of the window, not to resolution.
- **A window is kept only if all of these hold:**
  - The recorder's coverage log covers it: no single gap over 5 s, total gaps under 15 s, and the
    coverage segment that holds the open began at or before the open. This is `merge.py`'s rule.
  - It ends before both the market's `end_date` and `--cutoff`.
  - Its day is not excluded.
  - The fills source is complete over the window.
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
| contract_len_s | window_s. The env's length bit and the len_* one-hot only know 300, 900 and 14400. |
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

## Size

Each row takes 580 B, so one market-day (86,401 rows) is about 50 MB. All 276 recorded markets come
to about **14 GB per recorded day**. The disk has 27 GB free, so a full 18-day build of every market
is about 250 GB and does not fit. You will need to choose one of these:

- Build a subset of markets (`--top-n` or `--markets`; for example, 20 markets is about 1 GB per day).
- Have the v3 env read an event-driven or sparse book.

## Smoke test (2026-10-15, about 15 minutes recorded, `build_smoke/`)

| build | episodes | rows | fills | load.verify | C env |
|---|---|---|---|---|---|
| `w300` (5-minute windows, full coverage) | 580 | 174,580 | 16 legs | OK | opens and steps with len_mask 1; liq hook pays with size_mult 10 |
| `w4h` (`--allow-partial`, top 20) | 18 | 259,218 | 25 | OK | opens and steps with len_mask 4 |
| `day` (`--allow-partial`, top 3) | 1 | 86,401 | 19 | OK | refused: no length bit for 86400 |
| `w300_wide` (top 10) | 24 | 7,224 | 5 | OK | refused: MM_G = 44 by design |

In the `day` build, only 1 of the top 3 markets by rate had been recorded so far: 24 selected
markets were still pending subscription.

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
