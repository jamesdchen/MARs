# Track E baseline: no-RL liquidity-reward farming on the recorded midterm books

`baseline_farm.py` (plain numpy/pandas, one core, about 4.5 min) simulates a fixed quoting rule on every
recorded market. It uses the recorder's books (`rec/books`), the hourly reward snapshots (`rec/rewards` plus
`sel/selection.json`), the coverage log and data-api trade legs. The variants were fixed before any result was
seen, nothing was tuned, and every variant is reported.

**Run reported here:** window 2026-10-15 **20:21:26 to 22:16:00 UTC (1.91 h)**. That is everything recorded with
books and fills complete at run time (22:18). It covers 300 markets: 276 from the start, and the 24 late
markets from 20:46, since `rec/coverage` decides which minutes are quotable. 236 taker transactions fell
in the window, on only 66 of the 300 markets. Re-run with `python baseline_farm.py [--start ..] [--end ..]`.

## Method

- **Quotes.** Every UTC minute t, per market, one Yes bid and one Yes ask:
  - Price: adjusted_mid ∓ f·max_spread, rounded to the 1¢ tick toward the mid. The order is post-only, so it
    is never at or through the opposite best.
  - Size: k × rewards_min_size.
  - Variants: f ∈ {0.25, 0.5, 0.75}, where f = 0.5 is the spec's "max_spread / 2", and k ∈ {1, 2}.
  - The adjusted mid, competitors' Q per side and the program in force are imported from
    `build_elections.py`: `reward_stats`, `side_ladder`, `load_reward_cfg`, `cfg_at`,
    `coverage_segments` and `trades_data_api`. The import works and nothing was copied. The scorer is
    `analysis/rewards/reward_terms.py` (`S`, `q_min`, `competitor_q` 'hi').
  - The book is the latest uncrossed snapshot at or before t, in the Yes frame (the No book is mirrored),
    and only inside its coverage segment. A minute is quoted only if [t, t+60] is covered and the market
    is in its program.
- **Rewards.** One sample per minute, at build_elections' seeded sample second.
  - Our orders count only if they are still at least min size and at least 30 s old (`moas` = 30; a
    `moas0` sensitivity is included).
  - The adjusted mid includes our orders (cutoff → max(cutoff, our bid) or min(cutoff, our ask)).
  - Q_side = ((v−s)/v)² × size. Our side scores combine with `q_min`: single-sided /3 inside [0.10, 0.90],
    two-sided only outside.
  - Competitors are scored with `competitor_q` 'hi' on the visible book. Our own orders are not counted as
    competitors.
  - $ = share × daily_rate / 1440.
  - **Check:** 400 of 400 random snapshots match `reward_terms.reward_accrual` on the raw full ladder exactly,
    to a share difference of 4e-16. The queue lookup also matches exactly.
- **Fills.**
  - Source: data-api maker legs, in the Yes frame, with the aggressor taken from the taker leg.
  - A leg can fill an order only if its block time is more than 3 s after the order was placed. Block time
    trails the match by about 2 s; this is measured, and the book moved 2 s before the block in the one big
    sweep checked.
  - **Back of queue.** A leg at our exact price first eats the displayed size that was at that price when we
    joined. Volume through our price fills us, because the level was exhausted.
  - An unfilled order whose price is unchanged at the next minute keeps its queue position and age. Any other
    order is replaced at the back of the queue.
  - Inventory cap ±5 × min_size: a side is quoted only if a full fill stays within the cap.
- **P&L.**
  - spread = s·(mid_pre − px), where mid_pre is the top-of-book mid at block − 3 s.
  - markout_h = s·(mid(t+h) − mid_pre) for +1 m, +5 m, +1 h and to the end of the window. A horizon past
    the end uses the end mid.
  - net = rewards + spread + markout_end. This equals rewards + cash + inventory marked to the end mid.
- **Capital.** The per-minute maximum of (position at mid) + (collateral of the resting orders, where inventory
  covers the opposite order), summed over markets. This is only the collateral the quotes need. It does not
  include a buffer for the cap: a full ±5 × min_size inventory would be about 5× larger.
- **Comparators.** no_quote ($0). `*_nofills` is the same quotes with fills switched off: the rewards-only
  upper bound.
- **Extra variants and sensitivities.**
  - `*_join` never improves the visible touch: bid = min(target, best bid), ask = max(target, best ask).
    It was added after the first look, at the coordinator's request.
  - `frontq`: front-of-queue fills.
  - `moas0`: no minimum order age.

## Headline (1.91 h, 300 markets; $ are totals over the window unless marked /h)

| variant | rewards $ (inside touch) | spread $ | markout 1m / 5m / 1h / end $ | net $ | net $/h [30-min block CI] | capital $ | net %/h of capital [block CI] [market CI] | ×24 %/day | markets rewarded / filled | fills |
|---|---|---|---|---|---|---|---|---|---|---|
| no_quote | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 |
| f0.25_x1 | 50.30 (44.65) | 3.94 | -8.15 / -7.95 / -9.16 / -9.16 | 45.07 | 23.61 [20.46, 25.18] | 19,544 | 0.121 [0.105, 0.129] [0.074, 0.178] | 2.90 | 300 / 13 | 16 |
| f0.25_x1_nofills | 50.35 (44.70) | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 50.35 | 26.37 [22.58, 28.06] | 19,117 | 0.138 [0.118, 0.147] [0.090, 0.194] | 3.31 | 300 / 0 | 0 |
| f0.25_x2 | 61.34 (50.41) | 6.44 | -14.65 / -14.85 / -17.47 / -17.47 | 50.32 | 26.35 [23.39, 29.38] | 38,712 | 0.068 [0.060, 0.076] [0.039, 0.100] | 1.63 | 300 / 13 | 18 |
| f0.25_x2_nofills | 61.51 (50.57) | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 61.51 | 32.21 [27.52, 34.54] | 38,234 | 0.084 [0.072, 0.090] [0.060, 0.112] | 2.02 | 300 / 0 | 0 |
| f0.5_x1 | 42.42 (34.79) | 4.11 | -6.27 / -5.68 / -5.98 / -5.98 | 40.55 | 21.24 [18.16, 23.18] | 18,647 | 0.114 [0.097, 0.124] [0.066, 0.171] | 2.73 | 298 / 3 | 4 |
| f0.5_x1_nofills | 42.45 (34.83) | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 42.45 | 22.23 [19.14, 23.46] | 18,647 | 0.119 [0.103, 0.126] [0.073, 0.178] | 2.86 | 298 / 0 | 0 |
| f0.5_x2 | 49.67 (35.93) | 7.20 | -11.16 / -10.57 / -11.17 / -11.17 | 45.70 | 23.94 [19.75, 27.15] | 37,293 | 0.064 [0.053, 0.073] [0.039, 0.095] | 1.54 | 298 / 3 | 6 |
| f0.5_x2_nofills | 49.69 (35.95) | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 49.69 | 26.02 [22.34, 27.62] | 37,293 | 0.070 [0.060, 0.074] [0.045, 0.100] | 1.67 | 298 / 0 | 0 |
| f0.75_x1 | 36.03 (32.08) | 1.75 | -2.67 / -2.08 / -2.38 / -2.38 | 35.41 | 18.54 [16.01, 20.14] | 18,093 | 0.102 [0.088, 0.111] [0.055, 0.158] | 2.46 | 297 / 2 | 2 |
| f0.75_x1_nofills | 36.07 (32.11) | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 36.07 | 18.89 [16.33, 19.82] | 18,093 | 0.104 [0.090, 0.110] [0.058, 0.163] | 2.51 | 297 / 0 | 0 |
| f0.75_x2 | 39.63 (32.54) | 0.95 | -2.67 / -1.98 / -2.68 / -2.68 | 37.91 | 19.85 [16.66, 22.19] | 36,186 | 0.055 [0.046, 0.061] [0.030, 0.084] | 1.32 | 297 / 2 | 3 |
| f0.75_x2_nofills | 39.63 (32.54) | 0.00 | 0.00 / 0.00 / 0.00 / 0.00 | 39.63 | 20.76 [17.89, 21.85] | 36,186 | 0.057 [0.049, 0.060] [0.033, 0.087] | 1.38 | 297 / 0 | 0 |
| f0.25_x1_join | 16.23 (0.58) | 4.18 | -6.81 / -6.81 / -7.99 / -7.99 | 12.42 | 6.50 [4.86, 8.35] | 19,404 | 0.034 [0.025, 0.043] [0.014, 0.052] | 0.80 | 293 / 9 | 11 |
| f0.25_x2_join | 26.98 (0.77) | 7.09 | -12.91 / -12.90 / -15.44 / -15.44 | 18.64 | 9.76 [6.83, 13.32] | 38,478 | 0.025 [0.018, 0.035] [0.008, 0.041] | 0.61 | 293 / 9 | 13 |
| f0.5_x1_join | 11.61 (0.31) | 3.40 | -5.20 / -5.20 / -5.50 / -5.50 | 9.51 | 4.98 [3.04, 6.69] | 18,590 | 0.027 [0.016, 0.036] [0.013, 0.041] | 0.64 | 291 / 2 | 3 |
| f0.5_x2_join | 19.03 (0.37) | 6.49 | -10.09 / -10.09 / -10.69 / -10.69 | 14.83 | 7.77 [4.21, 10.76] | 37,180 | 0.021 [0.011, 0.029] [0.009, 0.032] | 0.50 | 291 / 2 | 5 |
| f0.75_x1_join | 6.24 (0.24) | 0.80 | -1.60 / -1.60 / -1.90 / -1.90 | 5.14 | 2.69 [1.43, 3.61] | 18,043 | 0.015 [0.008, 0.020] [0.004, 0.027] | 0.36 | 290 / 1 | 1 |
| f0.75_x2_join | 9.91 (0.28) | 0.00 | -1.60 / -1.50 / -2.20 / -2.20 | 7.71 | 4.04 [1.68, 5.65] | 36,086 | 0.011 [0.005, 0.016] [0.002, 0.020] | 0.27 | 290 / 1 | 2 |
| f0.5_x1_frontq | 42.41 (34.79) | 5.32 | -7.47 / -6.87 / -7.32 / -7.32 | 40.41 | 21.16 [18.13, 23.08] | 18,649 | 0.113 [0.097, 0.124] [0.066, 0.169] | 2.72 | 298 / 5 | 7 |
| f0.5_x1_moas0 | 45.08 (37.09) | 4.11 | -6.27 / -5.68 / -5.98 / -5.98 | 43.22 | 22.64 [19.86, 24.38] | 18,647 | 0.121 [0.107, 0.131] [0.072, 0.182] | 2.91 | 298 / 3 | 4 |

Notes on the confidence intervals:

- The block CI resamples 30-minute clock blocks. There are only 5 blocks, and the first one is short (8.6
  min), so this CI understates the uncertainty.
- The market CI resamples markets with their capital. It is the honest one here, because rewards are
  concentrated: in f0.5_x1 the top 5 markets earn 40% of rewards and the top 20 earn 81%.

**Comparison with the other worker's 0.12%/day.**

- f0.5_x1 makes **0.114%/h of capital** [market CI 0.066, 0.171]. Taken at face value that is about 2.7%/day,
  but the capital definitions differ. Ours is only the collateral of resting min-size quotes, about $62 per
  market.
- The join (never improve the touch) version makes **0.027%/h** [0.013, 0.041], about 0.64%/day on the same
  basis.

## Fill-logic check (coordinator's question 3)

- **The logic is right.** In a synthetic book (0.13 / 0.88, our bid 0.48 inside), the results are:
  - A taker selling at the old best of 0.13 fills our 0.48 bid for 20.
  - A taker buying at the old ask of 0.88 fills our 0.53 ask.
  - A sell at 0.50, which does not reach our bid, does not fill.

  In other words, legs at or through the old touch count as through our improved price
  (scratchpad test `filltest.py`, logic in `run_market`).
- **The low fill count is real for the recorded flow.**
  - The 20 markets where more than 50% of our scored samples improve the touch earn $33.7 of the $42.4 in
    f0.5_x1. In 1.91 h they had **2 taker transactions in total, about 1.0 per hour across all 20**. Only 2
    of the 20 markets traded at all, and **0** of those transactions traded at or through our inside quote.
    The top markets are books like 0.13 / 0.88 (`closer-governor-race-*`) with no trades.
  - The other 280 markets had about 122 taker transactions per hour together, on 64 markets. Our quotes
    there sit 1–4¢ from the mid, behind tight 0.4–1¢ touches. They were filled 4 times (f0.5_x1) to 16 times
    (f0.25_x1).
- **This flow is counterfactual.** Nobody trades a 75¢-wide book, and a tight quote placed into one would
  attract the flow it currently lacks, informed flow included. History cannot price the fills or the adverse
  selection of quotes that set the price. The **`*_inside` part of rewards (82% of the base variant) is
  therefore not validated** by this backtest. The `*_join` rows are the defensible lower bound: 0.011–0.034%/h.
- **Most of the pool is not reachable.** The $500+/day markets are 25 markets carrying $13.8k/day, more than
  half of the selected pool. They earn only $0.56/h in f0.5_x1, because competitor Q there is large.

## By reward-rate bucket (f0.5_x1 → f0.5_x1_join, $/h)

| rate $/day | markets | pool $/day | rewards $/h | net $/h | net %/h of capital | join: net $/h |
|---|---|---|---|---|---|---|
| 20–50 | 173 | 5,530 | 12.22 | 11.55 | 0.142 | 3.54 |
| 50–100 | 88 | 4,578 | 9.36 | 9.36 | 0.207 | 1.12 |
| 100–250 | 10 | 1,440 | 0.07 | 0.07 | 0.010 | 0.07 |
| 250–500 | 4 | 1,032 | 0.01 | 0.01 | 0.003 | 0.01 |
| 500+ | 25 | 13,833 | 0.56 | 0.24 | 0.005 | 0.24 |

**Top markets (f0.5_x1).** All have 0 taker transactions in the window, and all are inside the touch:

- `closer-governor-race-{georgia-or-wisconsin, new-york-or-pennsylvania, pennsylvania-or-illinois}` and
  `closer-senate-race-iowa-or-maine`: $3.6 each over 1.91 h. Each pays $50/day, and we take about 88% of it.
- `republican-senate-odds-hit-40-by-october-31`: $2.57.
- `will-amcha-yisrael-...-knesset`: $2.02.

**Top markets (f0.5_x1_join).** `republican-senate-odds-hit-40-by-october-31` $1.45 and
`democratic-senate-odds-60-62-on-october-23` $0.75, then four of the
`will-trump-say-*-georgia-october-16` markets at about $0.5 each.

## What a $/day number assumes

The `×24` column is just the hourly rate × 24. Reading it as $/day assumes all of the following:

1. **Stationary program rates.** The selected pool already moved from **$26,412/day (20:22) to $25,753/day
   (21:22)**, −2.5% in one hour, and all 300 markets are still in a program. The `closer-*` and
   `*-on-october-23` programs end with their markets.
2. **Unchanged competitor Q.** The inside-touch rewards exist because nobody quotes there. Any second farmer
   halves them.
3. **The window is representative.** It is 20:21–22:16 UTC, a US-evening window. Night and weekend flow and
   quoting differ. With only 5 half-hour blocks, intraday variation is unmeasured.
4. **Our quotes move nobody.** In reality competitors re-quote around us, and takers react to an improved touch.
5. **Fills stay as rare as recorded.** See the fill check above. This is the weakest assumption for the
   inside-touch markets.

## Caveats

- **Very little data:** 1.91 h and 236 taker transactions. The markout totals come from 4–18 fills, so they are
  anecdotes, not adverse-selection estimates. For comparison, the census measures 0.5–0.6¢/share for elections.
  In f0.5_x1 the fill markout is −5.98 on 244 shares, about −2.5¢/share, consistent with being picked off on
  sweeps.
- **Rewards are estimates.**
  - Competitor Q uses the visible book with the 'hi' bound, and a price level counts as one order.
  - The real scorer samples randomly. We use one seeded second per minute.
  - The `moas` = 30 s reading is an inference.
  - In 1% of samples our orders move the adjusted mid (0.8–5% across variants). There the competitor Q is still the value at the
    no-us mid.
  - Polymarket's payout rules (minimum payout, epoch accounting) are not modelled.
- **Queue position:** the displayed size at join is all ahead of us, and cancellations ahead are ignored. This
  is conservative. `frontq` shows the effect: 7 fills against 4.
- **data-api completeness.** No listing was truncated, and the listing counts as complete up to fetch − 120 s.
  Neg-risk conversions are invisible as fills, the same limitation as the census. 270 of the 300 markets are
  neg-risk.
- **Tick:** 1¢ is assumed everywhere. A sub-cent tick near 0.04 or 0.96 is not modelled.
- **Capital:** quote collateral only. See the method section.
- **v3 env limitations** (coordinator's point 4). The C env's liquidity hook:
  - does not let our own orders move the adjusted mid or the competitor Q;
  - has one order per side;
  - hard-codes v = 1.5¢ and min size = 50 (BUILD.md, env change 3).

  This baseline does move the adjusted mid with our orders, but still also uses one order per side and leaves
  competitor Q unchanged. Env results for the inside-touch markets will differ from these numbers and should
  be compared with care.

## Files

- `analysis/v3_elections/baseline_farm.py`: the script.
- `analysis/v3_elections/BASELINE.md`: this report.
- `analysis/v3_elections/baseline_summary.csv` and `.json`: the headline per variant, with the CIs and run
  metadata (window, reward-pool snapshots, trade counts). These are small and not ignored.
- `analysis/v3_elections/baseline_ci.csv`: bootstrap rows, including $/h per 30-minute block.
- `analysis/v3_elections/baseline_out/`:
  - `per_market.csv`: every market × variant.
  - `by_rate_bucket.csv`.
  - `fills.csv`: every simulated fill with its markouts.
  - `run_meta.json`.
  - `final_run.log`.
  - `cache/data_api/`: trade cache, git-ignored via `.gitignore`.
