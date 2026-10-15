# Pre-game sports env: spec of the changes to `mm.h` (track S, DECISIONS.md D11)

Worker w2, 2026-10-15. This is a spec, not code. It lists what the crypto replay env (`ENV.md`) needs
to replay pre-game sports markets from listing to kickoff. It reuses everything that is not
crypto-specific: the fill models, the latency/match-delay FIFO, accounting, the reward hook, the Log and the
split guard. Evidence comes from `REPORT.md` (book-free rule backtests, Sep 24 – Oct 7), from
`fair_value_anchor.md`, from w8's `analysis/sports/out/` and from the census.

## 0. What the data says the env must capture

- **Books are 1 tick wide and very deep.** On w8's recording, the pre-game spread is 1¢ ~100% of the time.
  The median depth within 1¢ is about 100k shares; NFL is in the millions. The mid barely moves (≈ 0
  changes/min). Our quote is therefore always at the touch or behind it, and fills depend almost entirely
  on **queue position**. The central fill model's queue tracking (displayed size at our price, pro-rata
  cancels) is required, not optional. Prints through the touch are rare sweeps and are toxic
  (REPORT.md §3).
- **Takers are delayed about 1 s (`secondsDelay` = 1 on ~97% of game markets); makers are not.** A maker
  can cancel within that second. The env should model this as a taker delay `D_take` that our
  cancels can beat, the opposite of crypto, where the taker delay was 50 ms.
- **Every resting order is wiped at kickoff** (`clearBookOnStart`). Holding into the game is not allowed
  (D11), and the edge to settlement is ±700 bps of noise per week (census).

## 1. Episode

- **One episode = one condition (outcome-0 frame), from `start` to kickoff.**
  - `start = max(listing, kickoff − H_max)`. `H_max` defaults to 24 h: more than 24 h out, the spread
    is 2.4¢ but adverse selection is 1.4¢ (census §3c) and volume is thin.
  - The decision interval is `dt = 30 s` beyond 2 h to kickoff and `dt = 2 s` inside 2 h. The step is
    variable, so episodes are about 2–3k steps, as for the 4h crypto markets.
  - A random start offset in the first 20% of the window is kept for training, as in crypto.
- **Terminal.** At `kickoff − T_pull` (default 5 min) quotes are forced off. At kickoff the inventory is
  flattened at the last pre-kickoff mid minus a crossing cost of half the spread plus one tick of slippage
  (`flat_c`, default 0.5¢/share). That step's reward uses this value instead of the mark.
  `kickoff_mode = 1` instead marks the inventory at the mid with no cost (for attribution only).
  Settlement is never used.
- **Kickoff time:** gamma `gameStartTime`. Each market keeps a `kickoff_err` flag, set when the first
  in-play print or the book wipe comes before `gameStartTime` (an early start; census caveat).
  Rows after the first sign of play are dropped. The env treats the earliest such row as kickoff.
- **Lengths.** `len_mask` is replaced by a `league_mask` (bit per league: soccer, NFL, CFB, MLB, NHL,
  NBA, tennis, esports, other) and a `type_mask` (moneyline, spreads, totals, other props). The same
  per-split lists in `MM_DATA` apply.
- **Splits (D11):** by game date. The oldest ~60% of days are train, the next ~20% val, the newest ~20%
  test, plus a forward test. `split = 2` keeps the `MM_FINAL` guard.

## 2. Data files (same layouts, new builder `data/build_sports.py`)

- `ticks/book.bin`: L2 at 1 s on change, from PolyOrderBooks (key) or our recorder, top 10 levels
  rather than 5. At ~1¢ spreads, 5 levels cover only 5¢. The flags are kept.
- `fills.bin` / `fill_index.bin`: on-chain maker legs (`onchain/extract.py`). The taker delay `D_take` is
  measured, not assumed: the median gap between the matched order's creation (calldata) and the block.
- `markets.bin` adds `kickoff_ts`, `listing_ts`, `league`, `market_type`, `event_id` (games share news),
  `fee_rate`/`rebate_rate` (sports v3 0.05/15%; NFL/CFB moneyline 0.03/15%; NFL/CFB props 0/0),
  `tick`, and the reward config (`rewards_max_spread`, `rewards_min_size`, `rate_per_day`).
- **No `p_model`.** There is no external pricer. An optional anchor column `p_ext` (Pinnacle no-vig or
  Smarkets mid) is filled where available; otherwise it is NaN and an observation flag says so
  (fair_value_anchor.md).

## 3. Observation changes (keep size 50; replace the crypto-only slots)

| obs | crypto now | sports |
|---|---|---|
| 0 | p_logit (pricer) | micro_logit: logit(book micro-price) / 3 |
| 2–3 | gap_c, dgap5_c (pricer − mark) | ext_gap_c: (p_ext − mid) in cents / 2, and ext_present |
| 4–6 | z, sdv_bp, sigma_1m_bp | realised mid vol over 10 min and 2 h; trades/min (log) |
| 7–9 | tau_frac, log_tau, in_twap | **ttk_log** = log1p(s to kickoff) / log1p(86400); **ttk_lt_5m**, **ttk_lt_1h** flags |
| 10 | delta_usd | league/news phase: **lineup_window** (soccer: kickoff −75 to −45 min; NFL inactives: −90 min) |
| 23–25 | Binance returns | sweep indicators: a print through the touch in the last 10/60/300 s (the toxic events of REPORT.md §3) |
| 26–28 | length one-hot | market type one-hot (moneyline / spread or total / prop) |
| new | – | league one-hot is too wide for 50 slots; use a 4-dim learned embedding index, or group as {soccer, US majors, college, other} |
| 40–41 | queue fraction | kept. It is the most important agent feature here. |
| 42–43 | rebate at our price | kept, with sports rates |
| new | – | **reward_score_share**: our share of the market's reward score in the last sampled minute |

Every feature stays causal (`<= t`). `p_ext` is used only with its own timestamp.

## 4. Actions

- Keep MultiDiscrete `{7, 7, 3}`. Pre-game books are one tick wide, so "improve 1–2 ticks" means
  crossing or locking: post-only pulls it back. The bid/ask head becomes
  {off, join, 1 behind, 2 behind, 4 behind, 10 behind, reward-edge}.
  `reward-edge` = the furthest price that still earns liquidity rewards (`max_spread` from the
  adjusted mid).
- Size head: {1x, 5x, 20x} × `min_order_size`. With 100k-deep queues, a 5-share order earns
  nothing. Reward eligibility needs `min_size` 20–100 (and 1000 for NFL/CFB first-half spreads).

## 5. Reward

`r = scale · (ΔW − inv_penalty · inv² · p(1−p) · Δt / H_ref − close_inv_coef · |inv| · ramp(ttk))`

- **ΔW** includes the **liquidity-rewards term**. The existing `liq_rewards` hook is used with the sports
  config:
  - `S = ((v − s)/v)² · b · size` for each of our orders within `max_spread` of the adjusted mid and with
    size ≥ `min_size`;
  - two-sided rule as in crypto;
  - one sample per minute; the competitor score `Q_comp` comes from the replay book (top 10 levels);
  - the payout is `rate_per_day / 1440` × our share. The "in-game multiplier" `b` is 1 pre-game.

  Sports pools are small: about $49k/day over 7.8k markets, $1–50 per market (docs_notes.md). The term
  matters mainly for esports and NFL first-half spreads. Rates come from the CLOB rewards snapshot of each
  replay day, so the recorder must snapshot `/rewards/markets/current` daily.
- **Maker rebates:** in the reward at each fill, as now. The pool is pro rata to fee-equivalent,
  `rebate_rate · fee_rate · p(1−p)` per share.
- **reward_mode 2** (fill markout, D5) with `markout_h` capped at kickoff. It is kept as the main
  shaping because it gives credit for adverse selection at the fill.
- **close_inv_coef with ramp on time to kickoff:** a penalty on |inventory| over the last
  `ramp_s` (default 30 min). This replaces `tau_frac`, because kickoff, not the window length, is what
  matters. The penalty is shaping only and is not in the P&L.
- Evaluation and selection use the flat-at-kickoff P&L (`pnl_kick`) and the D1.3/T5/T6 detectors.
  T6 counts kickoff-flatten P&L as inventory P&L.

## 6. Fill model changes

- Use the central model (queue = displayed size at our price, pro-rata cancels, through-prints fill us).
  Add the taker delay `D_take` (≈ 1 s): a taker print at u was submitted at about u − D − D_take.
  Our cancel that reaches the exchange before u − D wins.
- The participation cap (`part_cap`) is kept: at most 25% of taker volume per minute.
- Validation: the env's fills for a join-the-touch quoter should reproduce the on-chain maker-leg
  markouts of the census, about +0.3¢/share at 30 min before rewards.

## 7. Multi-market (phase 2)

Moneyline, spread and totals of one game are correlated. The census suggests a delta-flat book per game
to cut the to-kickoff variance. Phase 2 makes the episode a game, with one agent per condition and a
shared inventory penalty on game-level net exposure (outcome-0 deltas mapped by a simple model,
e.g. moneyline ≈ spread). Not needed for a first replay.
