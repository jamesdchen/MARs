# Track S: pre-game sports quoting rules, book-free on on-chain fills (w2)

> **CORRECTION (2026-10-15 ~22:30 UTC): see `RECONCILE.md`.**
>
> **Superseded:** the at-touch pool (−26/−33 bps) and every rule result in §2–3. They come from a book-free
> fill model that values all volume at or beyond our level at our price. That credits sweep volume, which in
> reality filled deeper makers at better prices, to the touch. It is biased against quoting by about 70 bps.
>
> **Corrected estimate:** census-definition legs at their own price, with the 30 min mark capped at kickoff.
> - Pre-game makers earn +31 bps [12, 50] to kickoff on Sep 24 – Oct 7.
> - At the touch: +20 [1, 37].
> - One tick behind: +81 [34, 124].
>
> The gap with the census is definitional, not a regime effect. The census's instability across weeks comes
> from its in-play 30 min mark.
>
> **Still valid:** the fair-value anchor survey, ENV_SPEC.md, and the observation that window and league
> choices don't transfer across weeks.


Worker w2, 2026-10-15. This answers the coordinator's track-S assignment (D11):
1. simple pre-game quoting rules, backtested without a book, with edge per $ and a CI;
2. a fair-value anchor;
3. the env spec.

**Data:**
- on-chain `OrderFilled` for every market, **Sep 24 – Oct 7 2026** (`onchain/extract.py --all --slim`, 14 days,
  0 range errors);
- gamma metadata for 385k tokens.

Pre-game = fills before `gameStartTime`. That gives 1.13M taker events on 44k conditions in 9.9k games.

**Hold-out:** every rule choice uses only **Sep 24 – Oct 2** (the selection period). **Oct 3 – 7** is a
check period, also older than the last 7 days. Oct 8 onward (the census week and later) was not read.

## TL;DR

1. **No simple pre-game quoting rule makes money on these two weeks, before rewards.**
   - 0 of 2,880 rules (pass 1) and 0 of 4,800 rules (pass 2, windows near kickoff) are positive on the
     selection period, under either fill model.
   - The rule with the best to-kickoff P&L, chosen on the selection period, still loses there and on the
     check period:
     - pass 2 (conservative fills): −$61 (−57 bps) on selection, then **−$2,063 (−1,740 bps [−3,291, −480])**
       on check;
     - pass 1 (conservative fills): −$222 (−647 bps), then −$21 (−165 bps, CI ±1,000).
   - The "best" rules are always the least active ones: they quote 3¢ off the micro-price, so only
     sweeps fill them.
2. **The census pre-game edge does not replicate on the two weeks before it.**
   - The pool of maker volume at the touch, valued like the census, makes
     **−26 bps [−43, −11] at 30 min and −33 bps [−55, −14] to kickoff**, over all 14 days.
   - It is positive at 30 min on only 2 of the 14 days.
   - The census measured +64 bps [1, 130] at 30 min on Oct 8–14. Its CI barely clears 0, and on these
     earlier weeks the pool is about 90 bps lower.
   - The spread is there (≈ 87–89 bps of notional ≈ 0.45¢/share). Adverse selection takes all of it,
     and more by kickoff.
3. **Where the pool is positive:**
   - **The last 5 min before kickoff:** +35 bps [12, 58]. Selection +32 [12, 50], check +39 [−11, 85].
   - **NHL at 30 min:** +32 [16, 48]. To kickoff it is −65 [−203, 21].
   - **NFL is ≈ 0.** It is the cleanest big league: +8 [−1, 18] at 30 min, −5 [−23, 10] to kickoff.

   As strategies, these don't survive a realistic per-quote size:
   - joining the touch only in the last 5–10 min loses −57 to −75 bps on selection and −92 to −166 bps on
     check (optimistic fills, 0.5¢ flatten cost);
   - with no flatten cost it is still −10 bps [−22, 1] on selection and −58 [−139, 1] on check.

   The pool's positive number is carried by large at-touch prints, which a quoter with a 50-share
   size doesn't get.
4. **Fills away from the touch are toxic.** Pre-game books are one tick wide and about 100k shares deep
   (w8's `live_books.txt`). A quote 1–3¢ away is filled only by sweeps, which lose 300–2,000 bps
   within 60 s. Pulling quotes when the trade micro-price is stale (more than 30 s since the last
   print) is the most useful filter in every top rule.
5. **Rule choices don't transfer from the selection period to the check period,** except as "quote
   less".
   - The global grid's ranking is stable between the two periods (rank correlation 0.97 in pass 1,
     0.93 in pass 2), because every rule is ranked by how little it trades.
   - Per-league selection overfits. For example, "other" goes from +$167 [CI above 0] on selection to
     −$3,436 on check, and CFB from +$4 to −$238. NFL (+$93 → +$420) and soccer (+$160 → +$50) stay
     positive, but each comes from 40–760 sweep fills with CIs of ±1,000 bps or more.
6. **Fair-value anchor** (`fair_value_anchor.md`):
   - No free external price beats Polymarket's own mid as a quote centre. Pinnacle no-vig, Smarkets,
     DraftKings (ESPN) and Kalshi all agree with it within about 0.5¢ in the big leagues.
   - Use an anchor only as a guard rail: pull quotes when it disagrees with Polymarket by more than about
     1.5–2¢.
   - **Keyless sources that work:**
     - Pinnacle guest API: live, but cached up to 15 min.
     - Smarkets: live.
     - ESPN core API: historical DraftKings open and close.
     - SportsbookReview line history: US leagues.
     - football-data.co.uk: soccer closing odds.
   - **Not available:** free timestamped Pinnacle or Betfair history.
7. **Env spec** (`ENV_SPEC.md`):
   - episode = 24 h before kickoff (or listing) to kickoff, flattened at kickoff;
   - the queue-aware central fill model is mandatory;
   - a 1 s taker delay (sports `secondsDelay`);
   - time-to-kickoff and sweep features;
   - a liquidity-rewards term through the existing hook.

**Recommendation:**
- **Quoting rule:** none. A simple rule does not earn the spread pre-game on these weeks.
- **v3 sports RL:** proceed only if the L2 replay shows that queue-front fills behave better than the
  at-touch pool. The pool is −33 bps to kickoff here.
- **Where to look first:** the last 5–30 min before kickoff, NFL and NHL moneylines, and quoting large
  size at the touch, since large at-touch prints were favourable. Check these on the next unseen week
  before building anything.
- **Rewards:** sports reward pools are small (about $49k/day over 7.8k markets). A rewards term would
  change these conclusions only for esports and NFL/CFB first-half spreads.

## 1. Method

`build_events.py` (DuckDB):
- Every fill of a sports market with a `gameStartTime` is put into the outcome-0 frame.
- The trade micro-price is the census definition: the mean of the latest taker-buy and taker-sell
  prints, at the last block before the event.
- A **taker event** is one transaction × condition × direction. For each event the script keeps:
  - its maker legs (prices and sizes);
  - the volume traded at or beyond each candidate quote level (d = 0, 0.5, 1, 1.5, 2, 3, 5¢ from the
    micro-price, rounded outward to 1¢);
  - the micro-price at +60 s, +5 min and +30 min, capped at kickoff;
  - the last micro-price before kickoff;
  - league, market type, fee and rebate rates.
- Nothing is read past the window end, Oct 8 00:00 UTC.

`rules.py`:
- **The rule.** Quote both sides at the micro-price ± d with size S and an inventory cap Q per condition.
  Pull T min before kickoff, quote only within A h of kickoff, and pull when the last print is older
  than `age` s.
- **Fills.** An event fills min(S, volume at or beyond our level, cap room) at our price, with sequential
  inventory per condition (numba). Two fill models bound reality:
  - `ge`: at-price volume fills us, as if we were at the front of the queue (optimistic);
  - `gt`: only volume strictly beyond our price fills us (conservative, close to reality with
    100k-deep queues).
- **P&L to kickoff (`pnl_kick`).** Fills are marked at the last pre-kickoff micro-price. Add rebates
  (sports v3: 0.05 × 15% × p(1−p)) and subtract a flatten cost of 0.5¢/share of net inventory per
  condition. Edge per $ = P&L / cost of tokens bought.
- **CIs.** Poisson bootstrap over gamma events (games), 1,000 replicates.

Grids:
- pass 1: d {0, 0.5, 1, 2, 3}, S {10, 50, 250}, Q {100, 500, 2,500}, T {0, 5, 15, 60} min,
  A {∞, 24, 6} h, age {30, 300, ∞} s, fill {gt, ge}; 2,880 rules per period;
- pass 2 (added after pass 1 and the selection-period touch table): A {∞, 24, 6, 2, 0.5} h,
  T {0, 1, 5, 15} min;
- per-league selection in pass 1: 8 leagues × 1,440 rules;
- the post-hoc last-5/10-min check: `out_pass2/last5min.txt`.

All of these are looks at the selection period. The check period was used only to evaluate.

## 2. The at-touch pool (`touch_edge.py` → `out/touch_edge.csv`)

This is maker volume at the touch level, valued at the touch, census-style. It is the pool, not a
strategy. Cells are bps of notional at 30 min | to kickoff, with [95% CI] over games.

| group | sel 30 min | sel kickoff | check 30 min | check kickoff | all 30 min | all kickoff | notional $k (all) |
|---|---|---|---|---|---|---|---|
| all=all | -28 [-51, -8] | -40 [-75, -15] | -22 [-50, -2] | -21 [-56, +9] | -26 [-43, -11] | -33 [-55, -14] | 202098 |
| league=soccer | -46 [-95, +5] | -30 [-75, +8] | -52 [-109, -7] | -38 [-97, +17] | -48 [-87, -4] | -32 [-67, -2] | 52288 |
| league=esports | -37 [-105, +17] | -110 [-263, +0] | +13 [-15, +41] | +37 [-24, +112] | -11 [-53, +19] | -34 [-130, +36] | 37012 |
| league=tennis | -139 [-206, -81] | -153 [-262, -63] | -27 [-132, +71] | -11 [-152, +117] | -92 [-156, -41] | -97 [-178, -19] | 10751 |
| league=nfl | -0 [-12, +11] | -7 [-39, +13] | +20 [+5, +35] | -2 [-30, +22] | +8 [-1, +18] | -5 [-23, +10] | 60079 |
| league=cfb | -19 [-34, -4] | -56 [-90, -25] | -93 [-191, -37] | -113 [-217, -50] | -46 [-82, -23] | -77 [-119, -46] | 20048 |
| league=mlb | +7 [-22, +34] | -26 [-68, +10] | +42 [-26, +99] | +20 [-83, +118] | +17 [-11, +43] | -14 [-52, +22] | 12279 |
| league=other | -134 [-220, -37] | +305 [-55, +904] | -480 [-911, +23] | -328 [-1214, +253] | -340 [-614, -55] | -93 [-558, +335] | 3252 |
| league=basketball | +19 [-17, +47] | +1 [-43, +32] | -62 [-210, +19] | -88 [-485, +141] | +3 [-31, +31] | -16 [-84, +39] | 1647 |
| league=nhl | +42 [+22, +58] | -6 [-46, +33] | +24 [+0, +46] | -116 [-359, +34] | +32 [+16, +48] | -65 [-203, +21] | 3860 |
| league=baseball | -202 [-353, -126] | -121 [-408, +18] | -263 [-567, -29] | -89 [-276, +117] | -224 [-377, -128] | -109 [-269, +3] | 172 |
| ttk=6-24h | -29 [-50, -9] | -37 [-79, +3] | -13 [-36, +8] | -5 [-102, +133] | -23 [-38, -7] | -26 [-71, +26] | 35791 |
| ttk=2-6h | -63 [-136, -14] | -48 [-93, -7] | -16 [-64, +23] | -41 [-119, +49] | -46 [-94, -9] | -45 [-87, -4] | 39787 |
| ttk=30m-2h | -53 [-104, -15] | -100 [-207, -35] | -49 [-91, -13] | -35 [-106, +33] | -51 [-87, -24] | -74 [-143, -24] | 48270 |
| ttk=5-30m | +15 [-25, +56] | +15 [-25, +56] | -18 [-94, +51] | -18 [-95, +51] | +1 [-38, +36] | +1 [-39, +36] | 47873 |
| ttk=>24h | -50 [-82, -23] | -83 [-185, +18] | -148 [-272, -54] | -213 [-366, -56] | -71 [-106, -41] | -100 [-186, -9] | 11556 |
| ttk=0-5m | +32 [+12, +50] | +32 [+12, +50] | +39 [-11, +85] | +39 [-11, +85] | +35 [+12, +58] | +35 [+12, +58] | 18821 |
| sports_type=moneyline | -21 [-51, +6] | -27 [-58, +2] | -4 [-41, +26] | +17 [-26, +59] | -14 [-38, +8] | -9 [-35, +16] | 121719 |
| sports_type=totals | -19 [-35, -5] | -32 [-63, +5] | -36 [-77, -5] | -88 [-182, -24] | -26 [-42, -12] | -53 [-90, -21] | 26045 |
| sports_type=spreads | -9 [-25, +5] | -19 [-51, +3] | -19 [-41, -2] | -58 [-96, -29] | -13 [-26, -2] | -32 [-57, -13] | 40396 |
| sports_type=soccer_team_totals | -62 [-107, -25] | -95 [-197, -3] | -85 [-113, -58] | -122 [-165, -71] | -70 [-100, -42] | -104 [-180, -43] | 702 |
| sports_type=both_teams_to_score | +33 [+16, +49] | -20 [-64, +19] | +25 [-15, +65] | +35 [-35, +102] | +30 [+13, +48] | +0 [-37, +38] | 1085 |
| sports_type=soccer_exact_score | +22 [+6, +34] | +12 [-5, +28] | -528 [-1145, -116] | +15 [+1, +27] | -135 [-292, -25] | +13 [-0, +25] | 1626 |
| sports_type=child_moneyline | -399 [-678, -143] | -532 [-914, -175] | +2 [-58, +70] | -31 [-128, +61] | -209 [-391, -53] | -294 [-553, -83] | 2488 |
| sports_type=map_handicap | -41 [-156, +48] | -121 [-490, +111] | +44 [+13, +66] | +26 [-27, +65] | +1 [-58, +50] | -49 [-223, +73] | 3892 |
| sports_type=first_half_totals | -45 [-92, -8] | -72 [-125, -30] | -26 [-144, +80] | -43 [-149, +51] | -36 [-100, +19] | -59 [-117, -7] | 522 |
| sports_type=soccer_first_half_team_totals | -50 [-90, -19] | -87 [-159, -5] | -61 [-106, -19] | -91 [-146, -42] | -55 [-85, -31] | -89 [-137, -41] | 97 |

By day, 30 min: positive on 2 of 14 days (Oct 4 +7, Oct 6 +9). The worst days are Sep 30 (−118), Oct 3 (−88)
and Sep 24 (−68).

## 3. Rules

Selected on the selection period by to-kickoff P&L; the check period is evaluation only.

| pass | fill | selected rule (d¢, S, Q, T min, A h, age s) | sel fills | sel P&L | sel bps [CI] | check fills | check P&L | check bps [CI] |
|---|---|---|---|---|---|---|---|---|
| 1 | gt | 3, 10, 2,500, 60, 6, 30 | 829 | −$222 | −647 [−978, −360] | 310 | −$21 | −165 [−1,396, 974] |
| 1 | ge | 3, 10, 100, 60, 6, 30 | 1,365 | −$214 | −383 [−577, −193] | 531 | −$19 | −89 [−883, 607] |
| 2 | gt | 3, 250, 2,500, 15, 0.5, 30 | 350 | −$61 | −57 [−523, 328] | 354 | −$2,063 | −1,740 [−3,291, −480] |
| 2 | ge | 3, 250, 500, 15, 0.5, 30 | 506 | −$156 | −98 [−399, 167] | 466 | −$1,254 | −797 [−1,906, 114] |
| ref | gt | join (d 0), 50, 500, 0, ∞, ∞ | 135k | −$65.8k | −389 [−413, −364] | 73k | −$35.4k | −411 [−464, −363] |
| ref | ge | join (d 0), 50, 500, 0, ∞, ∞ | 368k | −$54.9k | −110 [−122, −98] | 197k | −$30.8k | −120 [−149, −97] |

Joining the touch only in the last 30 min (Q 100, ge, selection period):
- spread +77 bps, rebate +27 bps;
- 60 s markout leaves +15 bps;
- at 30 min (capped at kickoff) it is −16 bps;
- −52 bps to kickoff after a 36 bps flatten cost.

The flatten cost is material at these margins. It is reported per rule in `grid_*.csv` (`flat_cost`).

Breakdowns of the selected rule by league, market type and time to kickoff:
`out/breakdown_selected_rule.json`. Per-league selections: `out/rules_summary.json`
(`per_league_selected`).

## 4. Caveats

- **No book.** Queue position is bracketed by `ge`/`gt`, not modelled. Fill size is capped by the
  printed volume, so there is no market impact. Quotes re-centre instantly on the trade micro-price,
  which a real quoter, seeing the book, would do better. The micro-price is a trade proxy: its median
  age is about 60 s, and it is stale in quiet markets.
- **Kickoff** is gamma `gameStartTime`, with no early-start detection. The 0.5¢ flatten cost is an
  assumption (half the 1¢ spread). The bps reported without flatten cost bound it.
- **Rewards** are not included. Sports pools are small and rarely reach the 5–250-share quotes simulated here.
- **Two weeks of late September – early October.** NFL is in season and European soccer in midweek
  rounds. The census week (Oct 8–14) was better for makers, which is a reason to treat any single week
  as noisy. Its CI was [1, 130].

## Files

- `build_events.py`: on-chain fills → pre-game taker events (`cache/events.parquet`, git-ignored).
- `rules.py`: the rule simulator and grids.
  - Pass 1: `out/`. Pass 2: `out_pass2/`.
  - Each holds `grid_sel.csv`, `grid_check.csv` and `rules_summary.json`.
- `touch_edge.py`: the pool table, `out/touch_edge.csv`.
- `gamma_tags.py`, `categories.py`: copies of the census scripts. The `gamma_tags.py` copy uses
  `as_completed` and shorter retries.
- `fair_value_anchor.md`: the odds-source survey (subagent, verified requests).
- `ENV_SPEC.md`: the env changes for a pre-game sports replay.

Reproduce:
1. `onchain/extract.py --start 2026-09-24 --end 2026-10-07 --all --events OrderFilled --slim`
2. `gamma_tags.py --fills ../../onchain/data/all/fills`
3. `build_events.py`
4. `rules.py`, run twice with the pass-2 arguments in §1 for the second run
5. `touch_edge.py`
