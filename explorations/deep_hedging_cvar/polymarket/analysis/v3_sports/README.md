# v3 track S: pre-game sports data pipeline (DECISIONS.md D11; worker w1)

| piece | file | status |
|---|---|---|
| Live recorder | `recorder_pregame.toml` (polyrec) | Running since 2026-10-15 20:22 UTC in w1's container. ~290 pre-game moneylines (top leagues by census volume) with kickoff in the next 48 h, rotating new games in every 5 min. Book rows run until kickoff + 5 min; trades and markets are finalized after kickoff. Output goes to `rec_out/` (git-ignored). |
| Compact daily files | `export_daily.py` → `rec_daily/<day>/` (committed hourly) | Outcome-0 book only (outcome 1 is its mirror), top 10 levels, zstd 19. Also trades, markets and coverage. About 40 MB/day. |
| Env binaries | `build_sports.py`, `README_build.md`, `env_sports/` | Raw data (recorder / backfill) becomes ticks/feat/fills/book/markets in the `data/build` schema. Episodes end at kickoff, and inventory settles at the last pre-kickoff mid. That settlement needs the patched env copy in `env_sports/` (fractional settlement, 3 lines). |
| Historical backfill | `backfill_pob.py`, `README_backfill.md`, `pob_openapi.json` | Ready for `POLYORDERBOOKS_API_KEY`. Run `--probe` first, then e.g. `--start 2026-09-16 --end 2026-10-14 --mirror --min-volume 50000`. It is resumable and writes `raw_pob/` in the recorder layout. Trades come from data-api, keyless (`--stage meta` works now). |

Pipeline: `rec_out/` and/or `raw_pob/` → `python build_sports.py --rec rec_out --rec raw_pob --out build --splits splits.json --verify`.
D11 fixes the splits by game date once the backfill range is known, and before any evaluation.

**The recorder dies with w1's container.** For days of unattended recording, deploy polyrec with
this config on a VPS (`recorder/README.md` "Deploy").

## Pre-game baseline

`baseline_pregame.py`: no-RL fixed-rule quoters on recorded games, played through the patched env
(`env_sports/`, the env's own fill model; not a standalone backtest). One command builds
`rec_out` with `build_sports.py` and runs the grid:

```bash
source ../../../.venv/bin/activate
python baseline_pregame.py --rec rec_out --out baseline_out      # after ~00:30 UTC Oct 16, ~1-3 min, 1 core
# outputs: baseline_out/summary.md (tables), summary.csv, per_market.csv, results.json (checks), build/, build.log
python baseline_pregame.py --rec rec_out --out /tmp/bt --test --slugs '^(nhl|nba)-'   # mechanics test on open games
```

**Eligible games** (real mode): kickoff (gamma `gameStartTime`) at least 30 min ago **and** the slug has
finalized trades in `rec_out/trades`. Trades come from the recorder only (`--no-fetch-trades`).
Esports slugs in the recorder (cs2/dota2/lol/val) mostly have a `gameStartTime` before the recorder
started (series already live), so their books are all "after kickoff" and the build drops them.

**Build**: dt 1 s, one 4 h segment ending at kickoff (`--seg 14400 --max-hours 4`), `--min-cover 0.05`
(the recorder started 20:22 UTC, so most of tonight's games have 0.5-4 h of book; rows before the first
book are no-book rows with no quotes and no legs). Settlement = the last pre-kickoff mid.
**Splits:** none are fixed yet (D11). Every game is put in split 1 only so the env opens it; this
is exploratory recorder data, not a val or test assignment. No test split is touched.

**Env**: `env_sports` (fractional settlement), latency 1 s, decisions every 1 s, Q = 100 shares, `part_cap`
0.25, liquidity rewards off. Fill model **central** is the headline. **Pessimistic** is the sensitivity check. One env
per market from offset 0 (the `baselines/runner.py` pattern), with the per-market Log read at the terminal.

**Strategy grid (fixed, 9 entries, nothing tuned)**: `no_quote`; `join_s{10,100}_pull{0,5}` (join the best bid
and the best ask); `inside_s{10,100}_pull{0,5}`. The inside strategy quotes one tick inside both touches
when the spread is >= 3 ticks. At exactly 2 ticks, quoting both sides one tick inside would lock the book, so
only the side that reduces |inventory| improves, and both join at zero inventory. At 1 tick it joins.
All strategies quote only when the book is two-sided. They pull both sides N min before kickoff (N=0 means at
kickoff, the episode end), using the obs-8 kickoff clock. Inventory is capped at 100 by the env. Terminal
inventory is marked at the last pre-kickoff mid. Sizes: 10 sh = size head 1 (2 x min size 5); 100 sh = head 2 x
`size_mult` 4. `per_market.csv` has the effective size. The baselines in `baselines/baselines.py` are not used:
their pull rule reads crypto `tau`.

**Fees and rebates (assumption)**: our quotes never take, so we pay no fees. The rebate is
`rebate_rate x fee_rate x q x p(1-p)` per maker fill, from gamma `feeSchedule` in markets.bin:
`sports_fees_v3` 0.05 x 15%, NFL/CFB moneylines 0.03 x 15% (`analysis/sports/out/docs_notes.md`).
In reality the rebate is our pro-rata share of the market's daily pool, by fee-equivalent, with a $1 minimum payout.
The env credits it as if our share of the pool equals our share of the fee-equivalent maker volume.

**Reported** per market and by league/overall: P&L = spread capture (vs mid at the match row) + 60 s
markout + inventory P&L (to the kickoff mark) + rebates; fills/market, fill volume, markouts
per share (maker-signed, mid-to-mid) at 30 s (env Log) and 5 min (`markout_300`). The 5-min markout is rebuilt
from the per-step inventory change with the env's own convention: match row tau = u - D, mid at tau + h,
the settlement past the end. Also reported: |inventory| at kickoff, and the D1.3 / T6 / T5 flags from
`train/evaluator.protocol_flags`.

### Mechanics test (2026-10-15 ~20:57 UTC, `--test`, NOT results)

These are open games via `build_sports --allow-open`: each game is cut at its last book row, which plays
"kickoff", so there is about 33 min of book per game. Trades come from data-api. The sample is 20 markets
(11 NHL, 2 NBA, 3 CFB, 3 ARG, 1 NFL) and 1090 maker legs. The whole run takes 49 s, with the build at 23 s.
- **Checks**:
  - The rebuilt 30 s markout equals the env Log's `markout_30` to within 2e-8 USD in every run, which validates
    the 5-min markout rebuild.
  - The obs-8 kickoff clock decodes to sports_feat `seconds_to_kickoff` within 0.001 s.
  - Every pull-5 run has 0 fill volume in the last 30 s.
  - `no_quote` gives P&L 0.
  - Both build verifies pass.
- **Central, 20 markets**: join_s10_pull0 makes 17 fills for +$0.10 in total (spread +0.30, mo60 -0.23,
  inv -0.07, rebates +0.10). join_s100_pull0 makes 21 fills for +$0.43. Markouts run -0.2 to -0.5 c/share at 30 s.
  All fills are in the 2 NBA games, and the NHL/CFB/ARG/NFL books saw no fills in 33 min.
  D1.3 fails overall (about 1 fill per market).
- **Pessimistic**: 1-5 fills in total and about -$0.1 to -0.3. **Optimistic**: 100-126 fills and +$5 to +20.
  The 20-60x gap between optimistic and central is the queue: at the touch, the displayed queue absorbs
  the at-price prints.
- `inside` = `join` in practice: 99.5% of two-sided seconds have a 1-tick spread, and 0.5% have 2 ticks.

### Caveats

- **Sample size**: one evening of recorder data, 0.5-4 h before kickoff, a few dozen games. Read
  the numbers as mechanics and order of magnitude, not edge.
- **Kickoff**: the schedule time (`gameStartTime`); see README_build "Kickoff accuracy".
- **No impact**: a 100-share quote joins the queue as if invisible, and nobody reacts to it (ENV.md section 4).
- **Rebates**: modelled as a share of the pool (above); the $1 daily minimum is ignored.
- **Gamma refetch**: `build_sports.py` refetches gamma for every slug that has no `closed` gamma cache, at
  about 0.5 s per slug. In `--test`, filter with `--slugs`, because otherwise all ~300 slugs are fetched (~2.5 min).
- **build_sports.py bug** (not edited, it is w1's file): with `--allow-open`, a market whose data-api trade
  history is empty crashes `api_trades` (`'DataFrame' object has no attribute 'outcome_idx'`). Seen in a 12-market test run (`--max-markets 12`, the first slugs alphabetically).
  Real mode uses `--no-fetch-trades` and does not hit it.

### Results (real kicked-off games)

To be filled from `baseline_out/summary.md` after the rerun.
