# v3 track S: pre-game sports data pipeline (DECISIONS.md D11; worker w1)

| piece | file | status |
|---|---|---|
| Live recorder | `recorder_pregame.toml` (polyrec) | Running since 2026-10-15 20:22 UTC in w1's container. ~290 pre-game moneylines (top leagues by census volume) with kickoff in the next 48 h, rotating new games in every 5 min. Book rows run until kickoff + 5 min; trades and markets are finalized after kickoff. Output goes to `rec_out/` (git-ignored). |
| Compact daily files | `export_daily.py` → `rec_daily/<day>/` (committed hourly) | Outcome-0 book only (outcome 1 is its mirror), top 10 levels, zstd 19. Also trades, markets and coverage. About 40 MB/day. |
| Env binaries | `build_sports.py`, `README_build.md`, `env_sports/` | Raw data (recorder / backfill) becomes ticks/feat/fills/book/markets in the `data/build` schema. Episodes end at kickoff, and inventory settles at the last pre-kickoff mid. That settlement needs the patched env copy in `env_sports/` (fractional settlement, 3 lines). |
| Reward programs | `rewards_snapshot.py` → `rewards/rewards_<day>.json` (git-tracked, small) | Daily CLOB `/rewards/markets/current` snapshot of the recorded markets (rate/day, max spread, min size); `build_sports.py --rewards-dir` turns it into per-row reward columns (feat `liq_*`, `sports_x3.bin`). First run 2026-10-15 21:24 UTC: 5 of 313 recorded markets in a program. Run it once a day. |
| Historical backfill | `backfill_pob.py`, `README_backfill.md`, `pob_openapi.json` | Ready for `POLYORDERBOOKS_API_KEY`. Run `--probe` first, then e.g. `--start 2026-09-16 --end 2026-10-14 --mirror --min-volume 50000`. It is resumable and writes `raw_pob/` in the recorder layout. Trades come from data-api, keyless (`--stage meta` works now). |

## Recorder outage 2026-10-16 04:51 → 2026-10-21 19:25 UTC (D16)

The session hit its weekly usage limit, so the recorder (and kickoff_watch) stopped at
2026-10-16T04:51:11Z (04:49:21Z for the watcher). They were restarted at 2026-10-21T19:25:58Z.
The outage is logged in `rec_out/GAPS.txt` (tracked copy: `kickoff/snapshot/RECORDER_GAPS.txt`),
so `build_sports.py` splits episodes there. Oct 16–21 have no or partial data and are ineligible
for val/test under D16. Since the restart the horizon is **14 d**: about 1,280 markets with
kickoff 0–14 days ahead (it was 7 d / 700).

## v3 data rule: episode cut (recorded 2026-10-15 ~21:15 UTC, before any v3 result)

- **The episode cut is the scheduled `gameStartTime`** (gamma). Polymarket's `clearBookOnStart` wipe
  fires at the scheduled time, not at the actual start. For example, val-flyr-mibrf was wiped at
  21:00:00.1 but went `running` at 21:08:15, 496 s late. So the book itself pins the scheduled cut.
- **The actual start from the sports websocket** (`kickoff_watch.py`) is used only to label games and
  to exclude early starts, where a game started before the scheduled time and the cut would contain
  in-play rows. It never moves the cut later.
- **Esports** (cs2, lol, dota2, val): gamma `endDate` is `gameStartTime + 6 h`. Since 21:14 UTC the
  recorder therefore uses a separate source with offsets shifted by −6 h, so book rows stop 5 min
  after the scheduled start instead of recording up to 6 h of in-play books.

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

**Update 2026-10-15 ~22:40 UTC (D15)**: the grid is now `no_quote` + {join, inside} x {10, 100 sh} x pull_s
{0, 300, 900, 1800, 3600} (pull = quotes off pull_s before kickoff; a val-selected hyperparameter, not a build
cut). Headline `pnl_flat`: the inventory left is crossed out at the pull (or the episode end if earlier),
marked at the mid there minus half the spread x |inv|; `pnl` is the env's kickoff-mid mark. Default
`--max-hours 24` (4 h segments tiling back from kickoff); every segment carries its majority time-to-kickoff
bucket (>24h, 6-24h, 1-6h, 1h-30m, 30-5m, 5-0m) from build_sports meta.json, and summary.md reports overall,
P&L flat by pull_s x bucket, by bucket and by league. The text below describes the first (pull {0, 5} min)
version.

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
- **build_sports.py bug** (fixed in sports-2): with `--allow-open`, a market whose data-api trade
  history is empty crashes `api_trades` (`'DataFrame' object has no attribute 'outcome_idx'`). Seen in a 12-market test run (`--max-markets 12`, the first slugs alphabetically).
  Real mode uses `--no-fetch-trades` and does not hit it.

### Results (real kicked-off games, run 2026-10-16 00:41 UTC)

These are exploratory runs on recorder data. No split is fixed yet and nothing here is a test
result. Full tables are in `baseline_out/summary.md`, and per-market rows in
`baseline_out/per_market.csv`.

**Sample.**
- 16 markets / 16 games whose kickoff was at least 30 min in the past and whose trades the recorder
  had finalized:
  - 9 NHL;
  - 3 CFB;
  - 1 NBA;
  - 3 outcome markets of one Argentine soccer game.
- Each game has only 2.6–3.6 h of book, because the recorder started at 20:22 UTC. So every episode
  falls in the **1–6 h** bucket. The finer buckets (1 h–30 m, 30–5 m, 5–0 m) are covered only
  through the pull grid, inside these 4 h episodes.
- Settings: central fills (headline), 1 s latency, Q 100.
- "P&L flat" crosses any leftover inventory at the pull, at half the spread.

**P&L flat, USD summed over the 16 markets, central fills:**

| strategy | pull 0 | 300 s | 900 s | 1800 s | 3600 s |
|---|---|---|---|---|---|
| join, 10 sh | −5.92 | −6.06 | −6.70 | −3.27 | +0.25 |
| inside, 10 sh | −5.51 | −5.65 | −6.30 | −2.96 | +0.24 |
| join, 100 sh | −6.28 | −6.36 | −4.91 | −4.19 | +0.38 |
| inside, 100 sh | −6.97 | −7.05 | −5.56 | −4.90 | −0.49 |
| no_quote | 0 | | | | |

**Reading.**
- **No fixed quoter beats no-quote.** The pull-3600 variants come out at about $0, but only because
  they barely trade: 2–3 fills per market, so they fail D1.3.
- Every variant that passes D1.3 (≥ 5 fills per market) loses $5.5–7 in total.
- **Edge per share.** Spread capture is 0.50 ¢ per share (the books are 1 tick wide almost all the
  time). Adverse selection outweighs it: the markout is −0.4 to −0.5 ¢ at 30 s and −0.5 to −0.7 ¢
  at 5 min. Rebates (about 0.9 / 4 USD at 10 / 100 shares) do not close the gap.
- **The loss sits near kickoff.** The more of the last hour a strategy quotes, the more it loses:
  pull 0 / 300 is worst and pull 3600 best.
  - That is the opposite of the D15 claim that the last 30 min is the best bucket.
  - It is one evening of 16 mostly NHL games, at the passive end of the queue (the env joins behind
    the displayed size). Treat it as one noisy data point, not evidence against D15.
- **By league:**
  - NBA (1 game) has the volume: 40–49 fills per market, and it loses −2.6 to −5 USD.
  - NHL fills about 1 per market.
  - The soccer (arg) three-way market loses −1.6 to −2.5 USD, mostly from inventory.
- **What would change the picture:** more days, and the > 24 h / 6–24 h buckets that D14/D15 rate
  highest. The recorder's 7 d horizon collects these from now on; the PolyOrderBooks backfill
  covers them historically.

### evaluator3 comparators, `--pull-grid`, env3 (run 2026-10-16 00:50 UTC)

**Setup.**
- Build: `baseline_out/build_seg0`, the same 16 games, `--seg 0` (one episode per game, up to 24 h,
  ending at kickoff), adapt3 sports-2.
- Env settings: `obs_size=80 liq_rewards=1 size_mode=1 Q=1000 flat_c=0.5 dt_other=1`.
- Split: `baseline_out/split3_explore.json`, an exploratory all-val map, not pre-registered.
- Ledger: `trials.jsonl`, tags `w1_s_comparators_oct15` (17 rows from a run killed by a container
  restart) and `w1_s_comparators_oct15_rerun`.
- Full output: `baseline_out/comparators3.json`.

**P&L, USD summed over 16 markets (fills per market in brackets):**

| comparator | fills | pull 0 | 300 | 900 | 1800 | 3600 |
|---|---|---|---|---|---|---|
| sym_touch | central | −1.87 (4.0) | −1.94 (3.8) | −1.08 (3.7) | −3.11 (2.7) | +1.75 (1.2) |
| sym_touch | pess | −5.06 | −5.06 | −4.79 | −4.79 | −2.89 |
| mid_value | central | −0.98 (2.9) | −1.00 | −0.33 | −3.04 | +1.12 (1.2) |
| as_mid | central | −2.49 (3.3) | −2.52 | −1.44 | −3.84 | +0.93 (1.1) |
| no_quote | | 0 | 0 | 0 | 0 | 0 |

**Reading.**
- This matches `baseline_pregame.py`: no comparator makes markets (all are below 5 fills per
  market) and none is positive on any pull before 3600 s.
- The pull-3600 positives come from about 1 fill per market and vanish under pessimistic fills.
- Rebates of about +1.9 USD (sym_touch at pull 0) do not cover the trading loss of −3.8 USD.

### Future baseline runs (D16, agreed with the coordinator 2026-10-16 02:30 UTC)

Under D16, recording gaps split episodes. `baseline_pregame.py` cannot follow that because it
runs on `env_sports/` (the v2 core), which only accepts 300/900/14400-row episodes, so every
4 h tile that spans a gap is dropped. On the 2026-10-16 NHL data that was 10 of 11 tiles.

Future pre-game baselines therefore use env3 with whole-window episodes:
```
python build_sports.py --rec rec_out --out B --seg 0 --max-hours 24 --min-cover 0.05 --no-fetch-trades --splits S --verify
python ../../v3env/adapt3.py B
cd ../../train3 && python evaluator3.py comparators --data-dir ../analysis/v3_sports/B --split-map S3 \
    --env3-kw obs_size=80 liq_rewards=1 size_mode=1 Q=1000 flat_c=0.5 dt_other=1 --variants central_1s pess_1s \
    --pull-grid --ledger ../analysis/v3_sports/trials.jsonl --tag w1_<name>
```
`evaluator3` chains gap pieces, carrying inventory across a gap as a forced pull, and applies the
D16 val/test exclusions. `baseline_pregame.py` stays as the 4 h-tile, env_sports reference for the
2026-10-16 run above.

### Kickoff results (2026-10-16 00:45 UTC)

From `kickoff/kickoffs.csv` and `kickoff/detect.csv`; their tracked copies are in `kickoff/snapshot/`.

**Actual starts (sports websocket).**
- No game started early. Every confidently timed start came **after** the scheduled
  `gameStartTime`:
  - cs2: +5 s / +5 s / +174 s;
  - lol: +416 s;
  - nhl: +318 s and +425 s;
  - val: +496 s.
- Most NHL, NBA and CFB games are only `upper_bound`: the websocket first reported them already in
  progress, 4–14 min after the scheduled time.

**Book wipe (`clearBookOnStart`).**
- Found **at the scheduled time** for nba-tor-nyk (+0.4 s), nba-okc-hou (−8.8 s) and cfb-gas-old
  (−7.7 s), as on Oct 15 at 21:00.
- **No wipe within +10 min** for 7 NHL games, nfl-sea-den and two other CFB games.
- **Outliers:**
  - cs2-fly-offcho was wiped 602 s *before* its scheduled time;
  - an MLB NRFI market was emptied 1,117 s after its scheduled time.
- So the wipe is not a reliable kickoff signal for every league. The v3 rule (cut at the scheduled
  `gameStartTime`) stands, and in this sample no early start needed excluding.

## Kickoff detection

Two tools find the ACTUAL game start. They exist because `clearBookOnStart` wipes the book when the game
begins, and that can happen before or after the scheduled `gameStartTime`. An early start means the
pre-game book already contains in-play quotes. A late start means the episode ends while the game is still pre-game.

| tool | what it does | output (`kickoff/`) |
|---|---|---|
| `kickoff_watch.py` (long-running) | Logs every message of `wss://sports-api.polymarket.com/ws` with its local receive time, starting a new file each UTC day and reconnecting with backoff. Polls gamma `/markets?slug=..` (20 per request) every 30 s for every recorded game, from scheduled −20 min to +40 min, and on up to +3 h while the game is not yet live. The recorded slugs come from `rec_out/coverage` "live" events. Every 5 min it re-derives `kickoffs.csv`. | `ws_<day>.jsonl`, `gamma_<day>.jsonl`, `meta.json` (slug → gameStartTime, endDate, event slug, gameId), `kickoffs.csv`, `watch.log` |
| `detect_kickoff.py` (offline) | Rebuilds each token's book from the recorder raw log (`book`/`rest_book` snapshots + `price_change` deltas). It looks for the wipe in [sched − 3 h, sched + 1 h]: a `book` snapshot with empty ladders, the book emptied on both sides, or a mass cancel (≥ 80 % of the levels on each side removed within 10 s). The books parquet (both ladders empty) is the fallback. It compares the wipe with the schedule and with kickoff_watch. | `detect.csv` (+ `detect_cache.json`) |

Run them:

    nohup nice -n 10 ../../../.venv/bin/python kickoff_watch.py >> kickoff/watch.log 2>&1 & echo $! > kickoff/watch.pid
    python kickoff_watch.py --derive        # rebuild kickoffs.csv from the logs
    python detect_kickoff.py                # rec_out -> kickoff/detect.csv, prints the games already past kickoff
    python detect_kickoff.py --synthetic    # self-test: early empty-book wipe, late per-level wipe, no wipe, a pre-game 60 % pull (must not fire)

How each tool reports an actual kickoff:
* `kickoffs.csv` takes the first live observation (ws `live`/`status` running/InProgress, or gamma `live`/`ended`/`elapsed>0`). It brackets that with the last not-live observation; ws sends `status: not_started, live: false` around the scheduled time. Confidence is
  `high` (bracket ≤ 60 s), `medium` (≤ 10 min), `low`, or `upper_bound` (already live when first seen, e.g. games that were running when the watcher started).
* `detect.csv` reports the wipe time and kind (`book_empty` | `emptied` | `books_empty` = strong; `mass_cancel` = medium; `mass_cancel+emptied`
  = a mass cancel that ends with an empty book, timed at its first cancel).

**Observed 2026-10-15 (first live test).** `val-flyr-mibrf` and `dota2-amaru-skyrea` were both scheduled for 21:00:00.
Their books collapsed at 21:00:00.06–0.32 (val: about 64 levels down to 3+3 within 0.3 s, then MMs re-quoted at wide spreads).
The games themselves started later: the ws sent `status: not_started` at 21:00:15, and val went `running` at 21:08:15 (+496 s), while dota2 had not started at 21:09.
**The clearBookOnStart wipe is driven by the scheduled gameStartTime, not by the actual start.** The wipe gives the cut time
the book actually saw, while kickoff_watch gives the game start. An early start (game before wipe) appears only in kickoff_watch, as the docs warn.
A late start leaves the wipe at the scheduled time, followed by a re-quoted pre-game book until the real start.

**Coverage limit.** `recorder_pregame.toml` writes book rows until gamma endDate + 300 s and stays subscribed until endDate + 600 s. Traditional sports have endDate = gameStartTime. **A start more than ~10 min late leaves no wipe in `rec_out`.** Such a start is visible only to kickoff_watch, whose ws runs all the time and whose gamma polls run until the game shows live, or for at most 3 h. Early starts are fully visible in the recorder data.
Esports are different: their gamma endDate is gameStartTime + 6 h (all 80 cs2/dota2/lol/val markets on 2026-10-15). The recorder therefore keeps up to 6 h of in-play esports book as well, and late esports starts appear in its data. build_sports.py already cuts at gameStartTime, not endDate.

**Hook for build_sports.py.** Not applied: build_sports.py is unchanged. Add 3 lines. Before the metadata loop (`# ---- metadata, kickoff, eligibility`):

    from detect_kickoff import actual_kickoffs; AK = actual_kickoffs([Path(r) for r in args.rec], HERE / "kickoff")

and right after `ko_true = ko`:

    if slug in AK and AK[slug]["t"] < ko: ko = ko_true = int(AK[slug]["t"])   # early start: cut at the observed wipe/start

Only EARLY starts move the cut. A late start leaves the cut at gameStartTime, which is where the wipe happens anyway.
`actual_kickoffs()` returns `{slug: {"t", "source": "wipe_<kind>" | "watch_ws" | "watch_gamma", "confidence", "wipe", "game_start", "scheduled", "delta_s"}}`.
`t` is the end of the pre-game regime: the earlier of the wipe and the game start observed by kickoff_watch (confidence ≥ medium).
A wipe counts as high when the book emptied or ≥ 20 levels were mass-cancelled. A smaller mass cancel counts only when nothing else exists, and its confidence is low. Use `kickoff_watch.load_actual_kickoffs()` alone to skip reading the raw log, which takes about 10 s of CPU per 35 min of raw log (about 7 min per recorded day).
