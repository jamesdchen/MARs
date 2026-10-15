# build_sports.py: pre-game sports books -> env binaries (Track S, D11)

`build_sports.py` turns normalized raw sports data (recorder `out_dir`, or the PolyOrderBooks
backfill in the same layout) into the files of `data/build`. The record layouts are identical and
the feat.bin column layout is the same frozen one, so `env.py`, `mm.h` and `train/envs.py` load it
with `data_dir=...`. The script never edits shared code.

```bash
source ../../../.venv/bin/activate                      # numpy, pandas, pyarrow, orjson (polyrec)
python build_sports.py --selftest                       # synthetic end to end incl. both envs (~10 s)
python build_sports.py --selftest --v3env DIR           # + adapt3/env3 probes (DIR needs a built binding3*.so)
python build_sports.py --rec rec_out --rec raw_pob --out build --verify            # finished games, 4 h segments
python build_sports.py --rec rec_out --seg 0 --out build_w --verify                # env3: 1 episode per game
python build_sports.py --rec rec_out --seg 0 --bucket-segments --out build_b       # env3: 1 episode per bucket
python ../../v3env/adapt3.py build_w                    # -> markets3.bin / meta3.json (v3env)
python rewards_snapshot.py                              # daily: rewards/rewards_<day>.json (git-tracked)
cd env_sports && python setup.py build_ext --inplace    # the patched v2 env (fractional settlement)
```

## sports-2 changes (2026-10-15, for v3env / ENV3.md proposals S1-S3, D14/D15)

| item | sports-2 |
|---|---|
| `--seg 0` (S1) | One episode per game: from the first valid book row (or `cut - --max-hours`) to the cut, any length (`n_rows` varies; `contract_len_s = n_rows - 1`). v3env/mm3 only: the v2 `mm.h` samples only 300/900/14400 rows, and the `len_*` one-hot is all 0 (train/envs.py MarketSetEnv asserts it). Games shorter than `--min-window-s` (600) are dropped. The 300/900/14400 modes are unchanged (default 14400). |
| `--bucket-segments` | With `--seg 0`: split each window at 24 h / 6 h / 1 h / 30 m / 5 m before kickoff, so every episode lies in one bucket. Each piece settles at its own last-row mid; `#k0` is the piece ending at the cut. |
| buckets (D14/D15) | `meta.segments[i].bucket` = the time-to-kickoff bucket holding most of the episode's decision rows: `>24h`, `6-24h`, `1-6h`, `1h-30m`, `30-5m`, `5-0m`; also `bucket_pure`, `bucket_rows`, `to_kickoff_start_s`, `to_kickoff_end_s`. Counts in `meta.sports.buckets`. |
| cut | `--cut-before-kickoff-s` (default **0** = at kickoff, D15; D14's 3600 was withdrawn): episodes end, and settle, at `cut_ts = T - cut`. Book rows / legs after the cut are dropped (`stats.*_after_cut_dropped`). The quote pull before kickoff is a policy / env (`pull_s`) hyperparameter, not a build cut. |
| dt | Default 1. `--dt > 1` prints a warning: latency, match delay, markouts, `part_cap` and the T5 flags count rows (ENV3.md issue 3). Use mm3 `dt_far` / `far_s` for the far-from-kickoff cadence instead. |
| meta.json `segments` (S3) | One record per markets.bin row (same order): `sid`, `slug`, `condition_id`, `row_offset`, `n_rows`, `open_ts`, `end_ts`, `row_s`, `kickoff_ts` (= the cut kickoff T: gamma `gameStartTime`; with `--allow-open` the last book row if earlier; equals `open_ts + sports_feat.seconds_to_kickoff[row_offset]`, i.e. adapt3's current `event_ts`), `kickoff_scheduled_ts`, `kickoff_source`, `cut_ts`, `event_ts` (= `end_ts`, where the episode settles), `settle_mode` 1, `settle_price`, `at_cut`, bucket fields, `actual_start` (kickoff_watch label: ts, source, confidence, delta; null if none), `early_start`, `rw` (program summary), `flags`. |
| settlement (S3) | Primary: `segments[i].settle_price` (settle_mode 1) = the feat mark at the last row. The legacy copy (`log_strike_vs_ul_open` + flag 16, read by adapt3.py and env_sports/) is still written; `--no-legacy-settle` drops it (then `winner_up = -1`, which mm3 settles at the last mark = the same price). `meta.settle_source` declares which. |
| flags | Bit 32 is no longer used: TRADES_TRUNCATED moved to **1024** (32 = UNRESOLVED in elections). 16 SETTLE_PX only as the legacy settle marker. New: 2048 IN_PROGRAM, 4096 RW_LATER_SNAPSHOT, 8192 EARLY_START. 128 is now named AT_CUT (= AT_KICKOFF with cut 0). |
| reward columns (S2) | feat `liq_mid` / `liq_comp` / `liq_pi` (no longer inert) and `sports_x3.bin` (float32 [N, 8] in v3env's x3.bin layout: `rw_v_c`, `rw_min_size`, `rw_rate_day`, `rw_in_program`, `liq_mid`, `liq_comp`, `liq_pi`, pad). Scorer as the elections builder (`reward_terms.py`): on the **full recorded ladder** of the snapshot in use (each token's own ladder, mapped to the Up frame), size-cutoff adjusted mid, competitors' 'hi' Q_min within v and >= min size; `liq_pi = rate / 1440` on one seeded second per UTC minute, when in program and scorable (valid two-sided book, recorder live). markets.bin `rewards_min_size` / `rewards_max_spread` = the value at the last row. With v2 envs keep `liq_rewards = 0`: the v2 hook scores our orders with its 1.5 c / 50 constants. |
| ladder depth | Recorder / backfill `books` rows carry every level (rec_out tonight: median 19, p99 76, max 91 levels per side); `rec_daily/` compact exports keep 10. `meta.sports.ladder` reports the depth seen and warns when it is <= 10. |
| programs (`--rewards-dir`, default `rewards/`) | `rewards_snapshot.py` files: per UTC day of the row, the latest snapshot taken on or before it, else the earliest (flag RW_LATER_SNAPSHOT; in program then only from its `start_date`). In program: rate = `total_daily_rate`, v = `rewards_max_spread`, min size = `rewards_min_size`; out: rate 0, v / min size from gamma. |
| actual kickoff | `--kickoff-csv` (default `kickoff/kickoffs.csv`): labels only (v3 data rule). `--drop-early-starts` skips games seen live before the schedule. |
| fix | `--allow-open` crash on an empty data-api history (`'DataFrame' object has no attribute 'outcome_idx'`): an empty frame had object columns, and `t[empty object Series]` selects columns. Typed columns now. |

`rewards_snapshot.py`: CLOB `/rewards/markets/current` (all pages; the code of
`../census/rewards_current.py`), kept for the recorded markets only (condition ids from the gamma
cache, recorder `markets/`, and the slugs of `coverage/` and `kickoff/meta.json` resolved through
gamma). /current has no slug or tags, and the recorder records only sports. First run
(2026-10-15 21:24 UTC): 18,759 markets in programs ($204k/day); **only 5 of the 313 recorded
markets are in a program** ($4,076/day: e.g. cfb-gas-old $50/day v 2.5 c min 1000, lol-fur-mvk
$95/day v 2.5 c min 500). The NBA/NHL moneylines carry gamma `rewardsMinSize` 50 / `rewardsMaxSpread`
4.5 but no pool (checked with `/rewards/markets/<condition_id>`: empty). Run it daily (a second
run on the same day merges).

Inputs, per `--rec DIR` (repeatable):
- `books/` in the data/raw book schema; `reconciled` is optional;
- `trades/` in the data/raw trades schema;
- `markets/` (recorder / backfill markets rows);
- `coverage/` (recorder liveness log).

Gamma (`gameStartTime`, tokens, fees, tick) is fetched per slug into `cache/gamma/`. Trades come
from the recorder; for a slug without recorder trades they come from data-api `/trades`, cached
in `cache/trades/` once the game is over. `--offline` uses the cache only. Only games whose
kickoff is at least `--settle-wait` (30 min) in the past are built. `--allow-open` is for testing
only: it cuts open markets at their last book row.

## Schema decisions

| item | choice |
|---|---|
| frame | Outcome 0 (the first `clobTokenId`: a team, or Yes) plays Up. Outcome-1 snapshots are mirrored as in `data/build.py` (`merged_snapshots`, imported). |
| kickoff | Gamma `gameStartTime`, else the backfill's `kickoff` column. Fallback: `eventStartTime`, then `endDate`, then recorder `event_start`/`end`, then books `market_end_at`. A fallback sets flag `KICKOFF_FALLBACK`. For atp/wta/mlb, `endDate` is kickoff + 6 h to 7 d. |
| cut | Book rows with `captured_at >= kickoff` and legs with block time `>= kickoff` are dropped. They are counted in `meta.sports.stats`. |
| grid | `--dt` seconds per row (default 1). Row j of a segment is `open_ts + j*dt`. Every env quantity counted in rows is in units of dt: latency, match delay, the `dt*` params, `markout_h`, the flow/mret windows and the "minute" of `part_cap`. Book = the latest snapshot `<= t`. Legs are those in `(t-dt, t]`. |
| episode | (`--seg 0`: see above.) A segment of exactly `--seg` rows + 1 (300, 900 or **14400**; `mm.h` samples only these lengths). Segments tile `[kickoff - max_hours, kickoff]` backwards from kickoff. `#k0` is the segment ending **at kickoff** (flag `AT_KICKOFF`). Defaults: dt 1, seg 14400, 24 h, which gives 6 episodes of 4 h per game, exactly the crypto 4h semantics. `--dt 6` gives one 24 h episode. |
| head | A segment is kept if at least `--min-cover` (0.8) of its rows have a book. Rows before the first book are no-book rows: `book_valid` 0, p_model 0.5, `mark_is_fv` 1, no legs. Flag `HEAD_PADDED`. Lag features (`dgap5`, `mret*`) are 0 until a two-sided book has been seen. |
| `len_*` one-hot | Set from seg: seg 14400 gives `len_4h` = 1. This is required, because `train/envs.py` MarketSetEnv asserts the one-hot against `contract_len_s`. In the trainer's groups, sports segments therefore count as "4h". |
| `tau_frac` (obs 7) | Rows to the segment (episode) end / seg, going 1 to 0, like crypto. |
| `log_tau` (obs 8) | `log1p(seconds to KICKOFF) / log1p(14400)`. It reaches 1.19 at 24 h, still inside the clip. This is the kickoff clock. |
| `p_model` | No underlying. The fair value is the book **mid** (`--p-model micro`: top-of-book microprice) of the last two-sided valid book, forward-filled, and 0.5 before the first one. With mid, gap_c/dgap5 are 0 and `bid/ask_minus_p_c` are minus/plus the half spread. |
| zeroed columns | `z`, `sdv_bp`, `sigma_1m_bp`, `in_twap`, `delta_usd`, `ulz5/30/120`, aux `delta_usd_raw`, `sigma`, `sqrt_v` (obs 35 inv_delta is then 0). ticks `ul_*` are 0. |
| liquidity rewards | sports-2: full-ladder reward columns (see above). sports-1 had the hook inert. |
| stale flag (book bit2) | Markets with a recorder coverage log (flag `COVERAGE`): stale means the recorder was not live at t. Otherwise stale means `book_age > --stale-s` (600 s). Sports books are quiet, and the crypto rule (10 s since the last change) would block every pessimistic fill. The `book_age` feature itself is unchanged (seconds since the last book row). |
| tick | The data rule of `features.py`: 0.01 until a sub-cent price appears, then 0.001 (sticky). Markets whose gamma tick is not on the 0.001 grid (0.0025) are skipped, because prices are int16 mils. |
| fees | Gamma `feeSchedule`: rate, exponent, rebateRate (sports_fees_v3: 0.05 / 1 / 0.15). The rate is 0 when fees are disabled. |
| trades | Maker legs as in `features.py`. A tx is dropped if its maker legs do not sum to the taker size (abs dev >= 0.01) or disagree on the aggressor. ticks volumes are computed from the kept legs, so the volume check holds. |
| markets.bin | `coin_id` = league index (`meta.coin_ids`). `contract_len_s` = episode rows - 1. `open_ts`/`end_ts` are real unix seconds (`end - open = contract_len_s*dt`). `tick_size_final` = gamma tick. Flags: 1 HAS_TRADES, 16 SETTLE_PX (legacy), 64 HEAD_PADDED, 128 AT_CUT, 256 KICKOFF_FALLBACK, 512 COVERAGE, 1024 TRADES_TRUNCATED, 2048 IN_PROGRAM, 4096 RW_LATER_SNAPSHOT, 8192 EARLY_START. |
| ticks.bin | Same 40 columns. `seconds_to_close` is in **rows** to the segment end, so `data/load.py verify()` passes for dt = 1. For dt > 1 it fails on `end - open == contract_len_s`; use `build_sports.verify`. |
| ids | `meta.slugs` / `splits.json` keys are `<slug>#k<i>`, where i = segments before kickoff. |
| side files | `sports_x3.bin` (above) and `sports_feat.bin`, float32 [N, 4]: `seconds_to_kickoff`, `seconds_to_seg_end`, `microprice` (-1 if not two-sided), `hours_since_first_book`. `mm.h` does not read it, because the frozen 44-column feat.bin has no free observed slot. |
| splits | By **game day** (the slug date, else the kickoff UTC date). All segments of a game take the game's split, and markets are sorted by (split, end_ts, id), so each split's rows are contiguous. The test-row sha256 is in `meta.test_rows`. Default is the D11 auto rule: oldest ~60% of days train, next ~20% val, newest ~20% test. With 3 days that is 2/1/0, so give `--splits` for real use. Example: `{"days": {"2026-09-20..2026-10-04": "train", "2026-10-05..2026-10-09": "val", "2026-10-10..2026-10-14": "test"}, "default": "exclude"}`. Days not listed become split 3 (purged); use `"default": "auto"` to auto-split them instead. The day-to-split map is in `meta.splits.day_split`. Writing into a `*jun15*` path is refused. |

## Settlement (important)

`mm.h` settles the terminal inventory at `markets.winner_up`. That field is `int32`, and `m->winner`
is an `int`, so the **stock env cannot settle at a fractional price**. The build therefore writes:
- `log_strike_vs_ul_open` (a float that `mm.h` never reads) = the settle price = the feat `mark`
  at the segment's last row (the last pre-kickoff mid), plus flag `SETTLE_PX`;
- `winner_up = int(settle >= 0.5)`. This is only a fallback so that the stock env runs. It is
  **not** the game result, which is never read.

With the stock env, every terminal P&L, every Log `pnl`/`pnl_trading` and the markouts past the
end are wrong by `inv * (winner_up - settle)`. On the real rec_out sample, one market with 25
shares long showed +8.79 USD in the stock env against +0.17 USD in the patched env. Use the stock
env only for shape and loading checks.

**Minimal fix:** `env_sports/` holds a copy of `mm.h`, `binding.c`, `setup.py` and `env.py` with 3 changes to `mm.h`:
1. `#define MM_FLAG_SETTLE_PX 16`;
2. `int winner` becomes `float winner` (same size, so `Mm` keeps its layout);
3. `mm_begin` sets `m->winner = (M->flags & MM_FLAG_SETTLE_PX) ? M->log_strike_vs_ul_open : (float)M->winner_up`.

Crypto builds never set flag 16, so on them the copy behaves exactly like `mm.h`; this was not
re-run against `test_env.py`. The `env.py` copy loads its own `binding*.so` by path, so it cannot
collide with `polymarket/binding*.so`. To use it with `train/envs.py`, swap the adapter's module,
as the selftest does:

```python
ad = envs.get_adapter('real'); ad.mod = <import env_sports/env.py>   # then make_train/make_eval(data_dir=build)
```

If the main env adopts the patch, the copy can go. Note: `RealAdapter.identity()` hashes the
stock `mm.h`/`env.py`, so a sports adapter should hash the `env_sports/` files instead.

## v3env (env3) and adapt3.py

`adapt3.py` (coordinator's, unedited) loads sports-2 builds as is: it still reads flag 16 +
`log_strike_vs_ul_open` (kept) and `event_ts = open_ts + sports_feat.seconds_to_kickoff[row 0]`
(= `meta.segments[i].kickoff_ts`, asserted by `verify`). Without `--rewards` it writes no x3.bin, so
mm3 pays the hook from feat `liq_*` (full ladder) with markets3 `rw_v_c` / `rw_min_size` from
markets.bin, but `rw_rate_day` is 0, so the reward observations (slots 55-77) stay 0. With
`--rewards` it overwrites that with its top-5 approximation. Proposed adapt3 change (tested in the
selftest by symlinking `sports_x3.bin` as `x3.bin`: same rewards and fills, `rw_active` = 1 on
in-program games): for build_version `sports-2`, copy `sports_x3.bin` to `x3.bin`, take
`settle_price` / `event_ts` / `kickoff_ts` from `meta.segments`, `rw_rate_day` from
`segments[i].rw.rate_day_last`, and stop requiring flag 16 (then `--no-legacy-settle` builds load too).

Recommended mm3 parameters for `--seg 0`: `len_mask` 15 (bit 8, `dt_other`), `dt_other` 1-2,
`dt_far` 30 / `far_s` 7200, `pull_s` and `flat_c` from val, `obs_size` 80, `liq_rewards` 1,
`size_mode` 1, Q >= 4-8x the largest reward min size (min sizes up to 1000 shares here).

## Recommended env parameters (v2 / env_sports)

- dt = 1: the crypto defaults apply (`dt_4h` = 5, so 2880 decisions per 4 h segment).
- dt = 6 (one 24 h episode per game): `match_delay=0`, which makes D = 1 row = 6 s instead of
  4 rows = 24 s. `latency=1` row = 6 s. `dt_4h` = 5 rows means 30 s per decision.
- `mark_mode` 0 and 1 coincide with p_model = mid (when the book is one-sided, both are the
  last two-sided mid).

## Size (`420 B/row`: ticks 160 + feat 176 + book 64 + fill_index 4 + sports_feat 16; plus 20 B per leg)

| window | dt 1 | dt 6 |
|---|---|---|
| 24 h per game | 86,401 rows, **36.3 MB** | 14,401 rows, **6.0 MB** |
| 25 days x 60 games | ~54 GB | ~9 GB |

Build speed: about 1 s per game-day at dt 1, measured on a synthetic 24 h game with a book every
3 s, single thread. The arrays are spooled per segment and streamed into the output, so memory
holds one game window at a time plus the input books.

## Tests run (2026-10-15, sports-2)

- `--selftest` (~10 s; also run by `v3env/test_env3.py`, which passed: `--quick`, 43 s, all checks ok
  incl. the C == reference3 runs on the selftest build): everything below for sports-1, plus
  - synthetic reward snapshots (a later snapshot, a program not yet started, unlisted games): the
    program flags per game are as designed; paid sample rows = one per UTC minute (+-2 at the ends)
    in program, 0 out of program;
  - one paid row rebuilt independently from the raw ladder with plain loops: adjusted mid and
    competitors' Q match (`comp_full` 459.35); the top-5 approximation gives 431.47 (v = 9 c reaches
    levels 6-8);
  - `--seg 0` (4 games, one 1001-row episode each; the 500 s game dropped), `--no-legacy-settle`,
    `--cut-before-kickoff-s 300` (episodes end 300 s earlier, bucket `30-5m`), `--bucket-segments`
    (8 pure pieces);
  - adapt3.py on the `--seg 0` build: `markets3.settle_price` == `segments.settle_price`,
    `event_ts` == `kickoff_ts`; env3 (obs 80, liq on, size_mode 1, join policy, fixed market): every
    episode runs `n_rows - 1` rows, Log `settle_px` == settle_price, rewards > 0 exactly on the
    in-program games; the x3 = `sports_x3.bin` variant gives identical rewards and fills and
    `rw_active` 1; `dt_far 30 / far_s 300` cuts 1000 steps to 304.
- Real data (`rec_out`, live recorder, `--allow-open`, nhl/nba/cfb/lol, data-api trades; 24 markets
  with an empty data-api history no longer crash): `--seg 0` 112 games (968-4302 rows), `--seg 0
  --bucket-segments` 493 pieces (136 games; 1-6h 112, 1h-30m 111, 30-5m 135, 5-0m 135), default
  `--seg 14400` (`--max-hours 4` and 24); both verifies pass; adapt3 + env3 step all of them
  (random actions, 32 envs, obs 80, liq on, dt_far; join episodes on the 2 in-program games: settle
  == meta, rewards paid).
- `baseline_pregame.py --test --slugs '^(nhl|nba)-' --fill-models central`: 31 markets, 21 strategies,
  160 s.

## Tests run (2026-10-15, sports-1)

- `--selftest`: 5 synthetic games on 5 days. It covers a crossed snapshot, maker legs on both
  outcome tokens, book rows and legs after kickoff (dropped), and a game listed late (padded or
  dropped).
  - Build at dt 1 / seg 300 / 15 min: 14 segments, auto split 8/3/3. Both `data/load.py
    verify()` and the sports `verify()` pass: settle equals the last-row mark, AT_KICKOFF
    segments end at kickoff, no row after kickoff, leg volume equals ticks volume, and the
    feat_meta layout is identical to data/build's.
  - Build at dt 6.
  - Stock env through `train/envs.py` RealAdapter (`make_train` with data_dir, which passes
    the layout check; `make_eval` MarketSetEnv, which passes the one-hot and episode-length
    asserts): random actions and a join policy run.
  - Flipping `winner_up` changes the stock env's P&L (sum |diff| 461 USD) but leaves `env_sports`
    unchanged (diff 0). Fills are identical between the two envs.
  - The dt 6 build loads in the stock env.
- **Real data** (`rec_out`, recorder started 20:21 UTC; NHL/NBA/NFL, `--allow-open`, seg 900, 30 min):
  - 31 games, 31 segments; trades from data-api; gamma kickoffs.
  - Both verifies pass, and both envs run.
  - Coverage flag set; mark is the fallback (`mark_is_fv`) on 15% of rows.
  - Observed columns stay within the clip.

## Open problems

- **Kickoff accuracy.** `gameStartTime` is the schedule. If a game starts late, we drop valid
  pre-game rows (conservative). If it starts early, in-play rows before the scheduled time stay
  in. Possible fix: detect the `clearBookOnStart` wipe (the book empties) and cut at
  min(schedule, wipe).
- **Trade history from data-api is capped.** It covers 20k rows per query, or 40k with the
  BUY/SELL split, newest first. For busy games the in-play prints push out the **oldest pre-game**
  legs. Such markets get `TRADES_TRUNCATED`, and segments starting before the earliest fetched leg
  are dropped (`--keep-truncated` keeps them). The recorder's incremental polling avoids this for
  recorded games. The on-chain fills (`onchain/`) would be the complete source for the backfill.
- **Out-of-distribution observations for the crypto policy.** Nine observed columns are constant
  0, and the one-hot is pinned to 4h. A crypto-trained policy will see out-of-distribution
  inputs; sports needs its own training (D11 v3).
- **Seconds to kickoff.** Only `log_tau` carries it inside the frozen observation. Its raw value
  is in `sports_feat.bin`, which no env reads.
- **Intermediate segments.** Segments that do not end at kickoff (dt 1 default) settle at the mid
  at their end. That is an artificial boundary; at dt 6, 24 h is one episode.
- **dt > 1.** The env's "minute" (`part_cap`) and the T5 `last30`/`last10` flags count rows, not
  seconds.
- **Reward programs** come from daily snapshots of the CURRENT programs only (no history before
  2026-10-15); for older games the program is inferred from the earliest snapshot and its
  `start_date` (flag RW_LATER_SNAPSHOT). Competitors' Q uses the 'hi' bound, b = 1.
