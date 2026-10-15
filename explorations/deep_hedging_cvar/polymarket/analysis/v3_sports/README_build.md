# build_sports.py: pre-game sports books -> env binaries (Track S, D11)

`build_sports.py` turns normalized raw sports data (recorder `out_dir`, or the PolyOrderBooks
backfill in the same layout) into the files of `data/build`. The record layouts are identical and
the feat.bin column layout is the same frozen one, so `env.py`, `mm.h` and `train/envs.py` load it
with `data_dir=...`. The script never edits shared code.

```bash
source ../../../.venv/bin/activate                      # numpy, pandas, pyarrow, orjson (polyrec)
python build_sports.py --selftest                       # synthetic end to end incl. both envs (~1 min)
python build_sports.py --rec rec_out --rec raw_pob --out build --verify            # finished games, defaults
python build_sports.py --rec rec_out --dt 6 --max-hours 24 --splits splits_v3.json  # one 24 h episode per game
cd env_sports && python setup.py build_ext --inplace    # the patched env (fractional settlement)
```

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
| episode | A segment of exactly `--seg` rows + 1 (300, 900 or **14400**; `mm.h` samples only these lengths). Segments tile `[kickoff - max_hours, kickoff]` backwards from kickoff. `#k0` is the segment ending **at kickoff** (flag `AT_KICKOFF`). Defaults: dt 1, seg 14400, 24 h, which gives 6 episodes of 4 h per game, exactly the crypto 4h semantics. `--dt 6` gives one 24 h episode. |
| head | A segment is kept if at least `--min-cover` (0.8) of its rows have a book. Rows before the first book are no-book rows: `book_valid` 0, p_model 0.5, `mark_is_fv` 1, no legs. Flag `HEAD_PADDED`. Lag features (`dgap5`, `mret*`) are 0 until a two-sided book has been seen. |
| `len_*` one-hot | Set from seg: seg 14400 gives `len_4h` = 1. This is required, because `train/envs.py` MarketSetEnv asserts the one-hot against `contract_len_s`. In the trainer's groups, sports segments therefore count as "4h". |
| `tau_frac` (obs 7) | Rows to the segment (episode) end / seg, going 1 to 0, like crypto. |
| `log_tau` (obs 8) | `log1p(seconds to KICKOFF) / log1p(14400)`. It reaches 1.19 at 24 h, still inside the clip. This is the kickoff clock. |
| `p_model` | No underlying. The fair value is the book **mid** (`--p-model micro`: top-of-book microprice) of the last two-sided valid book, forward-filled, and 0.5 before the first one. With mid, gap_c/dgap5 are 0 and `bid/ask_minus_p_c` are minus/plus the half spread. |
| zeroed columns | `z`, `sdv_bp`, `sigma_1m_bp`, `in_twap`, `delta_usd`, `ulz5/30/120`, aux `delta_usd_raw`, `sigma`, `sqrt_v` (obs 35 inv_delta is then 0). ticks `ul_*` are 0. |
| liquidity rewards | Hook inert: `liq_mid` = -1, `liq_comp` = 0, `liq_pi` = 0. No sports reward pools are modelled; `liq_rewards=1` adds nothing. |
| stale flag (book bit2) | Markets with a recorder coverage log (flag `COVERAGE`): stale means the recorder was not live at t. Otherwise stale means `book_age > --stale-s` (600 s). Sports books are quiet, and the crypto rule (10 s since the last change) would block every pessimistic fill. The `book_age` feature itself is unchanged (seconds since the last book row). |
| tick | The data rule of `features.py`: 0.01 until a sub-cent price appears, then 0.001 (sticky). Markets whose gamma tick is not on the 0.001 grid (0.0025) are skipped, because prices are int16 mils. |
| fees | Gamma `feeSchedule`: rate, exponent, rebateRate (sports_fees_v3: 0.05 / 1 / 0.15). The rate is 0 when fees are disabled. |
| trades | Maker legs as in `features.py`. A tx is dropped if its maker legs do not sum to the taker size (abs dev >= 0.01) or disagree on the aggressor. ticks volumes are computed from the kept legs, so the volume check holds. |
| markets.bin | `coin_id` = league index (`meta.coin_ids`). `contract_len_s` = seg. `open_ts`/`end_ts` are real unix seconds (`end - open = seg*dt`). `tick_size_final` = gamma tick. Flags: 1 HAS_TRADES, 16 SETTLE_PX, 32 TRADES_TRUNCATED, 64 HEAD_PADDED, 128 AT_KICKOFF, 256 KICKOFF_FALLBACK, 512 COVERAGE. |
| ticks.bin | Same 40 columns. `seconds_to_close` is in **rows** to the segment end, so `data/load.py verify()` passes for dt = 1. For dt > 1 it fails on `end - open == contract_len_s`; use `build_sports.verify`. |
| ids | `meta.slugs` / `splits.json` keys are `<slug>#k<i>`, where i = segments before kickoff. |
| side file | `sports_feat.bin`, float32 [N, 4]: `seconds_to_kickoff`, `seconds_to_seg_end`, `microprice` (-1 if not two-sided), `hours_since_first_book`. `mm.h` does not read it, because the frozen 44-column feat.bin has no free observed slot. |
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

## Recommended env parameters

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

## Tests run (2026-10-15)

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
- **Unchecked liquidity programs.** Sports reward programs and the gamma fields
  `rewardsMinSize` / `rewardsMaxSpread` are stored in markets.bin but not modelled.
