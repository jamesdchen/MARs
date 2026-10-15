# train3: training and evaluation for protocol v3 (pre-game sports, election reward farming)

**Build the env first** (once per machine and after every `v3env/mm3.h` / `binding3.c` change; the
`.so` is git-ignored): `../../.venv/bin/python ../v3env/setup.py build_ext --inplace` (from `train3/`).

This folder connects the v3 replay env (`../v3env/`, mm3.h; ENV3.md) to the PPO trainer of
`../train/` for the two tracks of DECISIONS.md D11. `../train/*.py` is imported, never edited:
the crypto v2 freeze uses it.

| file | role |
|---|---|
| `envs3.py` | `V3Adapter`: the train/ adapter interface around `v3env/env3.py` MmEnv3. It hashes the v3env/ code and the build's data, groups markets by category (sports league, elections), refuses the test split unless `MM_FINAL=1`, refuses `jun15` paths, and applies a split map. `MarketSetEnv3` plays every market of a split once. |
| `evaluator3.py` | per-market and per-category metrics, T5 / T6 / D1.3 (T6 is `final/reduce.py directional_verdict`, imported), seed-pooled verdicts, the reward-income line, D15 netting / flatten drag (`net/*`) and time buckets (`tk/*`, `bucket/*`), and the v3 comparators (no_quote, sym_touch, mid_value, as_mid, reward_farmer; `--pull-grid` sweeps pull_s). CLI: `comparators`, `policy`, `pool`. |
| `train_ppo3.py` | a thin wrapper over `train/train_ppo.py`: V3 adapter, evaluator3, `--data-dir`, `--split-map`, `--env3-kw`, ledger `train3/trials.jsonl`. |
| `split3.py` | lists a build's dates per track and writes the D11 split file, with its sha256 recorded. |
| `smoke3.py` | an end-to-end mechanics check (fixtures, and optionally tiny real builds). |
| `trials.jsonl`, `runs.jsonl` | the v3 ledger: one row per validation evaluation, every variant, comparators included. A row is written only by real v3 runs; final/common.py's trial count globs every `trials.jsonl`. |
| `smoke_trials.jsonl`, `smoke_runs.jsonl` | smoke and mechanics runs. They are never counted, never selected from, and no conclusions are drawn from them. |
| `split_registry.jsonl` | one row per pre-registered split file (path, sha256, counts). |
| `views/`, `runs/`, `.hashcache.json` | local, git-ignored: split views, run directories, cached data hashes. |

All commands below run from `polymarket/train3/` with `../../.venv/bin/python`. Other trainings
share the machine, so run one process at a time with `nice -n 10` and `--threads 1`.

## Track S: pre-game sports, end to end

```bash
# 1. build: recorder output and/or the PolyOrderBooks backfill -> data/build schema
#    (analysis/v3_sports/README_build.md, sports-2): one episode per game to kickoff (--seg 0), or one per
#    D15 bucket (--seg 0 --bucket-segments); reward columns from rewards/ snapshots (sports_x3.bin)
cd ../analysis/v3_sports
python build_sports.py --rec rec_out --rec raw_pob --seg 0 --out /data/v3/sports --verify
# 2. make it loadable by mm3: markets3.bin (settle at the kickoff mark, event = kickoff), x3.bin = sports_x3.bin
cd ../../v3env && python adapt3.py /data/v3/sports           # --rewards only for sports-1 builds
# 3. pre-register the split ONCE, before any model is evaluated on these dates (D11):
cd ../train3 && python split3.py /data/v3/sports --dry-run            # look at the dates first
python split3.py /data/v3/sports --out splits/sports_v3.json           # writes the file, .sha256, registry row
git add splits/sports_v3.json splits/sports_v3.json.sha256 split_registry.jsonl   # commit it before training
# 4. comparators on val (no_quote is the D1 primary candidate), swept over the D15 pull grid 0/300/900/1800/3600 s
nice -n 10 python evaluator3.py comparators --data-dir /data/v3/sports --split-map splits/sports_v3.json \
    --env3-kw obs_size=80 liq_rewards=1 size_mode=1 Q=1000 flat_c=0.5 dt_other=1 dt_far=30 far_s=7200 \
    --variants central_1s pess_1s --pull-grid
# 5. train (each periodic and final val evaluation is a ledger row); pull_s is a val-selected hyperparameter
nice -n 10 python train_ppo3.py --data-dir /data/v3/sports --split-map splits/sports_v3.json \
    --env3-kw obs_size=80 liq_rewards=1 size_mode=1 Q=1000 pull_s=900 flat_c=0.5 close_inv_coef=0.05 \
    close_ramp_s=1800 dt_other=1 dt_far=30 far_s=7200 --timesteps 25e6 --threads 1 --seed 0 --eval-every 5e6 --tag s_main
# 6. evaluate a checkpoint (per-market rows with --markets), and pool the seeds of one config (D13 reading C)
nice -n 10 python evaluator3.py policy --ckpt runs/<run>/final.pt --data-dir /data/v3/sports \
    --split-map splits/sports_v3.json --env3-kw obs_size=80 liq_rewards=1 size_mode=1 Q=1000 pull_s=900 flat_c=0.5 \
    dt_other=1 dt_far=30 far_s=7200 --markets
python evaluator3.py pool runs/<run_s0> runs/<run_s1> runs/<run_s2>
```

**D15: the episode runs from the start of data to kickoff** (D14's T − 1 h cut is withdrawn). The
env settles each episode at its last pre-kickoff mark (`settle_mode` 1 from markets3.bin) minus
`flat_c` (half a spread: D15's flatten cost; a full spread is the reported, not judged,
sensitivity). So:

- `pull_s` (quotes forced off that many seconds before kickoff) is a hyperparameter selected on val.
  The candidates are {300, 900, 1800, 3600}, plus 0 = quote to kickoff. `evaluator3.py comparators
  --pull-grid` evaluates every comparator at each one. The env pull and the comparator's own
  `pull_s` are set together, and each grid point is its own ledger row.
- `close_inv_coef` with `close_ramp_s=1800` (training-only shaping) ramps an inventory penalty over
  the final 30 minutes.
- **Netting is the central problem** (RECONCILE.md section 6: the pool nets 10–25% of its fills
  before kickoff, and a +20 to +30 bps edge needs 70–80%). Every evaluation reports `net/*`:
  - the netted share, 1 − Σ|inventory at kickoff| / Σ gross filled shares;
  - its per-episode mean;
  - |inventory| at kickoff;
  - the flatten drag in USD and in bps of the filled notional.
- **Bucket builds.** A `--bucket-segments` build makes every episode one bucket. Its time to event
  is still the time to kickoff (adapt3 sets `event_ts` = kickoff), so `pull_s` and obs 50 mean the
  same thing in both build kinds.
- **D14 lines.** `d14/*` (marked / flat at T − 1 h) is still reported, for comparison.

`split3.py` keeps a sports game day only if at least one of its episodes starts more than
`--min-pre-hours` (default 1, D14's rule; pass 0 under D15) before kickoff. The file records, per
day, how many episodes reach into each D14 bucket. Sports-2 builds carry the reward programs
(`sports_x3.bin`), so `liq_rewards=1` pays the hook on in-program games. obs_size 80 adds the
kickoff clock (obs 50), the category one-hot and the reward slots.

**`--seg 0` and the length one-hot.** `train/envs.py` MarketSetEnv asserts the crypto
`len_5m/15m/4h` one-hot, which is all 0 for `--seg 0` episodes. train3 never reaches it:
- `V3Adapter.make_eval` builds its own `MarketSetEnv3`, which checks the category one-hot and the
  decision count instead;
- `make_train` builds MmEnv3 directly;
- train_ppo3 replaces `train_ppo.get_adapter` and `train_ppo.evaluate`.

The smoke runs below train and evaluate on `--seg 0`-style fixtures: episode lengths 1200–3000 rows
(mm3 length bit 8), and the bucket pieces of 200–1500 rows.

## Track E: election reward farming, end to end

```bash
# 1. build from the recorded books + hourly reward snapshots + fills (analysis/v3_elections/BUILD.md).
#    Pass the split file made in step 3 as --split-days once it exists; before that, build every day as purged.
cd ../analysis/v3_elections
python build_elections.py --window-s 14400 --trades-source onchain --trades-path <extract.py out> --out /data/v3/elections
# 2. markets3.bin (settle_mode 2 = mark at the window end) + x3.bin (the per-row reward program)
cd ../../v3env && python adapt3.py /data/v3/elections
# 3. pre-register the split: the last recorded days before Nov 3 are the untouched test
cd ../train3 && python split3.py /data/v3/elections --dry-run --exclude 2026-10-15   # Oct 15 is a partial day
python split3.py /data/v3/elections --exclude 2026-10-15 --out splits/elections_v3.json
# 4. comparators: reward_farmer is the reward-only rule baseline
nice -n 10 python evaluator3.py comparators --data-dir /data/v3/elections --split-map splits/elections_v3.json \
    --env3-kw obs_size=80 liq_rewards=1 size_mode=1 Q=1000 --variants central_1s pess_1s
# 5. train
nice -n 10 python train_ppo3.py --data-dir /data/v3/elections --split-map splits/elections_v3.json \
    --env3-kw obs_size=80 liq_rewards=1 size_mode=1 Q=1000 dt_other=5 \
    --timesteps 25e6 --threads 1 --seed 0 --eval-every 5e6 --tag e_main
# 6. evaluate / pool as for sports
```

The election summary keeps the reward income apart from trading. `income/rewards` is the hook's
payout. `income/trading` is spread capture + 60 s markout + inventory (the inventory term includes
the flattening cost), and `income/rebates` is the maker rebate. `income/residual` checks that the
lines add up to the P&L.

## What the evaluation reports

From `evaluator3.summarize3`, for every evaluation, both overall and per group (`group/sports/nba`,
`group/elections`, `cat/<category>`):

- **P&L.** Total, mean and CVaR95 per market, plus every Log field as a sum and a mean.
- **Fills per market.**
- **T6.** `final/reduce.py directional_verdict` on the attribution totals: spread capture, adverse
  selection (60 s markout), inventory, rebates and rewards. A policy is directional when its
  inventory P&L is > 0 and is more than 50% of the total, or more than 50% of the positive
  components.
- **T5 (D2 artifact flags).** The share of fill P&L under each of inspread, crossed, stale,
  onesided and last30. The Log has no crossed-or-stale union, so those two are judged separately,
  as `train/evaluator.py` does. `outside` (a sweep through the touch) is reported only.
- **D1.3.** ≥ 5 fills per market, not directional, and T5 ≤ 25%.
- **D1.2 (`d12/active`).** ≥ 5 fills per market and a two-sided uptime ≥ 0.25, for secondary
  comparators.
- **Netting (`net/*`, D15).** These come from the env Log's `net_*` fields (ENV3.md section 9):
  - `net/netted_share` = 1 − Σ|end inv| / Σ gross filled shares (pooled), and
    `net/netted_share_ep` = its mean over episodes with fills;
  - `net/end_abs_inv` = |inventory| at the settlement row (kickoff), and `net/gross_filled_sh`;
  - `net/flat_drag_usd` = flat_c × |end inv| (already in the P&L), and `net/flat_drag_bps` = the
    same per filled notional;
  - they are reported overall, per market (`--markets`), per group / category and per bucket.
- **D15 time buckets (`tk/<bucket>/*`, `d14/*`).** P&L, decision steps and quoting share per
  time-to-kickoff bucket: `gt24h`, `6_24h`, `1_6h`, `1h_30m`, `30_5m` and `5_0m`. The bounds are
  build_sports' and are upper-inclusive: a decision exactly 1800 s out is `30_5m`. `lt1h` is the
  last three summed.
  - Each step's change of the observed wealth (obs 34) goes to the bucket of the step's decision
    time (obs 50). The unobserved terminal step takes the remainder, so the buckets add up to the
    P&L exactly.
  - A step counts as quoting when the action quotes and the env's `pull_s` has not forced the
    quotes off.
- **Episode buckets (`bucket/<b>/*`).** The same per-group lines (P&L, verdicts, income, netting,
  time buckets), grouped by the episode's bucket: `meta.json segments[i].bucket` (the bucket holding
  most of its decision rows). sports-1 builds compute the same rule. With `--bucket-segments`
  builds this is exact.
  - Wealth in the observation clips at ±0.25 Q. `tk/clip_steps` counts the steps where it does,
    and those steps make the split between buckets unreliable (the total stays exact), so use
    Q ≥ 1000.
  - The buckets need obs_size > 50.
  - The same lines are reported per league group.

With `--markets`, one row per market carries its bucket, P&L lines, netting (`net_*`) and its own
T5 / T6 numbers.

`pool` applies D13 reading C to a multi-seed method: each component is averaged over the seeds,
and the verdicts are taken on those averages.

## Comparators (D1, D11)

None of the comparators is tuned yet. They run at the defaults in `evaluator3.BASELINE_DEFAULTS`.
D1.4 asks for each one to be re-tuned on the frozen v3 env and data with the same 64-trial budget,
and every trial goes in the ledger. That tuning has not been written yet.

Every comparator reads the event clock from obs 50, so it is evaluated with obs_size ≥ 80, which
does not change the dynamics.

- `no_quote`: never quotes.
- `sym_touch`: joins both touches, with an inventory cap of `q_cap_lots` × the order size.
- `mid_value`: the fair-value quoter with the fair value replaced by the book mid, because there is
  no Binance underlying. It skews by inventory and pulls `pull_s` before the event.
- `as_mid`: Avellaneda-Stoikov in price units around the mid, with the horizon min(h_risk, time to
  event).
- `reward_farmer` (elections): while the market is in a reward program, it quotes both sides at
  max_spread / 2 from the mid with size = the reward min size (size_mode 1, head 0). It pulls a side
  once |inventory| would pass `inv_cap_lots` × min size. It needs `liq_rewards=1`.

## Split maps (`split3.py`, envs3 `split_map`)

- **Dates.** The date of a sports market is its game day: the slug date, else the kickoff UTC date,
  as in `build_sports.game_day`. The date of an election market is the UTC day of its window open.
- **Rule (D11).** The oldest ~60% of days are train, the next ~20% val and the newest ~20% test,
  using the sports builder's own arithmetic. For elections, days at or after `--cutoff`
  (2026-11-03) are dropped first.
- **File format.** The file is in the builders' own split format:
  - sports: `{"days": {...}, "default": "exclude"}`, for `build_sports.py --splits`;
  - elections: a flat `{day: name}`, for `build_elections.py --split-days`.

  Keys starting with `_` hold provenance.
- **How the adapter applies it.** The adapter does not rebuild. It writes a view directory
  (`views/<build>_<hash>/`) with symlinks to the build and a markets.bin whose split field comes
  from the map. Days missing from the map load as split 3 (purged); for sports these include the
  "extra forward test" days recorded after the split was fixed.
- **Immutability.** A split file is never overwritten, and its sha256 goes into
  `split_registry.jsonl` and into every run's config hash.
- **Dry-run files.** `--dry-run` files are marked `_dry_run`. V3Adapter refuses them except in
  `--smoke` runs.

## Smoke test (2026-10-15, mechanics only)

`python smoke3.py --work <scratch> --real-sports <B1> --real-elections <B2>`

For each build, smoke3 runs:
1. the fixtures (`v3env/fixtures3.py`, which calls `adapt3.py`);
2. `split3 --dry-run`;
3. 1M PPO steps (1024 envs, 1 thread, eval every 0.5M);
4. `evaluator3 policy` on central_1s and pess_1s;
5. every comparator, and for sports the `--pull-grid` sweep (central_1s).

**Rerun 2026-10-15 ~23:05 UTC on the sports-2 fixtures** (`python smoke3.py --work <scratch>`, fixtures
only). All three tracks ran without errors, and rows are in `smoke_trials.jsonl`:

| build | markets (train / val / test days) | train sps | wall (train + evals) |
|---|---|---|---|
| synthetic sports, sports-2 schema, one episode per game to kickoff (6 games, 1201–3001 rows, length bit 8, rewards from `sports_x3.bin`) | 4 / 1 / 1 days | ~145k | 12 s + 12 s |
| the same games as `--bucket-segments` pieces (16 episodes, 1h-30m / 30-5m / 5-0m) | 4 / 1 / 1 days | ~150k | 11 s + 11 s |
| synthetic elections (5 one-hour windows) | 3 / 1 / 1 days | ~163k | 11 s + 8 s |

Each sports pull grid gave 20 rows: 4 comparators × pull_s {0, 300, 900, 1800, 3600}. As the grid
point rises, the quoting share in the pulled buckets drops to 0. The `net/*` and `bucket/<b>/*`
lines are present in every evaluation. No conclusions are drawn from the numbers.

The run before this one is described below. It used sports-1 fixtures and the D14 buckets.

The smoke builds hold less than an hour per game, so `split3 --min-pre-hours 0` turns off the D14
day rule. The D14 bucket attribution was also checked on the synthetic sports games, with the
bucket edges scaled down to 30 / 20 / 10 minutes: the buckets add up to each market's P&L, and the
marked-at-T−1h P&L equals the sum of the earlier buckets.

All four builds ran without errors. Rows are in `smoke_trials.jsonl`.

| build | markets (train / val / test days) | train sps | wall (train + evals) |
|---|---|---|---|
| synthetic sports (6 games) | 4 / 1 / 1 days | ~139k | 12 s + 7 s |
| synthetic elections (5 one-hour windows) | 3 / 1 / 1 days | ~157k | 11 s + 8 s |
| real sports: `build_sports.py --rec rec_daily --allow-open --seg 300 --max-hours 0.25 --slugs '^(nhl\|nba\|nfl)-'`, 31 open games cut at their last book row (2026-10-15 20:22-20:36 UTC), 93 segments, 111 legs | 2 / 1 / 0 days | ~156k | 11 s + 7 s |
| real elections: `build_elections.py --window-s 900 --allow-partial --top-n 5 --trades-source data-api` on the 20:21-21:07 UTC recording, 16 windows, 2 legs | 1 / 0 / 0 days (evaluated on train, no ledger row) | ~144k | 12 s + 9 s |

The real-data runs check only that the stack runs. They are open games with ~15 minutes of book,
or a single 45-minute election recording, and no conclusions are drawn from them.

## Blocked on data

- **No v3 split can be pre-registered yet.**
  - Sports needs the PolyOrderBooks backfill (API key) or several days of recorder data.
  - Elections needs full recorded days, starting from Oct 16.
- **The recorder output is not in git.** `rec_out/` and the election `rec/raw` are local to the
  workers, and only one day of compact sports books (no trades / markets) and one hour of election
  books are in the repo.
- **Sports reward pools** come only from daily snapshots since 2026-10-15. Older games are
  inferred from the earliest snapshot (ENV3.md open issue 1).
- **Comparator tuning (D1.4) is not done yet.** It runs once the val split exists.
- **D14 needs pre-game history longer than 1 h per game.** Under the D14 rule, the current sports
  smoke data (≤ 15 min per game) has no quotable day at all.
