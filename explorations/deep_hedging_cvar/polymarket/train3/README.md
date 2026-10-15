# train3: training and evaluation for protocol v3 (pre-game sports, election reward farming)

This folder connects the v3 replay env (`../v3env/`, mm3.h; ENV3.md) to the PPO trainer of
`../train/` for the two tracks of DECISIONS.md D11. `../train/*.py` is imported, never edited:
the crypto v2 freeze uses it.

| file | role |
|---|---|
| `envs3.py` | `V3Adapter`: the train/ adapter interface around `v3env/env3.py` MmEnv3. It hashes the v3env/ code and the build's data, groups markets by category (sports league, elections), refuses the test split unless `MM_FINAL=1`, refuses `jun15` paths, and applies a split map. `MarketSetEnv3` plays every market of a split once. |
| `evaluator3.py` | per-market and per-category metrics, T5 / T6 / D1.3 (T6 is `final/reduce.py directional_verdict`, imported), seed-pooled verdicts, the reward-income line, and the v3 comparators (no_quote, sym_touch, mid_value, as_mid, reward_farmer). CLI: `comparators`, `policy`, `pool`. |
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
#    (analysis/v3_sports/README_build.md). One 4 h episode ending at kickoff, or --seg 0 once S1 lands.
cd ../analysis/v3_sports
python build_sports.py --rec rec_out --rec raw_pob --out /data/v3/sports --verify
# 2. make it loadable by mm3: markets3.bin (settle at the kickoff mark), optional x3.bin from reward pools
cd ../../v3env && python adapt3.py /data/v3/sports [--rewards rates.json]
# 3. pre-register the split ONCE, before any model is evaluated on these dates (D11):
cd ../train3 && python split3.py /data/v3/sports --dry-run            # look at the dates first
python split3.py /data/v3/sports --out splits/sports_v3.json           # writes the file, .sha256, registry row
git add splits/sports_v3.json splits/sports_v3.json.sha256 split_registry.jsonl   # commit it before training
# 4. comparators on val (no_quote is the D1 primary candidate)
nice -n 10 python evaluator3.py comparators --data-dir /data/v3/sports --split-map splits/sports_v3.json \
    --env3-kw obs_size=80 size_mode=1 Q=1000 pull_s=3600 flat_c=0.5 --variants central_1s pess_1s
# 5. train (each periodic and final val evaluation is a ledger row)
nice -n 10 python train_ppo3.py --data-dir /data/v3/sports --split-map splits/sports_v3.json \
    --env3-kw obs_size=80 size_mode=1 Q=1000 pull_s=3600 flat_c=0.5 close_inv_coef=0.05 close_ramp_s=7200 \
    --timesteps 25e6 --threads 1 --seed 0 --eval-every 5e6 --tag s_main
# 6. evaluate a checkpoint (per-market rows with --markets), and pool the seeds of one config (D13 reading C)
nice -n 10 python evaluator3.py policy --ckpt runs/<run>/final.pt --data-dir /data/v3/sports \
    --split-map splits/sports_v3.json --env3-kw obs_size=80 size_mode=1 Q=1000 pull_s=3600 flat_c=0.5 --markets
python evaluator3.py pool runs/<run_s0> runs/<run_s1> runs/<run_s2>
```

**D14: the quote window is listing to kickoff − 1 h.** The env runs each episode to kickoff and
settles there at the last pre-kickoff mark (`settle_mode` 1 from markets3.bin); it has no option
to end an episode at T − 1 h. So:

- `pull_s=3600` forces the quotes off in the last hour.
- `close_inv_coef` with `close_ramp_s` (shaping, training only) pushes the inventory towards 0
  before then.
- The evaluator reports the D14 P&L directly. `d14/pnl_marked_T1h_total` is the observed wealth at
  the first decision inside the last hour, which is the inventory marked at the mid at T − 1 h.
  `d14/pnl_flat_T1h_total` subtracts |inventory| × `flat_c` there.
- The full-episode P&L, which also carries the inventory through the last hour, stays the Log's
  `pnl`. The difference is the `tk/lt1h` bucket.

Ending episodes exactly at T − 1 h would need a build cut there: build_sports.py could cut at
kickoff − 3600 s, or adapt3.py could set `event_ts` and settle at the T − 1 h row. Neither builder
nor adapter was changed.

`split3.py` keeps a sports game day only if at least one of its episodes starts more than
`--min-pre-hours` (1) before kickoff. The file records, per day, how many episodes reach into each
D14 bucket. Sports have no reward pools in any build yet (ENV3.md open issue 1), so leave
`liq_rewards` off unless `adapt3.py --rewards` was given pools. obs_size 80 still adds the kickoff
clock (obs 50) and the category one-hot.

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
- **D14 time buckets (`tk/<bucket>/*`, `d14/*`).** P&L, decision steps and quoting share per
  time-to-kickoff bucket: `gt24h`, `6_24h`, `1_6h` and `lt1h` (the last hour). Each step's change
  of the observed wealth (obs 34) goes to the bucket of the step's decision time (obs 50). The
  unobserved terminal step takes the remainder, so the buckets add up to the P&L exactly.
  - Wealth in the observation clips at ±0.25 Q. `tk/clip_steps` counts the steps where it does,
    and those steps make the split between buckets unreliable (the total stays exact), so use
    Q ≥ 1000.
  - The buckets need obs_size > 50.
  - The same lines are reported per league group.

With `--markets`, one row per market carries its P&L lines and its own T5 / T6 numbers.

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
5. every comparator.

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
- **Sports reward pools are in no build** (ENV3.md open issue 1).
- **Comparator tuning (D1.4) is not done yet.** It runs once the val split exists.
- **D14 needs pre-game history longer than 1 h per game.** Under the D14 rule, the current sports
  smoke data (≤ 15 min per game) has no quotable day at all.
