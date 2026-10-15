# PPO training stack (PufferLib 3.0, CPU)

Trains the Polymarket market maker (`../env.py`, `../mm.h`) with PufferLib 3.0's
PPO on the train split, scores it greedily on every validation market, logs every
validation-scored run to the ledger, sweeps, selects by the seed mean and retrains
the selected config on fresh seeds. The protocol is `../eval/EVALUATION.md`.
5.0 portability follows `../PUFFER5.md`.

## Full run (one command, resumable)

```bash
cd explorations/deep_hedging_cvar/polymarket/train
source ../../.venv/bin/activate
python run_all.py --dry-run                 # print the stages and commands
python run_all.py                           # preflight -> probes -> sweep -> select -> seeds -> publish
# options: --cvar-floor <best baseline val CVaR95> --max-inv <limit> --require-mm  (selection rule)
#          --name main --trials 64 --steps 25e6 --jobs 4 --final-steps 100e6 --seed-jobs 5
```

Re-run the same command after a kill or crash: finished stages are skipped, the sweep
resumes its Optuna study (trials a kill interrupted are marked stale and their
parameters re-queued; their partial run dirs are kept as `*.killedN`), finished runs
are never retrained or re-ledgered, and seeds skip finished seeds. `--from select`
redoes the later stages. State and stage logs: `runs/pipeline/<name>/`.

Outputs: `runs/sweeps/<name>/` (Optuna study, `sweep_log.jsonl`, one dir per run with
`final.pt`, `log.jsonl`, `result.json`, `val_markets.jsonl`), `runs/sweeps/<name>/selected.json`
(select.py), `runs/seeds/<name>/` (5 seed runs, `seeds.json`), and
**`train/selected.json`** in the format `../final/make_plan.py` reads
(`{"selected": {"cfg", "cfg_hash", "runs": [{"seed", "ckpt", "ckpt_sha256"}]}}`, the
5 final seeds; checked with its `_seed_entries`), plus the ledger rows below.

**Wall time on 4 idle cores** (measured: 158k agent steps/s per 1-thread process,
unchanged with 2 concurrent; 4 parallel 1-thread runs measured 133-151k each):
preflight tests ~2 min; probes 3 x 2M steps ~2 min; sweep 64 trials x 25M steps at
4 in parallel, ~175 s per H=128 trial (H=256 slower), ~55-65 min; re-seeding the 4 best
configs to 3 seeds (8 runs) ~7 min; 5 final seeds x 100M steps in parallel ~15 min.
**Total ~1.4 h (range 1.3-1.6 h).** Other jobs on the box scale it up: at 05:30 three
other agents' single-core jobs cut each trial to 17-50k SPS (~3x slower).

The stand-in plumbing test of the same pipeline (4 trials, kill after 25 s, resume,
reseed, select, 5 seeds, publish) ran end to end in 1.2 min:
`python run_all.py --env standin --name pipe_smoke --trials 4 --steps 3e5 --jobs 2
--reseed-top 1 --final-steps 3e5 --seed-jobs 2 --probe-steps 2e5 --skip-tests
--train-args "--num-envs 512 --minibatch-size 8192"`.

## Single steps

```bash
python test_train.py                   # stack tests (stand-in env, ~2 min)
python probes.py --env real --steps 2e6                        # learnability probes
python train_ppo.py --env real --timesteps 50e6 --eval-every 10e6 --threads 1 --tag dev --out runs/dev1
python train_ppo.py --env real --timesteps 20e6 --no-val --out runs/dev2   # never touches val
python bc.py --env real --teacher fair_value --value --out runs/bc/fv.pt  # optional warm start
python train_ppo.py --env real --init runs/bc/fv.pt --kl-coef 0.1 --kl-decay 0.5 --tag bc ...
python sweep.py --name s2 --env real --trials 64 --steps 25e6 --jobs 4 --threads 1 --reseed-top 4 --reseed-n 3
python select.py --tag s2 [--cvar-floor X --max-inv Y --require-mm]   # -> runs/sweeps/s2/selected.json
python seeds.py --selected runs/sweeps/s2/selected.json --timesteps 100e6 --jobs 5 --threads 1
python summary.py --tag s2 --by-config  # table: val P&L, fills/mkt, attribution, T6/T5, diagnostics
python bench.py --env real --threads 1 --num-envs 4096 --horizon 64 --minibatch 8192
```

Everything accepts `--env standin` (`standin.py`, a numpy toy with the same
spaces) for testing without the real env. `--env auto` (default) picks the real
env when `../env.py` exists.

## Files

| file | what |
|---|---|
| `policy.py` | `MLPPolicy` (default; tanh 2x128, shared trunk by default, three categorical heads 7/7/3 + value) and `PufferNetPolicy` (PufferLib 5.0's PufferNet: bias-free encoder, MinGRU layers with highway, fused decoder; recurrent). PufferLib 3.0 contract: `forward_eval(obs, state) -> (logits tuple, value)`, `forward` for training. `greedy_actions`, `logprob_entropy`, `load_checkpoint`, `PufferNetPolicy.save_puffer5_bin` (5.0 `.bin`). |
| `puffer3.py` | `Trainer`, a subclass of PufferLib 3.0's `PuffeRL` (same PPO, buffers, clamp, advantage, losses) with: recurrent state carried across epochs, reset on terminals and stored at each segment's row 0 so training replays the rollout's recurrence (5.0's `reset_every_horizon=0`); per-epoch logs and diagnostics; optional kickstarting KL to a teacher; faster sampling/scoring with identical values. |
| `envs.py` | Adapters. `RealAdapter`: train env = `MmEnv(split=0)` (random market, random start in the first 20%); eval env = `MarketSetEnv`, every market of a split once from offset 0 (one 1-env handle per market sharing one interned config, seeded so handle j plays mm.h's list[j]; checks the contract-length one-hot at reset and the decision count at the end); fill/latency variants; obs-based groups (contract length) and teacher. `StandInAdapter` for the toy. Test split refused. |
| `evaluator.py` | `evaluate(policy, adapter, split)`: argmax actions until every market finished once; per-market records -> P&L total/mean/median, CVaR95 (`../eval/stats.cvar`), max inventory, attribution sums, greedy action histograms. |
| `train_ppo.py` | CLI training run: periodic eval (val headline + train split for the gap), checkpoints, `log.jsonl`, final eval under the variants, `result.json`; one ledger row per val evaluation, one run-index row. |
| `ledger.py` | `trials.jsonl` (one row per validation evaluation) and `runs.jsonl` (one row per run), appended under a file lock; `config_hash` = sha256 of the canonical config without the seed. |
| `run_all.py` | The resumable pipeline (above). |
| `sweep.py` / `select.py` / `seeds.py` | Optuna TPE sweep on val (space: lr, ent_coef, gamma, GAE lambda, inv_penalty, hidden, mark_mode), resumable; pick by seed mean over >= 3 seeds (run index); retrain the pick on 5 fresh seeds, optionally longer. |
| `bc.py` | Behaviour cloning of a teacher (baselines' `fair_value` through `../baselines/obs_spec.py`, else a simple obs quoter) on train markets; checkpoint loadable with `--init`. |
| `probes.py` | Learnability probes (join both sides; fixed action; obs-dependent teacher). |
| `bench.py` | Throughput grid. |
| `summary.py` | Markdown table of runs (by ledger tag or run dirs; `--by-config` for seed means): val P&L, CVaR95, fills/market, attribution, T6/T5 shares, D1 flag, greedy action mix, pessimistic P&L, train-split P&L, last diagnostics. |
| `standin.py`, `test_train.py` | Toy env; tests. |

## Training details

- **PPO** is PufferLib 3.0's: V-trace-clipped GAE per `bptt_horizon` segment
  (recomputed with the current ratios before every minibatch), clipped policy and
  value losses, entropy bonus, Adam, cosine LR annealing, rewards clamped to [-1, 1].
  Defaults (`train_ppo.DEFAULT_CFG`): MLP 2x128 tanh shared trunk, 4096 envs,
  horizon 64 (batch 262144), minibatch 8192, 1 update epoch, lr 3e-4, ent 0.01,
  gamma 0.995, lambda 0.95, clip 0.2, vf_coef 0.5, max_grad_norm 0.5, 2 threads.
- **Horizon** 64, a multiple of 8. The env starts each episode at a random second in
  the first 20% of the window and episodes are 300-2880 decisions, so terminals do
  not land on a segment's row 0 systematically: `term_row0_frac` (share of
  terminals on row 0, whose reward puff_advantage never reads) is logged every
  epoch; measured 0.009-0.034 on the real env vs 1/64 = 0.016.
- **Rewards.** The env scales by `reward_scale` (default 1/Q) and clips at +-1
  itself. Logged: `clip_frac_reward` (share of rewards at +-1, before PuffeRL's clamp),
  `clip_mass_reward`, and the env's own `env/reward_clip_frac`. Measured 0.0 with
  the defaults (per-step reward std 0.012 / 0.006 / 0.004 for 5m / 15m / 4h).
- **`mark_mode`** (reward and wealth mark: 0 mid, 1 fair value, 2 blend) is a
  passthrough: `--mark 1` (alias `--mark-mode`). Any other env parameter:
  `--env-kw name=value` (e.g. `reward_mode=1`, `start_frac=0.5`, `dt=2`).
- **Recurrent policy** (`--policy puffernet --layers 1`): use_rnn; see `puffer3.py`.
  Default layers 4 to match `../puffer5/polymarket_mm.ini`. `test_train.py` checks
  that training recomputes the rollout's log-probs exactly (ratio 1 before the
  first update) with terminals inside segments and carried state, that the
  sequence scan equals stepping, and that the forward equals
  `../puffer5/puffernet_mm.py` (the replica checked against 5.0's C forward) on
  the same weights.
- **Warm start** (`bc.py` + `--init`) and **kickstarting** (`--kl-coef c
  --kl-decay f`: loss += c_t KL(teacher || policy), c_t linear from c to 0 over the
  first fraction f of training; teacher = the `--init` checkpoint unless
  `--kl-teacher`). Both are part of the config hash (with the checkpoint sha256).

## Evaluation and the ledger (EVALUATION.md T4)

- Greedy (argmax per head) on **all** markets of the split, each exactly once,
  from start offset 0; deterministic (two runs give identical numbers). Val
  (247 markets) takes ~1.2 s, train (431) ~2 s.
- Variants: headline `central_1s` (fill_model 1, latency 1); the final evaluation
  also runs `pess_1s` (tie-break); `central_2s`, `central_0s`, `opt_1s` with
  `--variants`.
- **Ledger** `train/trials.jsonl`: one row per validation evaluation (kind `val_eval`,
  `split: 1`; periodic and final, every variant), written as it happens, so a crash
  cannot lose one. This is the unit `../final/common.py count_trials` counts (one val
  evaluation per record, every file named `trials.jsonl` under polymarket/). Run
  summaries (config, final checkpoint + sha256, final metrics per variant) go to
  `train/runs.jsonl`, which select/summary/seeds read; the sweep's own log is
  `sweep_log.jsonl` so it is not counted twice. Stand-in runs use
  `runs/standin_trials.jsonl` / `runs/standin_runs.jsonl`. `--no-val` dev runs never
  touch val and write neither. Note: EVALUATION.md T4 names
  `polymarket/results/trials.jsonl`; final/ globs every `trials.jsonl`, so this file is
  counted either way.
- The ledger was migrated on 2026-10-15 from one row per run to one row per evaluation
  (regenerated from each run's `log.jsonl`; 27 evaluations from 7 runs, backup in
  `runs/trials.jsonl.bak_before_migration`). Sweep `s1` (25M steps/trial) was killed by
  the user's pause at ~trial 5: its 2 finished trials (4 val evaluations) are in the
  ledger with `status: aborted by user pause`, plus an annotation row; trials 0, 1, 4, 5
  were killed before any val evaluation (partial logs in `runs/sweeps/s1/`). The
  pipeline uses a new name (`main`) by default; `--name s1` would resume s1 instead
  (same 25M budget, its 4 killed trials re-queued).
- **Config hash** includes the resolved env parameters and the sha256 of mm.h,
  binding.c, env.py and the binding .so, so runs on different env builds never pool.
- `select.py`: seed mean of val `pnl_total` (central, 1 s) over configs with >= 3
  seeds, optional `--cvar-floor` (best baseline's CVaR95) and `--max-inv`,
  tie-break on the pessimistic seed mean; reruns of a seed are ignored; writes
  N (ledger rows) and the total number of val evaluations.
- `seeds.py`: 5 fresh seeds (100-104, unused by the sweep) of the selected
  config; mean / min / max / std across seeds per variant.

## Protocol flags in every evaluation

From the per-market Log records (`evaluator.protocol_flags`): `fills_per_market`;
`t6_share` = inventory/resolution P&L / total P&L when the total is positive
(T6: > 0.5 = directional) and `t6_share_pos` (positive inventory P&L over the sum of
positive components, defined for any total); `t5_max_share` = the largest share of
fill P&L (spread capture + 60 s markout) under one artifact flag kept by
DECISIONS.md D2 (inspread, crossed, stale, onesided, last30; NaN when fill P&L <= 0);
`makes_markets` = D1.3 (fills >= 5 per market, T6 <= 0.5, T5 <= 0.25).
`select.py --require-mm` restricts the choice to market-making configs.

## Diagnostics (log.jsonl)

Every epoch (`type: train`): `sps`, `sps_rollout`, `sps_update`, `losses/*`
(policy, value, `entropy` and `entropy_head0..2`, approx_kl, clipfrac,
explained_variance, kl_teacher), `clip_frac_reward`, `clip_mass_reward`,
`term_row0_frac`, `rew/<5m|15m|4h>/{mean,std}` (raw reward per contract length,
credited to the market the action was taken in), `act/<bid|ask>/<off|improve|join|behind>`
and `act/size/<1x|2x|5x>` (rollout actions), `env/*` (the env Log: pnl, attribution,
exploit-flag shares, reward_clip_frac, ...).
Every evaluation (`type: eval`): `diag/*` = the above averaged since the
previous evaluation, `val/<variant>/*` and `train/*` (greedy metrics and greedy
action histograms), `gap/score_mean` and `gap/pnl_mean` (train minus val, per market).

## Throughput (CPU, 4 shared cores)

Agent steps/s of a full PPO epoch (rollout + update), median of 2 epochs after a
warm-up, real env, from `bench.py` (`runs/bench.jsonl`). The 4 cores are shared with
other agents; `load` is the 1-min load average at the start of each setting, and
numbers move +-15% with it. Env alone (random actions, no policy): 3.6-5.2 M steps/s.

| policy | threads | envs x horizon | minibatch | epochs | total | rollout | update | load |
|---|---|---|---|---|---|---|---|---|
| MLP separate (stock-like start) | 1 | 4096 x 64 | 16384 | 2 | 63k | 316k | 78k | 3.4 |
| MLP separate | 1 | 4096 x 64 | 16384 | 1 | 107k | 315k | 163k | 8.4 |
| **MLP shared** | 1 | 4096 x 64 | 16384 | 1 | 133-143k | 424-439k | 193-213k | 4-9 |
| MLP shared | 1 | 4096 x 128 | 16384 | 1 | 134k | 423k | 197k | 4.2 |
| MLP shared | 1 | 16384 x 64 | 16384 | 1 | 124k | 331k | 198k | 3.8 |
| MLP shared | 1 | 16384 x 128 | 16384 | 1 | 132k | 421k | 191k | 3.2 |
| MLP shared | 1 | 4096 x 64 | 4096 / 8192 / 32768 / 65536 | 1 | 148k / 135k / 163k / 132k | ~440k | | 2-3 |
| **MLP shared** | **2** | 4096 x 64 | 16384 | 1 | **189k** | 512k | 300k | 2.1 |
| MLP shared | 2 | 16384 x 128 | 16384 | 1 | 178k | 573k | 258k | 1.9 |
| MLP shared | 4 | 4096 x 64 | 16384 | 1 | 176k | 465k | 284k | 1.5 |
| MLP shared + torch.compile (train and rollout) | 1 / 2 | 4096 x 64 | 16384 | 1 | 146k / 179k | 376k / 411k | 238k / 319k | 2.2 |
| PufferNet 1 layer, H 128 | 1 / 2 | 4096 x 64 | 16384 | 1 | 67k / 73k | 230k | 94k / 108k | 2.1 |
| PufferNet 4 layers, H 128 (5.0 ini) | 1 / 2 | 4096 x 64 | 16384 | 1 | 23k / 33k | 84k / 120k | 31k / 46k | 2.2 |
| end-to-end `train_ppo.py` defaults (incl. evals) | 2 | 4096 x 64 | 8192 | 1 | 159k median | | | ~3 |
| **current defaults, new sampler** (1 process / each of 2 concurrent) | 1 | 4096 x 64 | 8192 | 1 | **158k / 157k** | 592k / 603k | 216k / 212k | 0.8 |

What moved the number:
- **The PPO update dominates.** Shared trunk (~1.3x over separate actor/critic) and
  1 update epoch (~2x over 2) are the big levers. Minibatch size does not change
  throughput (within noise), so the default is 8192 (32 Adam steps per 262k batch).
- **Sampling and scoring actions**: `pufferlib.pytorch.sample_logits` pads the three
  heads to (3, B, 7) with -inf. Rollout: `policy.sample_actions` samples each head
  by inverse CDF from one uniform per head, reusing the log-softmax for the log-prob
  (per 4096-row step, 1 thread: sample_logits 7.6 ms, multinomial per head 3.7 ms,
  inverse CDF 1.9 ms; forward 3.8 ms; env 0.8 ms). Update: per-head log-softmax
  gives identical values at 1/3 of the cost (`policy.logprob_entropy`, ~20 -> 6 ms
  per 16k minibatch). `test_train.py` checks values and sampling frequencies.
- **Parallel runs, not threads**: 4 processes x 1 torch thread gave 133-151k SPS
  each (~550k aggregate) while 1 process x 2 threads gives ~160-190k. Sweeps and
  seeds use `--jobs 4 --threads 1`.
- **Threads**: 2 is best here; 4 gives nothing more. With the default OpenMP spin
  wait, 4 threads on a loaded box collapsed to 2.9k SPS; every entry point sets
  `OMP_WAIT_POLICY=PASSIVE` (4 threads then: 46k under load 7, vs 2.9k).
- **num_envs / horizon**: 4096-16384 and 64-128 are all within noise; 4096 x 64
  keeps batches (262k) small enough for frequent updates.
- **torch.compile** works on CPU (inductor, Python 3.13) but gains nothing net
  (stock 3.0 only compiles the training forward; `compile_eval` also compiles the
  rollout's, which got slower). Off by default.
- **PufferNet**: the MinGRU scan is sequential on CPU. Two fixes took 1 layer from
  30k to 67k: `unbind` instead of per-step slicing (a slice's backward wrote a full
  (S, T, H) zero tensor per step, quadratic in T) and contiguous gate matmuls.

## Smoke-tested

All on 2026-10-15 (`runs/`, git-ignored), env frozen at mm.h `746dd7de` unless noted.
- `test_train.py`: 10/10 pass (sampler/scorer values and sampling frequencies vs
  `sample_logits`; PufferNet scan == stepping; PufferNet == `../puffer5/puffernet_mm.py`
  on the same weights; training recomputes rollout log-probs and values exactly for
  MLP and PufferNet with terminals in segments and carried state; reward-clip stats;
  evaluator covers every val market once, deterministic, refuses split 2; config hash;
  PPO learns the join probe in 1M steps).
- `run_all.py` end to end on the stand-in, including kill + resume (1.2 min).
- Real env:
  - `../puffer5/score_val.py` path: a 5.0 `.bin` written by `PufferNetPolicy.save_puffer5_bin`,
    loaded by its `EvaluatorPolicy`, scored by `evaluator.evaluate` through `RealAdapter`
    on all 431 train markets: identical P&L to the same weights in `PufferNetPolicy`.
  - Probes: join (env-native `reward_mode=1`) 1.00 at 0.5M and 2M steps; const 1.00
    at 2M; teacher (tuned fair_value) 0.8995 at both 2M and 6M steps, the same ceiling BC
    reaches on that teacher (~0.95 per price head): the teacher depends on something the
    MLP does not recover from the obs, not a PPO defect.
  - `bc.py --teacher fair_value` (params from `../baselines/tuned.json`, re-tuned on the
    frozen env): 65k-262k samples, 79-90% joint accuracy. Earlier (pre-freeze env)
    PPO `--init` + KL 0.1: train-market greedy P&L -131 at 2M, -29 at 4M steps vs
    -3233 at 6M from scratch.
  - Throughput: 158k SPS end to end per 1-thread process; 2 concurrent: 157k each.

## Results so far (validation, central fills / 1 s, greedy, all 247 markets)

Single seeds, all in the ledger (27 val evaluations). Bar (DECISIONS.md D1): ~$0 per
market (no-quote). From `python summary.py`:

| run | lr | ent | mark | H | steps | val P&L | /market | CVaR95 | fills/mkt | spread | markout60 | inventory | rebates | T6+ share | 2-sided uptime | bid off | pess P&L | train P&L/mkt | entropy/head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| explore mm1_lr1e-3 | 1e-3 | 0.01 | fv | 128 | 40M | -177 | -0.72 | -45.2 | 32.4 | 254 | -639 | 174 | 33.8 | 0.38 | 0.23 | 0.28 | -280 | -1.67 | 1.88/1.88/1.06 |
| explore mm0_lr1e-3 | 1e-3 | 0.01 | mid | 128 | 40M | -239 | -0.97 | -47.2 | 40.0 | 352 | -771 | 140 | 39.8 | 0.26 | 0.44 | 0.12 | -253 | -2.29 | 1.89/1.88/1.06 |
| explore mm1_lr3e-4 | 3e-4 | 0.01 | fv | 128 | 40M | -404 | -1.63 | -41.4 | 33.0 | 268 | -825 | 116 | 36.5 | 0.28 | 0.32 | 0.28 | -270 | -1.72 | 1.89/1.91/1.07 |
| explore mm0_lr3e-4 | 3e-4 | 0.01 | mid | 128 | 40M | -540 | -2.19 | -34.6 | 38.0 | 304 | -845 | -38 | 38.5 | 0 | 0.44 | 0.09 | -327 | -2.47 | 1.92/1.93/1.08 |
| s1 t003 (aborted sweep) | 8.8e-4 | 2.4e-4 | mid | 64 | 25M | -654 | -2.65 | -52.1 | 39.1 | 370 | -1271 | 197 | 50.7 | 0.32 | 0.18 | 0.24 | -534 | -7.47 | 1.50/1.75/1.04 |
| s1 t002 (aborted sweep) | 1.3e-4 | 0.022 | mid | 64 | 25M | -748 | -3.03 | -48.7 | 47.0 | 498 | -1471 | 150 | 75.3 | 0.21 | 0.41 | 0.45 | -644 | -8.49 | 1.94/1.94/1.10 |

What they show: PPO learns slowly and still loses at 25-40M steps. The greedy policies
keep quoting (27-47 fills/market, mostly 'behind' the touch, two-sided uptime
0.2-0.4) and lose to adverse selection exactly like the baselines: the 60 s markout
(-640 to -1470) exceeds spread capture (+250 to +500). They do not learn not to quote.
The sampling policy stays near uniform (price-head entropy 1.88-1.94 of 1.95), and
approx_kl/clipfrac are tiny at lr 3e-4; lr 1e-3 and the fair-value mark (mark_mode 1)
were the best of the four (P&L improving -740 -> -177 over training).
What I would try, in order: the sweep as planned (lr up to 3e-3, low ent_coef,
inv_penalty, mark_mode); BC warm start from the fair-value / no-quote teacher with
the decaying KL (`bc.py`, `--init`, `--kl-coef`), which started near $0 on train
markets; a stronger 'off' prior at init; longer budgets for the selected config.

## Open items

- **Selection constraints** need inputs: `run_all.py --cvar-floor <best baseline val
  CVaR95> --max-inv <limit>` (Q = 100 shares is the env's hard cap); `--require-mm`
  applies D1.3 (fills >= 5/market, T6 <= 0.5).
- **Ledger location**: EVALUATION.md T4 names `polymarket/results/trials.jsonl`; this
  writes `train/trials.jsonl` (final/ counts every `trials.jsonl`).
- PufferNet is 2-6x slower than the MLP on CPU; 4 layers (5.0 parity) ~30k SPS. The
  pipeline sweeps the MLP only.
