# PPO training stack (PufferLib 3.0, CPU)

Trains the Polymarket market maker (`../env.py`, `../mm.h`) with PufferLib 3.0's
PPO on the train split, scores it greedily on every validation market, logs every
validation-scored run to the ledger, sweeps, selects by the seed mean and retrains
the selected config on fresh seeds. The protocol is `../eval/EVALUATION.md`.
5.0 portability follows `../PUFFER5.md`.

```bash
source ../../.venv/bin/activate        # from polymarket/train
python test_train.py                   # stack tests (stand-in env, ~2 min)
python probes.py --env real            # learnability probes on the real env's obs
python train_ppo.py --env real --timesteps 50e6 --eval-every 10e6 --out runs/dev1
python bc.py --env real --teacher fair_value --value --out runs/bc/fv.pt            # optional warm start
python train_ppo.py --env real --init runs/bc/fv.pt --kl-coef 0.1 --kl-decay 0.5 ...
python sweep.py --name s1 --env real --trials 24 --steps 20e6 --jobs 2 --threads 1 \
                --reseed-top 3 --reseed-n 3
python select.py --tag s1              # -> runs/sweeps/s1/selected.json
python seeds.py --selected runs/sweeps/s1/selected.json --jobs 2 --threads 1
python bench.py --env real --threads 1 2 4 --num-envs 4096 16384 --horizon 64 128
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
| `train_ppo.py` | CLI training run: periodic eval (val headline + train split for the gap), checkpoints, `log.jsonl`, final eval under the variants, `result.json`, one ledger row. |
| `ledger.py` | `trials.jsonl` (append under a file lock), `config_hash` = sha256 of the canonical config without the seed. |
| `sweep.py` / `select.py` / `seeds.py` | Optuna TPE sweep on val; pick by seed mean over >= 3 seeds; retrain the pick on 5 fresh seeds. |
| `bc.py` | Behaviour cloning of a teacher (baselines' `fair_value` through `../baselines/obs_spec.py`, else a simple obs quoter) on train markets; checkpoint loadable with `--init`. |
| `probes.py` | Learnability probes (join both sides; fixed action; obs-dependent teacher). |
| `bench.py` | Throughput grid. |
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
- One ledger row per run (`trials.jsonl`; stand-in runs go to
  `runs/standin_trials.jsonl`) with config hash, seed, full config, step/time, the
  final checkpoint and its sha256, all val metrics per variant, the train-split
  metrics, and `n_val_evals` (periodic val evals count; selection uses the final
  checkpoint only). `--no-val` dev runs never touch val and write no row.
  Note: EVALUATION.md names `polymarket/results/trials.jsonl` as the ledger; this
  stack writes `train/trials.jsonl` as instructed (`--ledger` to change). Merge
  or point both at one file before the final count N.
- `select.py`: seed mean of val `pnl_total` (central, 1 s) over configs with >= 3
  seeds, optional `--cvar-floor` (best baseline's CVaR95) and `--max-inv`,
  tie-break on the pessimistic seed mean; reruns of a seed are ignored; writes
  N (ledger rows) and the total number of val evaluations.
- `seeds.py`: 5 fresh seeds (100-104, unused by the sweep) of the selected
  config; mean / min / max / std across seeds per variant.

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

What moved the number:
- **The PPO update dominates.** Shared trunk (~1.3x over separate actor/critic) and
  1 update epoch (~2x over 2) are the big levers. Minibatch size does not change
  throughput (within noise), so the default is 8192 (32 Adam steps per 262k batch).
- **Scoring actions**: `pufferlib.pytorch.sample_logits` pads the three heads to
  (3, B, 7) with -inf; per-head log-softmax gives identical values at 1/3 of the
  cost (`policy.logprob_entropy`, ~20 -> 6 ms per 16k minibatch).
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

All on 2026-10-15 (`runs/`, git-ignored).
- `test_train.py`: 9/9 pass (sampler and scorer agree with `sample_logits`;
  PufferNet scan == stepping; PufferNet == `../puffer5/puffernet_mm.py` on the same
  weights; training recomputes rollout log-probs and values exactly for MLP and
  PufferNet with terminals in segments and carried state; reward-clip stats;
  evaluator covers every val market once, deterministic, refuses split 2; config
  hash; PPO learns the join probe).
- **Real env** (`../test_env.py` passing at 05:0x, obs 50):
  - `train_ppo.py --env real --timesteps 6e6 --eval-every 3e6 --tag smoke`: 44 s,
    159k SPS median; periodic + final val evals (central_1s, pess_1s), train-split
    evals, 247 per-market val records, one ledger row in `trials.jsonl`.
  - Probes, 2M steps each (~33 s): join (env-native `reward_mode=1`) 1.00, const 1.00,
    teacher (tuned fair_value baseline) 0.90 (bar 0.9; BC on the same teacher
    plateaus at 0.95 / 0.95 per price head).
  - `bc.py --teacher fair_value` (tuned params from `../baselines/tuned.json`):
    262k samples, 90% joint / 95% per-head accuracy on held-out envs. PPO
    `--init` + KL 0.1 decaying over the first half: train-market greedy P&L -131
    at 2M steps, -29 at 4M (cold start: -3696 at 3M, -3233 at 6M).
  - Shared vs separate MLP, 20M steps, `--no-val`, train-market greedy P&L
    (obs-48 env of 04:40): shared -1728 (48k SPS under load 10), separate -2699
    (44k). Shared became the default.
- **Stand-in**: `sweep.py` (3 trials + reseed of the best to 3 seeds) ->
  `select.py` (picked the only config with 3 seeds; its seeds scored 118 / 60 / 125)
  -> `seeds.py` (2 fresh seeds, same config hash reproduced), BC (99.4%) +
  `--init` + KL.

## Open items

- **Learning speed with the defaults is poor.** Cold-start PPO at lr 3e-4 barely
  moves the policy: after 20M steps head entropies were 1.93 / 1.93 / 1.09 (max
  1.95 / 1.95 / 1.10), approx_kl ~1e-4, clipfrac 0, explained variance ~0, and
  train-market P&L still negative. The sweep's lr range goes to 3e-3, and the
  smaller default minibatch doubles the Adam steps; the BC warm start is the other
  route (see above). Nothing was swept yet: no val number exists except the one
  `smoke` row.
- **Ledger location**: EVALUATION.md T4 names `polymarket/results/trials.jsonl`;
  this writes `train/trials.jsonl` as instructed. Merge before counting N.
- **Selection constraints** need inputs: `select.py --cvar-floor <best baseline
  CVaR95 on val>` and `--max-inv <risk limit>` (Q = 100 shares is the env's hard cap).
- **Env drift**: the env changed under this work (obs 48 -> 50, config interning,
  new Log fields). The config hash now includes the resolved env parameters and
  the sha256 of mm.h / binding.c / env.py / binding .so, so runs on different env
  versions never pool in select.py. Checkpoints from the obs-48 env are unusable.
- The teacher probe sits at the 0.9 bar after 2M steps; a 6M-step run is in
  `runs/probes_real_v2.jsonl`.
- PufferNet is 2-6x slower than the MLP on CPU; 4 layers (5.0 parity) is ~30k SPS.
