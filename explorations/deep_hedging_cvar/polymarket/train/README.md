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
  Defaults (`train_ppo.DEFAULT_CFG`): 4096 envs, horizon 64 (batch 262144),
  minibatch 16384, 1 update epoch, lr 3e-4, ent 0.01, gamma 0.995, lambda 0.95,
  clip 0.2, vf_coef 0.5, max_grad_norm 0.5.
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

THROUGHPUT_TABLE

## Smoke-tested

SMOKE_LIST

## Open items

OPEN_ITEMS
