# Improvement rounds (DECISIONS.md D3/D4)

`rounds.py` runs the D4 improvement rounds after the main pipeline. Each round
trains a few configs x 3 seeds, logs every val evaluation to `trials.jsonl` with the
round's tag, and compares itself with the best so far. The best so far is the
highest seed-mean val P&L (central fills, 1 s, all 247 val markets, greedy) among
configs with >= 3 seeds that pass D1.3 (fills/market >= 5, T6 share <= 0.5).

```bash
cd explorations/deep_hedging_cvar/polymarket/train && source ../../.venv/bin/activate

python rounds.py status                            # best so far, rounds done, stopping rule, next round
python rounds.py auto --after-main --detach        # overnight: wait for run_all.py, then R1..R6 until the rule stops it
python rounds.py run R1 --detach                   # or one round at a time
python rounds.py run R2 --dry-run                  # plan and time estimate only
python rounds.py finalize --after-main --detach    # D4 finish: 5 seeds (100-104) of the best -> train/selected.json
```

- **Detached runs** log to `runs/rounds/<mode>_<round>_<time>.log`. Per-round state is in
  `runs/rounds/<tag>/round.json` and run dirs `runs/rounds/<tag>/<variant>_s<seed>/`.
  `runs/rounds/history.json` lists the rounds; `runs/rounds/best.json` is set when a
  round improves.
- **Resuming**: run the same command again. Finished runs are reused, never retrained
  or re-ledgered. Killed runs restart, and their partial logs are kept as `*.killedN`. A
  finished round is not redone.
- **Concurrency**: a round refuses to start while `run_all.py`, `sweep.py`, `seeds.py`
  or another round is running. `--after-main` waits for them instead, and
  `--allow-concurrent` overrides the check. Each round runs `--jobs 4` processes with 1
  torch thread each; add `--nice 15` to yield to other work.
- **Base config**: `best.json` if a round set it, else the best D1.3-passing config with
  >= 3 seeds in `runs.jsonl` (e.g. the main sweep's selection or its final seeds). Every
  run trains `--steps` (default 25M, the main sweep's budget), except in R4.
- **Improvement and stopping rule.** A round improves if its best config beats the best
  before it by more than that config's seed standard error (std over seeds / sqrt(3));
  `round.json` also reports the pooled SE. `auto` stops at whichever comes first:
  - two non-improving rounds in a row;
  - `--deadline 2026-10-15T15:30:00+00:00`. Rounds whose estimate would run past it are
    skipped.

  Each finished round appends an `annotation` row to the ledger. Those rows do not
  count towards N.

| round | tag | what | configs x seeds | est. (4 idle cores) |
|---|---|---|---|---|
| R1 | `R1_bc_warmstart` | Behaviour cloning of the tuned fair_value baseline, half the configs from `baselines/tuned.json`, half from `tuned_active.json` (bc.py, 1M samples, value head too), then PPO from it with decaying KL to the BC policy. Sampled: KL coef 0.01-0.3, KL decay {0.25, 0.5, 1}, lr 0.2-2x base, ent_coef 1e-4-1e-2. | 16 x 3 | ~40 min + ~4 min BC |
| R2 | `R2_off_prior` | Fresh MLP with an 'off' prior: the initial logit of 'off' is raised by {0.5, 1, 2, 3} on both price heads, crossed with lr {0.5, 1, 2}x base. With any prior > 0 the greedy policy starts as no-quote ($0, 0 fills). | 12 x 3 | ~30 min |
| R3 | `R3_mark_mode` | mark_mode 1 and 2 (fair value, blend) on the overall best and the best configs of R1 and R2. Skips a mark the config already has. | <= 6 x 3 | <= 15 min |
| R4 | `R4_longer` | The 2 best configs at `--long-steps` (200M). | 2 x 3 | ~40 min |
| R5 | `R5_sim_pretrain` | `--pretrain-steps` {50M, 100M} on the simulator (`data/build_sim`, same obs layout, checked), one pretraining run per variant shared by its seeds, then a 25M fine-tune on the replay. Refused unless `sim/validation_env.json` has `"overall": true` (currently false); `--force-sim` overrides. | 2 x 3 | ~25 min |
| R6 | `R6_recurrent` | PufferNet replica (MinGRU, carried state), layers {1, 2} x hidden {64, 128}, fresh init. ~65k SPS per process. | 4 x 3 | ~21 min |

`auto` over all six takes about 2.5-3 h.

`finalize` trains 5 fresh seeds (100-104) of the best config at its own budget
(`--final-steps` to change it). It writes `train/selected.json` in the format
`final/make_plan.py` reads; any previous file is kept as `selected.json.before_finalize_<time>`.

## Smoke-tested (2026-10-15)

- **Stand-in, every round end to end** at `nice -n 15`, < 1M steps per run, on the
  stand-in ledgers: BC, KL, off prior, mark, longer, pretrain then fine-tune,
  PufferNet, comparison, history, stopping rule, `auto` stop and `finalize` publish.
  Command: `python rounds.py run R1 --env standin --steps 2e5 --configs 2 --bc-envs 64
  --bc-steps 64 --jobs 2 --nice 15 --suffix smoke` (and the same for R2-R6).
- **Real env, with `--no-val`** (nothing ledgered):
  - pretraining on `data/build_sim`, whose markets.bin and meta.json hashes are pinned
    in the config, then fine-tuning on the replay from that checkpoint;
  - BC with the tuned_active teacher;
  - `--off-prior 2`: the greedy policy quotes nothing at init.
- **Dry runs on the real env** print the plans and the estimates above.
