# Remote worker protocol (protocol v2, DECISIONS.md D7)

You are one of several parallel cloud sessions working on the Polymarket market-making RL project.
Each worker runs in its own container. The coordinating session is
`session_01V31NsSvTVCgJBT18w8sxcp`; it aggregates everyone's results and picks the final
method. You get a worker name (`w1`…`w5`) and an assignment. Everything below applies to every
worker.

Read these first:
- `polymarket/eval/DECISIONS.md` (D1–D7);
- `polymarket/final/RESULTS.md` (v1 test: no edge, adverse selection dominates);
- `polymarket/train/OVERNIGHT.md`;
- `polymarket/ENV.md`;
- `polymarket/train/README.md`.

## Hard rules

1. **Never touch the v2 test set.** That means `data/build_jun15/`, the Marketlens Jun 15 data
   and any script that builds it. Never set `MM_FINAL=1` and never pass `allow_test=1`.
   `train/envs.py` refuses `build_jun15` paths; don't work around that.
2. **Select only on val v2:** `data_dir=<abs path>/polymarket/data/build_v2`, split 1. That is
   Aug 23 + Aug 24, 374 markets. Train on split 0, Aug 21–22.
3. **Log every val evaluation** to your own ledger with
   `--ledger polymarket/train/workers/<name>/trials.jsonl` and `--tag <name>_<experiment>`. Never
   delete ledger rows, including rows from failed or aborted runs.
4. **Commit only your own folder,** `polymarket/train/workers/<name>/`. Never run
   `git add -A`. Never edit shared code (`mm.h`, `train/*.py`, `data/*`, `final/*`,
   `baselines/*`). If you need a code change, copy the module into your folder and change the copy.
5. **Push safely.** Push to branch `claude/eloquent-feynman-n78cn1` with
   `git pull --rebase origin claude/eloquent-feynman-n78cn1 && git push origin HEAD:claude/eloquent-feynman-n78cn1`,
   retrying on rejection. Never force-push and never rewrite history.
6. **CPU.** 4 cores. Run at most 4 training processes, with `--threads 1` and
   `OMP_WAIT_POLICY=passive OMP_NUM_THREADS=1`.
7. **Stay alive while runs are in flight.** The container is reclaimed when your session is idle,
   and that kills detached runs. While runs are in flight, keep a harness-tracked
   `run_in_background` Bash loop that waits while `pgrep -f "[t]rain_ppo.py"` matches, with timeout
   7200000. Re-arm it before it expires. Don't end your turn while runs are in flight unless they
   can resume.
8. **Deadline: 2026-10-16 02:00 UTC.** Push your final report by then; the coordinator freezes at
   03:00.

## Setup (about 15 min)

Run from the repo root.

```bash
cd explorations/deep_hedging_cvar/polymarket
bash setup.sh                        # ../.venv: torch 2.7.1 CPU + PufferLib 3.0 (numpy pin relaxed)
source ../.venv/bin/activate && pip install -q scipy optuna
cd data && python3 build.py && python3 load.py && /usr/bin/python3 features.py && python3 build_v2.py && cd ..
python setup.py build_ext --inplace  # C env binding
python test_env.py                   # must print ok (~2 min)
```

Check these sha256 values (`sha256sum`). If any differs, stop and report it, because results
wouldn't be comparable:

| file | sha256 |
|---|---|
| `mm.h` | `9189f83adbb3f42ab803aca998ff97954c7cc092ffd65401d2b835abaa98a8e9` |
| `data/build/ticks.bin` | `8539fe962cd5cc5f6d0934291ea9aa0b8ea98b8096f073ea29f9bd05c4437521` |
| `data/build/feat.bin` | `122176d8a2795551ed84376a34e1236f88e1cedc7bb7ccef046616767dd6435c` |
| `data/build/fills.bin` | `8c59307b8eed1df7ff53db9176d91bff4be9f9925ddddc29ecf1e4092769542f` |
| `data/build/book.bin` | `c9caad1c3c01edaaf6b358504a42f4b0032aa6737d1b06d8757431693634d934` |
| `data/build_v2/markets.bin` | `a35032d146d7802ec68b63a938fcfb0e37b322132c64efcee5df4e40b9e7d1f8` |

## How to train and evaluate

`train/train_ppo.py --help` lists the knobs:
- reward and inventory: `--reward-mode 2 --markout-h`, `--inv-penalty`, `--close-inv-coef/--close-ramp`;
- marking: `--mark-mode`;
- architecture: `--policy`, `--hidden`, `--layers`;
- warm start: `--init` (BC checkpoints from `bc.py`), `--kl-coef/--kl-decay/--kl-teacher`;
- env kwargs: `--env-kw key=value`.

Pass `--env-kw data_dir=$PWD/data/build_v2` so that val evaluations use v2. Check that the val
evaluation reports 374 markets.

The current best (v1, R7 `c02_h120`) is the config in `train/selected.json`: reward_mode 2,
markout_h 120. Its v2 val numbers are being computed by the coordinator.

Baselines on v1 val (central fills, 1 s):
- no-quote: $0;
- best forced-to-trade baseline (as_logit, active): −$0.30 per market.

Every method is judged on **central fills, 1 s latency, val v2, seed mean of at least 3 seeds**.
It must also pass D1.3: at least 5 fills per market, T6 directional share ≤ 50%, T5 artifact
share ≤ 25%.

## Outputs (all in `polymarket/train/workers/<name>/`)

- `trials.jsonl`: your ledger.
- `REPORT.md`: the table of every config you tried. For each: P&L per market (seed mean ± SE),
  fills per market, T6, D1.3 status, and per-length P&L (5m, 15m, 4h). Also: what worked, what
  didn't, and your recommended candidate.
- `ckpt/<config>_s<seed>.pt`: checkpoints of your top 3 configs, all seeds. They're small; use
  `git add -f`. Include the matching `config.json`.
- `runs/` is git-ignored scratch space.

Push intermediate results at least every hour. When you finish, also send the coordinator a short
summary with the `send_message` tool (claude-code-remote) to `session_01V31NsSvTVCgJBT18w8sxcp`.
