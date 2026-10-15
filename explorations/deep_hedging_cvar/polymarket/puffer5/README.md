# Polymarket market making on PufferLib 5.0

The replay env of `../mm.h` (one agent quotes a bid and an ask in one market per episode;
heads `ACT_SIZES = {7, 7, 3}`: bid level, ask level, size) behind PufferLib 5.0's native CUDA
trainer, pinned to `ea7f989` on the 5.0 branch (`6ffa5b1` plus per-stream cuBLAS handles; the
env API, `ini.h`, `puffercpu.c` and `build.sh` are identical). The rules the core follows for
this are in `../PUFFER5.md`; the bates port in `../../bates/puffer5/` is the prior art.

| file | what |
|---|---|
| `polymarket_mm.h` | the 5.0 wrapper: `Env` struct, `puf_init` (reads `[env]`), `puf_reset`, `puf_step` (float actions to int), `puf_log` |
| `make_ini.py` | writes `polymarket_mm.ini` (`[env]` = every knob of `mm.h` at its default, plus `data_dir`, `seed`, `log_episodes`; `[eval_env]`; PPO defaults) and `sweep.ini` (Protein space) |
| `patch_pufferlib.py` | two idempotent edits to a 5.0 checkout: `[eval_env]` overrides for the final evaluation, and no generic `[sweep.*]` blocks in `default.ini`; `--selftest` checks them on CPU |
| `puffernet_mm.py` | torch replica of 5.0's PufferNet with discrete heads, `.bin` checkpoint loader, `policy_fn`, and `EvaluatorPolicy` for `../train/evaluator.py` |
| `score_val.py` | scores 5.0 checkpoints (single runs, or Protein's top-k of a sweep) on validation with the Python evaluator; writes the trial ledger |
| `make_notebook.py` -> `colab.ipynb` | the whole GPU pipeline on Colab |
| `test_wrapper.c` | the wrapper steps bit-for-bit like the core (CPU, no CUDA) |
| `test_puffernet_mm.py`, `puffernet_mm_harness.c` | the torch replica against 5.0's own C forward pass (`src/puffercpu.c`) |

## Run on a GPU

**Colab:** open `colab.ipynb` (GPU runtime), Run all, download
`polymarket_puffer5_results.zip`. It clones this branch and PufferLib, rebuilds the data from
the committed parquet files (`data/build.py`, `data/features.py`, `data/load.py`), patches and
builds PufferLib, runs the CPU checks, trains on the train split, sweeps, and scores on
validation. Settings at the top: `PPO_TIMESTEPS`, `SWEEP_RUNS`, `TOP_K`, `USE_FLOAT`.

**Own box** (Linux, CUDA toolkit with `nvcc`, clang, libomp, ccache), with this repo cloned
next to PufferLib:

```bash
git clone -b 5.0 https://github.com/PufferAI/PufferLib.git && (cd PufferLib && git checkout ea7f989)
PL=$PWD/PufferLib PM=$PWD/MARs/explorations/deep_hedging_cvar/polymarket
(cd $PM/data && python build.py && python features.py && python load.py)
cd $PM/puffer5
python patch_pufferlib.py --selftest $PL && python patch_pufferlib.py $PL
python make_ini.py --data-dir $PM/data/build --out-dir /tmp/cfg   # mkdir -p /tmp/cfg first
mkdir -p $PL/ocean/polymarket_mm && cp ../mm.h polymarket_mm.h $PL/ocean/polymarket_mm/
cp /tmp/cfg/polymarket_mm.ini $PL/config/
cd $PL && ./build.sh polymarket_mm --float            # drop --float for bf16 (see below)
./puffer train --headless                             # train split; final eval on validation
cat /tmp/cfg/sweep.ini >> config/polymarket_mm.ini && ./puffer sweep --headless
```

The committed `polymarket_mm.ini` has `data_dir` relative to the PufferLib root
(`../MARs/...`, the repo cloned next to PufferLib); `--data-dir` makes it absolute. Without `--headless`, `./puffer eval` opens a window (render mode). Checks on CPU:

```bash
PL=/path/to/PufferLib
cc -O2 -fopenmp -I$PL/src -I$PL/raylib-5.5_linux_amd64/include -I.. test_wrapper.c \
   -o /tmp/test_wrapper -lm && OMP_WAIT_POLICY=passive /tmp/test_wrapper polymarket_mm.ini ../data/build
PUFFERLIB_DIR=$PL python test_puffernet_mm.py
python patch_pufferlib.py --selftest $PL
```

(raylib's headers come with `build.sh`'s download; any raylib 5.5 `include/` works, the
wrapper does not draw.)

## How runs are scored, and the splits

- Training: `[env] split = 0`, random market and a random start in the first 20% of its
  window (`start_frac`), so episode ends are not in sync with `horizon` (`PUFFER5.md` rule 7).
- The trainer's final evaluation of every run, the score of a sweep trial, is on
  **validation**: `patch_pufferlib.py` applies the ini's `[eval_env]` (`split = 1`,
  `fixed_start = 0`, `cycle_stride = 1`, `log_episodes = 1`) before it and restores `[env]`
  after. The wrapper seeds env i with `(seed << 32) + i`, so env i plays validation market
  i mod n_val from the open, and logs only that first episode. `eval_agents = eval_episodes =
  16 * 247 = 3952`: each validation market is played 16 times (sampled actions), weighted
  equally. The trainer's eval loop stops after `eval_episodes` finished episodes; without
  `log_episodes` the 5-minute markets, which finish first, would make up most of them.
  The patched trainer prints `eval_env: final eval with [env] split = 1 (was 0)`; the notebook
  checks for it.
- Protein sees one point per trial (`sweep.downsample = 1`): with more, the earlier points
  are bin means of the training curve, i.e. train-split P&L.
- `score_val.py` re-scores with the Python evaluator (`../train/evaluator.py`: every
  validation market once, argmax of each head, headline variant) and appends every
  evaluation, and every sweep trial, to the trial ledger (`EVALUATION.md` T4).
- Test (split 2) is refused by the core unless `allow_test = 1` and `MM_FINAL=1` in the
  process environment; the Python adapters refuse it outright. Only the last notebook cell
  can reach it, off by default.

## Throughput

- The core alone steps about **2.9M steps/s on one CPU thread** (this container, 256 envs,
  random actions, default knobs: 1 s steps on 5m/15m markets, 5 s on 4h). 5.0 steps envs on
  `vec.num_threads` OpenMP threads, so a 2-vCPU Colab VM caps env stepping at roughly 5M
  steps/s before the trainer's own CPU work; on most GPUs the network, not the env, should
  bound throughput for `hidden_size = 128, num_layers = 4`.
- Not yet measured on a GPU: the notebook records the wall time and steps/s of every run
  (`results/puffer5/*.json`, including start-up and the final validation evaluation).
- The data is small: one pass over the train split is ~0.3M decisions from the open
  (431 markets), so 100M steps are ~300 passes. Watch the validation score of the
  intermediate checkpoints (`checkpoint_interval = 50` epochs) for overfitting.

## 5.0 gotchas (and what this port does about them)

1. **Precision.** The default build stores observations, rewards and values in the rollout
   as bf16 and runs the network in bf16 (fp32 master weights). bf16 keeps 8 significant bits:
   relative error up to 0.4%. Our features are relative and clipped to [-5, 5], so
   observations are fine; but a value estimate V stored in bf16 carries noise of ~0.004 |V|,
   comparable to a per-step P&L reward (reward_scale = 1/Q). `--float` (fp32 everywhere, as
   PufferLib 3.0) is the notebook's default (`USE_FLOAT = True`); bf16 is the faster option on
   Ampere or newer (A100, L4). On a T4 (Turing, no bf16 tensor cores) keep `--float`.
2. **Rewards are clamped to [-1, 1]** by the 5.0 trainer too (`bates/README.md` item 5 is
   wrong); the Log's `reward_clip_frac` shows how often the core's own clip bites.
3. **Muon alignment.** The allocator pads every tensor to 16 bytes but `muon_step` walks the
   gradients unpadded. With discrete heads every tensor is a multiple of `hidden_size`
   values, so `hidden_size % 8 == 0` suffices (all sweep values are powers of 2 >= 32).
4. **Shapes** (asserted in `launch_train`): `horizon % 8 == 0` (4 with `--float`),
   `minibatch_size % horizon == 0`, `minibatch_size <= horizon * total_agents`,
   `total_agents % (minibatch_size / horizon) == 0`. `make_ini.py` checks every
   horizon/minibatch pair the sweep can draw with `total_agents = 4096`.
5. **The reward on a segment's row 0 is never used** (GAE runs per `horizon` segment): one
   transition in `horizon` is dropped. Random start offsets keep episode ends out of sync.
6. **No truncation flag**: every episode end is a terminal without bootstrap. The core ends
   episodes at resolution (inventory settles), a true terminal.
7. **Recurrent state** carries across segments (`reset_every_horizon = 0`) and is zeroed on a
   terminal, which the env raises with the next episode's first observation.
8. **The final evaluation reuses `[env]`** (the training split) and its loop counts finished
   episodes; both handled above (`[eval_env]`, `log_episodes`). The trainer's evaluation
   samples actions; the Python evaluator takes the argmax.
9. **Config strings.** Every ini value keeps its raw text (`DictItem.str`), so `data_dir` is a
   plain `[env]` string; relative paths resolve against the PufferLib root; `#` or `;`
   outside quotes start a comment (`make_ini.py` quotes the path); `--env.key=value` only
   overrides keys that exist in the ini.
10. **`puf_init` runs more than once per process** (eval trainer after training) and envs
    are `calloc`ed before agents are counted: the core maps the data once behind a static
    guard (and refuses a different `data_dir` in the same process), interns configs (at most
    256 distinct ones per process), and `init` never touches agent buffers (bound later).
11. **Concurrency**: `puf_step` runs on many envs at once (OpenMP and one thread per buffer);
    the core has no mutable globals after init and no `rand()`. `test_wrapper.c` steps under
    OpenMP against a serial core run.
12. **Sweeps**: `default.ini`'s generic `[sweep.*]` blocks (e.g. `vec.total_agents`,
    `num_layers` up to 8) merge into ours unless removed (`patch_pufferlib.py`); every default
    must lie inside its range or `./puffer sweep` exits; trial stdout goes to `/dev/null`, so a
    trial leaves only `logs/polymarket_mm/<run_id>.ini` and its checkpoints. The `[metrics]`
    of a run log are bin means over the run's history (train curve and the final evaluation
    mixed), not the validation score: use the sweep log's `sweep run=... score=...` lines.
13. **Eval agent count**: `eval_agents` must split evenly over `vec.num_buffers` (2); 3952
    does. The minibatch shape rules are only asserted for training.
14. **Checkpoints** (`checkpoints/polymarket_mm/<run_id>/<step>.bin`) are the fp32 master
    weights cut at the unpadded count: encoder `(H, 48)`, decoder `(18, H)`, then `(3H, H)`
    per MinGRU layer, no logstd. With `H % 8 == 0` the layout is the same in both builds and
    the length determines `obs_size` (`puffernet_mm.infer_obs_size`).
15. **Build on Colab**: `build.sh` links `-lomp5` (symlink Ubuntu's `libomp.so.5`), needs
    `-lGL`, NCCL from the `nvidia-nccl-cu12` wheel, and the CUDA stubs on `LIBRARY_PATH`
    (all in the notebook, as for bates).
16. **OpenMP spin-waits**: on an oversubscribed CPU (more threads than free cores), set
    `OMP_WAIT_POLICY=passive` or lower `vec.num_threads`; the test above runs 30x slower
    with spinning threads on a loaded 4-core box.
