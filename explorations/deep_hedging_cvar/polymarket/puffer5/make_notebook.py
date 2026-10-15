"""Writes colab.ipynb (kept as a script so the notebook diff stays readable).

    python make_notebook.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BRANCH = 'claude/eloquent-feynman-n78cn1'
PUFFER_COMMIT = 'ea7f989'   # 5.0 branch, 2026-10-14; 6ffa5b1 + per-stream cuBLAS handles

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip()})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip()})


md(f"""
# Polymarket market making: PPO on PufferLib 5.0 (GPU)

A replay env of Polymarket's crypto up/down markets (5m / 15m / 4h, Aug 21-24 2026):
one agent quotes a bid and an ask per market, three discrete heads (bid level,
ask level, size; `ACT_SIZES = {{7, 7, 3}}`), fills from the recorded taker prints
under a queue model, settlement at resolution. The env is C (`polymarket/mm.h`);
this notebook builds it into PufferLib 5.0's native CUDA trainer, trains PPO on
the **train** split, sweeps hyperparameters with Protein scored on the
**validation** split, and scores the selected checkpoints with our Python
evaluator on validation. The **test** split is touched only by the last cell,
which is off by default and needs `MM_FINAL=1` (eval/EVALUATION.md section 5).

1. **Runtime -> Change runtime type -> GPU** (T4 works; L4 / A100 are faster).
2. **Runtime -> Run all.**
3. At the end, download `polymarket_puffer5_results.zip` and send it back.

Data: the raw parquet files are in the repo; `data/build.py` and
`data/features.py` rebuild the memory-mapped arrays (~2 min). Build and a
single run take minutes; a 32-trial sweep a few hours on a T4 (lower
`SWEEP_RUNS` for a quick pass). Results are zipped after every stage, and with
`SAVE_TO_DRIVE = True` copied to Google Drive.

Code: `explorations/deep_hedging_cvar/polymarket` on branch `{BRANCH}` of
github.com/jamesdchen/MARs. PufferLib 5.0 is pinned to `{PUFFER_COMMIT}`.
""")

code("""
# Settings
PPO_TIMESTEPS = 100_000_000   # one training run on the train split; 0 to skip
SWEEP_RUNS = 32               # Protein trials, each scored on validation; 0 to skip
TOP_K = 5                     # sweep runs re-scored by the Python evaluator
# --float: observations, rewards and values are stored in fp32 in the rollout
# (as in PufferLib 3.0). False: the default bf16 build, faster, 8-bit mantissa
# (relative error <= 0.4% on stored obs/rewards/values; see puffer5/README.md).
USE_FLOAT = True
SAVE_TO_DRIVE = False
SEED = 0
# The single pre-registered test run (EVALUATION.md section 5). Leave False.
RUN_FINAL_TEST = False
""")

code("""
!nvidia-smi
!nvcc --version | tail -2
""")

code(f"""
%cd /content
!rm -rf MARs PufferLib
!git clone -q --depth 1 -b {BRANCH} https://github.com/jamesdchen/MARs.git
!git clone -q -b 5.0 https://github.com/PufferAI/PufferLib.git
!cd PufferLib && git checkout -q {PUFFER_COMMIT} && git log -1 --format='PufferLib %h %cd'
import os
PM = '/content/MARs/explorations/deep_hedging_cvar/polymarket'
P5 = f'{{PM}}/puffer5'
PL = '/content/PufferLib'
DATA = f'{{PM}}/data/build'
""")

md("""
## Data

`build.py` turns the committed parquet files (L2 snapshots, fills, Binance
1 s prices, market metadata) into `ticks.bin` / `markets.bin`; `features.py`
adds the env's inputs (`feat.bin`, `book.bin`, `fills.bin`, `fill_index.bin`);
`load.py` checks integrity (no NaN, offsets, the test rows' sha256).
""")

code("""
!pip -q install pyarrow scipy pandas
!cd {PM}/data && python build.py && python features.py && python load.py
!ls -la {DATA}
""")

md("""
## Build PufferLib 5.0 with the env

- `patch_pufferlib.py`: the trainer's final evaluation of every run applies
  the ini's `[eval_env]` overrides (validation split, every market once from the
  open), so the score of a run, and Protein's metric, are validation P&L; and
  the generic `[sweep.*]` blocks of `default.ini` go, so `sweep.ini` is the
  whole search space. Its self-test runs first (CPU only).
- `make_ini.py`: `config/polymarket_mm.ini` from the core's knobs (`mm.h`),
  with `data_dir` pointing at the build above.
- `build.sh` links LLVM OpenMP as `-lomp5` and `-lGL` (raylib); NCCL comes from
  the `nvidia-nccl-cu12` wheel.
""")

code("""
!apt-get -qq install -y ccache clang libomp-dev libgl1-mesa-dev > /dev/null
!pip -q install nvidia-nccl-cu12
!ln -sf $(ls /usr/lib/x86_64-linux-gnu/libomp.so.5 /usr/lib/llvm-*/lib/libomp.so.5 2>/dev/null | head -1) /usr/lib/x86_64-linux-gnu/libomp5.so
os.environ['LIBRARY_PATH'] = ':'.join(filter(None, [
    '/usr/local/cuda/lib64/stubs', '/usr/lib64-nvidia', os.environ.get('LIBRARY_PATH')]))
!cd {P5} && python patch_pufferlib.py --selftest {PL} && python patch_pufferlib.py {PL}
!mkdir -p /content/cfg && cd {P5} && python make_ini.py --data-dir {DATA} --out-dir /content/cfg
!mkdir -p {PL}/ocean/polymarket_mm
!cp {PM}/mm.h {P5}/polymarket_mm.h {PL}/ocean/polymarket_mm/
!cp /content/cfg/polymarket_mm.ini {PL}/config/
FLOAT = '--float' if USE_FLOAT else ''
!cd {PL} && ./build.sh polymarket_mm {FLOAT} 2>&1 | tail -20
assert os.path.exists(f'{PL}/puffer'), 'build failed: see the output above'
""")

md("""
## Python evaluator

`train/evaluator.py` plays every validation market once from the open with the
argmax of each head, through the PufferLib 3.0 binding of the same core
(`binding.c`), and appends each evaluation to the trial ledger
(`train/trials.jsonl`, EVALUATION.md T4). 3.0 is built CPU-only here (nvcc
hidden), with the numpy pin relaxed as in `polymarket/setup.sh`.
""")

code("""
import subprocess
!curl -sSL https://files.pythonhosted.org/packages/source/p/pufferlib/pufferlib-3.0.0.tar.gz | tar xz -C /tmp
setup_py = '/tmp/pufferlib-3.0.0/setup.py'
text = open(setup_py).read().replace("'numpy<2.0'", "'numpy'")
text = text.replace('\\nc_extensions = []\\n', '\\nc_extensions = []\\nc_extension_paths = []\\n', 1)
open(setup_py, 'w').write(text)
cpu_env = dict(os.environ, NO_OCEAN='1', PATH=':'.join(
    p for p in os.environ['PATH'].split(':') if 'cuda' not in p))   # CPU-only _C, no nvcc
subprocess.run(['pip', '-q', 'install', '--no-build-isolation', '/tmp/pufferlib-3.0.0'],
               env=cpu_env, check=True)
!cd {PM} && python setup.py build_ext --inplace 2>&1 | tail -2
""")

md("""
## Checks (CPU)

- `test_env.py`: the C core against its numpy reference.
- `test_wrapper.c`: the 5.0 wrapper steps bit-for-bit like the core (same seeds
  and actions), on this data, and its `log_episodes` knob logs only the first
  episode of each env.
- `test_puffernet_mm.py`: the torch replica of 5.0's PufferNet against
  PufferLib's own C forward pass (`src/puffercpu.c`), checkpoint layout, policy_fn.
""")

code("""
!cd {PM} && python test_env.py 2>&1 | tail -5
!cd {P5} && cc -O2 -fopenmp -I{PL}/src -I{PL}/raylib-5.5_linux_amd64/include -I.. test_wrapper.c -o /tmp/test_wrapper -lm && OMP_WAIT_POLICY=passive /tmp/test_wrapper /content/cfg/polymarket_mm.ini
!cd {P5} && PUFFERLIB_DIR={PL} python test_puffernet_mm.py 2>&1 | tail -3
""")

md("""
## Train with PufferLib 5.0 (train split)

One run of `PPO_TIMESTEPS` steps. The trainer's live dashboard goes to a log
file; we print its last frame. At the end the patched trainer evaluates the
final checkpoint on validation (`eval_env:` lines in the log). Every checkpoint
of the run (one per `checkpoint_interval` epochs) is then scored on validation
by the Python evaluator; the final one is the run's result, the others are
monitoring (all are counted in the ledger).
""")

code("""
import glob, json, re, shutil, subprocess, sys, time, configparser
import torch

OUT = f'{PM}/results/puffer5'
os.makedirs(OUT, exist_ok=True)
GPU = torch.cuda.get_device_name(0)
CKPTS = f'{PL}/checkpoints/polymarket_mm'
LOGS = f'{PL}/logs/polymarket_mm'
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')

def save(stage):
    zip_path = '/content/polymarket_puffer5_results.zip'
    subprocess.run(f'rm -f {zip_path} && cd {PM} && zip -qr {zip_path} results/puffer5 '
                   '$(ls train/trials.jsonl 2>/dev/null)', shell=True, check=True)
    if SAVE_TO_DRIVE:
        shutil.copy(zip_path, '/content/drive/MyDrive/polymarket_puffer5_results.zip')
    print(f'results saved after: {stage}')

def run_logged(cmd, log_path, cwd, env=None):
    with open(log_path, 'w') as log:
        return subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env)

def strip_ansi(text):
    return re.sub(r'\\x1b\\[[0-9;?]*[A-Za-z]', '', text)

def train_puffer5(name, steps):
    before = set(glob.glob(f'{CKPTS}/*'))
    t0 = time.time()
    log_path = f'{OUT}/{name}_train.log'
    proc = run_logged(['./puffer', 'train', '--headless', f'--train.total_timesteps={steps}',
                       f'--base.seed={SEED}', f'--vec.num_threads={os.cpu_count()}'], log_path, PL)
    wall = time.time() - t0
    text = strip_ansi(open(log_path, errors='replace').read())
    print(text[-3000:])
    assert proc.returncode == 0, f'training failed ({proc.returncode})'
    assert 'eval_env: final eval with [env] split = 1' in text, 'final eval not on validation'
    run_dir = max(set(glob.glob(f'{CKPTS}/*')) - before, key=os.path.getmtime)
    run_id = os.path.basename(run_dir)
    os.makedirs(f'{OUT}/{name}', exist_ok=True)
    ckpts = []
    for c in sorted(glob.glob(f'{run_dir}/*.bin')):
        ckpts.append(shutil.copy(c, f'{OUT}/{name}/'))
    shutil.copy(f'{LOGS}/{run_id}.ini', f'{OUT}/{name}/run.ini')
    shutil.copy(ckpts[-1], f'{OUT}/{name}.bin')
    meta = {'wall_time': wall, 'steps': steps, 'sps': steps / wall, 'device': GPU,
            'float': USE_FLOAT, 'run_id': run_id, 'checkpoints': ckpts}
    json.dump(meta, open(f'{OUT}/{name}.json', 'w'), indent=2)
    print(f'{name}: {steps / 1e6:.0f}M steps in {wall:.0f} s = {steps / wall / 1e6:.2f}M steps/s '
          f'(incl. start-up and the final validation eval), {len(ckpts)} checkpoints')
    return ckpts, f'{OUT}/{name}/run.ini'

def score_val(args, log_name):
    proc = run_logged([sys.executable, 'score_val.py', '--out', OUT, *args], f'{OUT}/{log_name}', P5)
    print(open(f'{OUT}/{log_name}').read()[-3000:])
    assert proc.returncode == 0, f'score_val failed: see results/puffer5/{log_name}'

if PPO_TIMESTEPS:
    ckpts, run_ini = train_puffer5('puffer5', PPO_TIMESTEPS)
    names = [f'puffer5_{os.path.basename(c)[:-4].lstrip("0")}' for c in ckpts]
    score_val(['--checkpoint', *ckpts, '--run-ini', *[run_ini] * len(ckpts), '--name', *names],
              'puffer5_val.log')
    save('PufferLib 5.0 run')
""")

md("""
## Sweep with PufferLib's tuner (Protein), scored on validation

Search space: `puffer5/sweep.ini` (learning rate, entropy coefficient, gamma,
GAE lambda, horizon 64-512, minibatch, hidden size 32-512, layers 1-4, ...;
the run length is the cost axis). Every trial trains on the train split; its
one Protein point (`downsample = 1`) is the trainer's final evaluation, which
`[eval_env]` runs on the validation split: mean P&L per validation market,
every market weighted equally. `score_val.py` then re-scores Protein's top
`TOP_K` runs with the Python evaluator (argmax actions) and keeps the best;
every trial is a validation evaluation and goes to the ledger.
""")

code("""
if SWEEP_RUNS:
    assert 'polymarket_mm patch' in open(f'{PL}/src/pufferl.cu').read(), 'unpatched trainer'
    cfg = f'{PL}/config/polymarket_mm.ini'
    if '[sweep]' not in open(cfg).read():
        with open(cfg, 'a') as f:
            f.write('\\n' + open(f'{P5}/sweep.ini').read())
    t0 = time.time()
    proc = run_logged(['./puffer', 'sweep', '--headless', f'--sweep.max_runs={SWEEP_RUNS}',
                       f'--vec.num_threads={os.cpu_count()}'], f'{OUT}/sweep.log', PL)
    assert proc.returncode == 0, 'sweep failed: see results/puffer5/sweep.log'
    SWEEP_WALL = time.time() - t0
    runs = re.findall(r'sweep run=(\\d+) score=(\\S+)', strip_ansi(open(f'{OUT}/sweep.log').read()))
    print(f'sweep: {len(runs)} runs in {SWEEP_WALL / 60:.0f} min')
    json.dump({'wall_time': SWEEP_WALL, 'runs': len(runs), 'device': GPU, 'float': USE_FLOAT},
              open(f'{OUT}/sweep.json', 'w'), indent=2)
    score_val(['--pufferlib', PL, '--sweep-log', f'{OUT}/sweep.log', '--top-k', str(TOP_K)],
              'sweep_val.log')
    save('sweep')
""")

md("""
## Summary (validation)
""")

code("""
rows = []
for line in open(f'{OUT}/val_scores.json') if os.path.exists(f'{OUT}/val_scores.json') else []:
    for name, r in json.loads(line).items():
        rows.append((name, r['val']))
if os.path.exists(f'{OUT}/puffer5_sweep_best.json'):
    best = json.load(open(f'{OUT}/puffer5_sweep_best.json'))
    rows.append((f"sweep best ({best['run_id']})", best['val']))
keys = ['n_markets', 'pnl_total', 'pnl_mean', 'pnl_median', 'cvar95', 'frac_positive', 'max_abs_inv']
print(f"{'run':40s}" + ''.join(f'{k:>14s}' for k in keys))
for name, v in rows:
    print(f'{name:40s}' + ''.join(f'{v.get(k, float("nan")):14.3f}' for k in keys))
save('summary')
""")

md("""
## FINAL TEST RUN (off by default; run once, ever)

**Do not run this cell during development.** EVALUATION.md section 5: freeze
code, configs and checkpoints, write `TEST_PLAN.json` and record its sha256
first. The core refuses split 2 unless `allow_test=1` **and** the process
environment has `MM_FINAL=1`; this cell additionally needs
`RUN_FINAL_TEST = True` above, and `%env MM_FINAL=1` in a cell of its own. It
evaluates the frozen sweep selection once on the test split with the 5.0
trainer (each test market played 16 times from the open, sampled actions) and
records the checkpoint's sha256. The Python evaluator refuses split 2 by design.
""")

code("""
if not RUN_FINAL_TEST:
    print('final test run skipped (RUN_FINAL_TEST = False)')
else:
    assert os.environ.get('MM_FINAL') == '1', 'set %env MM_FINAL=1 in its own cell first'
    import hashlib, numpy as np
    ckpt = f'{OUT}/puffer5_sweep_best.bin'
    best = json.load(open(f'{OUT}/puffer5_sweep_best.json'))
    raw = np.fromfile(f'{DATA}/markets.bin', np.uint8).reshape(-1, 88)
    n_test = int((raw[:, 40:44].copy().view('<i4')[:, 0] == 2).sum())
    agents = 16 * n_test
    proc = subprocess.run(['./puffer', 'eval', ckpt, '--headless', '--env.split=2',
                           '--env.allow_test=1', '--env.fixed_start=0', '--env.cycle_stride=1',
                           '--env.log_episodes=1', f'--base.eval_agents={agents}',
                           f'--base.eval_episodes={agents}',
                           f"--policy.hidden_size={best['hidden']}",
                           f"--policy.num_layers={best['layers']}"],
                          cwd=PL, capture_output=True, text=True)
    out = strip_ansi(proc.stdout + proc.stderr)
    line = [l for l in out.splitlines() if l.startswith('CUDA_EVAL')]
    assert proc.returncode == 0 and line, out[-3000:]
    result = {'checkpoint_sha256': hashlib.sha256(open(ckpt, 'rb').read()).hexdigest(),
              'run_id': best['run_id'], 'n_test_markets': n_test, 'agents': agents,
              'cuda_eval': line[-1], 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    json.dump(result, open(f'{OUT}/FINAL_TEST_puffer5.json', 'w'), indent=2)
    print(result)
    save('FINAL TEST')
""")

code("""
save('end')
from google.colab import files
files.download('/content/polymarket_puffer5_results.zip')
""")

nb = {'cells': cells, 'metadata': {
    'accelerator': 'GPU',
    'colab': {'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
    'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}

for c in nb['cells']:
    c['source'] = [line + '\n' for line in c['source'].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')

with open(os.path.join(HERE, 'colab.ipynb'), 'w') as f:
    json.dump(nb, f, indent=1)
