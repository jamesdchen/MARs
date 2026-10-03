"""Writes colab.ipynb (kept as a script so the notebook diff stays readable).

    python make_notebook.py
"""

import json

BRANCH = 'claude/deep-hedging-cvar-i622m7'
PUFFER_COMMIT = '6ffa5b1'

cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip()})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip()})


md(f"""
# Deep hedging under CVaR, part 3: Bates book, PufferLib 5.0 on a GPU

A short call + put book under Bates dynamics (stochastic variance with
leverage, crash jumps), hedged with the stock (price impact, proportional
cost, fixed fee) and a variance swap. The env is C (`bates/hedge.h`); this
notebook builds it into PufferLib 5.0's native CUDA trainer, trains PPO,
tunes everything with the same trial budget, and scores all methods on the
same held-out paths.

1. **Runtime → Change runtime type → GPU** (a T4 works; L4 or A100 is faster).
2. **Runtime → Run all.**
3. At the end, download `deep_hedging_gpu_results.zip` and send it back
   (upload it to the Claude session or attach it to the pull request).

Run time: build and single 5.0 runs take minutes; each sweep of 32 trials
can take an hour or more on a T4. Results are zipped after every stage, and
with `SAVE_TO_DRIVE = True` also copied to your Google Drive (Colab asks for
access once). Lower `SWEEP_RUNS` for a quicker pass.

Code: `explorations/deep_hedging_cvar/bates` on branch `{BRANCH}` of
github.com/jamesdchen/MARs. PufferLib 5.0 is pinned to commit `{PUFFER_COMMIT}`.
""")

code("""
# Settings
PPO_TIMESTEPS = 200_000_000      # same budget as the CPU PufferLib 3.0 run
LONG_TIMESTEPS = 1_000_000_000   # a second, longer 5.0 run; 0 to skip
SWEEP_RUNS = 32                  # trials in each sweep (PPO, pathwise, band); 0 to skip
SAVE_TO_DRIVE = False            # also copy the results zip to Google Drive after each stage
SEED = 0
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
""")

md("""
## Build PufferLib 5.0 with the env

`build.sh` links LLVM OpenMP as `-lomp5` (Ubuntu ships `libomp.so.5`, so we
add that name) and `-lGL` (raylib). NCCL comes from the `nvidia-nccl-cu12`
wheel. `--float` builds in float32: wealth and positions need more precision
than bfloat16, and it is the layout `puffer5/puffernet.py` reads.
`make_ini.py` writes the env's config from `market.py`, derived numbers
included.
""")

code("""
!apt-get -qq install -y ccache clang libomp-dev libgl1-mesa-dev > /dev/null
!pip -q install optuna nvidia-nccl-cu12
!ln -sf $(ls /usr/lib/x86_64-linux-gnu/libomp.so.5 /usr/lib/llvm-*/lib/libomp.so.5 2>/dev/null | head -1) /usr/lib/x86_64-linux-gnu/libomp5.so
import os
os.environ['LIBRARY_PATH'] = ':'.join(filter(None, [
    '/usr/local/cuda/lib64/stubs', '/usr/lib64-nvidia', os.environ.get('LIBRARY_PATH')]))
SRC = '/content/MARs/explorations/deep_hedging_cvar/bates'
!cd {SRC}/puffer5 && python make_ini.py
!mkdir -p /content/PufferLib/ocean/deep_hedging
!cp {SRC}/hedge.h {SRC}/puffer5/deep_hedging.h /content/PufferLib/ocean/deep_hedging/
!cp {SRC}/puffer5/deep_hedging.ini /content/PufferLib/config/
!cd /content/PufferLib && ./build.sh deep_hedging --float 2>&1 | tail -20
assert os.path.exists('/content/PufferLib/puffer'), 'build failed: see the output above'
""")

md("""
## Checks

- `test_c.py`: the C core against the PyTorch reference model `market.py`
  (identical noise and actions; losses agree to about 1e-13).
- `test_wrapper.c`: the 5.0 wrapper steps exactly like the core.
- `test_puffernet.py`: the PyTorch loader for 5.0 checkpoints against
  PufferLib's own C forward pass, here also on the GPU.
""")

code("""
!cd {SRC} && python test_c.py 2>&1 | tail -4
!cd {SRC}/puffer5 && cc -O2 -I/content/PufferLib/src -I/content/PufferLib/raylib-5.5_linux_amd64/include -I.. test_wrapper.c -o /tmp/test_wrapper -lm && /tmp/test_wrapper deep_hedging.ini
!cd {SRC}/puffer5 && PUFFERLIB_DIR=/content/PufferLib python test_puffernet.py 2>&1 | tail -3
""")

md("""
## Train with PufferLib 5.0

The trainer's live dashboard goes to a log file; we print its last frame.
Wall time includes start-up.
""")

code("""
import glob, json, re, shutil, subprocess, time, configparser
import torch

OUT = f'{SRC}/results/gpu'
os.makedirs(OUT, exist_ok=True)
ini = configparser.ConfigParser()
ini.read('/content/PufferLib/config/deep_hedging.ini')
HIDDEN = int(ini['policy']['hidden_size'])
LAYERS = int(ini['policy']['num_layers'])
GPU = torch.cuda.get_device_name(0)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')

def save(stage):
    zip_path = '/content/deep_hedging_gpu_results.zip'
    subprocess.run(f'rm -f {zip_path} && cd {SRC} && zip -qr {zip_path} results/gpu '
                   '$(ls -d results/combined 2>/dev/null)', shell=True, check=True)
    if SAVE_TO_DRIVE:
        shutil.copy(zip_path, '/content/drive/MyDrive/deep_hedging_gpu_results.zip')
    print(f'results saved after: {stage}')

def run_logged(cmd, log_path, cwd):
    with open(log_path, 'w') as log:
        return subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)

def train_puffer5(name, steps):
    before = set(glob.glob('/content/PufferLib/checkpoints/deep_hedging/*/*.bin'))
    t0 = time.time()
    proc = run_logged(['./puffer', 'train', '--headless', f'--train.total_timesteps={steps}',
                       f'--base.seed={SEED}', f'--vec.num_threads={os.cpu_count()}'],
                      f'{OUT}/{name}_train.log', '/content/PufferLib')
    wall = time.time() - t0
    text = open(f'{OUT}/{name}_train.log', errors='replace').read()
    print(re.sub(r'\\x1b\\[[0-9;?]*[A-Za-z]', '', text)[-3000:])
    assert proc.returncode == 0, f'training failed ({proc.returncode})'
    new = sorted(set(glob.glob('/content/PufferLib/checkpoints/deep_hedging/*/*.bin')) - before,
                 key=os.path.getmtime)
    shutil.copy(new[-1], f'{OUT}/{name}.bin')
    meta = {'wall_time': wall, 'steps': steps, 'sps': steps / wall, 'device': GPU,
            'hidden': HIDDEN, 'layers': LAYERS, 'checkpoint': new[-1]}
    json.dump(meta, open(f'{OUT}/{name}.json', 'w'), indent=2)
    print(f'{name}: {steps / 1e6:.0f}M steps in {wall:.0f} s = {steps / wall / 1e6:.2f}M steps/s')

train_puffer5('puffer5', PPO_TIMESTEPS)
if LONG_TIMESTEPS:
    train_puffer5('puffer5_long', LONG_TIMESTEPS)
save('PufferLib 5.0 runs')
""")

md("""
## PPO sweep with PufferLib's tuner (Protein)

Search space: `puffer5/sweep.ini`. PufferLib's `default.ini` also sweeps
`train.horizon` and `vec.total_agents`, which this env must keep fixed, so
we drop its generic `[sweep.*]` blocks in our clone first. Protein ranks runs
by the env's score, minus the Rockafellar-Uryasev objective; then
`select_sweep.py` re-scores the top 5 on validation paths with deterministic
actions and keeps the best. Test paths are not used.
""")

code("""
if SWEEP_RUNS:
    path = '/content/PufferLib/config/default.ini'
    blocks = re.split(r'(?m)^(?=\\[)', open(path).read())
    open(path, 'w').write(''.join(b for b in blocks if not b.startswith('[sweep.')))
    with open('/content/PufferLib/config/deep_hedging.ini', 'a') as f:
        f.write('\\n' + open(f'{SRC}/puffer5/sweep.ini').read())
    t0 = time.time()
    proc = run_logged(['./puffer', 'sweep', '--headless', f'--sweep.max_runs={SWEEP_RUNS}',
                       f'--vec.num_threads={os.cpu_count()}'],
                      f'{OUT}/puffer5_sweep.log', '/content/PufferLib')
    assert proc.returncode == 0, 'sweep failed: see results/gpu/puffer5_sweep.log'
    SWEEP_WALL = time.time() - t0
    print(f'sweep: {SWEEP_RUNS} runs in {SWEEP_WALL / 60:.0f} min')
    subprocess.run(['python', 'puffer5/select_sweep.py', '--pufferlib', '/content/PufferLib',
                    '--sweep-log', f'{OUT}/puffer5_sweep.log', '--sweep-wall', str(SWEEP_WALL),
                    '--out', 'results/gpu'], cwd=SRC, check=True)
    save('PPO sweep')
""")

md("""
## Baselines on the same GPU, with the same tuning budget

- `tune_baselines.py`: Bates-model hedges (delta, delta-vega with the
  variance swap) and the no-trade band around them with gamma-dependent
  widths and partial adjustment, tuned with Optuna's TPE on validation CVaR.
- `train_pathwise.py`: pathwise deep hedging. The `hybrid` gate uses
  pathwise gradients for the targets and score-function gradients for the
  trade decisions; `ste` is the straight-through variant.
- `sweep_pathwise.py`: Optuna TPE over the hybrid pathwise hyperparameters.
""")

code("""
def run_py(args, log_name):
    proc = run_logged(['python', *args], f'{OUT}/{log_name}', SRC)
    lines = open(f'{OUT}/{log_name}').read().splitlines()
    print(log_name, '|', lines[-1] if lines else '')
    assert proc.returncode == 0, f'{args[0]} failed: see results/gpu/{log_name}'

run_py(['tune_baselines.py', '--trials', str(max(SWEEP_RUNS, 1)), '--device', 'cuda',
        '--out', 'results/gpu/baselines.json'], 'baselines_log.jsonl')
save('baselines')
for gate in ['ste', 'hybrid']:
    run_py(['train_pathwise.py', '--gate', gate, '--device', 'cuda',
            '--out', f'results/gpu/pathwise_{gate}.pt'], f'pathwise_{gate}_log.jsonl')
if SWEEP_RUNS:
    run_py(['sweep_pathwise.py', '--trials', str(SWEEP_RUNS), '--device', 'cuda',
            '--out-dir', 'results/gpu/pathwise_sweep'], 'pathwise_sweep_log.jsonl')
save('pathwise')
""")

md("""
## Score everything on the same held-out paths

CPU runs from the repo (PufferLib 3.0 PPO on the C env) and the GPU runs
above, through `evaluate.py`.
""")

code("""
runs = ['baselines|baselines|results/gpu/baselines.json',
        'ppo3|PPO, PufferLib 3.0 + C env (CPU)|results/ppo.pt',
        'ppo5|PPO, PufferLib 5.0 (GPU)|results/gpu/puffer5.bin']
if LONG_TIMESTEPS:
    runs.append('ppo5|PPO, PufferLib 5.0, long run (GPU)|results/gpu/puffer5_long.bin')
runs += [f'pathwise|pathwise, {g} gate (GPU)|results/gpu/pathwise_{g}.pt' for g in ['ste', 'hybrid']]
if SWEEP_RUNS:
    runs += ['ppo5|PPO, PufferLib 5.0, tuned (GPU)|results/gpu/puffer5_sweep_best.bin',
             'pathwise|pathwise, hybrid gate, tuned (GPU)|results/gpu/pathwise_sweep/best.pt']
subprocess.run(['python', 'evaluate.py', '--device', 'cuda', '--out', 'results/combined',
                '--runs', *runs], cwd=SRC, check=True)
print(open(f'{SRC}/results/combined/summary.md').read())
""")

code("""
from IPython.display import Image, display
for name in ['loss_hist', 'positions', 'training', 'ru_objective']:
    path = f'{SRC}/results/combined/{name}.png'
    if os.path.exists(path):
        display(Image(path))
""")

code("""
save('evaluation')
from google.colab import files
files.download('/content/deep_hedging_gpu_results.zip')
""")

nb = {'cells': cells, 'metadata': {
    'accelerator': 'GPU',
    'colab': {'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
    'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}

for c in nb['cells']:
    c['source'] = [line + '\n' for line in c['source'].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')

with open('colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
