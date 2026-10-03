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
# Deep hedging under CVaR: PufferLib 5.0 on a GPU

Trains the impact + fixed cost hedging env with PufferLib 5.0's native CUDA
trainer, trains the pathwise baselines on the same GPU, and scores everything
on the same held-out paths as the CPU runs in the repo.

1. **Runtime → Change runtime type → GPU** (a T4 works; L4 or A100 is faster).
2. **Runtime → Run all.**
3. At the end, download `deep_hedging_gpu_results.zip` and send it back
   (upload it to the Claude session or attach it to the pull request).

Code: `explorations/deep_hedging_cvar/impact` on branch `{BRANCH}` of
github.com/jamesdchen/MARs. PufferLib 5.0 is pinned to commit `{PUFFER_COMMIT}`.
""")

code("""
# Settings
PPO_TIMESTEPS = 120_000_000      # same budget as the CPU PufferLib 3.0 run
LONG_TIMESTEPS = 1_000_000_000   # a second, longer 5.0 run; set to 0 to skip
SWEEP_RUNS = 32                  # trials in each hyperparameter sweep; 0 to skip
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
## Build PufferLib 5.0 with the hedging env

`build.sh` links LLVM OpenMP as `-lomp5`; Ubuntu ships it as `libomp.so.5`,
so we add the `libomp5.so` name. It also links `-lGL` (raylib), hence the GL
development package. NCCL comes from the `nvidia-nccl-cu12` wheel
that Colab's PyTorch already installs. `--float` builds in float32: the
observations carry wealth and hedge positions that need more precision than
bfloat16.
""")

code("""
!apt-get -qq install -y ccache clang libomp-dev libgl1-mesa-dev > /dev/null
!pip -q install optuna
!ln -sf $(ls /usr/lib/x86_64-linux-gnu/libomp.so.5) /usr/lib/x86_64-linux-gnu/libomp5.so
!pip -q install nvidia-nccl-cu12
import os
os.environ['LIBRARY_PATH'] = ':'.join(filter(None, [
    '/usr/local/cuda/lib64/stubs', '/usr/lib64-nvidia', os.environ.get('LIBRARY_PATH')]))
SRC = '/content/MARs/explorations/deep_hedging_cvar/impact'
!mkdir -p /content/PufferLib/ocean/deep_hedging
!cp {SRC}/puffer5/deep_hedging.h /content/PufferLib/ocean/deep_hedging/
!cp {SRC}/puffer5/deep_hedging.ini /content/PufferLib/config/
!cd /content/PufferLib && ./build.sh deep_hedging --float 2>&1 | tail -20
assert os.path.exists('/content/PufferLib/puffer'), 'build failed: see the output above'
""")

md("""
## Check the C env against the reference model

Same noise and actions through the C env and through `market.simulate`;
observations, losses and rewards must match (see `puffer5/test_c_env.py`).
""")

code("""
!cd {SRC}/puffer5 && PUFFERLIB_DIR=/content/PufferLib python test_c_env.py
""")

md("""
## Train with PufferLib 5.0

The trainer prints a live dashboard; it goes to a log file, and we keep the
last frame. Wall time includes start-up.
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

def train_puffer5(name, steps):
    before = set(glob.glob('/content/PufferLib/checkpoints/deep_hedging/*/*.bin'))
    t0 = time.time()
    with open(f'{OUT}/{name}_train.log', 'w') as log:
        proc = subprocess.run(
            ['./puffer', 'train', '--headless', f'--train.total_timesteps={steps}',
             f'--base.seed={SEED}', f'--vec.num_threads={os.cpu_count()}'],
            cwd='/content/PufferLib', stdout=log, stderr=subprocess.STDOUT)
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
""")

md("""
## Hyperparameter sweep with PufferLib's tuner (Protein)

The search space is `puffer5/sweep.ini`. PufferLib's `default.ini` also
sweeps `train.horizon` and `vec.total_agents`, which this env must keep fixed
(32-step episodes), so we drop its generic `[sweep.*]` blocks from our clone
first. Protein ranks runs by the env's score, minus the Rockafellar-Uryasev
objective; `select_sweep.py` then re-scores the top 5 on validation paths
with deterministic actions and keeps the best. The test paths are not used.
""")

code("""
if SWEEP_RUNS:
    path = '/content/PufferLib/config/default.ini'
    text = open(path).read()
    blocks = re.split(r'(?m)^(?=\\[)', text)
    open(path, 'w').write(''.join(b for b in blocks if not b.startswith('[sweep.')))
    with open('/content/PufferLib/config/deep_hedging.ini', 'a') as f:
        f.write('\\n' + open(f'{SRC}/puffer5/sweep.ini').read())
    t0 = time.time()
    with open(f'{OUT}/puffer5_sweep.log', 'w') as log:
        subprocess.run(['./puffer', 'sweep', '--headless', f'--sweep.max_runs={SWEEP_RUNS}',
                        f'--vec.num_threads={os.cpu_count()}'],
                       cwd='/content/PufferLib', stdout=log, stderr=subprocess.STDOUT, check=True)
    SWEEP_WALL = time.time() - t0
    print(f'sweep: {SWEEP_RUNS} runs in {SWEEP_WALL / 60:.0f} min')
    subprocess.run(['python', 'puffer5/select_sweep.py', '--pufferlib', '/content/PufferLib',
                    '--sweep-log', f'{OUT}/puffer5_sweep.log', '--sweep-wall', str(SWEEP_WALL),
                    '--out', 'results/gpu'], cwd=SRC, check=True)
""")

md("""
## Pathwise baselines on the same GPU

The same script as the CPU runs, with `--device cuda`, so the wall-clock
comparison is on equal hardware. The straight-through gate is the serious
pathwise baseline, so it gets a small temperature sweep; `evaluate.py`
reports validation CVaR for each run so the choice is made on validation
paths. The sigmoid relaxation and the hard gate are there as failure modes.
""")

code("""
PATHWISE = [('ste', 0.03), ('ste', 0.1), ('ste', 0.3), ('sigmoid', 0.1), ('hard', 0.1)]
for gate, temp in PATHWISE:
    name = f'pathwise_{gate}_t{temp:g}'
    with open(f'{OUT}/{name}_log.jsonl', 'w') as log:
        subprocess.run(['python', 'train_pathwise.py', '--gate', gate, '--temp', str(temp),
                        '--device', 'cuda', '--out', f'results/gpu/{name}.pt'],
                       cwd=SRC, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, open(f'{OUT}/{name}_log.jsonl').read().splitlines()[-1])
""")

md("""
### Pathwise sweep with the same number of trials

Optuna's TPE (also a Bayesian tuner) over learning rate, straight-through
temperature, batch size, network size and iterations, scored on the same
validation paths (`sweep_pathwise.py`).
""")

code("""
if SWEEP_RUNS:
    with open(f'{OUT}/pathwise_sweep_log.jsonl', 'w') as log:
        subprocess.run(['python', 'sweep_pathwise.py', '--trials', str(SWEEP_RUNS),
                        '--device', 'cuda', '--out-dir', 'results/gpu/pathwise_sweep'],
                       cwd=SRC, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(open(f'{OUT}/pathwise_sweep_log.jsonl').read()[-2000:])
""")

md("""
## Score everything on the same held-out paths

CPU runs from the repo (PufferLib 3.0 + Cython env, pathwise on CPU) and the
GPU runs above, all through `evaluate.py`.
""")

code("""
runs = [
    'ppo3|PPO, PufferLib 3.0 + Cython env (CPU)|results/ppo.pt',
    'ppo5|PPO, PufferLib 5.0 (GPU)|results/gpu/puffer5.bin',
] + (['ppo5|PPO, PufferLib 5.0, long run (GPU)|results/gpu/puffer5_long.bin']
     if LONG_TIMESTEPS else []) + [
    f'pathwise|pathwise, {g} gate ({w})|results/pathwise_{g}.pt'
    for g in ['ste', 'sigmoid', 'hard'] for w in ['CPU']] + [
    f'pathwise|pathwise, {g} gate, temp {t:g} (GPU)|results/gpu/pathwise_{g}_t{t:g}.pt'
    for g, t in PATHWISE] + ([
    'ppo5|PPO, PufferLib 5.0, tuned (GPU)|results/gpu/puffer5_sweep_best.bin',
    'pathwise|pathwise, ste gate, tuned (GPU)|results/gpu/pathwise_sweep/best.pt',
] if SWEEP_RUNS else [])
subprocess.run(['python', 'evaluate.py', '--out', 'results/combined', '--runs', *runs],
               cwd=SRC, check=True)
print(open(f'{SRC}/results/combined/summary.md').read())
""")

code("""
from IPython.display import Image, display
for name in ['loss_hist', 'positions', 'training', 'ru_objective']:
    display(Image(f'{SRC}/results/combined/{name}.png'))
""")

code("""
%cd {SRC}
!zip -qr /content/deep_hedging_gpu_results.zip results/gpu results/combined
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
