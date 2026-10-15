"""w3 sweep driver: architecture + PPO-knob sweep on reward_mode 2 / markout_h 120.

    python train/workers/w3/sweep_w3.py list               # print the configs
    python train/workers/w3/sweep_w3.py run [--only a,b] [--seeds 0 1 2] [--jobs 4]

Every config starts from train/selected.json's cfg (reward_mode 2, markout_h 120,
MLP 64x2 shared) and changes one group of knobs. Runs are resumable: a run whose
result.json exists is skipped. Each run calls the shared train/train_ppo.py
(unchanged) with --env-kw data_dir=<build_v2> so val is v2 split 1, the w3 ledger
and tag w3_<config>.
"""
import argparse
import copy
import json
import re
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
TRAIN = os.path.join(PM, 'train')
PY = os.path.join(PM, '..', '.venv', 'bin', 'python')
DATA_V2 = os.path.join(PM, 'data', 'build_v2')
STEPS = 25_000_000


def base_cfg():
    sel = json.load(open(os.path.join(TRAIN, 'selected.json')))['selected']['cfg']
    c = {k: copy.deepcopy(sel[k]) for k in ('env', 'policy', 'ppo', 'num_envs', 'init', 'kl_teacher')}
    c['env'] = {'backend': 'real', 'kw': dict(sel['env']['kw'])}
    assert c['env']['kw']['reward_mode'] == 2 and c['env']['kw']['markout_h'] == 120
    c['env']['kw']['data_dir'] = DATA_V2
    c['total_timesteps'] = STEPS
    return c


def configs():
    V = {}

    def add(name, f):
        c = base_cfg()
        f(c)
        V[name] = c

    add('base', lambda c: None)
    # (a) recurrent PufferNet replica (R6), layers x hidden
    for L in (1, 2):
        for H in (64, 128, 256):
            add(f'pnet_L{L}_H{H}', lambda c, L=L, H=H: c.__setitem__(
                'policy', {'kind': 'puffernet', 'hidden': H, 'layers': L}))
    # (b) MLP hidden x layers x shared/separate
    for H in (128, 256):
        for L in (2, 3):
            for sh in (True, False):
                add(f'mlp_H{H}_L{L}_{"sh" if sh else "sep"}',
                    lambda c, H=H, L=L, sh=sh: c['policy'].update(hidden=H, layers=L, shared=sh))
    # (c) PPO knobs on the base MLP
    P = lambda **kw: (lambda c: c['ppo'].update(**kw))  # noqa: E731
    add('ep2', P(update_epochs=2))
    add('ep4', P(update_epochs=4))
    add('mb4k', P(minibatch_size=4096))
    add('mb32k', P(minibatch_size=32768))
    add('hz128', P(bptt_horizon=128))
    add('hz256', P(bptt_horizon=256))
    add('lam090', P(gae_lambda=0.90))
    add('lam0995', P(gae_lambda=0.995))
    add('env16k', lambda c: c.__setitem__('num_envs', 16384))
    add('clip01', P(clip_coef=0.1))
    add('clip03', P(clip_coef=0.3))
    add('mgn02', P(max_grad_norm=0.2))
    add('mgn20', P(max_grad_norm=2.0))
    add('vf025', P(vf_coef=0.25))
    add('vf10', P(vf_coef=1.0))
    add('vfclip02', P(vf_clip_coef=0.2))
    add('sep_vf10', lambda c: (c['policy'].update(shared=False), c['ppo'].update(vf_coef=1.0)))
    return V


def cost(name, c):
    """Rough relative wall time (for longest-first scheduling)."""
    k = 1.0
    if c['policy']['kind'] == 'puffernet':
        k = 2.2 * c['policy']['layers'] ** 0.5 * (1 + c['policy']['hidden'] / 256)
    else:
        k = 1 + (c['policy']['hidden'] * c['policy']['layers']) / 400
        if not c['policy'].get('shared', True):
            k *= 1.4
    return k * c['ppo']['update_epochs'] ** 0.7


def run_one(name, c, seed, steps=None):
    out = os.path.join(HERE, 'runs', f'{name}_s{seed}')
    if os.path.exists(os.path.join(out, 'result.json')):
        return name, seed, 'skip'
    if os.path.exists(out):  # killed earlier: keep it, start fresh
        os.rename(out, out + f'.killed{int(time.time())}')
    os.makedirs(out, exist_ok=True)
    c = copy.deepcopy(c)
    if steps:
        c['total_timesteps'] = int(steps)
    cfg_path = os.path.join(out, 'w3_cfg.json')
    json.dump(c, open(cfg_path, 'w'), indent=1)
    cmd = [PY, os.path.join(TRAIN, 'train_ppo.py'), '--config', cfg_path, '--seed', str(seed),
           '--threads', '1', '--out', out, '--tag', f'w3_{name}',
           '--ledger', os.path.join(HERE, 'trials.jsonl'), '--quiet']
    if c['policy']['kind'] == 'puffernet':
        cmd += ['--layers', str(c['policy']['layers'])]
    env = {**os.environ, 'OMP_WAIT_POLICY': 'passive', 'OMP_NUM_THREADS': '1'}
    t0 = time.time()
    with open(os.path.join(out, 'stdout.log'), 'w') as f:
        r = subprocess.run(cmd, cwd=TRAIN, stdout=f, stderr=subprocess.STDOUT, env=env)
    msg = f'{name} s{seed} rc={r.returncode} {time.time() - t0:.0f}s'
    print(time.strftime('%H:%M:%S'), msg, flush=True)
    return name, seed, r.returncode


def n_training():
    r = subprocess.run(['pgrep', '-f', '[t]rain_ppo.py'], capture_output=True, text=True)
    return len(r.stdout.split())


def top_configs(k=3, min_seeds=3):
    sys.path.insert(0, HERE)
    import analyze_w3
    A = {n: analyze_w3.agg(v) for n, v in analyze_w3.collect().items() if len(v) >= min_seeds}
    ok = [n for n, x in A.items() if x['d13']] or list(A)
    return sorted(ok, key=lambda n: -A[n]['pnl_mean'])[:k]


def follow(a):
    """Keep 4 training processes busy after the main sweep's queue has drained:
    1) seeds 3,4 of the top-3 configs (ranked when the first slot opens);
    2) then entries of queue.json ([{"name", "cfg"(full) or "base": name + "ppo"/"policy"/"top": ...,
       "seeds": [...]}]), re-read every poll, so it can be extended while running."""
    V = configs()
    n_main = len(V) * 3
    started, top3 = set(), None
    qpath = os.path.join(HERE, 'queue.json')
    pool = ThreadPoolExecutor(8)
    while True:
        dirs = [d for d in os.listdir(os.path.join(HERE, 'runs')) if re.match(r'.+_s\d+$', d)]
        main_started = sum(1 for k in V for s in (0, 1, 2) if f'{k}_s{s}' in dirs)
        if main_started >= n_main and n_training() < a.jobs:
            if top3 is None:
                top3 = top_configs()
                print(time.strftime('%H:%M:%S'), 'top3', top3, flush=True)
            jobs = [(k, V[k], s) for k in top3 for s in (3, 4)]
            try:
                Q = json.load(open(qpath)) if os.path.exists(qpath) else []
            except json.JSONDecodeError:
                Q = []
            for e in Q:
                c = e.get('cfg')
                if c is None:
                    c = copy.deepcopy(V[e['base']])
                    for grp in ('ppo', 'policy'):
                        c[grp].update(e.get(grp, {}))
                    if 'num_envs' in e:
                        c['num_envs'] = e['num_envs']
                    if 'kw' in e:
                        c['env']['kw'].update(e['kw'])
                for s in e.get('seeds', [0, 1, 2]):
                    jobs.append((e['name'], c, s))
            todo = [j for j in jobs if (j[0], j[2]) not in started
                    and not os.path.exists(os.path.join(HERE, 'runs', f'{j[0]}_s{j[2]}', 'result.json'))]
            if todo:
                name, c, s = todo[0]
                started.add((name, s))
                print(time.strftime('%H:%M:%S'), 'launch', name, s, flush=True)
                pool.submit(run_one, name, c, s)
                time.sleep(20)
                continue
        time.sleep(15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['list', 'run', 'follow'])
    p.add_argument('--only', default=None)
    p.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--steps', type=float, default=None)
    a = p.parse_args()
    V = configs()
    if a.only:
        V = {k: V[k] for k in a.only.split(',')}
    if a.cmd == 'follow':
        return follow(a)
    if a.cmd == 'list':
        for k, c in V.items():
            print(f'{k:22s} cost={cost(k, c):.2f} policy={c["policy"]} envs={c["num_envs"]}')
        print(len(V), 'configs')
        return
    jobs = sorted(((k, c, s) for k, c in V.items() for s in a.seeds), key=lambda t: -cost(t[0], t[1]))
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda t: run_one(*t, steps=a.steps), jobs))
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
