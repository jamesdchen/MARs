"""w5: BC warm start from three teachers + PPO fine-tune at reward_mode 2 (fill markout)
with a decaying KL to the BC teacher. Resumable job queue (4 x 1-thread processes).

    python sweep.py bc                       # one BC checkpoint per teacher (train split only)
    python sweep.py plan                     # write runs/plan.json (sampled configs)
    python sweep.py run --seeds 0 1 2        # train every config x seed (skip finished)
    python sweep.py run --only c03 c11 --seeds 3 4
"""
import os
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, copy, hashlib, json, subprocess, sys, time
import numpy as np

W5 = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(W5, '..', '..'))
PM = os.path.dirname(TRAIN)
PY = sys.executable
DATA = os.path.join(PM, 'data', 'build_v2')
RUNS = os.path.join(W5, 'runs')
BC = os.path.join(RUNS, 'bc')
LEDGER = os.path.join(W5, 'trials.jsonl')
TEACHERS = {'asl_act': ('as_logit', 'tuned_active'), 'fv': ('fair_value', 'tuned'),
            'fv_act': ('fair_value', 'tuned_active')}
BASE_ENV = {'mark_mode': 1, 'reward_mode': 2, 'data_dir': DATA}
POLICY = {'kind': 'mlp', 'hidden': 64, 'layers': 2, 'act': 'tanh', 'shared': True}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def bc_path(t, bias=0.0):
    return os.path.join(BC, f'{t}.pt' if not bias else f'{t}_off{bias:+g}.pt')


def do_bc(a):
    os.makedirs(BC, exist_ok=True)
    procs = []
    for t, (name, src) in TEACHERS.items():
        out = bc_path(t)
        if os.path.exists(out):
            continue
        kw = {**BASE_ENV, 'markout_h': 120, 'inv_penalty': 1e-4}
        cmd = [PY, 'bc.py', '--env', 'real', '--teacher', name, '--teacher-source', src,
               '--num-envs', str(a.bc_envs), '--steps', str(a.bc_steps), '--epochs', str(a.bc_epochs), '--value', '--threads', '1',
               '--policy', 'mlp', '--hidden', '64', '--layers', '2', '--out', out,
               '--env-kw', *[f'{k}={json.dumps(v)}' for k, v in kw.items()]]
        procs.append(subprocess.Popen(cmd, cwd=TRAIN, stdout=open(out + '.log', 'w'), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    # anti-'off' bias variants: lower the initial logit of 'off' on both price heads of the BC policy
    import torch
    for t in TEACHERS:
        for b in (-0.5, -1.0):
            out = bc_path(t, b)
            if os.path.exists(out) or not os.path.exists(bc_path(t)):
                continue
            ck = torch.load(bc_path(t), map_location='cpu', weights_only=False)
            nvec = ck['nvec']
            bias = ck['policy']['actor_head.bias']
            for h in (0, 1):
                bias[sum(nvec[:h])] += b
            ck['bc']['off_bias'] = b
            torch.save(ck, out)
            print('wrote', out)


def do_plan(a):
    rng = np.random.RandomState(a.plan_seed)
    plan = []
    tnames = list(TEACHERS)
    for i in range(a.configs):
        t = tnames[i % 3]
        h = (60, 120, 300)[(i // 3) % 3]
        plan.append({'name': f'c{i:02d}_{t}_h{h}', 'teacher': t, 'markout_h': h,
                     'kl_coef': float(10 ** rng.uniform(-2, np.log10(0.3))),
                     'kl_decay': float(rng.choice([0.25, 0.5, 1.0])),
                     'inv_penalty': float(10 ** rng.uniform(-5, -2.5)),
                     'off_bias': float(rng.choice([0.0, 0.0, -0.5, -1.0])),
                     'lr': float(6.8e-4 * 10 ** rng.uniform(-0.7, 0.2)),
                     'ent_coef': float(10 ** rng.uniform(-4, -2.5))})
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(W5, 'plan.json')
    if os.path.exists(path) and not a.force:
        raise SystemExit(f'{path} exists')
    json.dump(plan, open(path, 'w'), indent=1)
    print(json.dumps(plan, indent=1))


def job_cmd(c, seed, steps):
    rdir = os.path.join(RUNS, f'{c["name"]}_s{seed}')
    kw = {**BASE_ENV, 'markout_h': c['markout_h'], 'inv_penalty': c['inv_penalty']}
    if c.get('reward_scale'):
        kw['reward_scale'] = c['reward_scale']
    cfg = {'env': {'backend': 'real', 'kw': kw},
           'policy': dict(POLICY),
           'ppo': {'learning_rate': c['lr'], 'ent_coef': c['ent_coef'], 'gamma': 0.9988214282795336,
                   'gae_lambda': 0.9794307540969474, 'bptt_horizon': 64, 'minibatch_size': 8192,
                   'update_epochs': 1},
           'num_envs': 4096, 'total_timesteps': int(steps)}
    os.makedirs(os.path.join(RUNS, 'cfg'), exist_ok=True)
    cpath = os.path.join(RUNS, 'cfg', f'{c["name"]}.json')
    json.dump(cfg, open(cpath, 'w'), indent=1)
    cmd = [PY, 'train_ppo.py', '--config', cpath, '--seed', str(seed), '--threads', '1', '--out', rdir,
           '--tag', f'w5_bcmk_{c["name"]}', '--quiet', '--ledger', LEDGER,
           '--init', bc_path(c['teacher'], c['off_bias']), '--kl-coef', str(c['kl_coef']),
           '--kl-decay', str(c['kl_decay']), '--kl-teacher', bc_path(c['teacher'])]
    return rdir, cmd


def do_plan2(a):
    """Round 2 (appended to plan.json): larger reward_scale so the value head learns from the
    markout credit (round 1: explained variance ~0.03 at the default 1/Q), lower KL."""
    path = os.path.join(W5, 'plan.json')
    plan = json.load(open(path))
    if any(c.get('round') == 2 for c in plan):
        raise SystemExit('round 2 already planned')
    rng = np.random.RandomState(a.plan_seed + 2)
    i0 = len(plan)
    specs = [('asl_act', rs, h) for rs in (0.1, 1.0) for h in (60, 120, 300)] + \
            [('fv_act', rs, h) for rs in (0.1, 1.0) for h in (120, 300)] + \
            [('asl_act', 0.1, 120), ('asl_act', 0.1, 300), ('asl_act', 1.0, 120), ('fv', 0.1, 120)]
    for j, (t, rs, h) in enumerate(specs):
        plan.append({'name': f'c{i0 + j:02d}_{t}_h{h}_rs{rs:g}', 'round': 2, 'teacher': t, 'markout_h': h,
                     'reward_scale': rs,
                     'kl_coef': float(10 ** rng.uniform(-2, np.log10(0.05))),
                     'kl_decay': float(rng.choice([0.25, 0.5])),
                     'inv_penalty': float(10 ** rng.uniform(-5, -3)),
                     'off_bias': 0.0,
                     'lr': float(6.8e-4 * 10 ** rng.uniform(-0.5, 0.2)),
                     'ent_coef': float(10 ** rng.uniform(-4, -2.5))})
    json.dump(plan, open(path, 'w'), indent=1)
    print(json.dumps(plan[i0:], indent=1))


def do_run(a):
    plan = json.load(open(os.path.join(W5, 'plan.json')))
    if a.only:
        plan = [c for c in plan if any(c['name'].startswith(o) for o in a.only)]
    jobs = []
    for seed in a.seeds:          # seed-major so every config gets seed 0 first
        for c in plan:
            rdir, cmd = job_cmd(c, seed, a.steps)
            if os.path.exists(os.path.join(rdir, 'result.json')):
                continue
            if os.path.exists(rdir):  # killed earlier: keep the partial dir
                k = 0
                while os.path.exists(f'{rdir}.killed{k}'):
                    k += 1
                os.rename(rdir, f'{rdir}.killed{k}')
            jobs.append((rdir, cmd))
    print(f'{len(jobs)} jobs', flush=True)
    running = []
    env = {**os.environ, 'OMP_WAIT_POLICY': 'passive', 'OMP_NUM_THREADS': '1'}
    while jobs or running:
        while jobs and len(running) < a.jobs:
            rdir, cmd = jobs.pop(0)
            os.makedirs(RUNS, exist_ok=True)
            p = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=open(rdir + '.log', 'w'), stderr=subprocess.STDOUT)
            running.append((rdir, p, time.time()))
            print(time.strftime('%H:%M:%S'), 'start', os.path.basename(rdir), flush=True)
        time.sleep(5)
        for r in list(running):
            if r[1].poll() is not None:
                running.remove(r)
                res = os.path.join(r[0], 'result.json')
                msg = 'FAILED'
                if os.path.exists(res):
                    v = json.load(open(res))['val']['central_1s']
                    msg = f'pnl {v["pnl_total"]:+.1f} fills/mkt {v["fills_per_market"]:.1f} mm {v["makes_markets"]}'
                print(time.strftime('%H:%M:%S'), 'done', os.path.basename(r[0]), f'{time.time() - r[2]:.0f}s', msg, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['bc', 'plan', 'plan2', 'run'])
    p.add_argument('--bc-envs', type=int, default=1024)
    p.add_argument('--bc-steps', type=int, default=2048)
    p.add_argument('--bc-epochs', type=int, default=20)
    p.add_argument('--configs', type=int, default=36)
    p.add_argument('--plan-seed', type=int, default=5)
    p.add_argument('--force', action='store_true')
    p.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    p.add_argument('--only', nargs='*')
    p.add_argument('--steps', type=float, default=25e6)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    {'bc': do_bc, 'plan': do_plan, 'plan2': do_plan2, 'run': do_run}[a.mode](a)


if __name__ == '__main__':
    main()
