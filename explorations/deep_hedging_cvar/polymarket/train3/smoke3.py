"""End-to-end mechanics check of the v3 training stack, per track (no conclusions are drawn from it).

    python smoke3.py --work /scratch/smoke3                     # synthetic sports + elections builds
    python smoke3.py --work /scratch/smoke3 --real-sports B1 --real-elections B2   # + tiny real builds

For each track it runs, each step in its own process (mm3 maps one data directory per process),
at `nice -n 10` with one thread:
1. data: v3env/fixtures3.py (synthetic builds in the builders' schemas, then v3env/adapt3.py), or a
   real builder output (adapt3.py is run if markets3.bin is missing);
2. `split3.py --dry-run --out WORK/split_<name>.json` (a dry-run split, accepted only by --smoke);
3. `train_ppo3.py --smoke` for --timesteps (default 1M) with an eval every half, then the final
   eval under central_1s and pess_1s (ledger rows go to train3/smoke_trials.jsonl);
4. `evaluator3.py policy --smoke` on the final checkpoint (per-market rows) and
   `evaluator3.py comparators --smoke` (no-quote, sym_touch, mid_value, as_mid, + reward_farmer on
   elections).
If the dry-run split has no val day (a one-day real build), training runs with --no-val and the
evaluations use the train split (split 0); nothing then reaches a ledger from steps 3-4.
Finally one `smoke_check` row per track (headline numbers, wall times, errors) is appended to
train3/smoke_trials.jsonl. A failing step stops that track and is reported.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
PY = sys.executable
SMOKE_LEDGER = os.path.join(HERE, 'smoke_trials.jsonl')
ENV_KW = {'sports': ['obs_size=80', 'liq_rewards=1', 'size_mode=1', 'Q=1000', 'pull_s=60', 'flat_c=0.5'],
          'elections': ['obs_size=80', 'liq_rewards=1', 'size_mode=1', 'Q=1000']}


def run(cmd, cwd=HERE, log=None):
    t0 = time.time()
    full = ['nice', '-n', '10', PY] + cmd
    p = subprocess.run(full, cwd=cwd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if log:
        with open(log, 'a') as f:
            f.write(f'$ {" ".join(full)}\n{out}\n')
    if p.returncode:
        raise RuntimeError(f'{" ".join(cmd[:2])} failed ({p.returncode}): {out[-2000:]}')
    return out, time.time() - t0


def json_lines(out, key):
    rows = []
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith('{') and f'"{key}"' in ln:
            try:
                rows.append(json.loads(ln.replace('NaN', 'null')))
            except json.JSONDecodeError:
                pass
    return rows


def track(name, kind, build, work, timesteps, log):
    rec = {'kind': 'smoke_check', 'name': name, 'track': kind, 'build': os.path.realpath(build),
           'timesteps': timesteps, 'steps': {}, 'ok': False}
    try:
        if not os.path.exists(os.path.join(build, 'markets3.bin')):
            _, t = run([os.path.join(PM, 'v3env', 'adapt3.py'), build], log=log)
            rec['steps']['adapt3_s'] = t
        split = os.path.join(work, f'split_{name}.json')
        if os.path.exists(split):
            os.remove(split)
        out, t = run(['split3.py', build, '--dry-run', '--out', split], log=log)
        sp = json.load(open(split))
        counts = sp['_counts']
        rec['split'] = {k: len(v) for k, v in counts.items()}
        rec['steps']['split3_s'] = t
        has_val = bool(counts['val'])
        ev_split = '1' if has_val else '0'
        rd = os.path.join(work, 'runs', name)
        cmd = ['train_ppo3.py', '--smoke', '--data-dir', build, '--split-map', split, '--env3-kw', *ENV_KW[kind],
               '--timesteps', str(timesteps), '--num-envs', '1024', '--threads', '1', '--seed', '0',
               '--eval-every', str(timesteps // 2), '--tag', f'smoke_{name}', '--out', rd]
        if not has_val:
            cmd.append('--no-val')
        out, t = run(cmd, log=log)
        rec['steps']['train_s'] = t
        done = json_lines(out, 'done')
        rec['train'] = done[-1] if done else None
        res = json.load(open(os.path.join(rd, 'result.json')))
        rec['train_sps'] = res['sps_median']
        base = ['--smoke', '--data-dir', build, '--split-map', split, '--split', ev_split, '--env3-kw', *ENV_KW[kind],
                '--variants', 'central_1s', 'pess_1s', '--tag', f'smoke_{name}', '--markets']
        out, t = run(['evaluator3.py', 'policy', '--ckpt', os.path.join(rd, 'final.pt'), *base,
                      '--out', os.path.join(work, f'eval_policy_{name}.json')], log=log)
        rec['steps']['eval_policy_s'] = t
        rec['policy'] = json_lines(out, 'eval')
        out, t = run(['evaluator3.py', 'comparators', *base, '--out', os.path.join(work, f'eval_comp_{name}.json')],
                     log=log)
        rec['steps']['comparators_s'] = t
        rec['comparators'] = json_lines(out, 'eval')
        rec['eval_split'] = int(ev_split)
        rec['ok'] = True
    except Exception as e:  # noqa: BLE001
        rec['error'] = f'{type(e).__name__}: {e}'
    sys.path.insert(0, os.path.join(PM, 'train'))
    import ledger
    ledger.append(rec, SMOKE_LEDGER)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True, help='scratch directory (fixtures, splits, runs, logs)')
    ap.add_argument('--timesteps', type=int, default=1_000_000)
    ap.add_argument('--real-sports', help='a build_sports.py output (mechanics check only)')
    ap.add_argument('--real-elections', help='a build_elections.py output (mechanics check only)')
    ap.add_argument('--no-fixtures', action='store_true')
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    log = os.path.join(a.work, 'smoke3.log')
    jobs = []
    if not a.no_fixtures:
        fx = os.path.join(a.work, 'fixtures')
        if not os.path.exists(os.path.join(fx, 'elections', 'markets3.bin')):
            run([os.path.join(PM, 'v3env', 'fixtures3.py'), fx], cwd=os.path.join(PM, 'v3env'), log=log)
        jobs += [('fx_sports', 'sports', os.path.join(fx, 'sports')),
                 ('fx_elections', 'elections', os.path.join(fx, 'elections'))]
    if a.real_sports:
        jobs.append(('real_sports', 'sports', a.real_sports))
    if a.real_elections:
        jobs.append(('real_elections', 'elections', a.real_elections))
    out = []
    for name, kind, build in jobs:
        r = track(name, kind, build, a.work, a.timesteps, log)
        out.append(r)
        print(json.dumps({'track': name, 'ok': r['ok'], 'error': r.get('error'), 'split': r.get('split'),
                          'steps': {k: round(v, 1) for k, v in r['steps'].items()}, 'train_sps': r.get('train_sps'),
                          'policy': r.get('policy'), 'comparators': r.get('comparators')}, default=str), flush=True)
    with open(os.path.join(a.work, 'smoke3_summary.json'), 'w') as f:
        json.dump(out, f, indent=1, default=str)
    return 0 if all(r['ok'] for r in out) else 1


if __name__ == '__main__':
    sys.exit(main())
