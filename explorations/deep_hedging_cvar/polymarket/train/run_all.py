"""The whole training pipeline, resumable: preflight -> probes -> sweep -> select -> seeds -> publish.

    python run_all.py                          # full run (defaults below), 4 cores
    python run_all.py --dry-run                # print the stages and commands only
    python run_all.py --from select            # redo from a stage (later outputs are rebuilt)

Re-running continues where it stopped: every stage is idempotent (the sweep
resumes its Optuna study and re-queues trials a kill interrupted; seeds skip
finished seeds; finished runs are never retrained or re-ledgered).
State and logs: runs/pipeline/<name>/ (state.json, one log per stage).

Stages
  preflight  env frozen (mm.h sha256 = --env-sha256), binding fresh, quick unit tests
  probes     probes.py on the real env (join, const, teacher), --probe-steps each;
             failures are reported, not fatal unless --require-probes
  sweep      sweep.py --name <name>: --trials TPE trials of --steps agent steps,
             --jobs processes x 1 torch thread, then the --reseed-top best configs
             re-trained to 3 seeds
  select     select.py --tag <name> --min-seeds 3 (seed mean; optional --cvar-floor,
             --max-inv, --require-mm) -> runs/sweeps/<name>/selected.json
  seeds      seeds.py: the selected config on 5 fresh seeds (100-104) for
             --final-steps steps each -> runs/seeds/<hash>/seeds.json
  publish    train/selected.json in the format final/make_plan.py reads:
             {"selected": {"cfg", "cfg_hash", "runs": [{"seed", "ckpt", "ckpt_sha256"}]}}
             with the 5 final seeds, plus the selection record and the seed summary.

Every validation evaluation lands in train/trials.jsonl (one row each) and every
run summary in train/runs.jsonl. Nothing touches the test split.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
PY = sys.executable
STAGES = ('preflight', 'probes', 'sweep', 'select', 'seeds', 'publish')
FROZEN_MM_H = '746dd7dea250eee035a8cb08a7be479eacf2d44ac1b7251503bc0d0e468566a9'  # DECISIONS.md D1.4


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Pipeline:
    def __init__(self, a):
        self.a = a
        self.dir = os.path.join(HERE, 'runs', 'pipeline', a.name)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, 'state.json')
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.sweep_dir = os.path.join(HERE, 'runs', 'sweeps', a.name)
        self.real = a.env == 'real'
        self.ledger_args = [] if self.real else ['--ledger', os.path.join(HERE, 'runs', 'standin_trials.jsonl')]
        self.selected_path = os.path.join(self.sweep_dir, 'selected.json')

    def save(self):
        json.dump(self.state, open(self.state_path, 'w'), indent=1, default=str)

    def run(self, stage, cmd):
        log = os.path.join(self.dir, f'{stage}.log')
        print(f'[{time.strftime("%H:%M:%S")}] {stage}: {" ".join(shlex.quote(c) for c in cmd)}', flush=True)
        if self.a.dry_run:
            return 0
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE', 'PYTHONWARNINGS': 'ignore'}
        with open(log, 'a') as f:
            f.write(f'\n==== {time.strftime("%Y-%m-%dT%H:%M:%S")} {cmd}\n')
            f.flush()
            rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        print(f'[{time.strftime("%H:%M:%S")}] {stage}: exit {rc} (log {log})', flush=True)
        return rc

    def done(self, stage):
        return self.state.get(stage, {}).get('done', False)

    def mark(self, stage, **info):
        if self.a.dry_run:
            return
        self.state[stage] = {'done': True, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **info}
        self.save()

    # ---------------------------------------------------------------- stages
    def preflight(self):
        a = self.a
        mm = sha256(os.path.join(PM, 'mm.h'))
        if self.real and a.env_sha256 and mm != a.env_sha256:
            raise SystemExit(f'mm.h sha256 {mm} != frozen {a.env_sha256} (DECISIONS.md D1.4); '
                             'pass --env-sha256 "" to skip')
        so = [f for f in os.listdir(PM) if f.startswith('binding') and f.endswith('.so')]
        if self.real and not so or so and os.path.getmtime(os.path.join(PM, so[0])) < os.path.getmtime(os.path.join(PM, 'mm.h')):
            raise SystemExit('the binding .so is missing or older than mm.h: rebuild with '
                             '`python setup.py build_ext --inplace` in polymarket/')
        if not a.skip_tests and self.run('preflight', [PY, 'test_train.py']) != 0:
            raise SystemExit('test_train.py failed')
        self.mark('preflight', mm_h_sha256=mm)

    def probes(self):
        a = self.a
        out = os.path.join(self.dir, 'probes.jsonl')
        self.run('probes', [PY, 'probes.py', '--env', a.env, '--steps', str(a.probe_steps),
                            '--threads', '1', '--out', out])
        res = [json.loads(l) for l in open(out)] if os.path.exists(out) and not a.dry_run else []
        ok = bool(res) and all(r.get('pass') for r in res)
        if a.require_probes and not ok and not a.dry_run:
            raise SystemExit(f'probes failed: {res}')
        self.mark('probes', results=res, all_pass=ok)

    def sweep(self):
        a = self.a
        cmd = [PY, 'sweep.py', '--name', a.name, '--env', a.env, '--trials', str(a.trials), *self.ledger_args,
               '--steps', str(a.steps), '--jobs', str(a.jobs), '--threads', '1',
               '--reseed-top', str(a.reseed_top), '--reseed-n', '3']
        if a.base_config:
            cmd += ['--base-config', a.base_config]
        if a.train_args:
            cmd += ['--'] + shlex.split(a.train_args)
        if self.run('sweep', cmd) != 0:
            raise SystemExit('sweep failed (re-run run_all.py to resume)')
        self.mark('sweep')

    def select(self):
        a = self.a
        cmd = [PY, 'select.py', '--tag', a.name, '--min-seeds', '3', '--out', self.selected_path,
               *self.ledger_args]
        if not self.real:
            cmd += ['--tiebreak-variant', 'central_1s']
        if a.cvar_floor is not None:
            cmd += ['--cvar-floor', str(a.cvar_floor)]
        if a.max_inv is not None:
            cmd += ['--max-inv', str(a.max_inv)]
        if a.require_mm:
            cmd += ['--require-mm']
        if self.run('select', cmd) != 0:
            raise SystemExit('select found no eligible config (see select.log and selected.json)')
        if self.a.dry_run:
            return
        sel = json.load(open(self.selected_path))
        self.mark('select', cfg_hash=sel['selected']['cfg_hash'], mean=sel['selected']['mean'])

    def seeds(self):
        a = self.a
        cmd = [PY, 'seeds.py', '--selected', self.selected_path, '--jobs', str(a.seed_jobs), *self.ledger_args,
               '--threads', '1', '--seeds', *[str(s) for s in a.final_seeds]]
        if a.final_steps:
            cmd += ['--timesteps', str(a.final_steps)]
        out = self.seeds_out()
        cmd += ['--out', out]
        if self.run('seeds', cmd) != 0:
            raise SystemExit('seeds failed (re-run run_all.py to resume)')
        self.mark('seeds', out=out)

    def seeds_out(self):
        return os.path.join(HERE, 'runs', 'seeds', self.a.name)

    def publish(self):
        if self.a.dry_run:
            print('publish: train/selected.json from the selection and the final seeds', flush=True)
            return
        sel = json.load(open(self.selected_path))
        summ = json.load(open(os.path.join(self.seeds_out(), 'seeds.json')))
        runs = []
        for seed, r in sorted(summ['per_seed'].items(), key=lambda kv: int(kv[0])):
            if sha256(r['ckpt']) != r['ckpt_sha256']:
                raise SystemExit(f'checkpoint {r["ckpt"]} changed since training')
            runs.append({'seed': int(seed), 'ckpt': r['ckpt'], 'ckpt_sha256': r['ckpt_sha256'],
                         'cfg_hash': r['cfg_hash']})
        cfg = json.load(open(os.path.join(self.seeds_out(), 'config.json')))['cfg']
        hashes = sorted({r['cfg_hash'] for r in runs})
        doc = {
            'selected': {'cfg': cfg, 'cfg_hash': hashes[0] if len(hashes) == 1 else hashes,
                         'runs': runs, 'n_seeds': len(runs)},
            'selection': {'sweep': self.a.name, 'selected_json': self.selected_path,
                          'sweep_cfg_hash': sel['selected']['cfg_hash'], 'rule': sel['rule'],
                          'sweep_seed_mean': sel['selected']['mean'], 'sweep_seeds': sel['selected']['seeds'],
                          'n_val_evals_ledger_at_selection': sel.get('n_val_evals_ledger')},
            'final_seeds_val': summ['across_seeds'],
            'note': 'cfg = the selected sweep config with total_timesteps set for the final seeds '
                    '(hence its own cfg_hash); runs = the final seed checkpoints final/make_plan.py uses',
        }
        out = os.path.join(HERE, 'selected.json') if self.real else os.path.join(self.dir, 'selected.json')
        json.dump(doc, open(out, 'w'), indent=1, default=float)
        print(f'published {out}: {len(runs)} seeds', flush=True)
        self.mark('publish', path=out)

    def go(self):
        a = self.a
        start = STAGES.index(a.from_stage) if a.from_stage else 0
        for st in STAGES[:start]:
            if not self.done(st) and st not in a.skip:
                raise SystemExit(f'--from {a.from_stage}: stage {st} has not finished')
        for st in STAGES[start:]:
            if st in a.skip:
                print(f'skip {st}', flush=True)
                continue
            if self.done(st) and not (a.from_stage and STAGES.index(st) >= start):
                print(f'{st}: done ({self.state[st]["time"]})', flush=True)
                continue
            getattr(self, st)()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', default='main', help='sweep / pipeline name (ledger tag)')
    p.add_argument('--env', default='real', choices=['real', 'standin'],
                   help='standin: plumbing test (own ledger; publishes inside runs/pipeline/<name>)')
    p.add_argument('--trials', type=int, default=64, help='sweep trial budget (= the baselines\')')
    p.add_argument('--steps', type=float, default=25e6, help='agent steps per sweep trial')
    p.add_argument('--jobs', type=int, default=4, help='parallel training processes (1 torch thread each)')
    p.add_argument('--reseed-top', type=int, default=4)
    p.add_argument('--seed-jobs', type=int, default=5, help='final seeds in parallel (1 thread each)')
    p.add_argument('--final-steps', type=float, default=100e6, help='agent steps per final seed')
    p.add_argument('--final-seeds', type=int, nargs='*', default=[100, 101, 102, 103, 104])
    p.add_argument('--probe-steps', type=float, default=2e6)
    p.add_argument('--require-probes', action='store_true')
    p.add_argument('--cvar-floor', type=float, default=None)
    p.add_argument('--max-inv', type=float, default=None)
    p.add_argument('--require-mm', action='store_true')
    p.add_argument('--base-config', default=None, help='JSON config for train_ppo.py --config')
    p.add_argument('--train-args', default='', help='extra train_ppo.py flags for every sweep run')
    p.add_argument('--env-sha256', default=FROZEN_MM_H)
    p.add_argument('--skip-tests', action='store_true')
    p.add_argument('--skip', nargs='*', default=[], choices=STAGES)
    p.add_argument('--from', dest='from_stage', choices=STAGES, default=None)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    a.steps, a.final_steps, a.probe_steps = int(a.steps), int(a.final_steps), int(a.probe_steps)
    t0 = time.time()
    Pipeline(a).go()
    print(f'pipeline {a.name}: finished in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
