"""Named improvement rounds (DECISIONS.md D3/D4), resumable and detachable.

    python rounds.py status                         # best so far, rounds done, stopping rule
    python rounds.py run R1 --detach                # one round in the background
    python rounds.py auto --detach                  # R1..R6 in D4 order until the stopping rule
    python rounds.py finalize --detach              # 5 final seeds of the best -> train/selected.json
    python rounds.py run R2 --dry-run               # print the plan only

A round trains a small set of configs x 3 seeds (train_ppo.py, one process per
run, --jobs processes x 1 torch thread), each run's val evaluations go to the
ledger train/trials.jsonl tagged with the round name, and its summary to
train/runs.jsonl. At the end the round is compared with the best before it: the
best seed-mean val P&L (central fills, 1 s; all val markets; greedy) among configs
with >= 3 seeds that pass D1.3 (fills/market >= 5, T6 share <= 0.5). The round
"improves" if its best config beats that by more than its own seed standard error
(std over seeds / sqrt(n)). Two non-improving rounds in a row, or the deadline
(15:30 UTC 2026-10-15), stop `auto` (D4). Results: runs/rounds/<round>/round.json;
history: runs/rounds/history.json; the best so far: runs/rounds/best.json.

Base config of each round: the best so far (best.json, else the best D1.3-passing
3-seed config in runs.jsonl, e.g. the main sweep's selection), unless a round says
otherwise. Rounds (D4 order):
  R1  BC warm start from the tuned fair_value baseline (baselines/tuned.json and
      tuned_active.json) + decaying KL to the BC policy; sampled over KL coef
      [0.01, 0.3] / decay {0.25, 0.5, 1} / lr [0.2x, 2x base] / ent_coef [1e-4, 1e-2]
      (--configs, default 16).
  R2  'off' prior at init (MLP head bias of the two price heads; logit bonus
      {0.5,1,2,3} x lr {0.5x, 1x, 2x base}); fresh init, no BC.
Every run trains --steps (default 25M, the main sweep's budget) except R4.
  R3  mark_mode 1 and 2 on the best config of R1, of R2 and the overall best.
  R4  longer runs (--long-steps, default 200M) of the 2 best configs.
  R5  pretraining on the simulator (data/build_sim, same obs layout) then fine-tuning
      on the replay; refused unless sim/validation_env.json "overall" is true.
  R6  recurrent PufferNet replica (layers {1, 2} x hidden {64, 128}).
  R7  (DECISIONS.md D5; runs after R2) env reward_mode 2: each fill credited with its
      markout over markout_h s {30, 60, 120} plus rebate, inventory penalty, no MTM;
      on the best of R1/R2 (its BC init or off prior), sampled over inv_penalty, lr,
      ent_coef (--r7-configs, default 16). Selection and reports stay on true P&L.
      Refused until the env accepts reward_mode=2 (a test call); `auto` waits up to
      --r7-wait minutes for it.
Resuming: re-run the same command; finished runs are reused (never retrained or
re-ledgered), killed runs restart (partial logs kept as *.killedN).
"""

import argparse
import concurrent.futures as cf
import copy
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
PY = sys.executable
sys.path.insert(0, HERE)
import ledger  # noqa: E402

ORDER = ('R1', 'R2', 'R7', 'R3', 'R4', 'R5', 'R6')   # D4 + D5 (R7 right after R2)
TITLES = {'R1': 'bc_warmstart', 'R2': 'off_prior', 'R7': 'markout_reward', 'R3': 'mark_mode',
          'R4': 'longer', 'R5': 'sim_pretrain', 'R6': 'recurrent'}
DEADLINE = '2026-10-15T15:30:00+00:00'
ROOT = os.path.join(HERE, 'runs', 'rounds')
SIM_DIR = os.path.join(PM, 'data', 'build_sim')
SIM_GATE = os.path.join(PM, 'sim', 'validation_env.json')
SPS = {'mlp': 140e3, 'puffernet': 65e3}  # per 1-thread process, 4 in parallel (bench.py)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc)


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    json.dump(obj, open(tmp, 'w'), indent=1, default=float)
    os.replace(tmp, p)


# ----------------------------------------------------------------- scoring
def run_metrics(res, variant=None):
    v = (res.get('val') or {}).get(variant or res.get('headline') or 'central_1s') or {}
    pnl = v.get('pnl_total')
    t6 = v.get('t6_share')
    return {'pnl_total': pnl, 'fills': v.get('fills_per_market'),
            't6': 0.0 if t6 is None or (isinstance(t6, float) and math.isnan(t6)) else t6,
            'cvar95': v.get('cvar95'), 'pess': ((res.get('val') or {}).get('pess_1s') or {}).get('pnl_total')}


def group_stats(results):
    """Seed statistics of one config from its result dicts (first per seed)."""
    by_seed = {}
    for r in results:
        by_seed.setdefault(r['seed'], r)
    ms = [run_metrics(r) for r in by_seed.values()]
    vals = [m['pnl_total'] for m in ms if m['pnl_total'] is not None]
    if not vals:
        return None
    n = len(vals)
    std = float(np.std(vals, ddof=1)) if n > 1 else float('nan')
    fills = float(np.mean([m['fills'] or 0.0 for m in ms]))
    t6 = float(np.mean([m['t6'] for m in ms]))
    return {'n': n, 'seeds': sorted(by_seed), 'mean': float(np.mean(vals)), 'std': std,
            'se': std / math.sqrt(n) if n > 1 else float('nan'), 'per_seed': vals,
            'fills_per_market': fills, 't6_share': t6, 'd13': fills >= 5 and t6 <= 0.5,
            'cvar95_mean': float(np.mean([m['cvar95'] for m in ms if m['cvar95'] is not None] or [np.nan])),
            'pess_mean': float(np.mean([m['pess'] for m in ms if m['pess'] is not None] or [np.nan]))}


def configs_from_index(runs_path, tag=None, backend=None):
    groups = {}
    for r in ledger.read(runs_path):
        if r.get('kind') != 'train_run' or not r.get('val'):
            continue
        if tag is not None and r.get('tag') != tag:
            continue
        if backend and r.get('backend') != backend:
            continue
        groups.setdefault(r['cfg_hash'], []).append(r)
    out = []
    for h, rs in groups.items():
        st = group_stats(rs)
        if st:
            out.append({'cfg_hash': h, 'cfg': rs[0]['cfg'], 'tag': rs[0].get('tag'), **st,
                        'runs': [{'seed': r['seed'], 'ckpt': r.get('ckpt'), 'ckpt_sha256': r.get('ckpt_sha256'),
                                  'run_dir': r.get('run_dir')} for r in rs]})
    return out


def best_of(cands, min_seeds=3, require_d13=True):
    ok = [c for c in cands if c['n'] >= min_seeds and (c['d13'] or not require_d13)]
    return max(ok, key=lambda c: c['mean']) if ok else None


# ----------------------------------------------------------------- the launcher
class Rounds:
    def __init__(self, a):
        self.a = a
        self.standin = a.env == 'standin'
        self.root = os.path.join(ROOT, 'standin') if self.standin else ROOT
        self.ledger_path = ledger.STANDIN_LEDGER if self.standin else ledger.LEDGER
        self.runs_path = ledger.runs_path(self.ledger_path)
        self.hist_path = os.path.join(self.root, 'history.json')
        self.best_path = os.path.join(self.root, 'best.json')

    # -- state
    def history(self):
        return jload(self.hist_path, [])

    def best(self):
        """The best so far: best.json if a round set it, else the index."""
        b = jload(self.best_path)
        cands = configs_from_index(self.runs_path, backend='standin' if self.standin else 'real')
        ib = best_of(cands)
        if b and (ib is None or b['mean'] >= ib['mean']):
            return b
        return ib

    def fallback_base(self):
        cands = configs_from_index(self.runs_path, backend='standin' if self.standin else 'real')
        return best_of(cands) or best_of(cands, require_d13=False) or best_of(cands, min_seeds=1, require_d13=False)

    def stop_status(self):
        h = [r for r in self.history() if r.get('status') == 'done']
        streak = 0
        for r in reversed(h):
            if r.get('improved'):
                break
            streak += 1
        late = now() >= dt.datetime.fromisoformat(self.a.deadline)
        return {'non_improving_streak': streak, 'past_deadline': late,
                'stop': late or streak >= 2, 'deadline': self.a.deadline}

    # -- planning
    def plan(self, rnd, base):
        a = self.a
        bcfg = copy.deepcopy(base['cfg'])
        steps = int(a.steps)
        rng = np.random.RandomState(a.plan_seed + ORDER.index(rnd))
        lr0 = bcfg['ppo']['learning_rate']
        V = []

        def variant(name, cfg, pre=None):
            cfg = copy.deepcopy(cfg)
            cfg.setdefault('total_timesteps', steps)
            V.append({'name': name, 'cfg': cfg, 'pre': pre})

        def fresh(cfg):
            c = copy.deepcopy(cfg)
            c['init'], c['kl_teacher'] = None, None
            c['total_timesteps'] = steps
            return c

        if rnd == 'R1':
            for i in range(a.configs):
                src = ('tuned', 'tuned_active')[i % 2]
                c = fresh(bcfg)
                c['ppo']['learning_rate'] = float(lr0 * 10 ** rng.uniform(-0.7, 0.3))
                c['ppo']['ent_coef'] = float(10 ** rng.uniform(-4, -2))
                kl = float(10 ** rng.uniform(-2, -0.5))
                decay = float(rng.choice([0.25, 0.5, 1.0]))
                variant(f'c{i:02d}_{src}', c, pre={'kind': 'bc', 'source': src, 'kl_coef': kl, 'kl_decay': decay})
        elif rnd == 'R2':
            if bcfg['policy'].get('kind', 'mlp') != 'mlp':
                raise SystemExit('R2 needs an MLP base (the off prior is a head bias)')
            for prior in (0.5, 1.0, 2.0, 3.0):
                for lm in (0.5, 1.0, 2.0):
                    c = fresh(bcfg)
                    c['policy']['off_prior'] = prior
                    c['ppo']['learning_rate'] = float(lr0 * lm)
                    variant(f'prior{prior:g}_lr{lm:g}x', c)
        elif rnd == 'R3':
            bases = [('best', base)]
            for r in ('R1', 'R2'):
                rb = best_of(configs_from_index(self.runs_path, tag=self.tag(r)))
                if rb and rb['cfg_hash'] not in [b['cfg_hash'] for _, b in bases]:
                    bases.append((r, rb))
            for bname, b in bases:
                for mm in (1, 2):
                    c = copy.deepcopy(b['cfg'])
                    if c['env']['kw'].get('mark_mode', 0) == mm:
                        continue
                    c['env']['kw']['mark_mode'] = mm
                    c['total_timesteps'] = steps
                    variant(f'{bname}_mark{mm}', c)
        elif rnd == 'R4':
            cands = sorted([c for c in configs_from_index(self.runs_path, backend='standin' if self.standin else 'real')
                            if c['n'] >= 3 and c['d13']], key=lambda c: -c['mean'])[:2] or [base]
            for i, b in enumerate(cands):
                c = copy.deepcopy(b['cfg'])
                c['total_timesteps'] = int(a.long_steps)
                variant(f'top{i + 1}_{a.long_steps / 1e6:g}M', c)
        elif rnd == 'R5':
            gate = jload(SIM_GATE, {})
            if not self.standin and gate.get('overall') is not True and not a.force_sim:
                raise SystemExit(f'R5 refused: sim gate not passed ({SIM_GATE} overall = {gate.get("overall")})')
            for pre_steps in a.pretrain_steps:
                c = fresh(bcfg)
                variant(f'pre{pre_steps / 1e6:g}M', c, pre={'kind': 'pretrain', 'steps': int(pre_steps)})
        elif rnd == 'R7':
            init_base = self.r1r2_best() or base
            icfg = copy.deepcopy(init_base['cfg'])
            lr1 = icfg['ppo']['learning_rate']
            for i in range(a.r7_configs):
                c = copy.deepcopy(icfg)
                c['total_timesteps'] = steps
                h = (30, 60, 120)[i % 3]
                c['env']['kw'].update(reward_mode=2, markout_h=h,
                                      inv_penalty=float(10 ** rng.uniform(-5, -1.5)))
                c['ppo']['learning_rate'] = float(lr1 * 10 ** rng.uniform(-0.3, 0.3))
                c['ppo']['ent_coef'] = float(10 ** rng.uniform(-4, -2))
                variant(f'c{i:02d}_h{h}', c)
        elif rnd == 'R6':
            for layers in (1, 2):
                for hidden in (64, 128):
                    c = fresh(bcfg)
                    c['policy'] = {'kind': 'puffernet', 'hidden': hidden, 'layers': layers}
                    variant(f'pnet_L{layers}_H{hidden}', c)
        if a.max_configs:
            V = V[:a.max_configs]
        return V

    def r1r2_best(self):
        """The better of R1's and R2's best configs (D1.3 first, then any with 3 seeds)."""
        for need in (True, False):
            cands = [best_of(configs_from_index(self.runs_path, tag=self.tag(r)), require_d13=need)
                     for r in ('R1', 'R2')]
            cands = [c for c in cands if c]
            if cands:
                return max(cands, key=lambda c: c['mean'])
        return None

    def env_ready_r7(self):
        """Test call: does the installed env accept reward_mode=2 and markout_h?"""
        if self.standin:
            return True, 'stand-in'
        code = ('import numpy as np, env as E\n'
                'e = E.MmEnv(num_envs=2, seed=0, reward_mode=2, markout_h=60)\n'
                'e.reset(0)\n'
                'for _ in range(400): e.step(np.full((2, 3), 3, np.int32))\n'
                'e.close(); print("R7-ENV-OK")')
        r = subprocess.run([PY, '-c', code], cwd=PM, capture_output=True, text=True, timeout=300)
        return r.returncode == 0 and 'R7-ENV-OK' in r.stdout, (r.stderr or r.stdout)[-400:]

    def tag(self, rnd):
        return f'{rnd}_{TITLES[rnd]}' + (f'_{self.a.suffix}' if self.a.suffix else '')

    def estimate_min(self, V, seeds):
        tot = 0.0
        for v in V:
            kind = v['cfg']['policy'].get('kind', 'mlp')
            n = v['cfg']['total_timesteps'] + (v['pre']['steps'] / len(seeds) if v['pre'] and v['pre']['kind'] == 'pretrain' else 0)
            tot += len(seeds) * n / SPS[kind]
        return tot / min(self.a.jobs, max(1, len(V) * len(seeds))) / 60 * 1.1  # +10%: evals, start-up

    # -- running
    def sh(self, cmd, log):
        os.makedirs(os.path.dirname(log), exist_ok=True)
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE', 'PYTHONWARNINGS': 'ignore'}
        full = (['nice', '-n', str(self.a.nice)] if self.a.nice else []) + cmd
        with open(log, 'a') as f:
            f.write(f'\n==== {time.strftime("%Y-%m-%dT%H:%M:%S")} {" ".join(full)}\n')
            f.flush()
            rc = subprocess.run(full, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0:
            raise RuntimeError(f'exit {rc}: see {log}')

    def train(self, rdir, cfg, seed, tag, extra=()):
        res = os.path.join(rdir, 'result.json')
        if os.path.exists(res):
            return json.load(open(res))
        from sweep import set_aside
        set_aside(rdir)
        os.makedirs(rdir, exist_ok=True)
        cpath = os.path.join(rdir, 'round_config.json')
        jdump({'cfg': cfg}, cpath)
        cmd = [PY, 'train_ppo.py', '--config', cpath, '--seed', str(seed), '--threads', '1',
               '--out', rdir, '--tag', tag, '--quiet', '--ledger', self.ledger_path, *extra]
        if self.standin:
            cmd += ['--env', 'standin']
        self.sh(cmd, os.path.join(rdir + '.log'))
        return json.load(open(res)) if os.path.exists(res) else None

    def pre_bc(self, rdir, v):
        a, p = self.a, v['pre']
        out = os.path.join(rdir, f'bc_{p["source"]}.pt')
        if not os.path.exists(out):
            pk = v['cfg']['policy']
            cmd = [PY, 'bc.py', '--env', 'standin' if self.standin else 'real', '--teacher-source', p['source'],
                   '--num-envs', str(a.bc_envs), '--steps', str(a.bc_steps), '--value', '--threads', '1',
                   '--policy', pk.get('kind', 'mlp'), '--hidden', str(pk.get('hidden', 128)),
                   '--layers', str(pk.get('layers', 2)), '--out', out]
            if pk.get('kind', 'mlp') == 'mlp' and not pk.get('shared', True):
                cmd.append('--separate')
            kw = {k: v2 for k, v2 in v['cfg']['env']['kw'].items() if v2 is not None}
            if kw:
                cmd += ['--env-kw', *[f'{k}={json.dumps(x)}' for k, x in kw.items()]]
            self.sh(cmd, out + '.log')
        return out

    def pre_pretrain(self, rdir, v):
        """One simulator pretraining run per variant (seed 0), shared by the seeds."""
        p = v['pre']
        pdir = os.path.join(rdir, f'{v["name"]}_pretrain')
        ck = os.path.join(pdir, 'final.pt')
        if not os.path.exists(ck):
            c = copy.deepcopy(v['cfg'])
            c['total_timesteps'] = p['steps']
            if self.standin:
                c['env']['kw']['lengths'] = [100, 200, 400]   # a different 'simulator' for the toy
            else:
                c['env']['kw']['data_dir'] = SIM_DIR
            self.train(pdir, c, 0, self.tag('R5') + '_pretrain', extra=('--no-val', '--no-train-eval'))
        return ck

    def attach_init(self, cfg, ck, kl=None):
        import torch
        c = copy.deepcopy(cfg)
        c['init'] = {'path': os.path.abspath(ck), 'sha256': sha256(ck)}
        c['policy'] = dict(torch.load(ck, map_location='cpu', weights_only=False)['policy_kwargs'])
        if kl:
            c['kl_teacher'] = {'path': os.path.abspath(ck), 'sha256': sha256(ck),
                               'coef': kl['kl_coef'], 'decay': kl['kl_decay']}
        if self.standin:
            c['env']['kw'].pop('lengths', None)
        else:
            c['env']['kw'].pop('data_dir', None)
        return c

    def run_round(self, rnd):
        a = self.a
        tag = self.tag(rnd)
        rdir = os.path.join(self.root, tag)
        state_path = os.path.join(rdir, 'round.json')
        state = jload(state_path, {})
        if state.get('status') == 'done' and not a.redo_compare:
            print(json.dumps({'round': tag, 'status': 'done', 'improved': state.get('improved')}))
            return state
        before = state.get('best_before') or self.best()
        base = self.fallback_base() if before is None else before
        if base is None:
            raise SystemExit('no base config: run the main pipeline (or any 3-seed config) first')
        if rnd == 'R7' and not a.dry_run:
            waited = 0
            while True:
                ok, why = self.env_ready_r7()
                if ok:
                    break
                if a.mode != 'auto' or waited >= a.r7_wait:
                    raise SystemExit(f'R7 refused: the env does not accept reward_mode=2 yet ({why.strip()[-200:]})')
                print(json.dumps({'round': 'R7', 'waiting_for_env': waited}), flush=True)
                time.sleep(120)
                waited += 2
        V = self.plan(rnd, base)
        seeds = list(range(a.seeds))
        est = self.estimate_min(V, seeds)
        plan = {'round': tag, 'base_cfg_hash': base['cfg_hash'], 'base_mean': base.get('mean'),
                'variants': [{'name': v['name'], 'pre': v['pre'], 'steps': v['cfg']['total_timesteps']} for v in V],
                'seeds': seeds, 'estimate_min': round(est, 1), 'jobs': a.jobs}
        print(json.dumps(plan, default=float), flush=True)
        if a.dry_run:
            return plan
        left = (dt.datetime.fromisoformat(a.deadline) - now()).total_seconds() / 60
        if a.mode == 'auto' and est > left and not a.ignore_deadline:
            print(json.dumps({'round': tag, 'skipped': f'estimate {est:.0f} min > {left:.0f} min to the deadline'}))
            return None
        state = {'round': tag, 'type': rnd, 'status': 'running', 'started': state.get('started') or now().isoformat(),
                 'best_before': before, 'plan': plan, 'mm_h_sha256_start': self.mm_sha()}
        jdump(state, state_path)
        jdump({'pid': os.getpid(), 'round': tag, 'started': state['started']}, os.path.join(self.root, 'active.json'))
        self.append_history({k: state[k] for k in ('round', 'type', 'status', 'started')})

        # pre-steps (BC checkpoints, simulator pretraining), in parallel
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            futs = {}
            for v in V:
                if v['pre'] and v['pre']['kind'] == 'bc':
                    key = ('bc', v['pre']['source'])
                    if key not in futs:
                        futs[key] = ex.submit(self.pre_bc, rdir, v)
                elif v['pre'] and v['pre']['kind'] == 'pretrain':
                    futs[('pre', v['name'])] = ex.submit(self.pre_pretrain, rdir, v)
            pre = {k: f.result() for k, f in futs.items()}
        for v in V:
            if v['pre'] and v['pre']['kind'] == 'bc':
                v['cfg'] = self.attach_init(v['cfg'], pre[('bc', v['pre']['source'])], kl=v['pre'])
            elif v['pre'] and v['pre']['kind'] == 'pretrain':
                v['cfg'] = self.attach_init(v['cfg'], pre[('pre', v['name'])])

        results = {v['name']: [] for v in V}
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            futs = {ex.submit(self.train, os.path.join(rdir, f'{v["name"]}_s{s}'), v['cfg'], s, tag): (v['name'], s)
                    for v in V for s in seeds}
            for f in cf.as_completed(futs):
                name, s = futs[f]
                try:
                    r = f.result()
                    if r:
                        results[name].append(r)
                        m = run_metrics(r)
                        print(json.dumps({'round': tag, 'variant': name, 'seed': s, 'val_pnl': m['pnl_total'],
                                          'fills': m['fills'], 't6': m['t6']}, default=float), flush=True)
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({'round': tag, 'variant': name, 'seed': s, 'failed': str(e)}), flush=True)
        return self.compare(rnd, rdir, state, V, results)

    def compare(self, rnd, rdir, state, V, results):
        tag = self.tag(rnd)
        table = []
        for v in V:
            st = group_stats(results[v['name']])
            if st:
                h = results[v['name']][0]['cfg_hash']
                table.append({'variant': v['name'], 'cfg_hash': h, 'cfg': results[v['name']][0]['cfg'], **st,
                              'runs': [{'seed': r['seed'], 'ckpt': r['ckpt'], 'ckpt_sha256': r['ckpt_sha256'],
                                        'run_dir': r['run_dir']} for r in results[v['name']]]})
        table.sort(key=lambda t: -t['mean'])
        rb = best_of(table)
        before = state['best_before']
        delta = (rb['mean'] - before['mean']) if rb and before else None
        improved = bool(rb and (before is None or (rb['se'] == rb['se'] and delta > rb['se'])))
        pooled = math.sqrt(rb['se'] ** 2 + (before.get('se') or 0) ** 2) if rb and before and before.get('se') == before.get('se') else None
        try:
            os.remove(os.path.join(self.root, 'active.json'))
        except OSError:
            pass
        end_sha = self.mm_sha()
        state['mm_h_sha256_end'] = end_sha
        state['env_changed_during_round'] = end_sha != state.get('mm_h_sha256_start')
        state.update({'status': 'done', 'finished': now().isoformat(), 'table': table, 'round_best': rb,
                      'delta': delta, 'improved': improved, 'pooled_se': pooled,
                      'rule': 'improved iff round best (D1.3, >= 3 seeds) - best before > round best seed SE'})
        jdump(state, os.path.join(rdir, 'round.json'))
        if improved:
            jdump({**rb, 'round': tag, 'set': now().isoformat()}, self.best_path)
        self.append_history({'round': tag, 'type': rnd, 'status': 'done', 'finished': state['finished'],
                             'improved': improved, 'delta': delta, 'round_best_mean': rb and rb['mean'],
                             'round_best_se': rb and rb['se'], 'best_before_mean': before and before.get('mean')})
        ledger.append({'kind': 'annotation', 'tag': tag, 'note': f'round {tag} finished: improved={improved}',
                       'round_best': rb and {k: rb[k] for k in ('cfg_hash', 'mean', 'se', 'n', 'd13')},
                       'best_before': before and {k: before.get(k) for k in ('cfg_hash', 'mean', 'se', 'n')},
                       'delta': delta}, self.ledger_path)
        print(json.dumps({'round': tag, 'improved': improved, 'delta': delta,
                          'round_best': rb and {k: rb[k] for k in ('variant', 'mean', 'se', 'n', 'fills_per_market', 't6_share')},
                          'best_before': before and {k: before.get(k) for k in ('mean', 'se', 'n')},
                          'table': [{k: t[k] for k in ('variant', 'mean', 'se', 'n', 'fills_per_market', 't6_share', 'd13')}
                                    for t in table]}, default=float), flush=True)
        return state

    def mm_sha(self):
        return None if self.standin else sha256(os.path.join(PM, 'mm.h'))

    def append_history(self, row):
        h = self.history()
        h = [r for r in h if not (r['round'] == row['round'] and r.get('status') == 'running')] + [row]
        jdump(h, self.hist_path)

    # -- commands
    def status(self):
        b = self.best()
        print(json.dumps({'best': b and {k: b.get(k) for k in ('cfg_hash', 'tag', 'round', 'mean', 'se', 'n',
                                                                'fills_per_market', 't6_share')},
                          'history': self.history(), **self.stop_status(),
                          'next': self.next_round()}, indent=1, default=float))

    def next_round(self):
        done = {r['type'] for r in self.history() if r.get('status') in ('done', 'skipped')}
        return next((r for r in ORDER if r not in done), None)

    def auto(self):
        while True:
            st = self.stop_status()
            if st['stop'] and not self.a.ignore_deadline:
                print(json.dumps({'auto': 'stop', **st}), flush=True)
                return
            rnd = self.next_round()
            if rnd is None:
                print(json.dumps({'auto': 'all rounds done'}), flush=True)
                return
            try:
                out = self.run_round(rnd)
            except SystemExit as e:
                out = None
                print(json.dumps({'round': rnd, 'skipped': str(e)}), flush=True)
            if out is None:
                self.append_history({'round': self.tag(rnd), 'type': rnd, 'status': 'skipped',
                                     'time': now().isoformat()})

    def finalize(self):
        a = self.a
        b = self.best()
        if b is None:
            raise SystemExit('no best config')
        out = os.path.join(self.root, 'final_seeds')
        sel = os.path.join(self.root, 'final_selected_source.json')
        jdump({'cfg': b['cfg'], 'selected': b}, sel)
        cmd = [PY, 'seeds.py', '--selected', sel, '--jobs', str(a.jobs), '--threads', '1', '--out', out,
               '--ledger', self.ledger_path, '--seeds', '100', '101', '102', '103', '104']
        if a.final_steps:
            cmd += ['--timesteps', str(a.final_steps)]
        self.sh(cmd, os.path.join(self.root, 'final_seeds.log'))
        summ = json.load(open(os.path.join(out, 'seeds.json')))
        runs = []
        for seed, r in sorted(summ['per_seed'].items(), key=lambda kv: int(kv[0])):
            if sha256(r['ckpt']) != r['ckpt_sha256']:
                raise SystemExit(f'checkpoint {r["ckpt"]} changed')
            runs.append({'seed': int(seed), 'ckpt': r['ckpt'], 'ckpt_sha256': r['ckpt_sha256'], 'cfg_hash': r['cfg_hash']})
        cfg = json.load(open(os.path.join(out, 'config.json')))['cfg']
        hashes = sorted({r['cfg_hash'] for r in runs})
        doc = {'selected': {'cfg': cfg, 'cfg_hash': hashes[0] if len(hashes) == 1 else hashes, 'runs': runs,
                            'n_seeds': len(runs)},
               'selection': {'source': 'rounds.py (DECISIONS.md D4)', 'best': {k: b.get(k) for k in
                             ('cfg_hash', 'tag', 'round', 'mean', 'se', 'n', 'fills_per_market', 't6_share')},
                             'history': self.history()},
               'final_seeds_val': summ['across_seeds']}
        dest = os.path.join(self.root, 'selected.json') if self.standin else os.path.join(HERE, 'selected.json')
        if os.path.exists(dest) and not self.standin:
            os.replace(dest, dest + f'.before_finalize_{time.strftime("%Y%m%d_%H%M%S")}')
        jdump(doc, dest)
        print(json.dumps({'published': dest, 'seeds': [r['seed'] for r in runs]}), flush=True)


def busy():
    """PIDs of other training pipelines (run_all / sweep / seeds / another rounds run)."""
    me = {os.getpid(), os.getppid()}
    out = subprocess.run(['pgrep', '-f', r'run_all\.py|sweep\.py|seeds\.py|rounds\.py (run|auto|finalize)'],
                         capture_output=True, text=True).stdout.split()
    pids = []
    for x in out:
        try:
            argv = open(f'/proc/{x}/cmdline', 'rb').read().split(b'\0')
        except OSError:
            continue
        if int(x) in me or b'python' not in argv[0] or b'--detach' in argv:
            continue  # shell wrappers, our own launcher
        pids.append(int(x))
    return pids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['status', 'run', 'auto', 'finalize'])
    p.add_argument('round', nargs='?', choices=ORDER)
    p.add_argument('--env', default='real', choices=['real', 'standin'])
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--steps', type=float, default=25e6,
                   help='steps per run in R1-R3, R5 (fine-tune), R6 (the main sweep\'s budget)')
    p.add_argument('--configs', type=int, default=16, help='R1: sampled configs')
    p.add_argument('--r7-configs', type=int, default=16, help='R7: sampled configs')
    p.add_argument('--r7-wait', type=int, default=90, help='auto: minutes to wait for reward_mode 2')
    p.add_argument('--max-configs', type=int, default=None, help='cap the variants of a round (smoke)')
    p.add_argument('--long-steps', type=float, default=200e6, help='R4 steps per run')
    p.add_argument('--pretrain-steps', type=float, nargs='*', default=[50e6, 100e6], help='R5 variants')
    p.add_argument('--bc-envs', type=int, default=1024)
    p.add_argument('--bc-steps', type=int, default=1024)
    p.add_argument('--final-steps', type=float, default=None, help='finalize: steps per seed (default: the best\'s)')
    p.add_argument('--force-sim', action='store_true', help='R5 despite a failed sim gate')
    p.add_argument('--deadline', default=DEADLINE)
    p.add_argument('--ignore-deadline', action='store_true')
    p.add_argument('--redo-compare', action='store_true')
    p.add_argument('--suffix', default='', help='appended to round tags (e.g. smoke)')
    p.add_argument('--plan-seed', type=int, default=0)
    p.add_argument('--nice', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--detach', action='store_true', help='run in the background (log under runs/rounds/)')
    p.add_argument('--after-main', action='store_true',
                   help='wait until no other pipeline (run_all/sweep/seeds/rounds) is running, then start')
    p.add_argument('--allow-concurrent', action='store_true', help='start even if another pipeline runs')
    a = p.parse_args()
    if a.mode == 'run' and not a.round:
        p.error('run needs a round (R1..R6)')
    if a.detach:
        argv = [x for x in sys.argv[1:] if x != '--detach']
        os.makedirs(ROOT, exist_ok=True)
        log = os.path.join(ROOT, f'{a.mode}_{a.round or ""}_{time.strftime("%Y%m%d_%H%M%S")}.log')
        proc = subprocess.Popen([PY, os.path.abspath(__file__), *argv], cwd=HERE, stdout=open(log, 'a'),
                                stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'detached_pid': proc.pid, 'log': log}))
        return
    R = Rounds(a)
    if a.mode != 'status' and not a.dry_run and not a.allow_concurrent:
        while busy():
            if not a.after_main:
                raise SystemExit(f'another training pipeline is running (pids {busy()}); '
                                 'use --after-main to wait for it or --allow-concurrent')
            time.sleep(60)
    if a.mode == 'status':
        R.status()
    elif a.mode == 'run':
        R.run_round(a.round)
    elif a.mode == 'auto':
        R.auto()
    else:
        R.finalize()


if __name__ == '__main__':
    main()
