"""Protocol v2 search (DECISIONS.md D7): reward_mode 2 configs trained on train, scored on v2 val.

    python v2search.py run --detach          # 32 configs x 3 seeds (resumable)
    python v2search.py status                # progress + current table
    python v2search.py select                # seed-mean selection -> runs/v2/<name>_selected.json

Base: the R7 final config (train/selected.json: MLP with off prior, reward_mode 2, fair-value
mark). Each config samples (stratified for the categorical ones, fixed seed):
  markout_h {60, 120, 300}; inv_penalty log-uniform [1e-6, 1e-2]; lr log [1e-4, 3e-3];
  ent_coef log [1e-4, 3e-2]; hidden {64, 128}; close-inventory penalty (train_ppo.py
  CloseInventoryPenalty, CVaR-flavoured): off for half the configs, else coef log [1e-5, 3e-3]
  over the last 20% of the window; training markets len_mask {7 all, 4 4h only, 2 15m only}.
Every run trains --steps (25M) with data_dir=data/build_v2 (its split 0 = the same train
markets, split 1 = v2 val: Aug 23 + Aug 24) and is scored greedily on every v2 val market
(central and pessimistic fills, 1 s); its val evaluations go to train/trials.jsonl (tag
<name>), its summary to runs.jsonl. Length filters (quote only on some contract lengths) are
derived from each run's per-market records (evaluator.filter_records) and each one looked
at gets its own ledger row (tag <name>_filter).

Selection: the (config, filter) pair with the best seed mean of v2 val P&L over 3 seeds,
among pairs passing D1.3 (fills/market >= 5, T6 <= 0.5) on the seed means. The table
also shows the seed SE, the pessimistic P&L and P&L per length.
"""

import argparse
import concurrent.futures as cf
import copy
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
import evaluator  # noqa: E402,F401  (imports pufferlib.pufferl, which sets a SIGINT handler: main thread only)
from rounds import jdump, jload, sha256  # noqa: E402

V2_DIR = os.path.join(PM, 'data', 'build_v2')
FILTERS = (('5m', '15m', '4h'), ('4h',), ('15m', '4h'), ('15m',), ('5m',), ('5m', '15m'))
LEN_OF_MASK = {7: ('5m', '15m', '4h'), 4: ('4h',), 2: ('15m',)}


def sample_configs(base, n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        c = copy.deepcopy(base)
        c['init'], c['kl_teacher'] = None, None
        kw = c['env']['kw']
        kw.update(reward_mode=2, markout_h=(60, 120, 300)[i % 3],
                  inv_penalty=float(10 ** rng.uniform(-6, -2)),
                  len_mask=(7, 4, 2)[(i // 3) % 3])
        c['ppo']['learning_rate'] = float(10 ** rng.uniform(-4, math.log10(3e-3)))
        c['ppo']['ent_coef'] = float(10 ** rng.uniform(-4, math.log10(3e-2)))
        c['policy']['hidden'] = (64, 128)[(i // 9) % 2]
        if i % 2:
            c['shaping'] = {'close_inv_coef': float(10 ** rng.uniform(-5, math.log10(3e-3))), 'ramp': 0.2}
        else:
            c.pop('shaping', None)
        out.append({'name': f'c{i:02d}', 'cfg': c})
    return out


class Search:
    def __init__(self, a):
        self.a = a
        self.vroot = a.out_root or os.path.join(HERE, 'runs', 'v2')
        self.root = os.path.join(self.vroot, a.name)
        self.data_dir = a.data_dir or V2_DIR
        self.ledger = a.ledger or ledger.LEDGER

    def base(self):
        sel = json.load(open(os.path.join(HERE, 'selected.json')))
        b = copy.deepcopy(sel['selected']['cfg'])
        b['total_timesteps'] = int(self.a.steps)
        b['env']['kw']['data_dir'] = os.path.abspath(self.data_dir)
        return b

    def plan(self):
        p = os.path.join(self.root, 'plan.json')
        plan = jload(p)
        if plan is None:
            plan = {'name': self.a.name, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'configs': sample_configs(self.base(), self.a.configs, self.a.plan_seed),
                    'seeds': list(range(self.a.seeds))}
            jdump(plan, p)
        return plan

    def train(self, v, seed):
        rdir = os.path.join(self.root, f'{v["name"]}_s{seed}')
        res = os.path.join(rdir, 'result.json')
        if os.path.exists(res):
            return json.load(open(res))
        from sweep import set_aside
        set_aside(rdir)
        os.makedirs(rdir, exist_ok=True)
        cpath = os.path.join(rdir, 'v2_config.json')
        jdump({'cfg': v['cfg']}, cpath)
        cmd = (['nice', '-n', str(self.a.nice)] if self.a.nice else []) + [
            PY, 'train_ppo.py', '--config', cpath, '--seed', str(seed), '--threads', '1', '--out', rdir,
            '--tag', self.a.name, '--quiet', '--variants', 'central_1s', 'pess_1s', '--ledger', self.ledger]
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE', 'PYTHONWARNINGS': 'ignore'}
        with open(rdir + '.log', 'a') as f:
            rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0 or not os.path.exists(res):
            raise RuntimeError(f'{rdir}: exit {rc}')
        r = json.load(open(res))
        self.derive_filters(r)
        return r

    def derive_filters(self, r):
        """Ledger the length-filter variants of one finished run (once)."""
        from evaluator import filter_records, nan_to_none, summarize
        mark = os.path.join(r['run_dir'], 'filters.json')
        if os.path.exists(mark):
            return json.load(open(mark))
        recs = [json.loads(l) for l in open(os.path.join(r['run_dir'], 'val_markets.jsonl'))]
        out = {}
        for f in FILTERS:
            s = summarize(filter_records(recs, f), group_key='contract_len_s')
            out['+'.join(f)] = nan_to_none(s)
            if f != FILTERS[0]:
                ledger.append({'kind': 'val_eval', 'split': 1, 'protocol': 'v2', 'source': 'train/v2search.py',
                               'backend': 'real', 'tag': self.a.name + '_filter', 'cfg_hash': r['cfg_hash'],
                               'seed': r['seed'], 'variant': 'central_1s', 'derived': 'length_filter',
                               'quote_lengths': list(f), 'run_dir': r['run_dir'], 'data_dir': self.data_dir,
                               'val': out['+'.join(f)]}, self.ledger)
        jdump(out, mark)
        return out

    def run(self):
        if not os.path.exists(os.path.join(self.data_dir, 'markets.bin')):
            raise SystemExit(f'{self.data_dir} missing (v2 val data)')
        self.check_train_markets()
        plan = self.plan()
        tasks = [(v, s) for v in plan['configs'] for s in plan['seeds']]
        with cf.ThreadPoolExecutor(self.a.jobs) as ex:
            futs = {ex.submit(self.train, v, s): (v['name'], s) for v, s in tasks}
            for f in cf.as_completed(futs):
                name, s = futs[f]
                try:
                    r = f.result()
                    hv = r['val'][r['headline']]
                    print(json.dumps({'run': f'{name}_s{s}', 'pnl': hv.get('pnl_total'),
                                      'fills': hv.get('fills_per_market')}, default=float), flush=True)
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({'run': f'{name}_s{s}', 'failed': str(e)}), flush=True)
                self.write_progress()
        return self.select()

    def check_train_markets(self):
        """data/build_v2's split 0 must be exactly data/build's train markets."""
        from envs import MARKET_DTYPE
        a = np.fromfile(os.path.join(PM, 'data', 'build', 'markets.bin'), MARKET_DTYPE)
        b = np.fromfile(os.path.join(self.data_dir, 'markets.bin'), MARKET_DTYPE)
        if not np.array_equal(a[a['split'] == 0], b[b['split'] == 0]):
            raise SystemExit(f'{self.data_dir}: split 0 differs from data/build train markets')
        print(json.dumps({'v2_val_markets': int((b['split'] == 1).sum()), 'train_markets': int((b['split'] == 0).sum())}),
              flush=True)

    def table(self):
        plan = self.plan()
        rows = []
        for v in plan['configs']:
            runs = []
            for s in plan['seeds']:
                rd = os.path.join(self.root, f'{v["name"]}_s{s}')
                if os.path.exists(os.path.join(rd, 'result.json')):
                    r = json.load(open(os.path.join(rd, 'result.json')))
                    runs.append((r, self.derive_filters(r)))
            if not runs:
                continue
            for f in FILTERS:
                key = '+'.join(f)
                ms = [fl[key] for _, fl in runs]
                pnl = [m['pnl_total'] for m in ms]
                n = len(pnl)
                fills = float(np.mean([m.get('fills_per_market') or 0 for m in ms]))
                t6s = [m.get('t6_share') for m in ms]
                t6 = float(np.mean([0.0 if (t is None or t != t) else t for t in t6s]))
                pess = [r['val'].get('pess_1s', {}).get('pnl_total') for r, _ in runs] if key == '5m+15m+4h' else []
                rows.append({'config': v['name'], 'filter': key, 'n': n, 'mean': float(np.mean(pnl)),
                             'se': float(np.std(pnl, ddof=1) / math.sqrt(n)) if n > 1 else float('nan'),
                             'per_seed': pnl, 'fills': fills, 't6': t6, 'd13': fills >= 5 and t6 <= 0.5,
                             'pess_mean': float(np.mean(pess)) if pess and None not in pess else None,
                             'len_mask': v['cfg']['env']['kw'].get('len_mask'),
                             'markout_h': v['cfg']['env']['kw'].get('markout_h'),
                             'hidden': v['cfg']['policy'].get('hidden'),
                             'close': (v['cfg'].get('shaping') or {}).get('close_inv_coef'),
                             'lr': v['cfg']['ppo']['learning_rate'], 'ent': v['cfg']['ppo']['ent_coef'],
                             'inv_penalty': v['cfg']['env']['kw'].get('inv_penalty'),
                             **{f'pnl_{L}': float(np.mean([m.get(f'len/{L}/pnl_total') or 0 for m in ms]))
                                for L in ('5m', '15m', '4h')},
                             'runs': [{'seed': r['seed'], 'ckpt': r['ckpt'], 'ckpt_sha256': r['ckpt_sha256']}
                                      for r, _ in runs]})
        return rows

    def select(self):
        rows = self.table()
        ok = [r for r in rows if r['n'] >= 3 and r['d13']]
        best = max(ok, key=lambda r: r['mean']) if ok else None
        best_all = max([r for r in rows if r['filter'] == '5m+15m+4h' and r['n'] >= 3 and r['d13']],
                       key=lambda r: r['mean'], default=None)
        plan = self.plan()
        doc = {'name': self.a.name, 'rule': 'max seed mean (>= 3 seeds) of v2 val P&L among D1.3-passing '
                                            '(config, length filter) pairs', 'best': best,
               'best_unfiltered': best_all, 'n_rows': len(rows),
               'best_cfg': best and next(v['cfg'] for v in plan['configs'] if v['name'] == best['config']),
               'n_val_evals_ledger': ledger.n_val_evals(self.ledger)}
        jdump(doc, os.path.join(self.vroot, f'{self.a.name}_selected.json'))
        print(json.dumps({'best': best and {k: best[k] for k in ('config', 'filter', 'mean', 'se', 'fills', 't6')},
                          'best_unfiltered': best_all and {k: best_all[k] for k in ('config', 'mean', 'se')}},
                         default=float), flush=True)
        return doc

    def ensembles(self):
        """3-seed ensembles (logits and probabilities) of the top configs: the --top best by
        unfiltered seed mean, the --top best D1.3-passing (config, filter) pairs' configs, and
        the --top best by any filter; scored on v2 val with every length filter (v2eval.py,
        one ledger row each, tag <name>_ens). Results: runs/v2/<name>_ens.json."""
        rows = [r for r in self.table() if r['n'] >= 3]
        pick = []
        for key in (lambda r: r['filter'] == '5m+15m+4h', lambda r: r['d13'], lambda r: True):
            for r in sorted(filter(key, rows), key=lambda r: -r['mean'])[:self.a.top]:
                if r['config'] not in pick:
                    pick.append(r['config'])
        cmd = [PY, 'v2eval.py', 'ckpts', '--out', os.path.join(self.vroot, f'{self.a.name}_ens'),
               '--tag', f'{self.a.name}_ens', '--ensemble-only', '--ledger', self.ledger]
        if self.a.data_dir:
            cmd += ['--data-dir', self.a.data_dir]
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE', 'PYTHONWARNINGS': 'ignore'}
        def one(c):
            ps = [os.path.join(self.root, f'{c}_s{s}', 'final.pt') for s in self.plan()['seeds']]
            log = os.path.join(self.vroot, f'{self.a.name}_ens_{c}.log')
            with open(log, 'a') as f:
                subprocess.run(cmd + ['--name', c, '--ckpts', *ps], cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                               env=env, check=True)
            return c
        with cf.ThreadPoolExecutor(self.a.jobs) as ex:
            for c in ex.map(one, pick):
                print(json.dumps({'ensembled': c}), flush=True)
        out = {}
        d = os.path.join(self.vroot, f'{self.a.name}_ens')
        for c in pick:
            res = jload(os.path.join(d, f'{c}.json'), {}).get(c, {})
            out[c] = res
        jdump(out, os.path.join(self.vroot, f'{self.a.name}_ens.json'))
        return out

    def write_progress(self):
        rows = self.table()
        plan = self.plan()
        done = sum(os.path.exists(os.path.join(self.root, f'{v["name"]}_s{s}', 'result.json'))
                   for v in plan['configs'] for s in plan['seeds'])
        top = sorted([r for r in rows if r['n'] >= 1], key=lambda r: -r['mean'])[:15]
        lines = [f'## v2 search `{self.a.name}`: {done}/{len(plan["configs"]) * len(plan["seeds"])} runs done '
                 f'({time.strftime("%H:%M UTC", time.gmtime())})', '',
                 '| config | filter | seeds | mean P&L | SE | fills/mkt | T6 | D1.3 | 5m | 15m | 4h | len_mask | H_mo | hidden | close | lr | ent |',
                 '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in top:
            lines.append('| ' + ' | '.join(str(x) for x in (
                r['config'], r['filter'], r['n'], f'{r["mean"]:+.1f}', f'{r["se"]:.1f}' if r['se'] == r['se'] else '-',
                f'{r["fills"]:.1f}', f'{r["t6"]:.2f}', 'yes' if r['d13'] else 'no', f'{r["pnl_5m"]:+.0f}',
                f'{r["pnl_15m"]:+.0f}', f'{r["pnl_4h"]:+.0f}', r['len_mask'], r['markout_h'], r['hidden'],
                f'{r["close"]:.1e}' if r['close'] else '-', f'{r["lr"]:.1e}', f'{r["ent"]:.1e}')) + ' |')
        jdump({'done': done, 'top': top}, os.path.join(self.root, 'progress.json'))
        with open(os.path.join(self.root, 'progress.md'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def status(self):
        self.write_progress()
        print(open(os.path.join(self.root, 'progress.md')).read())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['run', 'status', 'select', 'ensembles'])
    p.add_argument('--top', type=int, default=4, help='ensembles: configs per ranking')
    p.add_argument('--name', default='v2s1')
    p.add_argument('--configs', type=int, default=32)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--steps', type=float, default=25e6)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--nice', type=int, default=0)
    p.add_argument('--plan-seed', type=int, default=7)
    p.add_argument('--data-dir', default=None)
    p.add_argument('--ledger', default=None, help='ledger (default train/trials.jsonl; smoke tests: elsewhere)')
    p.add_argument('--out-root', default=None, help='smoke tests: results root instead of runs/v2')
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    if a.detach:
        argv = [x for x in sys.argv[1:] if x != '--detach']
        os.makedirs(os.path.join(HERE, 'runs', 'v2'), exist_ok=True)
        log = os.path.join(HERE, 'runs', 'v2', f'{a.name}_{a.mode}.log')
        proc = subprocess.Popen([PY, os.path.abspath(__file__), *argv], cwd=HERE, stdout=open(log, 'a'),
                                stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'detached_pid': proc.pid, 'log': log}))
        return
    s = Search(a)
    if a.mode == 'run':
        s.run()
    elif a.mode == 'status':
        s.status()
    elif a.mode == 'ensembles':
        s.ensembles()
    else:
        s.select()


if __name__ == '__main__':
    main()
