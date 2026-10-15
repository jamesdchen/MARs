"""Protocol v2 (DECISIONS.md D7) evaluations on v2 validation: seed ensembles, baselines,
per-contract-length tables and length filters.

    python v2eval.py ensembles            # R7 final seeds, main final seeds, R1 tuned-BC configs
    python v2eval.py baselines            # tuned + activity-constrained baselines
    python v2eval.py ckpts --name X --ckpts a.pt b.pt c.pt     # any set: each alone + ensemble
    python v2eval.py report               # tables -> OVERNIGHT_V2.md section

v2 validation = data/build_v2 split 1 (Aug 23 + Aug 24, 374 markets); the script refuses
to run before data/build_v2 exists (--data-dir to override for smoke tests, which must use
--split 0 and --no-ledger). Greedy actions, central fills, 1 s latency, every market once
from offset 0 (evaluator.py). The env parameters of each checkpoint's own training config
are kept (mark_mode and Q change the observation), with data_dir set to v2.

Length filters: 'quote only on these contract lengths' (off elsewhere). Each market is its own
env and recurrent state, so a filtered evaluation is exactly the unfiltered per-market records
with the excluded markets set to no-quote (evaluator.filter_records); each filter variant
looked at is still a val evaluation and gets its own ledger row (derived: length_filter).

Every evaluation appends one row to train/trials.jsonl (kind val_eval, split 1, protocol v2,
tag v2_*). Results: runs/v2/<group>.json.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pufferlib.pufferl  # noqa: E402,F401
warnings.filterwarnings('default', category=RuntimeWarning)

import ledger  # noqa: E402
from envs import get_adapter, no_jun15  # noqa: E402
from evaluator import evaluate, filter_records, nan_to_none, summarize  # noqa: E402
from policy import EnsemblePolicy, load_checkpoint  # noqa: E402

V2_DIR = os.path.join(PM, 'data', 'build_v2')
OUT = os.path.join(HERE, 'runs', 'v2')
HEADLINE_KW = {'fill_model': 1, 'latency': 1}   # central fills, 1 s (D3)
FILTERS = (('5m', '15m', '4h'), ('4h',), ('15m', '4h'), ('15m',), ('5m',), ('5m', '15m'))
BASELINES = ('no_quote', 'sym_touch', 'as_logit', 'fair_value', 'fv_smart')
KEYS = ('n_markets', 'pnl_total', 'pnl_mean', 'cvar95', 'fills_per_market', 't6_share', 't6_share_pos',
        't5_max_share', 'makes_markets', 'mean/uptime_two_sided', 'sum/spread_capture', 'sum/markout_60',
        'sum/inventory_pnl', 'sum/rebates', 'len/5m/pnl_total', 'len/15m/pnl_total', 'len/4h/pnl_total',
        'len/5m/fills_per_market', 'len/15m/fills_per_market', 'len/4h/fills_per_market')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


class BaselinePolicy(torch.nn.Module):
    """A baselines.py quoter in the evaluator's policy contract (one-hot logits)."""
    recurrent = False

    def __init__(self, fn, nvec=(7, 7, 3)):
        super().__init__()
        self.fn, self.nvec = fn, tuple(nvec)

    def forward_eval(self, obs, state=None):
        a = torch.as_tensor(np.asarray(self.fn(obs.numpy()), dtype=np.int64))
        logits = tuple(torch.nn.functional.one_hot(a[:, h], n).float() * 20.0 for h, n in enumerate(self.nvec))
        return logits, torch.zeros(obs.shape[0], 1)


class V2:
    def __init__(self, a):
        self.a = a
        self.data_dir = a.data_dir or V2_DIR
        no_jun15(self.data_dir)
        if not os.path.exists(os.path.join(self.data_dir, 'markets.bin')):
            raise SystemExit(f'{self.data_dir} does not exist yet (v2 validation data); waiting for it')
        if a.split != 1 and not a.no_ledger:
            raise SystemExit('only split 1 (v2 val) is ledgered; use --no-ledger for smoke tests')
        self.adapter = get_adapter('real')
        self.out = a.out or OUT
        os.makedirs(self.out, exist_ok=True)

    def ledger_row(self, tag, name, spec, res, derived=None):
        if self.a.no_ledger:
            return
        ledger.append({'kind': 'val_eval', 'split': 1, 'protocol': 'v2', 'source': 'train/v2eval.py',
                       'backend': 'real', 'tag': tag, 'name': name, 'data_dir': os.path.abspath(self.data_dir),
                       'variant': 'central_1s', 'policy_spec': spec, 'derived': derived,
                       'config_hash': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                       'val': nan_to_none(res)}, ledger.LEDGER)

    def run(self, policy, env_kw, tag, name, spec, filters=True):
        kw = {**env_kw, **HEADLINE_KW, 'data_dir': self.data_dir}
        t0 = time.time()
        res, recs = evaluate(policy, self.adapter, split=self.a.split, env_kw=kw, return_records=True)
        self.ledger_row(tag, name, spec, res)
        out = {'name': name, 'spec': spec, 'all': {k: res.get(k) for k in KEYS}, 'seconds': time.time() - t0}
        if filters:
            out['filters'] = {}
            for f in FILTERS[1:]:
                r = summarize(filter_records(recs, f), group_key='contract_len_s')
                self.ledger_row(tag, f'{name}|quote:{"+".join(f)}', {**spec, 'quote_lengths': list(f)}, r,
                                derived='length_filter')
                out['filters']['+'.join(f)] = {k: r.get(k) for k in KEYS}
        print(json.dumps({'name': name, **{k: out['all'][k] for k in ('pnl_total', 'fills_per_market',
                                                                       't6_share', 'makes_markets')}},
                         default=float), flush=True)
        return out

    def ckpt_set(self, group, name, paths, tag):
        """Each checkpoint alone, then the logits ensemble (and the probability ensemble)."""
        path = os.path.join(self.out, f'{group}.json')
        done = json.load(open(path)) if os.path.exists(path) else {}
        mems, kw = [], None
        for p in paths:
            pol, ck = load_checkpoint(p)
            mems.append(pol)
            kw = {k: v for k, v in ck['cfg']['env']['kw'].items() if v is not None and k != 'data_dir'}
        results = done.get(name, {})
        specs = {os.path.basename(os.path.dirname(p)): {'checkpoints': [p], 'sha256': [sha256(p)]} for p in paths}
        for (sname, spec), pol in zip(specs.items(), mems):
            if sname not in results:
                results[sname] = self.run(pol, kw, tag, f'{name}/{sname}', spec)
        for comb in self.a.combine:
            ename = f'ensemble_{comb}_{len(paths)}'
            if ename not in results and len(paths) > 1:
                spec = {'checkpoints': list(paths), 'sha256': [sha256(p) for p in paths], 'combine': comb}
                results[ename] = self.run(EnsemblePolicy(mems, comb), kw, tag, f'{name}/{ename}', spec)
        done[name] = results
        json.dump(done, open(path, 'w'), indent=1, default=float)
        return results

    def ensembles(self):
        sel = json.load(open(os.path.join(HERE, 'selected.json')))
        self.ckpt_set('ensembles', 'r7_final', [r['ckpt'] for r in sel['selected']['runs']], 'v2_ensemble')
        main = sorted(os.path.join(HERE, 'runs', 'seeds', 'main', d, 'final.pt')
                      for d in os.listdir(os.path.join(HERE, 'runs', 'seeds', 'main')) if d.startswith('seed_'))
        main = [p for p in main if os.path.exists(p)]
        if main:
            self.ckpt_set('ensembles', 'main_final', main, 'v2_ensemble')
        r1 = os.path.join(HERE, 'runs', 'rounds', 'R1_bc_warmstart')
        cfgs = sorted({d.rsplit('_s', 1)[0] for d in os.listdir(r1)
                       if '_tuned_s' in d and not d.endswith('.log') and 'killed' not in d})
        for c in cfgs:
            ps = [os.path.join(r1, f'{c}_s{s}', 'final.pt') for s in range(3)]
            ps = [p for p in ps if os.path.exists(p)]
            if len(ps) == 3:
                self.ckpt_set('ensembles', f'r1_{c}', ps, 'v2_ensemble')

    def baselines(self):
        sys.path.insert(1, os.path.join(PM, 'baselines'))
        import baselines as bl
        from obs_spec import load_spec
        spec = load_spec(env_params={'Q': 100.0})
        path = os.path.join(self.out, 'baselines.json')
        done = json.load(open(path)) if os.path.exists(path) else {}
        for src in ('tuned', 'tuned_active'):
            tuned = json.load(open(os.path.join(PM, 'baselines', f'{src}.json')))
            for name in BASELINES:
                if name not in tuned and name != 'no_quote':
                    continue
                key = f'{name}' + ('' if src == 'tuned' else ' (active)')
                if key in done or (name == 'no_quote' and src != 'tuned'):
                    continue
                params = tuned.get(name, {}).get('best_params', {})
                pol = BaselinePolicy(bl.make(name, spec, params))
                done[key] = self.run(pol, {}, 'v2_baselines', key,
                                     {'baseline': name, 'source': f'baselines/{src}.json', 'params': params})
                json.dump(done, open(path, 'w'), indent=1, default=float)


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '-'
    return f'{x:+.{d}f}' if isinstance(x, float) else str(x)


def table(rows):
    hdr = ['policy', 'P&L', '/market', 'CVaR95', 'fills/mkt', 'T6', 'D1.3', 'P&L 5m', 'P&L 15m', 'P&L 4h']
    out = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for name, m in rows:
        out.append('| ' + ' | '.join([name, fmt(m['pnl_total']), fmt(m['pnl_mean'], 2), fmt(m['cvar95']),
                                      fmt(m['fills_per_market']), fmt(m['t6_share'], 2),
                                      'yes' if m.get('makes_markets') else 'no', fmt(m.get('len/5m/pnl_total')),
                                      fmt(m.get('len/15m/pnl_total')), fmt(m.get('len/4h/pnl_total'))]) + ' |')
    return '\n'.join(out)


def report(out=OUT):
    parts = []
    for group in ('ensembles', 'baselines'):
        p = os.path.join(out, f'{group}.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if group == 'ensembles':
            for name, res in d.items():
                parts.append(f'### {name}\n\n' + table([(k, v['all']) for k, v in res.items()]))
                ens = [k for k in res if k.startswith('ensemble')]
                if ens:
                    e = res[ens[0]]
                    parts.append(f'Length filters ({name}, {ens[0]}):\n\n' + table(
                        [('all lengths', e['all'])] + [(f'quote {k}', v) for k, v in e['filters'].items()]))
        else:
            parts.append('### baselines\n\n' + table([(k, v['all']) for k, v in d.items()]))
            parts.append('Baselines, length filters (best filter per baseline):\n\n' + table(
                [(f'{k} | quote {f}', m) for k, v in d.items() for f, m in
                 [max(v['filters'].items(), key=lambda kv: kv[1]['pnl_total'])]]))
    return '\n\n'.join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['ensembles', 'baselines', 'ckpts', 'report'])
    p.add_argument('--name', default=None)
    p.add_argument('--ckpts', nargs='*', default=[])
    p.add_argument('--tag', default='v2_ensemble')
    p.add_argument('--combine', nargs='*', default=['logits', 'probs'])
    p.add_argument('--data-dir', default=None)
    p.add_argument('--split', type=int, default=1)
    p.add_argument('--no-ledger', action='store_true')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', default=None, help='results dir (default runs/v2)')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.mode == 'report':
        print(report(a.out or OUT))
        return
    v = V2(a)
    if a.mode == 'ensembles':
        v.ensembles()
    elif a.mode == 'baselines':
        v.baselines()
    else:
        v.ckpt_set(a.name or 'custom', a.name or 'custom', a.ckpts, a.tag)


if __name__ == '__main__':
    main()
