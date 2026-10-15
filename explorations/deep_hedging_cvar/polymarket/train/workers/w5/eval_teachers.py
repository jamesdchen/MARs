"""w5: evaluate the BC teachers (baseline quoters) and BC students on val v2 (split 1),
central fills, 1 s latency. One ledger row per evaluation in w5/trials.jsonl.

    python eval_teachers.py teachers
    python eval_teachers.py ckpt --name bc_x path.pt
"""
import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
import hashlib, json, sys, time
import numpy as np
import torch

W5 = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.abspath(os.path.join(W5, '..', '..'))
PM = os.path.dirname(TRAIN)
sys.path.insert(0, TRAIN)
import pufferlib.pufferl  # noqa
import ledger
from envs import get_adapter, no_jun15
from evaluator import evaluate, nan_to_none
from policy import load_checkpoint
from v2eval import BaselinePolicy, KEYS

DATA = os.path.join(PM, 'data', 'build_v2')
LEDGER = os.path.join(W5, 'trials.jsonl')
OUT = os.path.join(W5, 'teachers.json')
TEACHERS = [('as_logit', 'tuned_active'), ('fair_value', 'tuned'), ('fair_value', 'tuned_active'),
            ('as_logit', 'tuned'), ('no_quote', 'tuned')]


def run(adapter, pol, env_kw, tag, name, spec):
    no_jun15(DATA)
    kw = {**env_kw, 'fill_model': 1, 'latency': 1, 'data_dir': DATA}
    t0 = time.time()
    res = evaluate(pol, adapter, split=1, env_kw=kw)
    assert res['n_markets'] == 374, res['n_markets']
    ledger.append({'kind': 'val_eval', 'split': 1, 'protocol': 'v2', 'source': 'workers/w5/eval_teachers.py',
                   'backend': 'real', 'tag': tag, 'name': name, 'data_dir': DATA, 'variant': 'central_1s',
                   'policy_spec': spec,
                   'config_hash': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                   'val': nan_to_none(res)}, LEDGER)
    out = {k: res.get(k) for k in KEYS}
    out['seconds'] = time.time() - t0
    print(json.dumps({'name': name, **{k: out[k] for k in ('pnl_total', 'pnl_mean', 'fills_per_market',
          't6_share', 'makes_markets', 'len/5m/pnl_total', 'len/15m/pnl_total', 'len/4h/pnl_total')}},
          default=float), flush=True)
    return out


def main():
    torch.set_num_threads(1)
    adapter = get_adapter('real')
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if sys.argv[1] == 'teachers':
        sys.path.insert(1, os.path.join(PM, 'baselines'))
        import baselines as bl
        from obs_spec import load_spec
        spec = load_spec(env_params={'Q': 100.0})
        for name, src in TEACHERS:
            key = f'{name} ({src})'
            if key in done:
                continue
            params = {} if name == 'no_quote' else json.load(open(os.path.join(PM, 'baselines', f'{src}.json')))[name]['best_params']
            pol = BaselinePolicy(bl.make(name, spec, params))
            done[key] = run(adapter, pol, {}, 'w5_teachers', key,
                            {'baseline': name, 'source': f'baselines/{src}.json', 'params': params})
            json.dump(done, open(OUT, 'w'), indent=1, default=float)
    elif sys.argv[1] == 'ckpt':
        name, path = sys.argv[3], sys.argv[4]
        pol, ck = load_checkpoint(path)
        kw = {k: v for k, v in (ck.get('cfg', {}).get('env', {}).get('kw', {}) or {}).items()
              if v is not None and k != 'data_dir'}
        h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        done[name] = run(adapter, pol, kw, 'w5_bc_students', name, {'checkpoint': os.path.relpath(path, PM), 'sha256': h})
        json.dump(done, open(OUT, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
