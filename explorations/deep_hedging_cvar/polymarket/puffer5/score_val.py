"""Score PufferLib 5.0 checkpoints on the validation split with our Python evaluator.

    # sweep: re-score Protein's top_k runs and keep the best
    python score_val.py --pufferlib /content/PufferLib --sweep-log sweep.log --top-k 5 \
        --out ../results/puffer5
    # single runs: one or more checkpoints (hidden/layers from the run's log ini, or given)
    python score_val.py --checkpoint run.bin --run-ini logs/polymarket_mm/<run_id>.ini \
        --name puffer5 --out ../results/puffer5

Every checkpoint goes through puffernet_mm.EvaluatorPolicy (the torch replica
of 5.0's PufferNet, checked against puffercpu.c) into ../train/evaluator.py:
every validation market played once from the open, argmax of each head, the
adapter's headline variant (central fills, 1 s latency). Each evaluation is
appended to the trial ledger (../train/ledger.py, EVALUATION.md T4), as is
every sweep trial, since Protein's score is itself a validation evaluation:
the trainer's final evaluation, which [eval_env] puts on split 1 (mean P&L
per validation market, sampled actions). Sweep mode copies the run with the
highest validation total P&L to <out>/puffer5_sweep_best.bin with a JSON of
the run facts. The test split is never used: the adapters refuse split 2.

--env standin runs the plumbing on ../train/standin.py (no data needed); the
real adapter wraps ../env.py (PufferLib 3.0 binding of ../mm.h).
"""

import argparse
import configparser
import glob
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PM, 'train'))
from puffernet_mm import EvaluatorPolicy  # noqa: E402

ENV_NAME = 'polymarket_mm'
CONFIG_SECTIONS = ('env', 'policy', 'train', 'vec')
RUN_ONLY_ENV = ('data_dir', 'log_episodes')


def read_ini(path):
    ini = configparser.ConfigParser(strict=False, interpolation=None)
    ini.optionxform = str   # keep 'Q'
    with open(path) as f:
        ini.read_string(''.join(line for line in f if not line.startswith('#')))
    return ini


def num(v):
    try:
        f = float(v)
    except ValueError:
        return v.strip('"\'')
    return int(f) if f.is_integer() else f


def run_config(ini):
    """The training-relevant config of a 5.0 run log, and its seed."""
    cfg = {'framework': 'pufferlib5'}
    for sec in CONFIG_SECTIONS:
        if ini.has_section(sec):
            cfg[sec] = {k: num(v) for k, v in ini[sec].items()
                        if not (sec == 'env' and k in RUN_ONLY_ENV)}
    if 'vec' in cfg:
        cfg['vec'] = {'total_agents': cfg['vec'].get('total_agents')}
    seed = num(ini['base']['seed']) if ini.has_section('base') and 'seed' in ini['base'] else None
    return cfg, seed


def check_run_splits(ini, path):
    env = ini['env'] if ini.has_section('env') else {}
    ev = ini['eval_env'] if ini.has_section('eval_env') else {}
    if num(env.get('split', '0')) != 0 or num(ev.get('split', '-1')) != 1:
        raise SystemExit(f'{path}: expected [env] split = 0 and [eval_env] split = 1')


def sweep_runs(pufferlib, sweep_log):
    """Finished sweep runs: Protein's (validation) score, cost, steps, run ini and
    final checkpoint."""
    pat = re.compile(r'sweep run=(\d+) score=(\S+) cost=(\S+) steps=(\S+)')
    text = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', open(sweep_log, errors='replace').read())
    runs = []
    for m in pat.finditer(text):
        run, score, cost, steps = int(m[1]), float(m[2]), float(m[3]), float(m[4])
        inis = sorted(glob.glob(f'{pufferlib}/logs/{ENV_NAME}/sweep_*_{run:04d}.ini'),
                      key=os.path.getmtime)
        if not inis:
            continue
        run_id = os.path.splitext(os.path.basename(inis[-1]))[0]
        bins = sorted(glob.glob(f'{pufferlib}/checkpoints/{ENV_NAME}/{run_id}/*.bin'))
        ini = read_ini(inis[-1])
        check_run_splits(ini, inis[-1])
        runs.append(dict(run=run, run_id=run_id, protein_score=score, cost=cost, steps=steps,
                         ini=inis[-1], checkpoint=bins[-1] if bins else None,
                         hidden=int(num(ini['policy']['hidden_size'])),
                         layers=int(num(ini['policy']['num_layers'])),
                         finite=bool(np.isfinite(score))))
    return runs


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


class Ledger:
    """../train/ledger.py when importable (standin rows go to its standin file)."""

    def __init__(self, standin, fallback, path=None):
        try:
            import ledger
            self.mod = ledger
            self.path = path or (ledger.STANDIN_LEDGER if standin else ledger.LEDGER)
        except ImportError:
            self.mod, self.path = None, path or fallback

    def config_hash(self, cfg):
        if self.mod:
            return self.mod.config_hash(cfg)
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def append(self, row):
        if self.mod:
            return self.mod.append(row, self.path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(json.dumps(row, sort_keys=True, default=float) + '\n')
        return row


def score(path, hidden, layers, adapter, device, env_kw=None):
    from evaluator import evaluate
    policy = EvaluatorPolicy.from_bin(path, hidden, layers, device=device)
    if policy.obs_size != adapter.obs_size or tuple(policy.nvec) != tuple(adapter.nvec):
        raise SystemExit(f'{path}: obs {policy.obs_size} / heads {policy.nvec} vs env '
                         f'{adapter.obs_size} / {adapter.nvec}')
    return evaluate(policy, adapter, split=1, env_kw=env_kw, device=device)


HEADLINE = ('n_markets', 'pnl_total', 'pnl_mean', 'pnl_median', 'cvar95', 'frac_positive',
            'max_abs_inv')


def brief(res):
    return {k: res[k] for k in HEADLINE if k in res}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pufferlib', default='/content/PufferLib')
    p.add_argument('--sweep-log', help='stdout of ./puffer sweep (sweep mode)')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--checkpoint', nargs='*', default=[], help='single mode: .bin files')
    p.add_argument('--run-ini', nargs='*', default=[], help='their run log inis (hidden, layers)')
    p.add_argument('--name', nargs='*', default=[])
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--env', default='real', choices=['real', 'standin'])
    p.add_argument('--out', default=os.path.join(PM, 'results', 'puffer5'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--ledger', help='trial ledger path (default: ../train/ledger.py\'s)')
    args = p.parse_args()

    from envs import get_adapter
    adapter = get_adapter(args.env)
    ledger = Ledger(args.env == 'standin', os.path.join(args.out, 'trials.jsonl'), args.ledger)
    os.makedirs(args.out, exist_ok=True)
    common = {'source': 'puffer5', 'env': adapter.name, 'split': 1}

    if args.sweep_log:
        runs = sweep_runs(args.pufferlib, args.sweep_log)
        assert runs, 'no finished sweep runs in the log'
        for r in runs:   # each trial was scored on validation by the trainer
            cfg, seed = run_config(read_ini(r['ini']))
            ledger.append({**common, 'kind': 'sweep_trial', 'run_id': r['run_id'],
                           'config': cfg, 'config_hash': ledger.config_hash(cfg), 'seed': seed,
                           'val_trainer_score': r['protein_score'], 'steps': r['steps'],
                           'cost_s': r['cost']})
        ok = [r for r in runs if r['finite'] and r['checkpoint']]
        top = sorted(ok, key=lambda r: -r['protein_score'])[:args.top_k]
        for r in top:
            res = score(r['checkpoint'], r['hidden'], r['layers'], adapter, args.device)
            cfg, seed = run_config(read_ini(r['ini']))
            ledger.append({**common, 'kind': 'eval', 'run_id': r['run_id'],
                           'checkpoint': r['checkpoint'], 'checkpoint_sha256': sha256(r['checkpoint']),
                           'config': cfg, 'config_hash': ledger.config_hash(cfg), 'seed': seed,
                           'val': res})
            r['val'] = brief(res)
            print(json.dumps({k: r[k] for k in ('run', 'run_id', 'protein_score', 'hidden',
                                                'layers', 'steps', 'val')}), flush=True)
        best = max(top, key=lambda r: r['val']['pnl_total'])
        shutil.copy(best['checkpoint'], os.path.join(args.out, 'puffer5_sweep_best.bin'))
        meta = {'run_id': best['run_id'], 'hidden': best['hidden'], 'layers': best['layers'],
                'steps': best['steps'], 'protein_score': best['protein_score'], 'val': best['val'],
                'trials': len(runs), 'scored': top, 'selection': 'max validation pnl_total of '
                'the top_k runs by the trainer\'s validation score'}
        with open(os.path.join(args.out, 'puffer5_sweep_best.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=float)
        print(f"selected sweep run {best['run']} ({best['run_id']}): validation P&L "
              f"{best['val']['pnl_total']:.2f} over {best['val']['n_markets']} markets")
        return

    assert args.checkpoint, 'give --sweep-log or --checkpoint'
    names = args.name or [os.path.splitext(os.path.basename(c))[0] for c in args.checkpoint]
    out = {}
    for i, (ckpt, name) in enumerate(zip(args.checkpoint, names)):
        hidden, layers, cfg, seed = args.hidden, args.layers, {'framework': 'pufferlib5'}, None
        if i < len(args.run_ini):
            ini = read_ini(args.run_ini[i])
            check_run_splits(ini, args.run_ini[i])
            hidden, layers = int(num(ini['policy']['hidden_size'])), int(num(ini['policy']['num_layers']))
            cfg, seed = run_config(ini)
        res = score(ckpt, hidden, layers, adapter, args.device)
        ledger.append({**common, 'kind': 'eval', 'name': name, 'checkpoint': ckpt,
                       'checkpoint_sha256': sha256(ckpt), 'config': cfg,
                       'config_hash': ledger.config_hash(cfg), 'seed': seed, 'val': res})
        out[name] = {'checkpoint': ckpt, 'hidden': hidden, 'layers': layers, 'val': res}
        print(name, json.dumps(brief(res)), flush=True)
    with open(os.path.join(args.out, 'val_scores.json'), 'a') as f:
        f.write(json.dumps(out, default=float) + '\n')


if __name__ == '__main__':
    main()
