"""Train the market maker with PufferLib 3.0's PPO on CPU (train split), with
periodic deterministic evaluation on every validation market.

    python train_ppo.py --env standin --timesteps 20e6 --out runs/smoke
    python train_ppo.py --config runs/sweep_x/selected.json --seed 3

Flow: build the train env (split 0, random start offsets) and the policy
(policy.py; or --init from a bc.py / train_ppo.py checkpoint), run
puffer3.Trainer (PuffeRL 3.0) for --timesteps agent steps, and every
--eval-every steps score the greedy policy on all validation markets (split 1,
start offset 0, headline fill/latency variant) and on all train markets (for the
train-vs-val gap). At the end: save final.pt, score it under every variant in
--variants, write result.json and append one row to the ledger
(train/trials.jsonl; the stand-in env writes runs/standin_trials.jsonl).
Selection uses the final checkpoint only; the periodic evaluations are
monitoring, counted in the row's n_val_evals.

Run directory (--out): config.json, log.jsonl (one 'train' row per epoch and an
'eval' row per evaluation, the latter with the averaged training diagnostics
since the previous evaluation), ckpt_<step>.pt, final.pt, val_markets.jsonl
(per-market records of the final headline evaluation), result.json.
"""

import os
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')  # shared CPUs: no spin-waiting threads (bench.py)

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pufferlib.pufferl  # noqa: E402,F401  (sets warnings to raise on RuntimeWarning)
warnings.filterwarnings('default', category=RuntimeWarning)

import ledger  # noqa: E402
from envs import get_adapter  # noqa: E402
from evaluator import evaluate, nan_to_none  # noqa: E402
from policy import load_checkpoint, make_policy  # noqa: E402
from puffer3 import Trainer, ppo_config  # noqa: E402

DEFAULT_CFG = {
    'env': {'backend': 'auto', 'kw': {}},
    'policy': {'kind': 'mlp', 'hidden': 128, 'layers': 2, 'act': 'tanh', 'shared': True},
    'ppo': {'learning_rate': 3e-4, 'ent_coef': 0.01, 'gamma': 0.995, 'gae_lambda': 0.95,
            'clip_coef': 0.2, 'vf_coef': 0.5, 'vf_clip_coef': 10.0, 'max_grad_norm': 0.5,
            'update_epochs': 1, 'bptt_horizon': 64, 'minibatch_size': 8192,
            'optimizer': 'adam', 'anneal_lr': True, 'adam_eps': 1e-8},
    'num_envs': 4096,
    'total_timesteps': 50_000_000,
    'init': None,
    'kl_teacher': None,
}

# CLI flag -> (config path, type)
FLAGS = {
    'timesteps': (('total_timesteps',), float),
    'num_envs': (('num_envs',), int),
    'policy': (('policy', 'kind'), str),
    'hidden': (('policy', 'hidden'), int),
    'layers': (('policy', 'layers'), int),
    'act': (('policy', 'act'), str),
    'lr': (('ppo', 'learning_rate'), float),
    'ent_coef': (('ppo', 'ent_coef'), float),
    'gamma': (('ppo', 'gamma'), float),
    'gae_lambda': (('ppo', 'gae_lambda'), float),
    'clip_coef': (('ppo', 'clip_coef'), float),
    'vf_coef': (('ppo', 'vf_coef'), float),
    'vf_clip_coef': (('ppo', 'vf_clip_coef'), float),
    'max_grad_norm': (('ppo', 'max_grad_norm'), float),
    'update_epochs': (('ppo', 'update_epochs'), int),
    'horizon': (('ppo', 'bptt_horizon'), int),
    'minibatch_size': (('ppo', 'minibatch_size'), int),
    'optimizer': (('ppo', 'optimizer'), str),
    # env kwargs passed through to the env (None = env default)
    'inv_penalty': (('env', 'kw', 'inv_penalty'), float),
    'reward_scale': (('env', 'kw', 'reward_scale'), float),
    'fill_model': (('env', 'kw', 'fill_model'), int),
    'latency': (('env', 'kw', 'latency'), float),
    'dt': (('env', 'kw', 'dt'), float),
    'Q': (('env', 'kw', 'Q'), float),
    'mark_mode': (('env', 'kw', 'mark_mode'), int),
}


def set_path(d, path, v):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = v


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_cfg(args):
    cfg = copy.deepcopy(DEFAULT_CFG)
    if args.config:
        loaded = json.load(open(args.config))
        loaded = loaded.get('cfg', loaded)  # selected.json / result.json wrap it
        for k, v in loaded.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    if args.env is not None:
        cfg['env']['backend'] = args.env
    for flag, (path, typ) in FLAGS.items():
        v = getattr(args, flag)
        if v is not None:
            set_path(cfg, path, typ(v))
    if args.shared is not None:
        cfg['policy']['shared'] = args.shared
    if cfg['policy']['kind'] == 'puffernet':
        # PufferNet has no activation choice or separate critic; default to the
        # 5.0 ini's 4 layers (puffer5/polymarket_mm.ini) unless --layers is given
        cfg['policy'] = {'kind': 'puffernet', 'hidden': cfg['policy']['hidden'],
                         'layers': args.layers if args.layers is not None else
                         (cfg['policy']['layers'] if args.config else 4)}
    cfg['env']['kw'].update(parse_kv(args.env_kw))
    cfg['total_timesteps'] = int(cfg['total_timesteps'])
    if args.init:
        cfg['init'] = {'path': os.path.abspath(args.init), 'sha256': sha256_file(args.init)}
        pk = torch.load(args.init, map_location='cpu', weights_only=False)['policy_kwargs']
        if pk != cfg['policy']:
            print(f'--init: policy {pk} replaces {cfg["policy"]}', file=sys.stderr)
            cfg['policy'] = dict(pk)
    if args.kl_coef:
        teacher = args.kl_teacher or args.init
        if not teacher:
            raise SystemExit('--kl-coef needs --kl-teacher or --init')
        cfg['kl_teacher'] = {'path': os.path.abspath(teacher), 'sha256': sha256_file(teacher),
                             'coef': args.kl_coef, 'decay': args.kl_decay}
    return cfg


def variants_for(adapter, names):
    table = getattr(adapter, 'variants', {'central_1s': {}})
    names = names or list(getattr(adapter, 'final_variants', table))
    return {n: table[n] for n in names}


def average_rows(rows):
    acc = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def save_ckpt(path, policy, cfg, cfg_hash, seed, trainer, extra=None):
    pk = dict(cfg['policy'])
    torch.save({'policy': policy.state_dict(), 'policy_kwargs': pk, 'obs_size': policy.obs_size,
                'nvec': list(policy.nvec), 'cfg': cfg, 'cfg_hash': cfg_hash, 'seed': seed,
                'step': trainer.global_step, 'epoch': trainer.epoch,
                'optimizer': trainer.optimizer.state_dict(), **(extra or {})}, path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', help='JSON config (or selected.json / result.json) to start from')
    p.add_argument('--env', choices=['auto', 'standin', 'real'])
    for flag, (_, typ) in FLAGS.items():
        p.add_argument('--' + flag.replace('_', '-'), dest=flag, type=typ if typ is not float else float)
    p.add_argument('--shared', dest='shared', action='store_true', default=None,
                   help='MLP: one trunk for actor and critic')
    p.add_argument('--separate', dest='shared', action='store_false', help='MLP: separate trunks')
    p.add_argument('--mark', dest='mark_mode', type=int,
                   help='alias of --mark-mode: reward/wealth mark, 0 mid, 1 fair value, 2 blend')
    p.add_argument('--env-kw', nargs='*', help='extra env kwargs key=value (JSON values)')
    p.add_argument('--init', help='checkpoint to initialize the policy from (bc.py or train_ppo.py)')
    p.add_argument('--kl-coef', type=float, default=0.0, help='KL(teacher||policy) coefficient')
    p.add_argument('--kl-decay', type=float, default=0.5, help='fraction of training to decay it to 0')
    p.add_argument('--kl-teacher', help='teacher checkpoint (default: --init)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=int(os.environ.get('TRAIN_THREADS', 2)))
    p.add_argument('--eval-every', type=float, default=0, help='agent steps between evaluations (0: final only)')
    p.add_argument('--no-train-eval', action='store_true', help='skip the train-split eval (gap)')
    p.add_argument('--no-val', action='store_true',
                   help='dev run: never touch validation (train-split evals only, no ledger row)')
    p.add_argument('--variants', nargs='*', help='final eval variants (default: all of the adapter)')
    p.add_argument('--max-epochs', type=int, default=None, help='stop after this many epochs (smoke)')
    p.add_argument('--out', default=None, help='run directory (default runs/<time>_s<seed>)')
    p.add_argument('--tag', default=None, help='free-form label stored in the ledger (e.g. sweep name)')
    p.add_argument('--ledger', default=None, help='ledger path (default by env backend)')
    p.add_argument('--result-json', default=None, help='also write the result here')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)

    cfg = build_cfg(args)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adapter = get_adapter(cfg['env']['backend'])
    cfg['env']['backend'] = adapter.name
    env_kw = {k: v for k, v in cfg['env']['kw'].items() if v is not None}
    ident = adapter.identity(env_kw)
    cfg['env']['resolved'] = ident['params']  # recomputed every run: env drift changes the hash
    cfg['env']['code_sha256'] = ident['code']
    cfg_hash = ledger.config_hash(cfg)
    ledger_path = args.ledger or (ledger.LEDGER if adapter.name == 'real' else ledger.STANDIN_LEDGER)
    out = args.out or os.path.join(HERE, 'runs', f'{time.strftime("%Y%m%d_%H%M%S")}_{cfg_hash[:8]}_s{args.seed}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({'cfg': cfg, 'cfg_hash': cfg_hash, 'seed': args.seed, 'argv': sys.argv}, f, indent=1)
    log_f = open(os.path.join(out, 'log.jsonl'), 'a')

    def log(row):
        log_f.write(json.dumps(nan_to_none(row), default=float) + '\n')
        log_f.flush()

    env = adapter.make_train(cfg['num_envs'], seed=args.seed, **env_kw)

    if cfg['init']:
        policy, _ = load_checkpoint(cfg['init']['path'])
        policy.train()
    else:
        pk = dict(cfg['policy'])
        policy = make_policy(pk.pop('kind'), env=env, **pk)
    teacher = None
    if cfg['kl_teacher']:
        teacher, _ = load_checkpoint(cfg['kl_teacher']['path'])

    rnn = bool(getattr(policy, 'recurrent', False))
    pc = ppo_config(seed=args.seed, total_timesteps=cfg['total_timesteps'], use_rnn=rnn,
                    data_dir=out, **cfg['ppo'])
    if cfg['kl_teacher']:
        pc.update(kl_teacher_coef=cfg['kl_teacher']['coef'], kl_teacher_decay=cfg['kl_teacher']['decay'])
    trainer = Trainer(pc, env, policy, teacher=teacher, action_labels=adapter.action_labels,
                      group_fn=adapter.group_fn(env), group_names=adapter.group_names)

    variants = variants_for(adapter, args.variants)
    headline = next(iter(variants))
    n_val_evals = 0
    since_eval = []
    t0 = time.time()

    def do_eval(final=False):
        nonlocal n_val_evals
        diag = average_rows(since_eval)
        since_eval.clear()
        tag = 'final' if final else f'{trainer.global_step:012d}'
        ck = save_ckpt(os.path.join(out, 'final.pt' if final else f'ckpt_{tag}.pt'),
                       trainer.uncompiled_policy, cfg, cfg_hash, args.seed, trainer)
        results = {}
        names = [] if args.no_val else (list(variants) if final else [headline])
        for name in names:
            kw = {**env_kw, **variants[name]}
            res, recs = evaluate(trainer.uncompiled_policy, adapter, split=1, env_kw=kw,
                                 seed=args.seed, return_records=True)
            n_val_evals += 1
            results[name] = res
            if final and name == headline:
                with open(os.path.join(out, 'val_markets.jsonl'), 'w') as f:
                    for r in recs:
                        f.write(json.dumps(r, default=float) + '\n')
        tr = None
        if not args.no_train_eval:
            tr = evaluate(trainer.uncompiled_policy, adapter, split=0,
                          env_kw={**env_kw, **variants[headline]}, seed=args.seed)
        row = {'type': 'eval', 'final': final, 'step': trainer.global_step, 'epoch': trainer.epoch,
               'wall': time.time() - t0, 'ckpt': ck,
               **{f'val/{n}/{k}': v for n, r in results.items() for k, v in r.items()},
               **{f'diag/{k}': v for k, v in diag.items()}}
        vh = results.get(headline, {})
        if tr is not None:
            row.update({f'train/{k}': v for k, v in tr.items()})
            for k in ('score_mean', 'pnl_mean'):
                if k in tr and k in vh:
                    row[f'gap/{k}'] = tr[k] - vh[k]
        log(row)
        if not args.quiet:
            print(json.dumps({'eval_step': trainer.global_step, 'val_pnl_total': vh.get('pnl_total'),
                              'val_cvar95': vh.get('cvar95'), 'val_score': vh.get('score_mean'),
                              'train_pnl_total': (tr or {}).get('pnl_total'),
                              'gap_score': row.get('gap/score_mean')}), flush=True)
        return results, tr, ck

    eval_every = int(args.eval_every)
    next_eval = eval_every if eval_every else None
    while trainer.global_step < cfg['total_timesteps']:
        trainer.evaluate()
        logs = trainer.train()
        since_eval.append(logs)
        log({'type': 'train', **logs})
        if not args.quiet and trainer.epoch % 10 == 1:
            keys = ('step', 'sps', 'losses/entropy', 'losses/value_loss', 'losses/clipfrac',
                    'losses/approx_kl', 'clip_frac_reward', 'term_row0_frac', 'env/score')
            print(json.dumps({k: round(logs[k], 5) if isinstance(logs.get(k), float) else logs.get(k)
                              for k in keys}), flush=True)
        if args.max_epochs and trainer.epoch >= args.max_epochs:
            break
        if next_eval and trainer.global_step >= next_eval and trainer.global_step < cfg['total_timesteps']:
            do_eval()
            next_eval += eval_every

    wall = time.time() - t0
    train_rows = [json.loads(l) for l in open(os.path.join(out, 'log.jsonl')) if '"type": "train"' in l]
    sps = float(np.median([r['sps'] for r in train_rows[1:]] or [train_rows[0]['sps']]))
    results, tr, ck = do_eval(final=True)
    env.close()

    result = {'kind': 'train_run', 'backend': adapter.name, 'tag': args.tag, 'cfg_hash': cfg_hash,
              'seed': args.seed, 'cfg': cfg, 'steps': trainer.global_step, 'epochs': trainer.epoch,
              'wall_s': wall, 'sps_median': sps, 'run_dir': out, 'ckpt': ck,
              'ckpt_sha256': sha256_file(ck), 'n_val_evals': n_val_evals, 'headline': None if args.no_val else headline,
              'val': {n: nan_to_none(r) for n, r in results.items()},
              'train_eval': nan_to_none(tr) if tr else None}
    with open(os.path.join(out, 'result.json'), 'w') as f:
        json.dump(result, f, indent=1, default=float)
    if args.result_json:
        with open(args.result_json, 'w') as f:
            json.dump(result, f, indent=1, default=float)
    if not args.no_val:
        ledger.append(result, ledger_path)
    if not args.quiet:
        h = results.get(headline, {})
        print(json.dumps({'done': out, 'steps': trainer.global_step, 'wall_s': round(wall, 1),
                          'sps_median': round(sps), 'val_pnl_total': h.get('pnl_total'),
                          'val_cvar95': h.get('cvar95'), 'ledger': ledger_path}), flush=True)
    return result


if __name__ == '__main__':
    main()
