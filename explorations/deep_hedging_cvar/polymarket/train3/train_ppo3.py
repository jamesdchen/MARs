"""PPO training on the v3 env: a thin wrapper over train/train_ppo.py (not copied, not edited).

    python train_ppo3.py --data-dir BUILD --split-map SPLIT.json \
        --env3-kw obs_size=80 liq_rewards=1 size_mode=1 Q=1000 settle_mode=2 \
        --timesteps 25e6 --num-envs 4096 --threads 1 --seed 0 --tag e_main

What it changes in train_ppo.main (by rebinding names in the imported module, so the file stays
as the crypto freeze uses it):
- the adapter is train3/envs3.py V3Adapter (backend 'v3'): data from --data-dir (a builder output
  made loadable by v3env/adapt3.py) and --split-map (split3.py), both stored in the config's env
  kwargs and therefore in the config hash, together with the v3env/ code and data sha256;
- evaluation is train3/evaluator3.py evaluate (per-category groups, T5 / T6 / D1.3 and the
  reward-income line in every val row);
- the ledger is train3/trials.jsonl (one row per validation evaluation, every variant) and the
  run index train3/runs.jsonl; `--smoke` sends both to train3/smoke_trials.jsonl /
  smoke_runs.jsonl (mechanics checks: never counted, never selected from) and accepts a split
  map written by `split3.py --dry-run`;
- run directories default to train3/runs/<time>_<tag>_s<seed>;
- `--env3-kw k=v ...` (alias `--env-kw`) passes any mm3 parameter (settle_mode, obs_size, size_mode, size_rel*,
  liq_rewards, liq_v_c, liq_min_size, flat_c, pull_s, dt_other, dt_far, far_s, cat_mask,
  close_inv_coef / close_ramp / close_ramp_s, reward_mode, fee / rebate overrides; ENV3.md
  section 10). `--close-inv-coef` / `--close-ramp` map to the core's shaping (mm3 has it built in;
  train_ppo's CloseInventoryPenalty wrapper reads crypto obs and is not used).
Every other train_ppo.py flag works unchanged (--timesteps, --num-envs, --lr, --ent-coef, --hidden,
--init, --kl-coef, --eval-every, --no-train-eval, --variants, --max-epochs, ...). --env is fixed to
v3.
"""

import os
import sys
import time

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import envs3  # noqa: E402  (train/ and v3env/ on sys.path)
import evaluator3  # noqa: E402

import train_ppo  # noqa: E402  (train/train_ppo.py)

LEDGER = evaluator3.LEDGER
SMOKE_LEDGER = evaluator3.SMOKE_LEDGER
OWN = {'--data-dir': 1, '--split-map': 1, '--smoke': 0, '--env3-kw': '*', '--env-kw': '*'}


def split_argv(argv):
    """(our options, train_ppo argv)."""
    ours, rest, i = {'env3_kw': []}, [], 0
    while i < len(argv):
        a = argv[i]
        key = a.split('=', 1)[0]
        if key in OWN:
            name = key[2:].replace('-', '_')
            if OWN[key] == 0:
                ours[name] = True
                i += 1
            elif OWN[key] == 1:
                if '=' in a:
                    ours[name] = a.split('=', 1)[1]
                    i += 1
                else:
                    ours[name] = argv[i + 1]
                    i += 2
            else:
                i += 1
                while i < len(argv) and not argv[i].startswith('--'):
                    ours['env3_kw'].append(argv[i])
                    i += 1
        else:
            rest.append(a)
            i += 1
    return ours, rest


def _flag_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return None


def _drop_flag(argv, flag):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == flag:
            skip = True
            continue
        if a.startswith(flag + '='):
            continue
        out.append(a)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, rest = split_argv(argv)
    if not ours.get('data_dir'):
        raise SystemExit('--data-dir BUILD is required (a builder output made loadable by v3env/adapt3.py)')
    if _flag_value(rest, '--env') not in (None, 'v3'):
        raise SystemExit('train_ppo3.py always uses the v3 env')
    rest = _drop_flag(rest, '--env')
    kw = list(ours['env3_kw'])
    for flag, key in (('--close-inv-coef', 'close_inv_coef'), ('--close-ramp', 'close_ramp')):
        v = _flag_value(rest, flag)
        if v is not None:
            kw.append(f'{key}={v}')
            rest = _drop_flag(rest, flag)
    kw.append(f'data_dir={os.path.abspath(ours["data_dir"])}')
    if ours.get('split_map'):
        kw.append(f'split_map={os.path.abspath(ours["split_map"])}')
    smoke = bool(ours.get('smoke'))
    if _flag_value(rest, '--ledger') is None:
        rest += ['--ledger', SMOKE_LEDGER if smoke else LEDGER]
    elif smoke and os.path.abspath(_flag_value(rest, '--ledger')) == LEDGER:
        raise SystemExit('--smoke never writes the main ledger')
    if _flag_value(rest, '--out') is None:
        tag = _flag_value(rest, '--tag') or ('smoke' if smoke else 'v3')
        seed = _flag_value(rest, '--seed') or '0'
        rest += ['--out', os.path.join(HERE, 'runs', f'{time.strftime("%Y%m%d_%H%M%S")}_{tag}_s{seed}')]
    rest += ['--env-kw'] + kw

    adapter = envs3.V3Adapter(allow_dry_run_split=smoke)
    train_ppo.get_adapter = lambda name='v3': adapter
    train_ppo.evaluate = evaluator3.evaluate
    return train_ppo.main(rest)


if __name__ == '__main__':
    main()
