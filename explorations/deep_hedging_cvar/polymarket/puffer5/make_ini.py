"""Write polymarket_mm.ini and sweep.ini, PufferLib 5.0's config for the MM env.

    python make_ini.py [--data-dir PATH]

polymarket_mm.ini goes to PufferLib/config/. sweep.ini is appended to it
before `./puffer sweep`, after patch_pufferlib.py has removed the generic
[sweep.*] blocks from PufferLib's config/default.ini.

[env] holds every knob of the core, read from ../mm.h (MM_PARAM_NAMES and
MM_DEFAULTS, documented in ../ENV.md), so the ini and the core cannot drift
apart, plus data_dir, seed and the wrapper's log_episodes. data_dir is a
string: 5.0 keeps the raw text of every ini value (DictItem.str), and a
relative path resolves against the PufferLib root, the trainer's working
directory. The default assumes the repo is cloned next to PufferLib.

[eval_env] overrides [env] for the trainer's final evaluation only (needs
patch_pufferlib.py). That evaluation becomes one full episode of every
validation market, each played k times: split = 1, episodes from the open
(fixed_start = 0), env i starts on validation market i mod n_val
(cycle_stride > 0; the wrapper seeds the core with the env index), and each
env logs only its first episode (log_episodes = 1). With eval_agents =
eval_episodes = k * n_val, the trainer's eval loop (which stops at
eval_episodes finished episodes) ends when every env has finished its market,
so the score is the mean P&L per validation market with every market weighted
equally; without log_episodes the first episodes to finish, the 5-minute
markets, would dominate it. That score is what a run reports and the only
point Protein sees (sweep.downsample = 1). Test (split 2) never appears here:
the core refuses it unless allow_test = 1 and the process has MM_FINAL=1.

check() asserts the trainer's shape rules (launch_train in pufferl.cu:
horizon % 8 == 0, minibatch_size % horizon == 0, total_agents %
(minibatch_size / horizon) == 0, minibatch_size <= horizon * total_agents)
for the defaults and for every horizon/minibatch pair the sweep can draw,
hidden_size % 8 == 0 for Muon (PUFFER5.md rule 5), and that every default
lies inside its sweep range (`./puffer sweep` exits otherwise).
"""

import argparse
import itertools
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, '..', 'mm.h')
LOCAL_DATA = os.path.join(HERE, '..', 'data', 'build')
DEFAULT_DATA_DIR = '../MARs/explorations/deep_hedging_cvar/polymarket/data/build'
MARKET_ITEMSIZE, SPLIT_OFFSET = 88, 40      # data/README.md: MarketRec, int32 split at byte 40


def core_params(path=CORE):
    """(name, default) of every MmParams field, in order, from mm.h."""
    text = re.sub(r'//[^\n]*|/\*.*?\*/', '', open(path).read(), flags=re.S)
    enums = {k: int(v) for k, v in re.findall(r'\b(MM_\w+)\s*=\s*(-?\d+)', text)}
    names = re.search(r'MM_PARAM_NAMES\[\]\s*=\s*\{(.*?)\};', text, re.S)[1]
    values = re.search(r'MM_DEFAULTS\[\]\s*=\s*\{(.*?)\};', text, re.S)[1]
    names = re.findall(r'"(\w+)"', names)
    values = [v.strip() for v in values.split(',') if v.strip()]
    assert len(names) == len(values), (len(names), len(values))
    out = []
    for n, v in zip(names, values):
        out.append((n, float(enums[v]) if v in enums else float(v)))
    return out


def val_market_count(data_dir=LOCAL_DATA):
    path = os.path.join(data_dir, 'markets.bin')
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found: build the data first (data/build.py)')
    raw = np.fromfile(path, np.uint8).reshape(-1, MARKET_ITEMSIZE)
    split = raw[:, SPLIT_OFFSET:SPLIT_OFFSET + 4].copy().view('<i4')[:, 0]
    return int((split == 1).sum())


def eval_agents(n_val, target=4096):
    """k * n_val, a multiple of 16 (equal buffers, aligned batches), near target."""
    base = n_val * 16 // math.gcd(n_val, 16)
    return base * max(1, round(target / base))


CORE_PARAMS = core_params()
# [env]: the core's defaults; the 5.0 wrapper adds data_dir, seed and log_episodes
# (0 = log every episode; the final evaluation sets 1).
ENV = {'data_dir': DEFAULT_DATA_DIR, **dict(CORE_PARAMS), 'seed': 0, 'log_episodes': 0}
EVAL_ENV = {'split': 1, 'fixed_start': 0, 'cycle_stride': 1, 'log_episodes': 1}

BASE = {
    'env_name': 'polymarket_mm',
    'eval_agents': None,       # k * n_val, set in main()
    'eval_episodes': None,     # = eval_agents (one logged episode per env)
    # Epochs between checkpoints (one epoch = horizon * total_agents steps), so a
    # run leaves intermediate checkpoints to score on validation.
    'checkpoint_interval': 50,
}
VEC = {'total_agents': 4096}
POLICY = {'hidden_size': 128, 'num_layers': 4}   # PufferNet's defaults (default.ini)
TRAIN = {
    'total_timesteps': 100_000_000,
    'horizon': 128,
    'minibatch_size': 16384,
    'gamma': 0.995,
    'gae_lambda': 0.95,
    'ent_coef': 0.01,
    'learning_rate': 0.015,
    'replay_ratio': 1.0,
    'clip_coef': 0.2,
    'vf_coef': 2.0,
    'max_grad_norm': 1.5,
    'momentum': 0.95,
}

SWEEP = {
    'metric': 'score',
    'goal': 'maximize',
    'max_runs': 32,
    'gpus': 1,
    # One Protein point per run: the final evaluation, on validation. With
    # downsample > 1 the earlier points come from the training curve (train split).
    'downsample': 1,
    'early_stop_quantile': 0.3,
    'max_suggestion_cost': 1800,
}

# (section, key): (distribution, min, max, scale). The run length is Protein's cost axis.
SPACE = {
    ('train', 'total_timesteps'): ('log_normal', 2e7, 5e8, 'time'),
    ('train', 'learning_rate'): ('log_normal', 3e-4, 0.1, 0.5),
    ('train', 'ent_coef'): ('log_normal', 1e-4, 0.05, 'auto'),
    ('train', 'gamma'): ('logit_normal', 0.98, 0.9995, 'auto'),
    ('train', 'gae_lambda'): ('logit_normal', 0.8, 0.995, 'auto'),
    ('train', 'horizon'): ('uniform_pow2', 64, 512, 'auto'),
    ('train', 'minibatch_size'): ('uniform_pow2', 8192, 65536, 'auto'),
    ('train', 'replay_ratio'): ('uniform', 0.5, 4.0, 'auto'),
    ('train', 'clip_coef'): ('uniform', 0.05, 0.5, 'auto'),
    ('train', 'vf_coef'): ('uniform', 0.1, 4.0, 'auto'),
    ('train', 'max_grad_norm'): ('uniform', 0.1, 1.5, 'auto'),
    ('train', 'momentum'): ('logit_normal', 0.5, 0.999, 'auto'),
    ('policy', 'hidden_size'): ('uniform_pow2', 32, 512, 'auto'),
    ('policy', 'num_layers'): ('int_uniform', 1, 4, 'auto'),
}
SECTIONS = {'base': BASE, 'vec': VEC, 'policy': POLICY, 'train': TRAIN}


def fmt(v):
    if isinstance(v, str):
        return v
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(float(v))


def pow2_range(lo, hi):
    return [2 ** k for k in range(int(math.log2(lo)), int(math.log2(hi)) + 1)]


def shape_ok(horizon, minibatch, agents):
    return (horizon % 8 == 0 and minibatch % horizon == 0 and minibatch <= horizon * agents
            and agents % (minibatch // horizon) == 0)


def check(n_val):
    agents = VEC['total_agents']
    assert shape_ok(TRAIN['horizon'], TRAIN['minibatch_size'], agents), 'default shapes'
    _, h_lo, h_hi, _ = SPACE['train', 'horizon']
    _, m_lo, m_hi, _ = SPACE['train', 'minibatch_size']
    for h, m in itertools.product(pow2_range(h_lo, h_hi), pow2_range(m_lo, m_hi)):
        assert shape_ok(h, m, agents), f'sweep can draw horizon={h} minibatch_size={m}'
        # at least one gradient step per epoch at the smallest replay ratio
        assert SPACE['train', 'replay_ratio'][1] * h * agents / m >= 1, (h, m)
    _, lo, hi, _ = SPACE['policy', 'hidden_size']
    assert all(hs % 8 == 0 for hs in pow2_range(lo, hi)) and POLICY['hidden_size'] % 8 == 0
    for (sec, key), (dist, lo, hi, _) in SPACE.items():
        v = SECTIONS[sec][key]
        assert lo <= v <= hi, f'default {sec}.{key} = {v} outside [{lo}, {hi}]'
        if dist == 'uniform_pow2':
            assert v & (v - 1) == 0, f'{sec}.{key} = {v} is not a power of 2'
    assert ENV['split'] == 0 and ENV['allow_test'] == 0 and EVAL_ENV['split'] == 1
    assert set(EVAL_ENV) <= set(ENV) and EVAL_ENV['cycle_stride'] > 0
    assert BASE['eval_episodes'] == BASE['eval_agents'] and BASE['eval_agents'] % n_val == 0
    assert BASE['eval_agents'] % 16 == 0
    assert SWEEP['downsample'] == 1, 'only the final validation eval may reach Protein'


def ini_text(data_dir, n_val):
    env = {**ENV, 'data_dir': f'"{data_dir}"'}   # quoted: ';' or '#' would start a comment
    k = BASE['eval_agents'] // n_val
    out = ['# Generated by make_ini.py from ../mm.h and the data build; do not edit by hand.']
    out += ['', '[base]'] + [f'{k_} = {fmt(v)}' for k_, v in BASE.items()]
    out[-3:-3] = [f'# Final evaluation: each of the {n_val} validation markets played {k} times,',
                  '# one logged episode per env ([eval_env]).']
    for name, sec in [('vec', VEC), ('policy', POLICY)]:
        out += ['', f'[{name}]'] + [f'{k_} = {fmt(v)}' for k_, v in sec.items()]
    out += ['', '# The core\'s knobs and defaults (../mm.h, ../ENV.md), data_dir (absolute, or',
            '# relative to the PufferLib root), seed (mixed with the env index) and the',
            '# wrapper\'s log_episodes. The trainer clamps rewards to [-1, 1].', '[env]']
    out += [f'{k_} = {fmt(v)}' for k_, v in env.items()]
    out += ['', '# [env] overrides for the trainer\'s final evaluation only (patch_pufferlib.py):',
            '# every train run and sweep trial is scored on the validation split.',
            '[eval_env]']
    out += [f'{k_} = {fmt(v)}' for k_, v in EVAL_ENV.items()]
    out += ['', '# 1 s steps (5 s on 4h markets), episodes of up to 300 (5m), 900 (15m) and',
            '# 2880 (4h) decisions, dense mark-to-market rewards: gamma 0.995 looks ~200',
            '# steps ahead, past the 60 s markout. horizon is a multiple of 8,',
            '# minibatch_size of horizon.', '[train]']
    out += [f'{k_} = {fmt(v)}' for k_, v in TRAIN.items()]
    return '\n'.join(out) + '\n'


def sweep_text():
    out = ['# Generated by make_ini.py; appended to config/polymarket_mm.ini before',
           '# `./puffer sweep`, after patch_pufferlib.py removed the generic [sweep.*]',
           '# blocks of default.ini, so this is the whole space. Not swept: vec.total_agents',
           '# (4096 keeps every horizon/minibatch pair valid) and the env knobs.',
           '# metric = score is the env Log\'s mean P&L per episode, in $, of the final',
           '# evaluation, which [eval_env] runs on the validation split; downsample = 1',
           '# keeps the training curve (train split) out of Protein.', '', '[sweep]']
    out += [f'{k} = {fmt(v)}' for k, v in SWEEP.items()]
    for (sec, key), (dist, lo, hi, scale) in SPACE.items():
        out += ['', f'[sweep.{sec}.{key}]', f'distribution = {dist}', f'min = {fmt(lo)}',
                f'max = {fmt(hi)}', f'scale = {scale}']
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                   help='data/build as the trainer sees it (from the PufferLib root)')
    p.add_argument('--out-dir', default=HERE)
    args = p.parse_args()
    n_val = val_market_count()
    BASE['eval_agents'] = BASE['eval_episodes'] = eval_agents(n_val)
    check(n_val)
    with open(os.path.join(args.out_dir, 'polymarket_mm.ini'), 'w') as f:
        f.write(ini_text(args.data_dir, n_val))
    with open(os.path.join(args.out_dir, 'sweep.ini'), 'w') as f:
        f.write(sweep_text())
    print(f'polymarket_mm.ini: {len(CORE_PARAMS)} core knobs; final eval on {n_val} validation '
          f'markets x {BASE["eval_agents"] // n_val} = {BASE["eval_agents"]} envs; sweep.ini: '
          f'{len(SPACE)} parameters')


if __name__ == '__main__':
    main()
