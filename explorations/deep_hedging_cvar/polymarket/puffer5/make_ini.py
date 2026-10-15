"""Write polymarket_mm.ini and sweep.ini, PufferLib 5.0's config for the MM env.

    python make_ini.py [--data-dir PATH]

polymarket_mm.ini goes to PufferLib/config/. sweep.ini is appended to it
before `./puffer sweep` (the notebook does this, after removing the generic
[sweep.*] blocks from PufferLib's config/default.ini; patch_pufferlib.py).

[env] holds every knob of the core (../mm.h, ENV.md), read by
polymarket_mm.h's puf_init. data_dir is a string: 5.0 keeps the raw text of
every ini value (DictItem.str), and a relative path resolves against the
PufferLib root, the trainer's working directory. The default assumes the
repo is cloned next to PufferLib, as in the notebook.

[eval_env] overrides [env] for the trainer's final evaluation only (needs
patch_pufferlib.py). It switches that evaluation to the validation split, so
the score each run reports, and the only point Protein sees (downsample = 1),
is validation P&L. Test (split 2) never appears here: the core refuses it
unless allow_test = 1 and the process has MM_FINAL=1.

check() asserts the trainer's shape rules (pufferl.cu: horizon % 8 == 0,
minibatch_size % horizon == 0, total_agents % (minibatch_size / horizon)
== 0, minibatch_size <= horizon * total_agents) for the defaults and for
every horizon/minibatch pair the sweep can draw, hidden_size % 8 == 0 for
Muon (PUFFER5.md rule 5), and that every default lies inside its sweep
range (`./puffer sweep` exits otherwise).
"""

import argparse
import itertools
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = '../MARs/explorations/deep_hedging_cvar/polymarket/data/build'

# Knobs of the core, in the order of ENV.md. Values are the core's defaults.
ENV = {
    'data_dir': DEFAULT_DATA_DIR,
    'split': 0,            # 0 train, 1 val, 2 test (guarded), see data/README.md
    'allow_test': 0,       # split 2 also needs MM_FINAL=1 in the environment
    'fill_model': 1,       # 0 optimistic, 1 central, 2 pessimistic (eval/EVALUATION.md T1)
    'latency': 1,          # quote placement and cancel delay, seconds (T2; headline 1 s)
    'dt': 1,               # seconds of replay per env step
    'Q': 10,               # shares per size unit of the size head
    'inv_penalty': 0.0,    # training-only shaping on |inventory|; not in score
    'reward_scale': 0.1,   # reward = reward_scale * (P&L change in $, minus shaping)
    'seed': 0,             # mixed with the env index (env->rng) per env
}
EVAL_ENV = {'split': 1}

BASE = {
    'env_name': 'polymarket_mm',
    # Episodes for the final (validation) evaluation of every train/sweep run.
    'eval_episodes': 4096,
    # Epochs between checkpoints (one epoch = horizon * total_agents steps),
    # so a run leaves intermediate checkpoints to score on validation.
    'checkpoint_interval': 50,
}
VEC = {'total_agents': 4096}
POLICY = {'hidden_size': 128, 'num_layers': 4}   # PufferNet defaults (default.ini)
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
    if isinstance(v, float) and v != int(v):
        return repr(v)
    return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(v)


def pow2_range(lo, hi):
    return [2 ** k for k in range(int(math.log2(lo)), int(math.log2(hi)) + 1)]


def shape_ok(horizon, minibatch, agents):
    return (horizon % 8 == 0 and minibatch % horizon == 0 and minibatch <= horizon * agents
            and agents % (minibatch // horizon) == 0)


def check():
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
    assert ENV['split'] != 2 and EVAL_ENV['split'] == 1 and ENV['allow_test'] == 0
    assert set(EVAL_ENV) <= set(ENV)
    assert SWEEP['downsample'] == 1, 'only the final validation eval may reach Protein'


def ini_text(data_dir):
    env = {**ENV, 'data_dir': f'"{data_dir}"'}   # quoted: ';' or '#' would start a comment
    out = ['# Generated by make_ini.py; do not edit by hand.']
    for name, sec in [('base', BASE), ('vec', VEC), ('policy', POLICY)]:
        out += ['', f'[{name}]'] + [f'{k} = {fmt(v)}' for k, v in sec.items()]
    out += ['', '# Every knob of ../mm.h (see ../ENV.md). data_dir: absolute, or relative to',
            '# the PufferLib root. Rewards are clamped to [-1, 1] by the trainer.', '[env]']
    out += [f'{k} = {fmt(v)}' for k, v in env.items()]
    out += ['', '# [env] overrides for the trainer\'s final evaluation only (patch_pufferlib.py):',
            '# every run, and every sweep trial, is scored on the validation split.',
            '[eval_env]']
    out += [f'{k} = {fmt(v)}' for k, v in EVAL_ENV.items()]
    out += ['', '# 1 s steps, episodes of 301 to 14401 steps (one market each) with dense',
            '# mark-to-market rewards: gamma 0.995 looks ~200 s ahead, past the 60 s',
            '# markout. horizon is a multiple of 8, minibatch_size of horizon.', '[train]']
    out += [f'{k} = {fmt(v)}' for k, v in TRAIN.items()]
    return '\n'.join(out) + '\n'


def sweep_text():
    out = ['# Generated by make_ini.py; appended to config/polymarket_mm.ini before',
           '# `./puffer sweep`, after the generic [sweep.*] blocks of default.ini are removed',
           '# (patch_pufferlib.py), so this is the whole space. Not swept: vec.total_agents',
           '# (4096 keeps every horizon/minibatch pair valid) and the env knobs.',
           '# metric = score is the env Log\'s per-episode P&L (scaled $) of the final',
           '# evaluation, which [eval_env] puts on the validation split; downsample = 1',
           '# keeps the training curve (train split) out of Protein.', '', '[sweep]']
    out += [f'{k} = {fmt(v)}' for k, v in SWEEP.items()]
    for (sec, key), (dist, lo, hi, scale) in SPACE.items():
        out += ['', f'[sweep.{sec}.{key}]', f'distribution = {dist}', f'min = {fmt(lo)}',
                f'max = {fmt(hi)}', f'scale = {scale}']
    return '\n'.join(out) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    p.add_argument('--out-dir', default=HERE)
    args = p.parse_args()
    check()
    with open(os.path.join(args.out_dir, 'polymarket_mm.ini'), 'w') as f:
        f.write(ini_text(args.data_dir))
    with open(os.path.join(args.out_dir, 'sweep.ini'), 'w') as f:
        f.write(sweep_text())


if __name__ == '__main__':
    main()
