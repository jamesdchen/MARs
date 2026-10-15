"""Env adapters: the one place that knows how to build the train / eval envs and
how to read per-market results back. Everything else in train/ talks to an
`Adapter`.

- `standin`: train/standin.py, a numpy toy with the real env's spaces.
- `real`:    polymarket/env.py (C core mm.h through PufferLib 3.0's native
             binding). See RealAdapter for the assumptions it checks.

Adapter interface:
    obs_size, nvec, action_labels (per head: category of each index),
    group_names (contract-length buckets for reward diagnostics), join (index)
    make_train(num_envs, seed, **kw)    split 0, random start offsets
    make_eval(split, seed, **kw)        -> (env, n_markets): every market of the
                                           split exactly once, start offset 0
    eval_records(env)                   per-market dicts finished since last call,
                                           each with 'market' and 'pnl'
    group_fn(env)                       callable -> (B,) int labels, or None
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
sys.path.insert(0, HERE)

TEST_SPLIT = 2


def _no_test(split):
    if int(split) == TEST_SPLIT:
        raise PermissionError('train/ never touches the test split (EVALUATION.md section 5)')


class StandInAdapter:
    name = 'standin'

    def __init__(self):
        import standin
        self.mod = standin
        self.action_labels = standin.ACTION_LABELS
        self.group_names = standin.GROUP_NAMES
        self.join = standin.JOIN
        self.nvec = (7, 7, 3)
        self.obs_size = 16
        self.variants = {'central_1s': {}}  # the toy has one fill model
        self.final_variants = ('central_1s',)

    def make_train(self, num_envs, seed=0, **kw):
        return self.mod.StandInMM(num_envs=num_envs, split=0, seed=seed, eval_mode=False,
                                  obs_size=self.obs_size, **kw)

    def make_eval(self, split=1, seed=0, **kw):
        _no_test(split)
        n = len(self.mod.build_split(split)['lens'])
        env = self.mod.StandInMM(num_envs=n, split=split, seed=seed, eval_mode=True,
                                 obs_size=self.obs_size, **kw)
        return env, n

    def eval_records(self, env):
        recs, env.episode_records = env.episode_records, []
        return recs

    def group_fn(self, env):
        return lambda obs: env.group_labels()

    def teacher(self, name='standin', params=None):
        return self.mod.teacher_policy


MARKET_DTYPE = np.dtype([
    ('row_offset', '<i8'), ('open_ts', '<i8'), ('end_ts', '<i8'), ('n_rows', '<i4'),
    ('contract_len_s', '<i4'), ('coin_id', '<i4'), ('winner_up', '<i4'), ('split', '<i4'),
    ('flags', '<i4'), ('tick_size', '<f4'), ('taker_fee_rate', '<f4'), ('fee_exponent', '<f4'),
    ('min_order_size', '<f4'), ('rewards_min_size', '<f4'), ('rewards_max_spread', '<f4'),
    ('log_strike_vs_ul_open', '<f4'), ('maker_rebate_rate', '<f4'), ('market_idx', '<i4'),
    ('tick_size_final', '<f4')])
assert MARKET_DTYPE.itemsize == 88


class MarketSetEnv:
    """Every market of a split once, from start offset 0, for evaluation.

    PufferLib 3.0's binding only reports the Log summed over all envs of a
    vec handle, and the 5m markets all end on the same step, so per-market
    results need one handle per market: one MmEnv(num_envs=1, fixed_market=mk,
    fixed_start=0) per market, all writing into slices of shared buffers.
    A step steps only the envs still on their first market; when one ends, its
    Log (n = 1: exactly that market) becomes the record. Deterministic: with a
    fixed market and start the core draws no random numbers."""

    def __init__(self, mod, markets, split, seed=0, **kw):
        import gymnasium
        n = len(markets)
        self.num_agents = n
        self.single_observation_space = gymnasium.spaces.Box(-5.0, 5.0, (mod.OBS_SIZE,), np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete(mod.ACT_SIZES)
        self.observations = np.zeros((n, mod.OBS_SIZE), np.float32)
        self.actions = np.zeros((n, len(mod.ACT_SIZES)), np.int32)
        self.rewards = np.zeros(n, np.float32)
        self.terminals = np.zeros(n, bool)
        self.truncations = np.zeros(n, bool)
        self.masks = np.ones(n, bool)
        self.markets = markets
        self.envs = []
        for i, mk in enumerate(markets['idx']):
            buf = {k: getattr(self, k)[i:i + 1] for k in
                   ('observations', 'actions', 'rewards', 'terminals', 'truncations', 'masks')}
            self.envs.append(mod.MmEnv(num_envs=1, seed=seed, split=split, fixed_market=int(mk),
                                       fixed_start=0, buf=buf, **kw))
        self.binding = self.envs[0]._binding
        self.active = np.ones(n, bool)
        self.records = []

    def reset(self, seed=0):
        for e in self.envs:
            self.binding.vec_reset(e.c_envs, seed)
        self.active[:] = True
        self.records = []
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.terminals[:] = False
        b = self.binding
        for i in np.flatnonzero(self.active):
            b.vec_step(self.envs[i].c_envs)
        for i in np.flatnonzero(self.active & self.terminals):
            log = b.vec_log(self.envs[i].c_envs)
            assert log.get('n') == 1.0, log
            m = self.markets[i]
            self.records.append({'env': int(i), 'market': int(m['idx']),
                                 'contract_len_s': int(m['contract_len_s']), 'coin_id': int(m['coin_id']),
                                 'open_ts': int(m['open_ts']), 'end_ts': int(m['end_ts']),
                                 **{k: float(v) for k, v in log.items() if k != 'n'}})
            self.active[i] = False
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def close(self):
        for e in self.envs:
            e.close()


REAL_LABELS = (('off', 'improve', 'improve', 'join', 'behind', 'behind', 'behind'),) * 2 + (
    ('1x', '2x', '5x'),)


class RealAdapter:
    """polymarket/env.py (MmEnv, mm.h through PufferLib 3.0's native binding).

    Train env: MmEnv(split=0) with its defaults: random market, random start in
    the first start_frac (20%) of the window. Eval: MarketSetEnv over every
    market of the split (fixed_start = 0). Fill/latency variants are env
    kwargs (fill_model 0 optimistic / 1 central / 2 pessimistic, latency s);
    the first variant is the headline (EVALUATION.md: central, 1 s)."""
    name = 'real'
    variants = {'central_1s': {'fill_model': 1, 'latency': 1},
                'pess_1s': {'fill_model': 2, 'latency': 1},
                'central_2s': {'fill_model': 1, 'latency': 2},
                'central_0s': {'fill_model': 1, 'latency': 0},
                'opt_1s': {'fill_model': 0, 'latency': 1}}
    final_variants = ('central_1s', 'pess_1s')
    native_probes = {'join': {'reward_mode': 1}}  # mm.h: reward 1 iff both heads 'join'

    def __init__(self):
        import importlib.util
        path = os.path.join(PM, 'env.py')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} does not exist yet; use --env standin')
        if PM not in sys.path:
            sys.path.insert(1, PM)  # binding*.so lives next to env.py
        spec = importlib.util.spec_from_file_location('pm_env', path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.obs_size = self.mod.OBS_SIZE
        self.nvec = tuple(self.mod.ACT_SIZES)
        self.action_labels = REAL_LABELS
        self.join = 3
        meta = json.load(open(os.path.join(PM, 'data', 'build', 'feat_meta.json')))
        ci = meta['column_index']
        self.len_cols = [ci['len_5m'], ci['len_15m'], ci['len_4h']]
        self.col = ci
        self.group_names = ('5m', '15m', '4h')
        self.markets_all = np.fromfile(os.path.join(PM, 'data', 'build', 'markets.bin'), MARKET_DTYPE)

    def split_markets(self, split):
        _no_test(split)
        idx = np.flatnonzero(self.markets_all['split'] == split)
        m = self.markets_all[idx]
        out = np.zeros(len(idx), dtype=[('idx', '<i8'), ('contract_len_s', '<i4'), ('coin_id', '<i4'),
                                         ('open_ts', '<i8'), ('end_ts', '<i8')])
        out['idx'] = idx
        for k in ('contract_len_s', 'coin_id', 'open_ts', 'end_ts'):
            out[k] = m[k]
        return out

    def make_train(self, num_envs, seed=0, **kw):
        kw.pop('split', None)
        return self.mod.MmEnv(num_envs=num_envs, seed=seed, split=0, **kw)

    def make_eval(self, split=1, seed=0, **kw):
        _no_test(split)
        for k in ('split', 'fixed_market', 'fixed_start', 'cycle_stride', 'allow_test'):
            kw.pop(k, None)
        mk = self.split_markets(split)
        return MarketSetEnv(self.mod, mk, split, seed=seed, **kw), len(mk)

    def eval_records(self, env):
        recs, env.records = env.records, []
        return recs

    def group_fn(self, env):
        cols = self.len_cols
        return lambda obs: np.argmax(obs[:, cols], axis=1)

    def teacher(self, name='fair_value', params=None):
        """The fair-value baseline (baselines/baselines.py) when its obs spec
        resolves on this env; otherwise a simple obs-based quoter."""
        try:
            sys.path.insert(1, os.path.join(PM, 'baselines'))
            import baselines as bl
            from obs_spec import load_spec
            return bl.make(name, load_spec(), params)
        except Exception as e:  # noqa: BLE001
            print(f'teacher: baselines.{name} unavailable ({type(e).__name__}: {e}); '
                  'using the simple obs quoter', file=sys.stderr)
            return self.simple_teacher

    def simple_teacher(self, obs):
        """Join both sides when present; pull the side the fair-value gap says
        will be picked off (|gap| > 1 cent); back off the side that adds to a
        position beyond half of Q; size 1x."""
        c = self.col
        gap_c = obs[:, c['gap_c']] * 5.0              # cents
        inv = obs[:, self.mod.OBS_SIZE - 15]          # inventory / Q (first state slot)
        a = np.zeros((obs.shape[0], 3), np.int32)
        a[:, 0] = np.where(obs[:, c['bid_present']] > 0.5, 3, 0)
        a[:, 1] = np.where(obs[:, c['ask_present']] > 0.5, 3, 0)
        a[gap_c < -1.0, 0] = 0
        a[gap_c > 1.0, 1] = 0
        a[(inv > 0.5) & (a[:, 0] > 0), 0] = 5
        a[(inv < -0.5) & (a[:, 1] > 0), 1] = 5
        return a


ADAPTERS = {'standin': StandInAdapter, 'real': RealAdapter}


def get_adapter(name='auto'):
    if name == 'auto':
        name = 'real' if os.path.exists(os.path.join(PM, 'env.py')) else 'standin'
    return ADAPTERS[name]()
