"""V3Adapter: the train/ adapter interface (train/envs.py) around the v3 replay env
(v3env/env3.py MmEnv3, mm3.h) for protocol v3 (DECISIONS.md D11): track S (pre-game sports) and
track E (election reward farming).

    from envs3 import V3Adapter
    ad = V3Adapter()
    env = ad.make_train(1024, seed=0, data_dir=BUILD, split_map=SPLIT_JSON, obs_size=80, liq_rewards=1)
    ev, n = ad.make_eval(1, data_dir=BUILD, split_map=SPLIT_JSON, obs_size=80, liq_rewards=1)

Differences from train/envs.py RealAdapter (which is not edited):
- `identity()` hashes the v3env/ code (every top-level file: mm3.h, binding3.c, env3.py, adapt3.py,
  the built binding3*.so, ...) and the data of the build actually loaded: the five v2-schema files,
  meta.json / feat_meta.json, and the v3 side files markets3.bin / x3.bin / meta3.json; with a split
  map also the split file and the split view's markets.bin. Hashes of big files are cached by
  (path, size, mtime) in train3/.hashcache.json.
- Markets are grouped by category (sports league, elections), not by crypto contract length:
  `market_table()` per market (category, group 'sports/<league>' or 'elections', game / window
  date, slug), training diagnostics by category one-hot (obs 52-54, needs obs_size >= 55).
- The test split (2) is refused unless the environment has MM_FINAL=1 (then allow_test=1 is
  passed, and mm3.h checks both again); any path containing 'jun15' is refused (D7).
- Split map: `split_map=FILE` (written by split3.py) assigns every market a split by its date
  (sports: game day = slug date, else kickoff UTC date; elections: UTC day of the window open).
  Days not in the map go to split 3 (purged). The builders' own split field is ignored: the
  adapter writes a *view* directory (train3/views/<build>_<hash>/) holding symlinks to every
  file of the build and a rewritten markets.bin, and points mm3 at it. A split file written by
  `split3.py --dry-run` is refused unless `allow_dry_run_split=True` (smoke runs only).

Eval env (MarketSetEnv3): every market of a split once from row 0, one MmEnv3(num_envs=1) handle
per market (cycle_stride = n, seed j -> list[j]), as train/envs.py MarketSetEnv. Each finished
episode is checked against its market: `episode_rows == n_rows - 1` and the decision count
(dt per length class, dt_far / far_s cadence), and the category one-hot of the first observation
(obs_size >= 55).
"""

import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.dirname(HERE)
TRAIN = os.path.join(PM, 'train')
V3 = os.path.join(PM, 'v3env')
for _p in (V3, TRAIN):
    if _p not in sys.path:
        sys.path.append(_p)

from envs import MARKET_DTYPE, REAL_LABELS, RealAdapter, no_jun15  # noqa: E402  (train/envs.py)

TEST_SPLIT = 2
SPLIT_IDS = {'train': 0, 'val': 1, 'test': 2, 'purged': 3, 'exclude': 3}
SPLIT_NAMES = {0: 'train', 1: 'val', 2: 'test', 3: 'purged'}
CAT_NAMES = ('crypto', 'sports', 'elections', 'other')
EXT_DTYPE = np.dtype([('settle_price', '<f8'), ('event_ts', '<i8'), ('settle_mode', '<i4'), ('category', '<i4'),
                      ('row_s', '<f4'), ('rw_v_c', '<f4'), ('rw_min_size', '<f4'), ('rw_rate_day', '<f4'),
                      ('fee_rate', '<f4'), ('fee_exp', '<f4'), ('rebate_rate', '<f4'), ('flags3', '<i4'),
                      ('pad', '<i4', 2)])
assert EXT_DTYPE.itemsize == 64
DATA_FILES = ('markets.bin', 'feat.bin', 'book.bin', 'fills.bin', 'fill_index.bin', 'markets3.bin', 'x3.bin',
              'meta.json', 'feat_meta.json', 'meta3.json')
VIEWS = os.path.join(HERE, 'views')
HASH_CACHE = os.path.join(HERE, '.hashcache.json')
OBS_CAT = 52            # ENV3.md section 7: obs 52-54 = category one-hot (crypto, sports, elections)
OBS_EV = 50             # ev_log = log1p(seconds to event) / log1p(86400)
# D15 time-to-kickoff buckets (build_sports.BUCKETS): name, the bound below it (s), build_sports' name
BUCKETS = (('gt24h', 86400, '>24h'), ('6_24h', 21600, '6-24h'), ('1_6h', 3600, '1-6h'), ('1h_30m', 1800, '1h-30m'),
           ('30_5m', 300, '30-5m'), ('5_0m', 0, '5-0m'))
BUCKET_FROM_BUILD = {b: n for n, _, b in BUCKETS}


def episode_bucket(lead_s, n_rows, row_s=1.0):
    """build_sports.bucket_info for an episode whose first row is lead_s seconds before the event: the
    D15 bucket holding most of its decision rows (all but the last), bounds upper-inclusive (a row
    exactly 1800 s out is 30_5m). Used when meta.json has no segments[i].bucket (sports-1 builds)."""
    nd = max(int(n_rows) - 1, 1)

    def n_le(b):                    # decision rows j in [0, nd) with lead_s - j * row_s <= b
        j0 = max(0, math.ceil((lead_s - b) / row_s - 1e-9))
        return max(0, nd - j0)
    le = [n_le(b) for _, b, _ in BUCKETS[:-1]]
    cnt = [nd - le[0]] + [le[k - 1] - le[k] for k in range(1, len(le))] + [le[-1]]
    return BUCKETS[int(np.argmax(cnt))][0]


# ----------------------------------------------------------------------------- hashing
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def cached_sha256(path):
    """sha256 of a file, cached by (realpath, size, mtime_ns): an 18-day election build holds 10^8
    rows, and identity() runs at every training start."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = f'{real}|{st.st_size}|{st.st_mtime_ns}'
    if st.st_size < (64 << 20):
        return sha256_file(real)
    try:
        cache = json.load(open(HASH_CACHE))
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key]
    h = sha256_file(real)
    try:
        with open(HASH_CACHE, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                cur = json.loads(f.read() or '{}')
            except ValueError:
                cur = {}
            cur[key] = h
            f.seek(0)
            f.truncate()
            f.write(json.dumps(cur, indent=0))
    except OSError:
        pass
    return h


def v3env_code_hashes():
    """Every top-level file of v3env/ (sources, docs, the built binding3*.so)."""
    out = {}
    for f in sorted(os.listdir(V3)):
        p = os.path.join(V3, f)
        if os.path.isfile(p) and not f.endswith('.pyc'):
            out[f'v3env/{f}'] = sha256_file(p)
    return out


# ----------------------------------------------------------------------------- build metadata
def detect_kind(build):
    """'sports' / 'elections' / 'crypto' (adapt3.detect, or meta3.json when present)."""
    m3 = os.path.join(build, 'meta3.json')
    if os.path.exists(m3):
        return json.load(open(m3))['kind']
    meta = json.load(open(os.path.join(build, 'meta.json')))
    if str(meta.get('build_version', '')).startswith('sports'):
        return 'sports'
    if 'v3_elections' in str(meta.get('builder', '')):
        return 'elections'
    return 'crypto'


def _utc_day(ts):
    return _dt.datetime.fromtimestamp(int(ts), _dt.timezone.utc).strftime('%Y-%m-%d')


def market_table(build):
    """Per-market facts of a build (markets.bin + markets3.bin + meta.json), in market order:
    idx, split (the build's), n_rows, open_ts, end_ts, event_ts, contract_len_s, category, league,
    group, date (the split-map key), slug, settle_mode, row_s, rw_rate_day, min_order_size, and for
    sports the episode's D15 bucket (meta.json segments[i].bucket, else episode_bucket)."""
    no_jun15(build)
    mk = np.fromfile(os.path.join(build, 'markets.bin'), MARKET_DTYPE)
    p3 = os.path.join(build, 'markets3.bin')
    ext = np.fromfile(p3, EXT_DTYPE) if os.path.exists(p3) else None
    if ext is not None and len(ext) != len(mk):
        raise ValueError(f'{p3}: {len(ext)} records for {len(mk)} markets (rerun adapt3.py)')
    meta = json.load(open(os.path.join(build, 'meta.json')))
    slugs = meta.get('slugs') or [f'm{i}' for i in range(len(mk))]
    segs = meta.get('segments')                     # build_sports sports-2: one record per market
    if segs is not None and len(segs) != len(mk):
        segs = None
    kind = detect_kind(build)
    rows = []
    for i, m in enumerate(mk):
        e = ext[i] if ext is not None else None
        ev = int(e['event_ts']) if e is not None and int(e['event_ts']) > 0 else int(m['end_ts'])
        cat = int(e['category']) if e is not None else 0
        slug = str(slugs[i]) if i < len(slugs) else f'm{i}'
        base = slug.split('#')[0].split('@')[0]
        if kind == 'sports' or cat == 1:
            # build_sports.game_day: the slug date, else the kickoff UTC date
            d = re.search(r'(\d{4}-\d{2}-\d{2})', base)
            date = d.group(1) if d else _utc_day(ev)
            league = base.split('-')[0] or 'unknown'
            group = f'sports/{league}'
        elif kind == 'elections' or cat == 2:
            date = _utc_day(m['open_ts'])            # build_elections: UTC day of the window open
            league = 'elections'
            group = 'elections'
        else:
            date = _utc_day(m['open_ts'])
            league = CAT_NAMES[cat]
            group = CAT_NAMES[cat]
        bucket = None
        if kind == 'sports' or cat == 1:
            bk = segs[i].get('bucket') if segs is not None else None
            bucket = BUCKET_FROM_BUILD.get(bk) or episode_bucket(ev - int(m['open_ts']), int(m['n_rows']),
                                                                 float(e['row_s']) if e is not None else 1.0)
        rows.append({'idx': i, 'split': int(m['split']), 'n_rows': int(m['n_rows']), 'bucket': bucket,
                     'open_ts': int(m['open_ts']), 'end_ts': int(m['end_ts']), 'event_ts': ev,
                     'contract_len_s': int(m['contract_len_s']), 'category': cat,
                     'category_name': CAT_NAMES[cat], 'league': league, 'group': group, 'date': date,
                     'slug': slug, 'settle_mode': int(e['settle_mode']) if e is not None else 0,
                     'row_s': float(e['row_s']) if e is not None else 1.0,
                     'rw_rate_day': float(e['rw_rate_day']) if e is not None else 0.0,
                     'rw_min_size': float(e['rw_min_size']) if e is not None else 0.0,
                     'min_order_size': float(m['min_order_size'])})
    return rows


# ----------------------------------------------------------------------------- split map
def load_split_map(path, allow_dry_run=False):
    """{day: split id} and the file's metadata from a split3.py file. Accepts the sports builder
    layout ({"days": {day|"a..b": name}, "default": ...}) and the elections builder layout (flat
    {day: name}); keys starting with '_' are metadata."""
    no_jun15(path)
    raw = open(path, 'rb').read()
    d = json.loads(raw)
    meta = {k: v for k, v in d.items() if k.startswith('_')}
    if meta.get('_dry_run') and not allow_dry_run:
        raise PermissionError(f'{path} was written by split3.py --dry-run: not a pre-registered split '
                              '(smoke runs only: --smoke)')
    days = d.get('days') if isinstance(d.get('days'), dict) else {k: v for k, v in d.items()
                                                                     if not k.startswith('_') and k != 'default'}
    default = d.get('default', 'exclude')
    if default not in ('exclude', 'purged'):
        raise ValueError(f'{path}: default {default!r}: a v3 split map lists every day explicitly')
    out = {}
    for k, v in days.items():
        sid = SPLIT_IDS[str(v).lower()] if not isinstance(v, int) else int(v)
        lo, _, hi = k.partition('..')
        if hi:
            day = _dt.date.fromisoformat(lo)
            while day <= _dt.date.fromisoformat(hi):
                out[day.isoformat()] = sid
                day += _dt.timedelta(days=1)
        else:
            _dt.date.fromisoformat(k)
            out[k] = sid
    meta['sha256'] = hashlib.sha256(raw).hexdigest()
    meta['path'] = os.path.abspath(path)
    return out, meta


def make_view(build, split_file, allow_dry_run=False):
    """A directory mm3 can open with the split map applied: symlinks to every file of `build`
    and a markets.bin whose split field comes from the map (unlisted days -> 3). Idempotent: the
    view name is a hash of the build path, its markets.bin and the split file."""
    no_jun15(build)
    no_jun15(split_file)
    day_split, smeta = load_split_map(split_file, allow_dry_run)
    track = smeta.get('_track')
    kind = detect_kind(build)
    if track and kind != 'crypto' and track != kind:
        raise ValueError(f'split map for track {track!r} applied to a {kind} build')
    mk_path = os.path.join(build, 'markets.bin')
    key = hashlib.sha256(f'{os.path.realpath(build)}|{sha256_file(mk_path)}|{smeta["sha256"]}'.encode()).hexdigest()
    view = os.path.join(VIEWS, f'{os.path.basename(os.path.normpath(build))}_{key[:12]}')
    done = os.path.join(view, 'view3.json')
    if os.path.exists(done):
        return view
    tab = market_table(build)
    mk = np.fromfile(mk_path, MARKET_DTYPE)
    new = np.array([day_split.get(r['date'], 3) for r in tab], np.int32)
    mk['split'] = new
    os.makedirs(view, exist_ok=True)
    for f in os.listdir(build):
        src = os.path.join(os.path.realpath(build), f)
        dst = os.path.join(view, f)
        if f == 'markets.bin' or not os.path.isfile(src):
            continue
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    tmp = os.path.join(view, f'.markets.bin.tmp{os.getpid()}')
    mk.tofile(tmp)
    os.replace(tmp, os.path.join(view, 'markets.bin'))
    counts = {SPLIT_NAMES[s]: {'markets': int((new == s).sum()),
                               'days': sorted({r['date'] for r, x in zip(tab, new) if x == s})} for s in range(4)}
    info = {'build': os.path.realpath(build), 'split_file': smeta['path'], 'split_sha256': smeta['sha256'],
            'track': track, 'kind': kind, 'dry_run': bool(smeta.get('_dry_run')), 'counts': counts,
            'markets_sha256': sha256_file(os.path.join(view, 'markets.bin'))}
    with open(done, 'w') as f:
        json.dump(info, f, indent=1)
    return view


# ----------------------------------------------------------------------------- eval env
def len_class_dt(L, p):
    if p['dt'] > 0:
        return int(p['dt'])
    return int({300: p['dt_5m'], 900: p['dt_15m'], 14400: p['dt_4h']}.get(int(L), p['dt_other']))


def expected_steps(m, p):
    """Decisions of an episode from row 0 (mm3.h m3_step: dt per length class, dt_far while more
    than far_s seconds remain to the event)."""
    L = m['n_rows'] - 1
    dt = len_class_dt(L, p)
    if p['dt_far'] <= 0:
        return int(math.ceil(L / dt))
    ev_off = m['event_ts'] - m['open_ts']
    t = n = 0
    while t < L:
        d = int(p['dt_far']) if ev_off - t * m['row_s'] > p['far_s'] else dt
        t = min(t + d, L)
        n += 1
    return n


def len_bit(L):
    return {300: 1, 900: 2, 14400: 4}.get(int(L), 8)


class MarketSetEnv3:
    """Every market of a split once, from row 0 (train/envs.py MarketSetEnv for mm3)."""

    def __init__(self, mod, markets, split, seed=0, **kw):
        import gymnasium
        n = len(markets)
        obs_size = int(kw.get('obs_size', mod.DEFAULTS['obs_size']))
        self.obs_size = obs_size
        self.num_agents = n
        self.single_observation_space = gymnasium.spaces.Box(-5.0, 5.0, (obs_size,), np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete(mod.ACT_SIZES)
        self.observations = np.zeros((n, obs_size), np.float32)
        self.actions = np.zeros((n, len(mod.ACT_SIZES)), np.int32)
        self.rewards = np.zeros(n, np.float32)
        self.terminals = np.zeros(n, bool)
        self.truncations = np.zeros(n, bool)
        self.masks = np.ones(n, bool)
        self.markets = markets
        self.envs = []
        for j in range(n):
            buf = {k: getattr(self, k)[j:j + 1] for k in
                   ('observations', 'actions', 'rewards', 'terminals', 'truncations', 'masks')}
            self.envs.append(mod.MmEnv3(num_envs=1, seed=j, split=split, cycle_stride=n, fixed_start=0,
                                        buf=buf, **kw))
        p = self.envs[0].params
        self.expected = np.array([expected_steps(m, p) for m in markets])
        self.binding = self.envs[0]._binding
        self.active = np.ones(n, bool)
        self.records = []

    def reset(self, seed=0):
        for j, e in enumerate(self.envs):
            self.binding.vec_reset(e.c_envs, j)
        if self.obs_size >= OBS_CAT + 3:
            got = np.argmax(self.observations[:, OBS_CAT:OBS_CAT + 3], 1)
            want = np.array([m['category'] for m in self.markets])
            assert np.array_equal(got, np.minimum(want, 2)), 'eval env: market order differs from mm3 list'
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
            assert int(round(log['episode_rows'])) == m['n_rows'] - 1, (i, log['episode_rows'], m['n_rows'])
            assert int(round(log['episode_length'])) == self.expected[i], (i, log['episode_length'], self.expected[i])
            self.records.append({'env': int(i), 'market': int(m['idx']),
                                 **{k: m[k] for k in ('contract_len_s', 'open_ts', 'end_ts', 'event_ts', 'category',
                                                      'category_name', 'league', 'group', 'date', 'slug',
                                                      'n_rows', 'rw_rate_day', 'bucket')},
                                 **{k: float(v) for k, v in log.items() if k != 'n'}})
            self.active[i] = False
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def close(self):
        for e in self.envs:
            e.close()


# ----------------------------------------------------------------------------- adapter
class V3Adapter:
    """train/ adapter for v3env/env3.py (MmEnv3). Data comes from env kwargs `data_dir` (a build
    made loadable by adapt3.py) and optionally `split_map` (split3.py output); both are part of
    the config hash through identity()."""
    name = 'v3'
    variants = dict(RealAdapter.variants)
    final_variants = ('central_1s', 'pess_1s')
    native_probes = {'join': {'reward_mode': 1}}
    action_labels = REAL_LABELS
    join = 3

    def __init__(self, allow_dry_run_split=False):
        import env3
        self.mod = env3
        self.nvec = tuple(env3.ACT_SIZES)
        self.obs_size = env3.OBS_SIZE
        self.group_names = ('all',)
        self.allow_dry_run_split = allow_dry_run_split
        self._views = {}
        self._tables = {}

    # -- data resolution
    def resolve(self, kw):
        """(env kwargs for MmEnv3 without data_dir / split_map, the directory mm3 opens, the build)."""
        kw = dict(kw)
        build = kw.pop('data_dir', None)
        smap = kw.pop('split_map', None)
        if not build:
            raise ValueError('V3Adapter needs data_dir (a builder output made loadable by v3env/adapt3.py)')
        no_jun15(build)
        no_jun15(smap)
        build = os.path.abspath(build)
        if not os.path.exists(os.path.join(build, 'markets3.bin')) and detect_kind(build) != 'crypto':
            raise FileNotFoundError(f'{build}: no markets3.bin; run python v3env/adapt3.py {build}')
        d = build
        if smap:
            k = (build, os.path.abspath(smap))
            if k not in self._views:
                self._views[k] = make_view(build, smap, self.allow_dry_run_split)
            d = self._views[k]
        no_jun15(d)
        if 'obs_size' in kw:
            self.obs_size = int(kw['obs_size'])
        return kw, d, build

    def table(self, d):
        if d not in self._tables:
            self._tables[d] = market_table(d)
        return self._tables[d]

    def _guard(self, split, kw):
        if int(split) == TEST_SPLIT:
            if os.environ.get('MM_FINAL') != '1':
                raise PermissionError('train3/ touches the test split only with MM_FINAL=1 (DECISIONS.md D11: '
                                      'each track\'s test is run once)')
            kw['allow_test'] = 1
        else:
            kw.pop('allow_test', None)

    def identity(self, kw):
        kw2, d, build = self.resolve(kw)
        p = self.mod.params(**{k: v for k, v in kw2.items() if k not in ('split', 'allow_test')})
        p.pop('split', None)
        p.pop('allow_test', None)
        code = v3env_code_hashes()
        for f in DATA_FILES:
            if os.path.exists(os.path.join(d, f)):
                code[f'data:{f}'] = cached_sha256(os.path.join(d, f))
        p['data_dir'] = build
        if kw.get('split_map'):
            v = json.load(open(os.path.join(d, 'view3.json')))
            code['split_map'] = v['split_sha256']
            p['split_map'] = v['split_file']
        return {'params': p, 'code': code}

    def split_markets(self, split, d, kw):
        p = self.mod.params(**{k: v for k, v in kw.items() if k not in ('split',)})
        lm, cm = int(p['len_mask']), int(p['cat_mask'])
        return [m for m in self.table(d) if m['split'] == split and len_bit(m['n_rows'] - 1) & lm
                and (1 << m['category']) & cm]

    def make_train(self, num_envs, seed=0, **kw):
        kw.pop('split', None)
        kw, d, _ = self.resolve(kw)
        self._guard(0, kw)
        self._check_layout(d)
        env = self.mod.MmEnv3(num_envs=num_envs, seed=seed, split=0, data_dir=d, **kw)
        self.obs_size = env.obs_size
        self.group_names = CAT_NAMES[:3] if env.obs_size >= OBS_CAT + 3 else ('all',)
        return env

    def make_eval(self, split=1, seed=0, **kw):
        for k in ('split', 'fixed_market', 'fixed_start', 'cycle_stride'):
            kw.pop(k, None)
        kw, d, _ = self.resolve(kw)
        self._guard(split, kw)
        mk = self.split_markets(split, d, kw)
        if not mk:
            raise ValueError(f'no markets of split {split} in {d}')
        return MarketSetEnv3(self.mod, mk, split, seed=seed, data_dir=d, **kw), len(mk)

    def eval_records(self, env):
        recs, env.records = env.records, []
        return recs

    def group_fn(self, env):
        w = int(np.prod(env.single_observation_space.shape))
        if w >= OBS_CAT + 3:
            return lambda obs: np.argmax(obs[:, OBS_CAT:OBS_CAT + 3], axis=1)
        return lambda obs: np.zeros(len(obs), np.int64)

    def close_risk(self, obs):
        raise NotImplementedError('v3: close-inventory shaping is in the core (env kwargs close_inv_coef, '
                                  'close_ramp, close_ramp_s); train_ppo3.py maps --close-inv-coef to it')

    def teacher(self, name='mid_value', params=None, env_kw=None):
        """A v3 rule baseline (evaluator3.BASELINES) as an obs -> actions function."""
        import evaluator3
        return evaluator3.make_baseline(name, params, env_kw or {})

    def _check_layout(self, d):
        ref = os.path.join(PM, 'data', 'build', 'feat_meta.json')
        mine = os.path.join(d, 'feat_meta.json')
        if os.path.exists(ref) and os.path.exists(mine):
            if json.load(open(ref))['column_index'] != json.load(open(mine))['column_index']:
                raise ValueError(f'{d}: feat.bin column layout differs from data/build')


def get_adapter(name='v3', **kw):
    if name not in ('v3', 'auto'):
        raise ValueError(f'train3 only drives the v3 env, not {name!r}')
    return V3Adapter(**kw)
