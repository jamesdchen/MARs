"""v3 evaluation: per-market and per-category metrics, the protocol flags (D1.3, T5 / D2, T6 / D13),
a separate reward-income line for elections, and the comparators of D11 / D1 adapted to v3.

    evaluate(policy_or_fn, adapter, split, env_kw)  -> summary (and per-market records)
    pooled_verdict([summary_seed0, summary_seed1, ...]) -> D1.3 on seed-pooled attribution
    make_baseline(name, params, env_kw)              -> obs -> actions function

CLI (every val evaluation appends one row to the ledger, default train3/trials.jsonl):

    python evaluator3.py comparators --data-dir BUILD [--split-map F] [--env3-kw k=v ...] [--names ...]
        [--pull-grid [S ...]]       # D15 sweep: each comparator at env pull_s in S (default 0 300 900 1800 3600)
    python evaluator3.py policy --ckpt RUN/final.pt --data-dir BUILD [--split-map F] [--env3-kw ...]
    python evaluator3.py pool RUN_DIR [RUN_DIR ...]          # seed-pooled D1.3 of several result.json

Metrics. `summarize3(records)` returns everything train/evaluator.summarize does (P&L totals, CVaR,
sums / means of every Log field) plus:
- `income/*`: spread_capture, markout_60 (adverse selection), inventory (incl. the flattening
  cost), rebates and **rewards** (the liquidity-reward hook's income), with `income/trading` =
  spread + markout + inventory and `income/residual` = pnl - sum of the five (0 up to float32);
  for elections the rewards line is the reward-farming income, reported apart from trading;
- `t5/*`: the share of fill P&L (spread capture + 60 s markout, the env's pnl_<flag>) under each
  artifact flag of D2 (inspread, crossed, stale, onesided, last30; the Log cannot form the
  crossed-or-stale union, so its parts are judged separately, as train/evaluator.py does), the
  worst one and `t5/invalid` (> 25 %); the through-the-touch flag `outside` is reported only;
- `t6/*`: final/reduce.py `directional_verdict` on the attribution totals (imported, not copied);
- `d13/*`: fills/market >= 5, not directional, T5 valid -> `d13/makes_markets`; `d12/active`: the D1.2
  activity requirement of a secondary comparator (>= 5 fills / market, two-sided uptime >= 0.25);
- `net/*` (DECISIONS.md D15, analysis/v3_sports_rules/RECONCILE.md section 6): gross filled shares,
  filled notional, |inventory| at the episode end (the settlement row: kickoff), the netted share
  1 - sum |end inv| / sum gross (pooled, `net/netted_share`) and its per-episode mean
  (`net/netted_share_ep`), and the flatten drag (flat_c x |end inv|, already in the P&L) in USD and
  in bps of the filled notional; from the env Log's net_* fields (v3env/mm3.h);
- `tk/<bucket>/*` (D15 time buckets, track S): P&L, steps and quoting share per time-to-kickoff
  bucket of the decision time: gt24h, 6_24h, 1_6h, 1h_30m, 30_5m, 5_0m (build_sports' > 24 h, 6-24 h,
  1-6 h, 1 h-30 m, 30-5 m, 5-0 m; from the observed wealth, adding up to the P&L exactly), `tk/lt1h/*`
  = the last three summed; and `d14/*`, the P&L marked at T - 1 h and flattened there (D14's
  withdrawn window, kept for comparison); see TimeBuckets;
- `group/<g>/*`, `cat/<c>/*` and `bucket/<b>/*`: the same per league group ('sports/nba',
  'elections'), per category and per episode bucket (the build's `segments[i].bucket`: the D15
  bucket holding most of the episode's decision rows; exact with --bucket-segments builds), and
  `per_market` rows (fills, P&L, components, netting, T5 / T6 per market) when asked.
For a multi-seed method, `pooled_verdict` averages each attribution component over the seeds
first and then applies directional_verdict and the T5 shares (DECISIONS.md D13 reading C).

D16 recording gaps (v3env/ENV3.md section 11a):
- gap chains are played as one continuous episode (envs3 MarketSetEnv3, gap_chain = 1 by default):
  each piece is a record with its own P&L increment, so per-market lines over a chain add up to the
  continuous episode (the price move across a gap on the carried inventory lands in the next
  piece; the time buckets credit it to that piece's first decision);
- val / test scoring (split 1 / 2) excludes the D16 items: pieces on a day that is not eligible
  (more than 30 min of gaps: train only) and sports pieces flagged excluded_from_eval_last_hour
  (a gap in the game's last pre-kickoff hour). The excluded pieces are still played (a chain's
  state runs through them); a last-hour-excluded piece's time buckets other than its
  eval_excluded_buckets still count in `tk/*` (D16: the earlier buckets remain usable);
- every result carries `gaps` (the build's per-day gap counts and seconds, the evaluated days) and
  `excl/*` (counts of excluded pieces and time-bucket cells), `chain/*` (chains played).

Comparators (D1 / D11: no-quote plus classical quoters; none is tuned here, the defaults below are
the untuned points; tuning on v3 val is a separate, ledgered step):
- `no_quote`;
- `sym_touch`: join both touches, pull the side whose fill would take |q| past `q_cap_lots` x the
  head-0 order size (reward min sizes run 20-1000 shares, so a cap in shares would never quote);
- `mid_value`: the fair-value quoter of baselines/baselines.py with the fair value replaced by the
  book mid (no Binance underlying in sports or elections): centre = mid - k_skew * q / lot,
  half-spread = spread_mult * max(tick, 1c, spread / 2), both sides pulled within `pull_s` seconds
  of the event (kickoff; elections: window end) and when the book is not two-sided;
- `as_mid`: Avellaneda-Stoikov in price units around the mid: reservation mid - q_lots * gamma *
  sig2 * h, half-spread gamma * sig2 * h / 2 + ln(1 + gamma / k) / gamma, h = min(h_risk, seconds to
  event);
- `reward_farmer` (elections): while the market is in a reward program, quote both sides at
  max_spread / 2 from the mid with size = the reward min size (size_mode 1 head 0; with
  size_mode 0 the smallest head >= min size), pulling the side that would take |q| past
  `inv_cap_lots` x min size. Needs obs_size >= 59 and liq_rewards = 1 (it reads the program from
  obs 55-58).
Every baseline reads the event clock from obs 50 (ev_log) and so runs with obs_size >= 80 (its
own eval env; dynamics do not depend on obs_size).
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import envs3  # noqa: E402  (puts train/ and v3env/ on sys.path)

PM = envs3.PM
for _d in (os.path.join(PM, 'baselines'),):
    if _d not in sys.path:
        sys.path.append(_d)

from evaluator import nan_to_none, summarize  # noqa: E402  (train/evaluator.py)
from puffer3 import action_histogram  # noqa: E402

LEDGER = os.path.join(HERE, 'trials.jsonl')
SMOKE_LEDGER = os.path.join(HERE, 'smoke_trials.jsonl')
T5_FLAGS = ('inspread', 'crossed', 'stale', 'onesided', 'last30')     # DECISIONS.md D2
REPORT_FLAGS = ('outside', 'last10', 'inside2', 'extreme')
COMPONENTS = (('spread_capture', 'spread_capture'), ('adverse_selection', 'markout_60'),
              ('inventory', 'inventory_pnl'), ('rebates', 'rebates'), ('rewards', 'liq_rewards'))
MIN_FILLS = 5.0
MAX_FLAG_SHARE = 0.25


def _load_reduce():
    """final/reduce.py, imported under a unique name (final/ appended to sys.path for its
    `from common import ...`), so T6 is exactly its directional_verdict."""
    name = 'pm_final_reduce'
    if name in sys.modules:
        return sys.modules[name]
    fdir = os.path.join(PM, 'final')
    if fdir not in sys.path:
        sys.path.append(fdir)
    spec = importlib.util.spec_from_file_location(name, os.path.join(fdir, 'reduce.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


REDUCE = _load_reduce()
directional_verdict = REDUCE.directional_verdict
assert REDUCE.MAX_INVENTORY_SHARE == 0.5 and REDUCE.MIN_FILLS_PER_MARKET == MIN_FILLS
assert REDUCE.MAX_FLAG_SHARE == MAX_FLAG_SHARE


# ----------------------------------------------------------------------------- protocol flags
def attribution(tot):
    """final/reduce.py attribution totals from summed Log fields ({field: sum})."""
    out = {name: float(tot.get(field, 0.0)) for name, field in COMPONENTS}
    out['total'] = float(tot.get('pnl', 0.0))
    return out


def verdicts(tot, n_markets):
    """T6 / T5 / D1.3 from summed (or seed-averaged) Log fields over n_markets markets."""
    att = attribution(tot)
    dv = directional_verdict(att)
    out = {f't6/{k}': (float(v) if not isinstance(v, bool) else v) for k, v in dv.items()}
    fill_pnl = float(tot.get('spread_capture', 0.0) + tot.get('markout_60', 0.0))
    shares = {}
    for f in T5_FLAGS + REPORT_FLAGS:
        if f'pnl_{f}' in tot:
            shares[f] = float(tot[f'pnl_{f}']) / fill_pnl if fill_pnl > 0 else float('nan')
            out[f't5/share/{f}'] = shares[f]
            vol = float(tot.get('fill_vol', 0.0))
            out[f't5/vol_share/{f}'] = float(tot.get(f'vol_{f}', 0.0)) / vol if vol > 0 else float('nan')
    rule = {f: shares[f] for f in T5_FLAGS if f in shares and shares[f] == shares[f]}
    worst = max(rule, key=rule.get) if rule else None
    out['t5/worst_flag'] = worst
    out['t5/worst_share'] = rule[worst] if worst else float('nan')
    out['t5/invalid'] = bool(worst and rule[worst] > MAX_FLAG_SHARE)
    fpm = float(tot.get('fills', 0.0)) / n_markets if n_markets else 0.0
    ok = {'fills': fpm >= MIN_FILLS, 't6': not dv['directional'], 't5': not out['t5/invalid']}
    out['d13/fills_per_market'] = fpm
    out.update({f'd13/ok_{k}': v for k, v in ok.items()})
    out['d13/makes_markets'] = all(ok.values())
    out.update(income(tot))
    out.update(netting(tot))
    return out


def netting(tot):
    """D15 netting / flatten lines from summed Log fields (v3env/env3.py netting, prefixed net/)."""
    import env3
    out = {f'net/{k}': v for k, v in env3.netting(tot).items()}
    out['net/notional'] = float(tot.get('net_notional', 0.0))
    return out


def income(tot):
    """The P&L lines: trading (spread + markout + inventory), rebates, liquidity-reward income."""
    sc, mo, inv = (float(tot.get(k, 0.0)) for k in ('spread_capture', 'markout_60', 'inventory_pnl'))
    rb, rw, pnl = float(tot.get('rebates', 0.0)), float(tot.get('liq_rewards', 0.0)), float(tot.get('pnl', 0.0))
    return {'income/spread_capture': sc, 'income/markout_60': mo, 'income/inventory': inv,
            'income/flat_cost': float(tot.get('flat_cost', 0.0)), 'income/trading': sc + mo + inv,
            'income/rebates': rb, 'income/rewards': rw, 'income/total': pnl,
            'income/residual': pnl - (sc + mo + inv + rb + rw),
            'income/rw_share_mean': (float(tot.get('rw_share', 0.0)) / float(tot['rw_scored'])
                                     if tot.get('rw_scored') else float('nan'))}


def _sums(recs):
    keys = [k for k, v in recs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {k: float(sum(r.get(k, 0.0) for r in recs)) for k in keys}


def per_market(recs):
    """One row per market: identity, P&L lines, fills, uptime and its own T5 / T6 numbers."""
    rows = []
    for r in recs:
        v = verdicts(r, 1)
        rows.append({k: r.get(k) for k in ('market', 'slug', 'group', 'category_name', 'bucket', 'date', 'n_rows',
                                             'pnl', 'fills', 'fill_vol', 'max_abs_inv', 'final_abs_inv',
                                             'uptime_two_sided', 'uptime_any', 'settle_px', 'rw_scored',
                                             'chain_id', 'chain_pos', 'chain_end', 'gap_cut', 'after_gap',
                                             'eval_excl_last_hour', 'eval_excluded_buckets', 'day_eligible')}
                    | {k.split('/', 1)[1]: v[k] for k in ('income/trading', 'income/rebates', 'income/rewards',
                                                          'income/spread_capture', 'income/markout_60',
                                                          'income/inventory', 't6/share_of_total',
                                                          't6/directional', 't5/worst_flag', 't5/worst_share')}
                    | {k.replace('/', '_'): v[k] for k in NET_KEYS})
    return rows


def summarize3(recs, tk_extra=()):
    """tk_extra: records whose time-bucket lines count in tk/* (and the group tk lines) but which are
    not scored otherwise (D16: last-hour-excluded pieces with their excluded buckets zeroed)."""
    recs = list(recs)
    tk_extra = list(tk_extra)
    out = summarize(recs)                       # train/evaluator.py (P&L, CVaR, sum/ mean/ fields)
    out = {k: v for k, v in out.items() if not k.startswith(('len/', 't5_', 't6_', 'makes_markets'))}
    if not recs:
        out.update(time_buckets(tk_extra))
        return out
    out.update(verdicts(_sums(recs), len(recs)))
    out.update(time_buckets(recs + tk_extra))
    for key, prefix in (('group', 'group'), ('category_name', 'cat'), ('bucket', 'bucket')):
        for g in sorted({r[key] for r in recs if r.get(key) is not None}):
            rg = [r for r in recs if r.get(key) == g]
            rx = [r for r in tk_extra if r.get(key) == g]
            pnl = np.array([r['pnl'] for r in rg])
            v = verdicts(_sums(rg), len(rg))
            out.update({f'{prefix}/{g}/n': len(rg), f'{prefix}/{g}/pnl_total': float(pnl.sum()),
                        f'{prefix}/{g}/pnl_mean': float(pnl.mean()),
                        f'{prefix}/{g}/fills_per_market': v['d13/fills_per_market'],
                        f'{prefix}/{g}/t6_share': v['t6/share_of_total'], f'{prefix}/{g}/t6_directional': v['t6/directional'],
                        f'{prefix}/{g}/t5_worst_share': v['t5/worst_share'],
                        f'{prefix}/{g}/makes_markets': v['d13/makes_markets'],
                        **{f'{prefix}/{g}/{k}': v[k] for k in ('income/trading', 'income/rebates', 'income/rewards')},
                        **{f'{prefix}/{g}/{k}': v[k] for k in NET_KEYS},
                        **{f'{prefix}/{g}/{k}': v for k, v in time_buckets(rg + rx).items()}})
    # train_ppo.py / rounds.py read these names
    out['fills_per_market'] = out['d13/fills_per_market']
    out['t6_share'] = out['t6/share_of_total']
    out['t5_max_share'] = out['t5/worst_share']
    out['makes_markets'] = float(out['d13/makes_markets'])
    # D1.2: a secondary comparator must keep trading (>= 5 fills / market, two-sided uptime >= 0.25)
    out['d12/active'] = bool(out['d13/fills_per_market'] >= MIN_FILLS and out.get('mean/uptime_two_sided', 0.0) >= 0.25)
    return out


NET_KEYS = ('net/gross_filled_sh', 'net/end_abs_inv', 'net/netted_share', 'net/netted_share_ep',
            'net/flat_drag_usd', 'net/flat_drag_bps')


def time_buckets(recs):
    """D15 lines: P&L, decision steps and quoting steps per time-to-event bucket (and the last hour
    summed), and the P&L marked / flattened at T - 1 h (TimeBuckets; absent when obs_size <= 50)."""
    if not recs or f'tk_pnl_{TK_BUCKETS[-1][0]}' not in recs[0]:
        return {}
    s = lambda k: float(sum(r.get(k, 0.0) for r in recs))  # noqa: E731
    out = {}
    for name, members in [(n, (n,)) for n, _ in TK_BUCKETS] + [('lt1h', TK_LAST_HOUR)]:
        out[f'tk/{name}/pnl_total'] = sum(s(f'tk_pnl_{m}') for m in members)
        st = sum(s(f'tk_steps_{m}') for m in members)
        out[f'tk/{name}/steps'] = st
        out[f'tk/{name}/quote_frac'] = sum(s(f'tk_quote_steps_{m}') for m in members) / st if st else float('nan')
    out['d14/pnl_marked_T1h_total'] = s('d14_pnl_marked_T1h')
    out['d14/pnl_flat_T1h_total'] = s('d14_pnl_flat_T1h')
    out['d14/abs_inv_T1h_mean'] = float(np.mean([abs(r['d14_inv_T1h']) for r in recs]))
    out['d14/markets_reaching_T1h'] = s('d14_reached_T1h')
    out['tk/clip_steps'] = s('tk_clip_steps')
    return out


def pooled_verdict(summaries):
    """D1.3 of a multi-seed method on seed-pooled attribution (D13 reading C): every summed Log
    field is averaged over the seeds, then T6 (directional_verdict), T5 and fills/market."""
    n = [s['n_markets'] for s in summaries]
    if len(set(n)) != 1:
        raise ValueError(f'seeds evaluated on different market sets: {n}')
    keys = sorted({k[4:] for s in summaries for k in s if k.startswith('sum/')})
    tot = {k: float(np.mean([s.get(f'sum/{k}', 0.0) for s in summaries])) for k in keys}
    tot['pnl'] = float(np.mean([s['pnl_total'] for s in summaries]))
    v = verdicts(tot, n[0])
    pnl = [s['pnl_total'] for s in summaries]
    v.update(n_seeds=len(summaries), pnl_mean_over_seeds=float(np.mean(pnl)),
             pnl_se=float(np.std(pnl, ddof=1) / math.sqrt(len(pnl))) if len(pnl) > 1 else float('nan'),
             per_seed=pnl)
    return v



# ----------------------------------------------------------------------------- D15 time buckets
# (name, lower bound in seconds to the event): a decision belongs to the first bucket whose bound it
# exceeds, so bounds are upper-inclusive as build_sports.bucket_of (1800 s is 30_5m, 300 s is 5_0m)
TK_BUCKETS = (('gt24h', 86400.0), ('6_24h', 21600.0), ('1_6h', 3600.0), ('1h_30m', 1800.0), ('30_5m', 300.0),
              ('5_0m', -1.0))
TK_LAST_HOUR = ('1h_30m', '30_5m', '5_0m')
assert tuple(n for n, _ in TK_BUCKETS) == tuple(n for n, _, _ in envs3.BUCKETS)   # record 'bucket' uses the same names
T1H = 3600.0
OBS_INV, OBS_WEALTH = 33, 34          # ENV.md section 6: inventory / Q, clip(W / (0.05 Q), +-5)
PULL_GRID = (0.0, 300.0, 900.0, 1800.0, 3600.0)              # D15 candidate pull times (and 0 = at kickoff)


class TimeBuckets:
    """DECISIONS.md D15: P&L per time-to-kickoff bucket (> 24 h, 6-24 h, 1-6 h, 1 h-30 m, 30-5 m,
    5-0 m) and D14's P&L marked (and flattened) at T - 1 h, from the observations of the eval loop.

    Each step's change of the observed wealth W = cash + rebates + rewards + inventory x mark (obs 34)
    is credited to the bucket of the step's decision time (obs 50, seconds to the event). The
    terminal step is not observed (auto-reset), so the market's Log P&L minus the observed sum goes
    to the last step's bucket: the buckets add up to the P&L exactly. W is clipped at +-0.25 Q in
    the observation: steps at the clip are counted (`tk_clip_steps`) and make the split between
    buckets (not the total) unreliable; use Q >= 1000. `d14_pnl_marked_T1h` is W at the first
    decision within 1 h of the event (the inventory marked at the mid there; the whole P&L if the
    episode never gets that close), `d14_inv_T1h` the inventory then, and `d14_pnl_flat_T1h` the
    marked P&L minus |inventory| x flat_c / 100 (the env's flattening cost). A step counts as quoting
    when the action quotes a side and the env's pull_s has not forced the quotes off. Needs obs_size > 50."""

    def __init__(self, env, N):
        p = env.envs[0].params
        self.on = env.obs_size > envs3.OBS_EV
        self.Q, self.flat = float(p['Q']), float(p['flat_c']) / 100.0
        self.pull = float(p['pull_s'])                # the env forces quotes off within pull_s of the event
        nb = len(TK_BUCKETS)
        self.pnl = np.zeros((N, nb))
        self.steps = np.zeros((N, nb))
        self.qsteps = np.zeros((N, nb))
        self.clip = np.zeros(N)
        self.w1h = np.full(N, np.nan)
        self.inv1h = np.full(N, np.nan)
        self.last_b = np.zeros(N, np.int64)
        self.carry = np.zeros(N)             # D16 chains: absolute wealth at the end of the handle's last piece
        self.fresh = np.zeros(N, bool)       # the handle just entered the next piece of its chain

    def before(self, obs, act, active):
        if not self.on:
            return
        sec = np.expm1(obs[:, envs3.OBS_EV].astype(np.float64) * LOG86400)
        b = np.select([sec > lo for _, lo in TK_BUCKETS[:-1]], list(range(len(TK_BUCKETS) - 1)), len(TK_BUCKETS) - 1)
        self.b, self.w0 = b, obs[:, OBS_WEALTH].astype(np.float64) * 0.05 * self.Q
        j = np.flatnonzero(active & self.fresh)  # the move across the gap on the carried inventory
        self.pnl[j, b[j]] += self.w0[j] - self.carry[j]
        self.fresh[j] = False
        self.clip += active & (np.abs(obs[:, OBS_WEALTH]) >= 4.999)
        i = np.flatnonzero(active)
        self.steps[i, b[i]] += 1
        q = ((act[:, 0] != 0) | (act[:, 1] != 0)) & ~((self.pull > 0) & (sec <= self.pull))
        self.qsteps[i, b[i]] += q[i]
        first = active & (sec <= T1H) & np.isnan(self.w1h)
        self.w1h[first] = self.w0[first]
        self.inv1h[first] = obs[first, OBS_INV].astype(np.float64) * self.Q
        self.last_b[i] = b[i]

    def after(self, obs, term, active):
        if not self.on:
            return
        i = np.flatnonzero(active & ~term)
        w1 = obs[:, OBS_WEALTH].astype(np.float64) * 0.05 * self.Q
        self.pnl[i, self.b[i]] += w1[i] - self.w0[i]

    def finish(self, i, pnl, cont=False):
        """The record lines of handle i's finished piece; `cont`: its chain goes on (D16), so the handle's
        rows restart and the next piece's lines are measured from this piece's end wealth."""
        if not self.on:
            return {}
        self.pnl[i, self.last_b[i]] += pnl - self.pnl[i].sum()
        out = {}
        for k, (name, _) in enumerate(TK_BUCKETS):
            out[f'tk_pnl_{name}'] = float(self.pnl[i, k])
            out[f'tk_steps_{name}'] = float(self.steps[i, k])
            out[f'tk_quote_steps_{name}'] = float(self.qsteps[i, k])
        reached = not np.isnan(self.w1h[i])
        out['d14_reached_T1h'] = float(reached)
        out['d14_pnl_marked_T1h'] = float(self.w1h[i] - self.carry[i]) if reached else float(pnl)
        out['d14_inv_T1h'] = float(self.inv1h[i]) if reached else 0.0
        out['d14_pnl_flat_T1h'] = out['d14_pnl_marked_T1h'] - (abs(self.inv1h[i]) * self.flat if reached else 0.0)
        out['tk_clip_steps'] = float(self.clip[i])
        self.pnl[i] = self.steps[i] = self.qsteps[i] = 0.0
        self.clip[i] = 0.0
        self.w1h[i] = self.inv1h[i] = np.nan
        if cont:
            self.carry[i] += pnl
            self.fresh[i] = True
        return out

# ----------------------------------------------------------------------------- evaluation
def evaluate(policy, adapter, split=1, env_kw=None, seed=0, device='cpu', max_steps=None,
             return_records=False, with_markets=False):
    """train/evaluator.py evaluate() for the v3 adapter; `policy` is a torch policy (greedy
    actions) or a function obs -> int actions (B, 3)."""
    import torch
    from policy import greedy_actions
    t0 = time.perf_counter()
    env, n_markets = adapter.make_eval(split, seed, **(env_kw or {}))
    obs, _ = env.reset(seed)
    N = env.num_agents
    is_torch = hasattr(policy, 'forward_eval')
    recurrent = is_torch and getattr(policy, 'recurrent', False)
    state = {'lstm_h': None} if recurrent else None
    done = torch.zeros(N)
    records, steps = {}, 0
    nvec = tuple(int(n) for n in env.single_action_space.nvec)
    hist = [np.zeros(n) for n in nvec]
    tk = TimeBuckets(env, N)
    was = policy.training if is_torch else None
    if is_torch:
        policy.eval()
    try:
        with torch.no_grad():
            while len(records) < n_markets:
                if is_torch:
                    if recurrent:
                        state['done'] = done
                    logits, _ = policy.forward_eval(torch.as_tensor(obs).to(device), state)
                    act = greedy_actions(logits).cpu().numpy()
                else:
                    act = np.asarray(policy(obs), np.int32)
                active = env.active.copy()           # handles still playing (a gap chain spans pieces)
                for h, n in enumerate(nvec):
                    hist[h] += np.bincount(act[active, h], minlength=n)[:n]
                tk.before(obs, act, active)
                obs, _, term, _, _ = env.step(act)
                tk.after(obs, np.asarray(term, bool), active)
                done = torch.as_tensor(np.asarray(term, dtype=np.float32))
                for rec in adapter.eval_records(env):
                    if rec['market'] not in records:
                        rec.update(tk.finish(rec['env'], rec['pnl'], cont=not rec.get('chain_end', True)))
                        records[rec['market']] = rec
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    raise RuntimeError(f'eval did not finish {n_markets} markets in {max_steps} steps')
    finally:
        if is_torch:
            policy.train(was)
        env.close()
    recs = [records[k] for k in sorted(records)]
    scored, partial, excl = d16_filter(recs, split)
    out = summarize3(scored, partial)
    out.update(excl)
    out.update(chain_report(recs))
    out['gaps'] = gaps_report(adapter, env_kw, recs)
    out.update(action_histogram(hist, getattr(adapter, 'action_labels', None), prefix='act/'))
    out['eval_steps'] = steps
    out['eval_seconds'] = time.perf_counter() - t0
    if with_markets:
        ids = {id(r) for r in scored}
        out['per_market'] = [dict(row, scored=id(r) in ids) for r, row in zip(recs, per_market(recs))]
    return (out, recs) if return_records else out


# ----------------------------------------------------------------------------- D16 filters and reports
def d16_filter(recs, split):
    """(scored, tk-only, excl/* counts). On val / test (split 1 / 2) a piece on a D16-ineligible day
    (train only) is not scored at all, and a sports piece flagged excluded_from_eval_last_hour is not
    scored; its time-bucket lines outside eval_excluded_buckets still count in tk/* (D16: the earlier
    buckets remain usable). On train every piece is scored."""
    out = {'excl/day_ineligible': 0, 'excl/last_hour': 0, 'excl/total': 0, 'excl/tk_cells': 0,
           'excl/pnl_total': 0.0, 'excl/markets': []}
    if int(split) not in (1, 2):
        return list(recs), [], out
    scored, partial = [], []
    for r in recs:
        if r.get('day_eligible', True) is False:
            out['excl/day_ineligible'] += 1
        elif r.get('eval_excl_last_hour'):
            out['excl/last_hour'] += 1
            p = dict(r)
            for b in r.get('eval_excluded_buckets') or ():
                if f'tk_pnl_{b}' in p:
                    out['excl/tk_cells'] += int(p.get(f'tk_steps_{b}', 0.0) > 0)
                    p[f'tk_pnl_{b}'] = p[f'tk_steps_{b}'] = p[f'tk_quote_steps_{b}'] = 0.0
            partial.append(p)
        else:
            scored.append(r)
            continue
        out['excl/total'] += 1
        out['excl/pnl_total'] += float(r.get('pnl', 0.0))
        out['excl/markets'].append(r.get('slug'))
    out['excl/n_played'] = len(recs)
    return scored, partial, out


def chain_report(recs):
    """chain/*: gap chains played (D16), pieces in them, their summed P&L."""
    ch = {}
    for r in recs:
        if r.get('chain_len', 1) > 1:
            ch.setdefault(r['chain_id'], []).append(r)
    return {'chain/n_chains': len(ch), 'chain/n_pieces': sum(len(v) for v in ch.values()),
            'chain/pnl_total': float(sum(r['pnl'] for v in ch.values() for r in v))}


def gaps_report(adapter, env_kw, recs):
    """D16: every result reports the gaps: the build's per-day gap counts and seconds (meta3.json gaps,
    else the builder's fields), the days of the evaluated markets, and the ineligible ones."""
    try:
        _, d, _ = adapter.resolve(env_kw or {})
        g = envs3.gap_summary(d)
    except Exception as e:                      # noqa: BLE001  (a report must not stop an evaluation)
        return {'error': str(e)}
    if not g:
        return {'source': None, 'days': {}, 'note': 'build without D16 gap fields'}
    days = {k: {f: v.get(f) for f in ('n_gaps', 'total_gap_s', 'eligible')} for k, v in (g.get('days') or {}).items()}
    seen = sorted({x for r in recs for x in envs3._utc_days(r['open_ts'], r['end_ts'])})
    return {'source': g.get('schema'), 'days': days, 'n_gaps_total': g.get('n_gaps_total'),
            'total_gap_s': g.get('total_gap_s'), 'days_ineligible': g.get('days_ineligible', []),
            'evaluated_days': {x: days.get(x, {'n_gaps': 0, 'total_gap_s': 0.0, 'eligible': True}) for x in seen},
            'n_links': g.get('n_links', 0)}


# ----------------------------------------------------------------------------- baselines (v3)
LOG86400 = math.log1p(86400.0)
BASELINE_DEFAULTS = {
    'no_quote': {},
    'sym_touch': {'size': 0, 'q_cap_lots': 4.0},
    'mid_value': {'k_skew': 0.001, 'spread_mult': 1.0, 'pull_s': 300.0, 'size': 0},
    'as_mid': {'gamma': 10.0, 'k': 100.0, 'sig2': 3e-8, 'h_risk': 3600.0, 'pull_s': 300.0, 'size': 0},
    'reward_farmer': {'inv_cap_lots': 2.0, 'pull_s': 0.0},
}
TRACK_COMPARATORS = {'sports': ('no_quote', 'sym_touch', 'mid_value', 'as_mid'),
                     'elections': ('no_quote', 'sym_touch', 'mid_value', 'as_mid', 'reward_farmer')}


class _Decoder:
    """Obs (v2 slots via baselines/obs_spec.py, the documented feat_meta / ENV.md mapping) plus the
    v3 slots of ENV3.md section 7: seconds to the event (obs 50) and the reward program (55-58)."""

    def __init__(self, env_kw):
        import env3
        from obs_spec import load_spec
        p = env3.params(**{k: v for k, v in env_kw.items() if k in env3.DEFAULTS})
        self.p = p
        self.Q = p['Q']
        self.spec = load_spec(env_params={'Q': self.Q})
        self.size_mode = int(p['size_mode'])
        self.rel = np.array([p['size_rel0'], p['size_rel1'], p['size_rel2']]) if self.size_mode else np.array([1., 2., 5.])
        self.mult = p['size_mult']
        self.liq = p['liq_rewards'] > 0

    def __call__(self, obs):
        obs = np.asarray(obs)
        st = self.spec.decode(obs)
        w = obs.shape[1]
        if w <= envs3.OBS_EV:
            raise ValueError('v3 baselines read the event clock (obs 50): evaluate them with obs_size >= 80')
        st['tau'] = np.expm1(obs[:, envs3.OBS_EV].astype(np.float64) * LOG86400)   # seconds to the event
        mo = st['min_size']                                                         # min_order_size (5)
        if w > 58:
            st['rw_active'] = obs[:, 55] > 0.5
            st['rw_v_c'] = 5.0 * obs[:, 57].astype(np.float64)
            st['rw_min'] = np.where(st['rw_active'], np.expm1(3.0 * obs[:, 58].astype(np.float64)) * mo, 0.0)
        else:
            st['rw_active'] = np.zeros(len(obs), bool)
            st['rw_v_c'] = st['rw_min'] = np.zeros(len(obs))
        base = np.maximum(st['rw_min'], mo) if self.size_mode else mo
        st['sizes'] = self.mult * base[:, None] * self.rel[None, :]                 # shares per size head
        return st


def make_baseline(name, params=None, env_kw=None):
    import baselines as bl
    p = {**BASELINE_DEFAULTS[name], **(params or {})}
    dec = _Decoder(env_kw or {})
    amap = bl.ActionMap(dec.spec)
    q_max = dec.Q

    def sized(st, k):
        return st['sizes'][np.arange(len(st['q'])), np.broadcast_to(k, st['q'].shape)]

    if name == 'no_quote':
        return bl.no_quote(dec.spec)

    if name == 'sym_touch':
        def fn(obs):
            st = dec(obs)
            sz = sized(st, int(p['size']))
            cap = np.minimum(p['q_cap_lots'] * st['sizes'][:, 0], q_max)
            a = np.empty((len(st['q']), 3), np.int32)
            a[:, 0] = np.where(st['bid_ok'] & (st['q'] + sz <= cap + 1e-9), amap.join, amap.off)
            a[:, 1] = np.where(st['ask_ok'] & (st['q'] - sz >= -cap - 1e-9), amap.join, amap.off)
            a[:, 2] = int(p['size'])
            return a
        return fn

    if name == 'mid_value':
        def fn(obs):
            st = dec(obs)
            both, mid, spread = bl._book(st)
            k = int(p['size'])
            sz = sized(st, k)
            h = p['spread_mult'] * np.maximum(np.maximum(st['tick'], 0.01), 0.5 * spread)
            c = np.clip(np.nan_to_num(mid, nan=0.5), 0.01, 0.99) - p['k_skew'] * st['q'] / np.maximum(sz, 1e-9)
            live = both & st['two'] & (st['tau'] >= p['pull_s'])
            bid, ask = np.where(live, c - h, np.nan), np.where(live, c + h, np.nan)
            bid, ask = bl.inventory_limit(bid, ask, st['q'], sz, q_max)
            return bl.to_actions(amap, st, bid, ask, k)
        return fn

    if name == 'as_mid':
        g, kk = float(p['gamma']), float(p['k'])
        base = math.log1p(g / kk) / g

        def fn(obs):
            st = dec(obs)
            both, mid, _ = bl._book(st)
            k = int(p['size'])
            sz = sized(st, k)
            m = np.clip(np.nan_to_num(mid, nan=0.5), 0.005, 0.995)
            var = p['sig2'] * np.minimum(p['h_risk'], np.maximum(st['tau'], 0.0))
            r = m - (st['q'] / np.maximum(sz, 1e-9)) * g * var
            d = 0.5 * g * var + base
            live = both & st['two'] & (st['tau'] >= p['pull_s'])
            bid, ask = np.where(live, r - d, np.nan), np.where(live, r + d, np.nan)
            bid, ask = bl.inventory_limit(bid, ask, st['q'], sz, q_max)
            return bl.to_actions(amap, st, bid, ask, k)
        return fn

    if name == 'reward_farmer':
        if not dec.liq:
            raise ValueError('reward_farmer needs liq_rewards=1 (it reads the program from obs 55-58)')

        def fn(obs):
            st = dec(obs)
            both, mid, _ = bl._book(st)
            if dec.size_mode:
                k = np.zeros(len(st['q']), np.int64)          # head 0 = max(min size, min order) x size_rel0
            else:                                             # smallest v2 head that reaches the min size
                ok = st['sizes'] >= st['rw_min'][:, None] - 1e-9
                k = np.where(ok.any(1), np.argmax(ok, 1), 2)
            sz = sized(st, k)
            cap = np.minimum(p['inv_cap_lots'] * np.maximum(st['rw_min'], st['min_size']), q_max)
            half = 0.5 * st['rw_v_c'] / 100.0
            live = both & st['two'] & st['rw_active'] & (st['rw_v_c'] > 0) & (st['tau'] >= p['pull_s'])
            m = np.nan_to_num(mid, nan=0.5)
            bid = np.where(live & (st['q'] + sz <= cap + 1e-9), m - half, np.nan)
            ask = np.where(live & (st['q'] - sz >= -cap - 1e-9), m + half, np.nan)
            a = bl.to_actions(amap, st, bid, ask, 0)
            a[:, 2] = k
            return a
        return fn
    raise KeyError(name)


# ----------------------------------------------------------------------------- CLI
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


def _ledger_append(row, path):
    import ledger                                # train/ledger.py: locked append, git info
    return ledger.append(row, path)


def run_comparators(adapter, env_kw, names, split=1, variants=('central_1s',), ledger_path=LEDGER, tag=None,
                    params=None, with_markets=False, pull_grid=None):
    """Evaluate each comparator under each variant; one ledger row per val evaluation.

    pull_grid (D15): evaluate every comparator at each quote-pull time in the grid (seconds before
    the event; 0 = quote to kickoff). The env's pull_s forces the quotes off, and a comparator with
    its own pull_s parameter (mid_value, as_mid, reward_farmer) gets the same value, so the grid
    point alone decides when quoting stops. Results are keyed `<name>/<variant>/pull<s>`, and each
    is its own ledger row (the grid is a val-selected hyperparameter, D15: never chosen on test)."""
    out = {}
    kw0 = dict(env_kw)
    kw0['obs_size'] = max(int(kw0.get('obs_size', 50)), 80)
    pulls = [None] if pull_grid is None else [float(x) for x in pull_grid]
    for pull in pulls:
        kwp = dict(kw0) if pull is None else {**kw0, 'pull_s': pull}
        ident = adapter.identity(kwp)
        for name in names:
            prm = dict((params or {}).get(name) or {})
            if pull is not None and 'pull_s' in BASELINE_DEFAULTS[name]:
                prm['pull_s'] = pull
            for var in variants:
                kw = {**kwp, **adapter.variants[var]}
                fn = make_baseline(name, prm, kw)
                res = evaluate(fn, adapter, split=split, env_kw=kw, with_markets=with_markets)
                key = f'{name}/{var}' if pull is None else f'{name}/{var}/pull{int(pull)}'
                out[key] = res
                if split == 1 and ledger_path:
                    _ledger_append({'kind': 'val_eval', 'split': 1, 'source': 'comparator', 'backend': adapter.name,
                                    'tag': tag, 'method': name, 'params': {**BASELINE_DEFAULTS[name], **prm},
                                    'variant': var, 'variant_kw': adapter.variants[var], 'env': ident,
                                    'pull_s': pull, 'pull_grid': list(pulls) if pull is not None else None,
                                    'val': nan_to_none({k: v for k, v in res.items() if k != 'per_market'})},
                                   ledger_path)
    return out


def headline(res):
    keys = ('n_markets', 'pnl_total', 'pnl_mean', 'fills_per_market', 'income/trading', 'income/rebates',
            'income/rewards', 'income/flat_cost', 't6/share_of_total', 't6/directional', 't5/worst_flag',
            't5/worst_share', 'd13/makes_markets', 'd12/active', 'mean/uptime_two_sided', 'net/netted_share',
            'net/flat_drag_bps', 'net/end_abs_inv', 'd14/pnl_flat_T1h_total', 'excl/total', 'excl/day_ineligible',
            'excl/last_hour', 'chain/n_chains') + tuple(
            f'tk/{n}/pnl_total' for n, _ in TK_BUCKETS)
    return {k: (round(res[k], 4) if isinstance(res.get(k), float) else res.get(k)) for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for c in ('comparators', 'policy'):
        s = sub.add_parser(c)
        s.add_argument('--data-dir', required=True)
        s.add_argument('--split-map')
        s.add_argument('--split', type=int, default=1)
        s.add_argument('--env3-kw', nargs='*', help='env kwargs key=value (JSON values)')
        s.add_argument('--variants', nargs='*', default=['central_1s'])
        s.add_argument('--ledger', default=LEDGER)
        s.add_argument('--smoke', action='store_true', help='ledger train3/smoke_trials.jsonl; dry-run split maps ok')
        s.add_argument('--tag')
        s.add_argument('--out', help='write the full results JSON here')
        s.add_argument('--markets', action='store_true', help='include per-market rows')
        s.add_argument('--threads', type=int, default=1)
    sub.choices['comparators'].add_argument('--names', nargs='*')
    sub.choices['comparators'].add_argument('--params', help='JSON {name: params}')
    sub.choices['comparators'].add_argument('--pull-grid', nargs='*', type=float, metavar='S',
                                            help='D15 sweep: evaluate each comparator at env pull_s = each S '
                                                 '(seconds before kickoff; no values = 0 300 900 1800 3600)')
    sub.choices['policy'].add_argument('--ckpt', required=True)
    s = sub.add_parser('pool')
    s.add_argument('runs', nargs='+', help='run dirs (result.json) or result.json files')
    s.add_argument('--variant', default='central_1s')
    a = ap.parse_args(argv)
    if a.cmd == 'pool':
        sums = []
        for r in a.runs:
            p = os.path.join(r, 'result.json') if os.path.isdir(r) else r
            sums.append(json.load(open(p))['val'][a.variant])
        v = pooled_verdict(sums)
        print(json.dumps(nan_to_none(v), indent=1, default=str))
        return v
    import torch
    torch.set_num_threads(a.threads)
    ledger_path = SMOKE_LEDGER if a.smoke else a.ledger
    adapter = envs3.V3Adapter(allow_dry_run_split=a.smoke)
    kw = parse_kv(a.env3_kw)
    kw['data_dir'] = a.data_dir
    if a.split_map:
        kw['split_map'] = a.split_map
    if a.cmd == 'comparators':
        names = a.names or TRACK_COMPARATORS.get(envs3.detect_kind(a.data_dir), TRACK_COMPARATORS['sports'])
        grid = None if a.pull_grid is None else (a.pull_grid or list(PULL_GRID))
        res = run_comparators(adapter, kw, names, a.split, a.variants, ledger_path, a.tag,
                              json.loads(a.params) if a.params else None, a.markets, grid)
    else:
        from policy import load_checkpoint
        pol, ck = load_checkpoint(a.ckpt)
        kw.setdefault('obs_size', int(ck['obs_size']))
        res = {}
        ident = adapter.identity(kw)
        for var in a.variants:
            r = evaluate(pol, adapter, a.split, {**kw, **adapter.variants[var]}, with_markets=a.markets)
            res[f'policy/{var}'] = r
            if a.split == 1 and ledger_path:
                _ledger_append({'kind': 'val_eval', 'split': 1, 'source': 'evaluator3', 'backend': adapter.name,
                                'tag': a.tag, 'ckpt': os.path.abspath(a.ckpt), 'cfg_hash': ck.get('cfg_hash'),
                                'seed': ck.get('seed'), 'variant': var, 'env': ident,
                                'val': nan_to_none({k: v for k, v in r.items() if k != 'per_market'})}, ledger_path)
    for k, r in res.items():
        print(json.dumps({'eval': k, **headline(r)}, default=str), flush=True)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump({k: nan_to_none(v) for k, v in res.items()}, f, indent=1, default=str)
    return res


if __name__ == '__main__':
    main()
