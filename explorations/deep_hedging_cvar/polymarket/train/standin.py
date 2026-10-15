"""A small stand-in for polymarket/env.py, for testing the training stack.

Same interface as the real env: a native pufferlib.PufferEnv with float32
observations whose last feature is the constant 1, MultiDiscrete([7, 7, 3])
actions (bid choice, ask choice, size), a dense per-step reward clipped to
[-1, 1], terminal + auto-reset inside step, and a Log-style info dict with
`pnl`, attribution fields, `score` and `n`. Nothing here models Polymarket;
it is only a cheap, deterministic, learnable market-making toy:

- Each split has its own synthetic "markets" (lengths 300, 900 and 3600
  steps, i.e. 5m/15m/1h at dt = 1 s), precomputed like a replay: a fair value
  `fv` (random walk), a sticky mid that chases fv with a lag, and taker
  arrivals (uniforms and sweep depths) whose direction leans toward fv - mid,
  so quotes near a stale mid get adversely selected.
- Action: bid/ask choice 0 = no quote, k = 1..6 = quote k-1 ticks behind the
  mid (k = 1 at the mid); size 0/1/2 = 1/2/4 lots. A taker sell (buy) fills
  our bid (ask) when its sweep depth reaches our level. |inventory| <= Q.
- Reward per step = reward_scale * (change of mark-to-market P&L
  - inv_penalty * q^2); at the end of the market the inventory settles at the
  binary outcome 1{fv_end > 0.5}, which is a terminal (no truncation).
- Training: random market and random start offset in [0, len/2). Eval mode
  (`eval_mode=True`): env i plays market i of the split from offset 0, then
  idles on it; `episode_records` collects one dict per finished market.
"""

import gymnasium
import numpy as np

import pufferlib

TICK = 0.01
LENGTHS = (300, 900, 3600)
N_MARKETS = {0: 96, 1: 48, 2: 24}
LOG_KEYS = ('pnl', 'spread_pnl', 'inventory_pnl', 'fills', 'volume', 'max_abs_inv',
            'quote_uptime', 'episode_length', 'score', 'perf')


def build_split(split, n_markets=None, seed=1234):
    """Deterministic synthetic market data for a split (a flat row table + offsets)."""
    rng = np.random.default_rng(seed + 7919 * split)
    n = n_markets or N_MARKETS[split]
    lens = rng.choice(LENGTHS, size=n, p=(0.6, 0.3, 0.1))
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    rows = int(off[-1])
    fv = np.empty(rows, np.float32)
    mid = np.empty(rows, np.float32)
    for m in range(n):
        L = lens[m]
        x = np.cumsum(rng.normal(0, 0.03, L)) + rng.normal(0, 0.5)  # logit fair value
        f = 1 / (1 + np.exp(-x))
        md = np.empty(L)
        md[0] = f[0]
        for t in range(1, L):  # sticky mid: updates w.p. 0.3, toward lagged fv
            md[t] = md[t - 1] + (0.6 * (f[t - 1] - md[t - 1]) if rng.random() < 0.3 else 0.0)
        fv[off[m]:off[m + 1]] = f
        mid[off[m]:off[m + 1]] = np.clip(np.round(md / TICK) * TICK, 0.02, 0.98)
    gap = (fv - mid) / TICK
    p_buy = np.clip(0.15 + 0.05 * gap, 0.02, 0.6)    # takers lean toward fv
    p_sell = np.clip(0.15 - 0.05 * gap, 0.02, 0.6)
    u = rng.random((2, rows))
    depth = rng.exponential(1.5, (2, rows))
    buy_depth = np.where(u[0] < p_buy, depth[0], -1.0).astype(np.float32)
    sell_depth = np.where(u[1] < p_sell, depth[1], -1.0).astype(np.float32)
    return dict(lens=lens.astype(np.int64), off=off, fv=fv, mid=mid,
                buy_depth=buy_depth, sell_depth=sell_depth)


class StandInMM(pufferlib.PufferEnv):
    def __init__(self, num_envs=1024, split=0, seed=0, eval_mode=False, obs_size=16,
                 Q=20.0, inv_penalty=0.0, reward_scale=1.0, n_markets=None, log_interval=64,
                 buf=None, **unused):
        if split == 2:
            raise ValueError('the stand-in has no test split for the trainer to touch')
        assert obs_size >= 12
        self.num_agents = num_envs
        self.single_observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 7, 3])
        super().__init__(buf)
        self.d = build_split(split, n_markets)
        self.n_markets = len(self.d['lens'])
        self.eval_mode, self.Q = eval_mode, float(Q)
        self.inv_penalty, self.reward_scale = float(inv_penalty), float(reward_scale)
        self.log_interval = log_interval
        self.rng = np.random.default_rng(seed)
        n = num_envs
        self.m = np.zeros(n, np.int64)
        self.t = np.zeros(n, np.int64)
        self.t0 = np.zeros(n, np.int64)
        self.q = np.zeros(n)
        self.cash = np.zeros(n)
        self.acc = {k: np.zeros(n) for k in ('spread_pnl', 'fills', 'volume', 'max_abs_inv', 'quoted')}
        self.log_sum = {k: 0.0 for k in LOG_KEYS}
        self.log_n = 0
        self.tick = 0
        self.played = np.zeros(n, bool)
        self.episode_records = []

    # --- episode bookkeeping ---
    def _start(self, idx):
        if self.eval_mode:
            self.m[idx] = idx % self.n_markets
            self.t0[idx] = 0
        else:
            self.m[idx] = self.rng.integers(0, self.n_markets, len(idx))
            self.t0[idx] = (self.rng.random(len(idx)) * self.d['lens'][self.m[idx]] // 2).astype(np.int64)
        self.t[idx] = self.t0[idx]
        self.q[idx] = 0
        self.cash[idx] = 0
        for a in self.acc.values():
            a[idx] = 0

    def _obs(self):
        d, o = self.d, self.observations
        r = d['off'][self.m] + self.t
        prev = np.maximum(r - 1, d['off'][self.m])
        L = d['lens'][self.m]
        mid, fv = d['mid'][r], d['fv'][r]
        o[:] = 0
        o[:, 0] = 2 * (L - self.t) / 3600.0 - 1          # time to close
        o[:, 1] = (mid - 0.5) * 4
        o[:, 2] = np.clip((fv - mid) / TICK / 5, -5, 5)     # "Binance" fair-value gap
        o[:, 3] = np.clip((mid - d['mid'][prev]) / TICK, -5, 5)
        o[:, 4] = self.q / self.Q
        o[:, 5] = np.sign(self.q)
        o[:, 6] = np.clip(np.log1p(L / 300.0) - 1, -5, 5)
        o[:, 7] = np.minimum(mid, 1 - mid) * 10
        o[:, -1] = 1.0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._start(np.arange(self.num_agents))
        self.played[:] = False
        self.episode_records = []
        self._obs()
        return self.observations, []

    def step(self, actions):
        d = self.d
        a = np.asarray(actions)
        bid_c, ask_c, sz_c = a[:, 0], a[:, 1], a[:, 2]
        r = d['off'][self.m] + self.t
        mid = d['mid'][r].astype(np.float64)
        size = np.array([1.0, 2.0, 4.0])[sz_c]
        pnl_before = self.cash + self.q * mid
        bid_k = bid_c - 1.0  # ticks behind mid; -1 = no quote
        ask_k = ask_c - 1.0
        can_buy = self.q + size <= self.Q
        can_sell = self.q - size >= -self.Q
        fill_b = (bid_c > 0) & can_buy & (d['sell_depth'][r] >= bid_k)
        fill_a = (ask_c > 0) & can_sell & (d['buy_depth'][r] >= ask_k)
        pb, pa = mid - bid_k * TICK, mid + ask_k * TICK
        self.cash += np.where(fill_b, -pb * size, 0) + np.where(fill_a, pa * size, 0)
        self.q += np.where(fill_b, size, 0) - np.where(fill_a, size, 0)
        acc = self.acc
        acc['spread_pnl'] += np.where(fill_b, (mid - pb) * size, 0) + np.where(fill_a, (pa - mid) * size, 0)
        acc['fills'] += fill_b.astype(float) + fill_a
        acc['volume'] += (fill_b + fill_a) * size
        acc['max_abs_inv'] = np.maximum(acc['max_abs_inv'], np.abs(self.q))
        acc['quoted'] += (bid_c > 0) & (ask_c > 0)
        self.t += 1
        L = d['lens'][self.m]
        end = self.t >= L - 1
        r1 = d['off'][self.m] + np.minimum(self.t, L - 1)
        mark = np.where(end, (d['fv'][r1] > 0.5).astype(np.float64), d['mid'][r1])
        pnl_after = self.cash + self.q * mark
        rew = (pnl_after - pnl_before) - self.inv_penalty * self.q ** 2
        self.rewards[:] = np.clip(self.reward_scale * rew, -1, 1)
        self.terminals[:] = end
        info = []
        if end.any():
            idx = np.flatnonzero(end)
            ep_len = (self.t - self.t0)[idx]
            rec = dict(pnl=pnl_after[idx], spread_pnl=acc['spread_pnl'][idx],
                       fills=acc['fills'][idx], volume=acc['volume'][idx],
                       max_abs_inv=acc['max_abs_inv'][idx],
                       quote_uptime=acc['quoted'][idx] / ep_len, episode_length=ep_len)
            rec['inventory_pnl'] = rec['pnl'] - rec['spread_pnl']
            rec['score'] = rec['pnl']
            rec['perf'] = (rec['pnl'] > 0).astype(float)
            for k in LOG_KEYS:
                self.log_sum[k] += float(np.sum(rec[k]))
            self.log_n += len(idx)
            if self.eval_mode:
                for j, i in enumerate(idx):
                    if not self.played[i]:
                        self.played[i] = True
                        self.episode_records.append(
                            dict(env=int(i), market=int(self.m[i]),
                                 **{k: float(v[j]) for k, v in rec.items()}))
            self._start(idx)
        self.tick += 1
        if self.tick % self.log_interval == 0 and self.log_n:
            info.append({**{k: v / self.log_n for k, v in self.log_sum.items()}, 'n': float(self.log_n)})
            self.log_sum = {k: 0.0 for k in LOG_KEYS}
            self.log_n = 0
        self._obs()
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        pass
