#!/usr/bin/env python3
"""Reference implementation of the fee / rebate / liquidity-reward terms for the
market-making RL env (see REPORT.md section 4).  Pure functions + two presets.

Conventions
  * Everything is in the Up frame: the dataset's Up and Down books are exact
    mirrors, so a Down bid at q is an Up ask at 1-q.  Our quotes are given as Up-frame
    (price, size) lists for the bid side and the ask side.
  * Prices in dollars (0..1), distances for the reward scorer in cents.

Usage in an env step (dt = 1 s):
    r_t = mark_to_market_pnl_t                      # spread + inventory (existing env)
        - taker_fee(our_aggressive_fills)           # only if the agent crosses
        + rebate_accrual(our_maker_fills, cfg)      # section 4.1
        + reward_accrual(...) if a sample falls in (t-dt, t]   # section 4.2
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# ------------------------------------------------------------------ config


@dataclass(frozen=True)
class RewardConfig:
    name: str
    # taker fee: fee = C * fee_rate * p * (1-p) ** fee_exponent  (gamma feeSchedule)
    fee_rate: float = 0.07
    fee_exponent: float = 1.0
    # maker rebate as a fraction of the fee-equivalent of OUR maker fills
    rebate_rate: float = 0.20
    # liquidity rewards
    reward_enabled: bool = False
    reward_min_size: float = 50.0          # rewardsMinSize (shares, per order)
    reward_max_spread_c: float = 4.5       # rewardsMaxSpread (cents from adjusted mid)
    reward_c: float = 3.0                  # single-sided divisor
    two_sided_band: tuple = (0.10, 0.90)   # outside: Q_min = min(Q_one, Q_two)
    reward_min_order_age_s: float = 0.0    # clob-markets r.moas (undocumented; 1 in Aug, 30 now)
    sample_period_s: float = 60.0          # one sample per minute at a random offset
    # USD per market over its whole life (accepting-orders -> end); per sample = pool / life_minutes
    pool_usd_by_len_coin: dict = field(default_factory=dict)
    competitor_bound: str = "hi"           # "hi": competitors two-sided (conservative), "lo": one-sided
    tick: float = 0.01
    fine_tick: float = 0.001               # applies when price < 0.04 or > 0.96


_POOL_MONTH = {
    "5m": {"btc": 300e3, "sol": 50e3, "eth": 50e3, "hype": 50e3, "xrp": 50e3, "bnb": 25e3, "doge": 25e3},
    "15m": {"btc": 225e3, "sol": 25e3, "eth": 25e3, "hype": 25e3, "xrp": 25e3, "bnb": 12.5e3, "doge": 12.5e3},
    "4h": {"btc": 50e3, "sol": 10e3, "eth": 10e3, "hype": 10e3, "xrp": 10e3, "bnb": 5e3, "doge": 5e3},
}
_PER_DAY = {"5m": 288, "15m": 96, "4h": 6}

# Replay fidelity for the Aug 21-24 2026 data.
AUG_2026 = RewardConfig(
    name="aug2026_replay", rebate_rate=0.20, reward_enabled=True,
    reward_min_size=50.0, reward_max_spread_c=1.5, reward_min_order_age_s=1.0,
    pool_usd_by_len_coin={cl: {c: v / (_PER_DAY[cl] * 31) for c, v in d.items()} for cl, d in _POOL_MONTH.items()},
)
# Polymarket (global CLOB) as of 2026-10-15: same fees/rebate, no reward allocation on
# crypto up/down (not in /rewards/markets/current); scorer params are today's defaults.
LIVE_2026_10 = RewardConfig(
    name="live_2026_10", rebate_rate=0.20, reward_enabled=False,
    reward_min_size=50.0, reward_max_spread_c=4.5, reward_min_order_age_s=30.0,
)
# Sensitivity presets
AUG_2026_BONUS_REBATE = replace(AUG_2026, name="aug2026_bonus", rebate_rate=0.35)
AUG_2026_WIDE = replace(AUG_2026, name="aug2026_maxspread4.5", reward_max_spread_c=4.5)

# ------------------------------------------------------------------ fees / rebates


def fee_equivalent(price, size, cfg: RewardConfig):
    p = np.asarray(price, float)
    return np.asarray(size, float) * cfg.fee_rate * (p * (1 - p)) ** cfg.fee_exponent


def taker_fee(fills, cfg: RewardConfig) -> float:
    """fills: iterable of (price_of_token_traded, size). Cost (positive)."""
    return float(sum(fee_equivalent(p, s, cfg) for p, s in fills))


def rebate_accrual(maker_fills, cfg: RewardConfig) -> float:
    """Our maker fills this step -> rebate (credited at fill; paid daily, >= $1).
    Equals the docs' pro-rata rule  rebate_pool * my_fee_eq / total_fee_eq  because the
    pool is rebate_rate * taker fees and total maker fee_eq == total taker fees."""
    return float(cfg.rebate_rate * sum(fee_equivalent(p, s, cfg) for p, s in maker_fills))

# ------------------------------------------------------------------ liquidity reward


def S(dist_c, v):
    d = np.asarray(dist_c, float)
    return np.where(d < v, ((v - d) / v) ** 2, 0.0)


def adjusted_mid(bid_p, bid_s, ask_p, ask_s, min_size):
    """Size-cutoff-adjusted midpoint: mid of the prices where cumulative depth reaches
    min_size on each side (interpretation; docs only name the concept)."""
    if len(bid_p) == 0 or len(ask_p) == 0:
        return np.nan

    def side(p, s):
        k = np.searchsorted(np.cumsum(s), min_size)
        return p[min(k, len(p) - 1)]
    return 0.5 * (side(np.asarray(bid_p), np.asarray(bid_s)) + side(np.asarray(ask_p), np.asarray(ask_s)))


def side_scores(bid_p, bid_s, ask_p, ask_s, mid, cfg: RewardConfig, ages=None):
    """(Q_one, Q_two) for a set of orders; orders below min size (or too young) score 0."""
    v = cfg.reward_max_spread_c
    bp, bs, ap, as_ = (np.asarray(x, float) for x in (bid_p, bid_s, ask_p, ask_s))
    ok_b = bs >= cfg.reward_min_size
    ok_a = as_ >= cfg.reward_min_size
    if ages is not None:
        ok_b &= np.asarray(ages[0]) >= cfg.reward_min_order_age_s
        ok_a &= np.asarray(ages[1]) >= cfg.reward_min_order_age_s
    q1 = float((S(np.abs(mid - bp) * 100, v) * bs * ok_b).sum()) if len(bp) else 0.0
    q2 = float((S(np.abs(ap - mid) * 100, v) * as_ * ok_a).sum()) if len(ap) else 0.0
    return q1, q2


def q_min(q1, q2, mid, cfg: RewardConfig):
    lo, hi = cfg.two_sided_band
    if lo <= mid <= hi:
        return max(min(q1, q2), max(q1, q2) / cfg.reward_c)
    return min(q1, q2)


def competitor_q(q1, q2, mid, cfg: RewardConfig):
    """Sum over other makers of Q_min, from the anonymous replay book.  Bracketed:
    'hi' = book liquidity as two-sided as possible, 'lo' = every maker one-sided."""
    lo, hi = cfg.two_sided_band
    inside = lo <= mid <= hi
    if cfg.competitor_bound == "hi":
        return min(q1, q2) + (abs(q1 - q2) / cfg.reward_c if inside else 0.0)
    return (q1 + q2) / cfg.reward_c if inside else 0.0


def reward_accrual(book, ours, cfg: RewardConfig, pool_usd: float, life_minutes: float, our_ages=None):
    """Reward credited at one sample.
    book: (bid_p, bid_s, ask_p, ask_s) replay book (Up frame) WITHOUT our orders.
    ours: (bid_p, bid_s, ask_p, ask_s) our resting orders (Up frame).
    Returns (usd, info) where info carries the observation features of section 4.3."""
    if not cfg.reward_enabled or pool_usd <= 0:
        return 0.0, {}
    # the adjusted mid is computed on the merged book (our orders move it too)
    mb = [np.concatenate([np.asarray(book[0], float), np.asarray(ours[0], float)]),
          np.concatenate([np.asarray(book[1], float), np.asarray(ours[1], float)])]
    ma = [np.concatenate([np.asarray(book[2], float), np.asarray(ours[2], float)]),
          np.concatenate([np.asarray(book[3], float), np.asarray(ours[3], float)])]
    ob, oa = np.argsort(-mb[0], kind="stable"), np.argsort(ma[0], kind="stable")
    mid = adjusted_mid(mb[0][ob], mb[1][ob], ma[0][oa], ma[1][oa], cfg.reward_min_size)
    if not np.isfinite(mid):
        return 0.0, {"scorable": False}
    c1, c2 = side_scores(*book, mid, cfg)
    o1, o2 = side_scores(*ours, mid, cfg, ages=our_ages)
    mine = q_min(o1, o2, mid, cfg)
    comp = competitor_q(c1, c2, mid, cfg)
    share = mine / (mine + comp) if mine > 0 else 0.0
    pi = pool_usd / life_minutes * (cfg.sample_period_s / 60.0)
    return pi * share, dict(scorable=True, adj_mid=mid, q_one=o1, q_two=o2, q_min=mine,
                            comp_q=comp, share=share, single_sided_allowed=cfg.two_sided_band[0] <= mid <= cfg.two_sided_band[1])


def sample_times(t_open: float, t_end: float, rng: np.random.Generator, period=60.0):
    """One sample per minute at a uniform random offset (docs: 'random offset')."""
    m0 = np.arange(np.floor(t_open / period) * period, t_end, period)
    t = m0 + rng.uniform(0, period, len(m0))
    return t[(t >= t_open) & (t < t_end)]


if __name__ == "__main__":
    # docs worked example (mid 0.50, v = 3c, b = 1):  Q_one = (2/3)^2*100 + (1/3)^2*200 + (2/3)^2*100
    cfg = replace(AUG_2026, reward_max_spread_c=3.0, reward_min_size=0.0)
    q1, _ = side_scores([0.49, 0.48], [100, 200], [], [], 0.50, cfg)
    q1b, _ = side_scores([1 - 0.51], [100], [], [], 0.50, cfg)   # ask on m' at .51 == Up bid at .49
    assert abs(q1 + q1b - ((2 / 3) ** 2 * 100 + (1 / 3) ** 2 * 200 + (2 / 3) ** 2 * 100)) < 1e-9
    # fee table check (100 shares @ 0.50 -> $1.75; @0.10 -> $0.63)
    assert abs(taker_fee([(0.5, 100)], AUG_2026) - 1.75) < 1e-9
    assert abs(round(taker_fee([(0.1, 100)], AUG_2026), 2) - 0.63) < 1e-9
    usd, info = reward_accrual(([0.49, 0.48], [50, 300], [0.51, 0.52], [50, 300]),
                               ([0.495], [50], [0.505], [50]), AUG_2026,
                               pool_usd=AUG_2026.pool_usd_by_len_coin["5m"]["btc"], life_minutes=1445)
    print(f"example sample: ${usd:.5f}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})
    print("self-test ok")
