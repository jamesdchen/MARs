"""Classical market-making baselines for the Polymarket replay env (EVALUATION.md section 4).

Every factory takes an `ObsSpec` (obs_spec.py: the env's documented feature-name -> obs-index
mapping) plus parameters and returns a deterministic, vectorized policy

    fn(obs: float32 (B, OBS_SIZE)) -> int32 actions (B, 3) = (bid choice, ask choice, size choice)

for the env's MultiDiscrete([7, 7, 3]) action: price choice in {off, improve 2 ticks, improve 1,
join best, 1 behind, 2 behind, 4 behind} relative to the current best quote on that side (ask
symmetric: "improve" = lower), size in {1x, 2x, 5x} of the min order size (5 shares).

Each quoter computes a desired bid and ask price and `to_actions` maps each to the nearest
discrete choice relative to the current best quotes (all Up frame):
  - an improving quote never locks or crosses: improvement is capped at spread - 1 tick, so with a
    1-tick spread the best a side can do is join;
  - ties go to the more passive choice;
  - a desired price more than FAR_TICKS beyond the deepest choice (4 behind) is "off", so a quoter
    that wants to be far away is not forced to sit 4 ticks behind, which would be more aggressive
    than it asked for;
  - offsets are relative to the env's reference price (ENV.md section 3): the touch, or for an
    empty side of a one-sided book p_model floored / ceiled to the tick (obs 47-48 give the present
    side's touch relative to p_model); with no book at all, or a present touch whose obs 47-48 is
    clipped (> 25c from p_model) on a one-sided book, the side is off.
  - the price quoters (as_logit, fair_value, fv_smart) centre on the env's mark: the mid when the
    book is two-sided, p_model when it is one-sided; sym_touch only joins a touch that exists.
Inventory `q` is in shares (Up frame); parameters that scale with inventory use lots of the min
order size (q_lots = q / min_size) so they do not depend on the share count.

Baselines
---------
1. no_quote        always off on both sides (P&L 0; reward-only and sanity anchor).
2. sym_touch       join best bid and best ask, fixed size; inventory cap: the side whose fill would
                   take |q| beyond q_cap lots is pulled (cap_mode="pull") or backed off to the
                   deepest choice, 4 ticks behind (cap_mode="back").
3. as_logit        Avellaneda & Stoikov (2008) around the mid in logit units, x = logit(mid):
                     reservation  x_r = x - q_lots * gamma * v_x
                     half-spread  d   = gamma * v_x / 2 + ln(1 + gamma / k) / gamma
                     bid, ask     = sigmoid(x_r -+ d)
                   v_x = v_p / (m(1-m))^2 is the delta-method logit variance of the mid over the
                   risk horizon min(h_risk, tau), and v_p the exact variance of a digital's price
                   over that horizon (see `digital_var`): the "sigma^2 (T - t)" of AS for a binary.
                   It depends only on the mid and the time left (the underlying's vol cancels),
                   and includes the closing 60 s TWAP window. h_risk >= tau is classical
                   finite-horizon AS (risk to resolution); h_risk << tau is the stationary,
                   GLFT-like regime (Gueant, Lehalle & Fernandez-Tapia 2013) in which the
                   inventory penalty uses a fixed horizon. In logit units the quotes widen as
                   m(1-m) -> 0, and a desired quote beyond the ladder is off.
4. fair_value      the quoter of analysis/fair_value/REPORT.md section 6:
                     centre c = clip((p_model + mid)/2, 0.01, 0.99);
                            5m: c = sigmoid(0.6 logit p_model + 0.32 logit mid)
                     half-spread h = spread_mult * (max(tick, 1c, spread/2) + 0.5 |gap|),
                            gap = p_model - mid
                     skew   bid/ask = c - k_skew * q_lots -+ h
                     pull both sides when tau < pull_s or |gap| > gap_max.
                   Tuned: k_skew, spread_mult, gap_max, pull_s, size.
5. fv_smart        the fair-value quoter that only quotes a side when its quote would sit at or
                   inside the touch (join / improve; otherwise off) and pulls the side the gap
                   says is about to be picked off: bid off when gap < -g_adv, ask off when
                   gap > g_adv. Also pulls when tau < pull_s.

`DEFAULTS` holds the untuned point of each baseline (the REPORT spec values for fair_value,
enqueued as trial 0 by tune.py); `SEARCH` the Optuna search space.
"""
from __future__ import annotations

import math

import numpy as np

from obs_spec import ObsSpec

FAR_TICKS = 2.0          # beyond (deepest choice - FAR_TICKS) ticks a desired quote is "off"
TWAP_S = 60.0            # closing TWAP window of the resolution rule (fair_value REPORT section 1)
P_CLIP = 0.005           # mids are clipped to [P_CLIP, 1 - P_CLIP] before taking logits

# Untuned points and search spaces. Search entries: (kind, low, high, log) or ("cat", choices).
DEFAULTS = {
    "no_quote": {},
    "sym_touch": {"size": 0, "q_cap": 10.0, "cap_mode": "pull"},
    "as_logit": {"gamma": 0.01, "k": 50.0, "h_risk": 60.0, "size": 0},
    "fair_value": {"k_skew": 0.001, "spread_mult": 1.0, "gap_max": 0.10, "pull_s": 30.0, "size": 0},
    "fv_smart": {"k_skew": 0.001, "spread_mult": 1.0, "g_adv": 0.03, "pull_s": 30.0, "size": 0},
}
SEARCH = {
    "no_quote": {},
    "sym_touch": {"size": ("cat", [0, 1, 2]), "q_cap": ("float", 1.0, 200.0, True),
                  "cap_mode": ("cat", ["pull", "back"])},
    "as_logit": {"gamma": ("float", 1e-4, 10.0, True), "k": ("float", 2.0, 2000.0, True),
                 "h_risk": ("float", 5.0, 14400.0, True), "size": ("cat", [0, 1, 2])},
    "fair_value": {"k_skew": ("float", 1e-5, 0.02, True), "spread_mult": ("float", 0.3, 4.0, True),
                   "gap_max": ("float", 0.02, 0.5, True), "pull_s": ("float", 0.0, 120.0, False),
                   "size": ("cat", [0, 1, 2])},
    "fv_smart": {"k_skew": ("float", 1e-5, 0.02, True), "spread_mult": ("float", 0.3, 4.0, True),
                 "g_adv": ("float", 0.005, 0.3, True), "pull_s": ("float", 0.0, 120.0, False),
                 "size": ("cat", [0, 1, 2])},
}


# ----------------------------------------------------------------------------- math helpers
def logit(p):
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def norm_ppf(p):
    """Inverse standard normal CDF, vectorized (Acklam 2003, |rel err| < 1.2e-9); p in (0, 1)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p = np.asarray(p, dtype=np.float64)
    lo, hi = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    m = (p >= lo) & (p <= hi)
    q = p[m] - 0.5
    r = q * q
    out[m] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
              / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    for mask, sgn in ((p < lo, 1.0), (p > hi, -1.0)):
        qq = np.sqrt(-2 * np.log(np.where(sgn > 0, p[mask], 1 - p[mask])))
        out[mask] = sgn * ((((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5])
                           / ((((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1))
    return out


def remaining_var_units(tau):
    """Remaining variance of the settlement statistic in units of sigma^2 * seconds:
    tau - 40 for tau >= 60 (spot to the TWAP window, then a/3 of it), tau^3 / (3 * 60^2) inside
    the window (fair_value.py `_forward_and_var`). Continuous at tau = 60."""
    tau = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)
    return np.where(tau >= TWAP_S, tau - TWAP_S + TWAP_S / 3, tau ** 3 / (3 * TWAP_S ** 2))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def digital_var(m, tau, h):
    """Var[p_{t+h} | p_t = m] for a digital whose price is Phi(X / sqrt(V)) with X Gaussian:
    with f the fraction of the remaining variance V accrued over min(h, tau),
        Var = Phi2(d, d; f) - m^2 = (1 / 2 pi) * int_0^{asin f} exp(-d^2 / (1 + sin t)) dt,
    d = Phi^-1(m). f = 1 gives the full m(1 - m) of a Bernoulli settlement. The underlying's vol
    cancels; only the mid and the time left matter."""
    m = np.asarray(m, dtype=np.float64)
    V = remaining_var_units(tau)
    V_after = remaining_var_units(np.asarray(tau, dtype=np.float64) - h)
    f = np.where(V > 0, 1.0 - V_after / np.maximum(V, 1e-12), 1.0)
    f = np.clip(f, 0.0, 1.0)
    d2 = norm_ppf(np.clip(m, 1e-9, 1 - 1e-9)) ** 2
    ub = np.arcsin(f)                                    # (B,)
    th = 0.5 * ub[:, None] * (_GL_X[None, :] + 1.0)      # (B, 16) nodes on [0, ub]
    integrand = np.exp(-d2[:, None] / (1.0 + np.sin(th)))
    return 0.5 * ub * (integrand * _GL_W[None, :]).sum(1) / (2 * math.pi)


# ----------------------------------------------------------------------------- action mapping
class ActionMap:
    """Index bookkeeping for one price head and the size head, from the spec."""

    def __init__(self, spec: ObsSpec):
        offs = list(spec.offsets)
        self.off = offs.index(None)
        self.idx = np.array([i for i, o in enumerate(offs) if o is not None])
        self.ticks = np.array([offs[i] for i in self.idx], dtype=np.float64)
        self.join = int(self.idx[self.ticks == 0][0])
        self.deepest = int(self.idx[np.argmin(self.ticks)])
        self.n_sizes = len(spec.size_mults)
        self.size_mults = np.asarray(spec.size_mults, dtype=np.float64)


def price_to_choice(amap: ActionMap, desired, best, other_best, tick, side: int, ok,
                    inside_only: bool = False, far_ticks: float = FAR_TICKS):
    """Nearest discrete choice for a desired price on one side.

    side +1: bid (improve = higher), -1: ask (improve = lower). `best` is that side's best quote,
    `other_best` the opposite side's (caps improvement at spread - 1 tick). NaN desired or not ok
    -> off."""
    desired = np.asarray(desired, dtype=np.float64)
    o = side * (desired - best) / tick                   # ticks of improvement wanted
    o = np.round(o, 6)
    spread_t = np.rint(np.abs(other_best - best) / tick)     # quotes sit on the tick grid
    cap = np.where(np.isfinite(spread_t), spread_t - 1.0, amap.ticks.max())
    cap = np.maximum(cap, 0.0)                           # joining is always allowed
    allowed = amap.ticks[None, :] <= cap[:, None]
    if inside_only:
        allowed &= amap.ticks[None, :] >= 0
    dist = np.abs(o[:, None] - amap.ticks[None, :]) + 1e-6 * amap.ticks[None, :]
    dist = np.where(allowed, dist, np.inf)
    choice = amap.idx[np.argmin(dist, axis=1)]
    off = ~ok | ~np.isfinite(o) | (o < amap.ticks.min() - far_ticks)
    if inside_only:
        off |= o < -0.5
    return np.where(off, amap.off, choice).astype(np.int32)


def _uncross(amap: ActionMap, st: dict, a: np.ndarray, bid_px, ask_px, inside_only=False):
    """Our own bid must stay below our own ask (the core otherwise moves the ask to bid + tick).
    While both sides are on and the chosen prices touch or cross, step the side whose desired price
    is less hurt by it one choice more passive (ties: the ask, as the core would). With
    inside_only, a side pushed behind the touch goes off instead."""
    order = np.full(7, -1)                               # choice index -> rank (0 most aggressive)
    ranks = amap.idx[np.argsort(-amap.ticks)]
    order[ranks] = np.arange(len(ranks))
    by_rank = ranks
    tk = np.zeros(7)
    tk[amap.idx] = amap.ticks
    for _ in range(len(ranks)):
        on = (a[:, 0] != amap.off) & (a[:, 1] != amap.off)
        pb = st["bid_ref"] + tk[a[:, 0]] * st["tick"]
        pa = st["ask_ref"] - tk[a[:, 1]] * st["tick"]
        bad = on & (pb >= pa - 1e-9)
        if not bad.any():
            break
        rb, ra = order[a[:, 0]], order[a[:, 1]]
        nb = np.where(rb + 1 < len(ranks), by_rank[np.minimum(rb + 1, len(ranks) - 1)], amap.off)
        na = np.where(ra + 1 < len(ranks), by_rank[np.minimum(ra + 1, len(ranks) - 1)], amap.off)
        if inside_only:
            nb = np.where(tk[nb] < 0, amap.off, nb)
            na = np.where(tk[na] < 0, amap.off, na)
        cost_b = np.abs(bid_px - (st["bid_ref"] + tk[nb] * st["tick"])) - np.abs(bid_px - pb)
        cost_a = np.abs(ask_px - (st["ask_ref"] - tk[na] * st["tick"])) - np.abs(ask_px - pa)
        move_b = bad & (cost_b < cost_a)
        move_a = bad & ~move_b
        a[:, 0] = np.where(move_b, nb, a[:, 0])
        a[:, 1] = np.where(move_a, na, a[:, 1])
    return a


def to_actions(amap: ActionMap, st: dict, bid_px, ask_px, size_idx, inside_only=False):
    """(B, 3) int32 actions from desired prices (NaN = off) and a size index."""
    B = st["bid_ref"].shape[0]
    a = np.empty((B, 3), dtype=np.int32)
    a[:, 0] = price_to_choice(amap, bid_px, st["bid_ref"], np.where(st["ask_ok"], st["ask"], np.nan),
                              st["tick"], +1, np.isfinite(st["bid_ref"]), inside_only)
    a[:, 1] = price_to_choice(amap, ask_px, st["ask_ref"], np.where(st["bid_ok"], st["bid"], np.nan),
                              st["tick"], -1, np.isfinite(st["ask_ref"]), inside_only)
    with np.errstate(invalid="ignore"):
        a = _uncross(amap, st, a, np.asarray(bid_px, np.float64), np.asarray(ask_px, np.float64),
                     inside_only)
    a[:, 2] = np.broadcast_to(np.asarray(size_idx, dtype=np.int32), (B,))
    return a


def order_size(amap: ActionMap, st: dict, size_idx: int):
    return amap.size_mults[size_idx] * st["min_size"]


def inventory_limit(bid_px, ask_px, q, sz, q_max):
    """Pull the side whose fill would take q beyond +-q_max (shares)."""
    if not np.isfinite(q_max):
        return bid_px, ask_px
    return (np.where(q + sz > q_max + 1e-9, np.nan, bid_px),
            np.where(q - sz < -q_max - 1e-9, np.nan, ask_px))


def _book(st):
    """(quotable, mark, spread): quotable = the book has a side (the env quotes an empty side
    from p_model rounded to the tick); mark = mid if two-sided else p_model (the env's mark);
    spread = ask - bid if two-sided else 0."""
    quotable = (st["bid_ok"] | st["ask_ok"]) & np.isfinite(st["mark"])
    spread = np.where(st["two"], st["ask"] - st["bid"], 0.0)
    return quotable, st["mark"], spread


def _q_max(spec, params):
    v = params.get("q_max", spec.meta.get("q_max", spec.meta.get("inventory_limit", math.inf)))
    return float(v) if v is not None else math.inf


# ----------------------------------------------------------------------------- baselines
def no_quote(spec: ObsSpec, params=None):
    amap = ActionMap(spec)

    def fn(obs):
        a = np.full((np.asarray(obs).shape[0], 3), amap.off, dtype=np.int32)
        a[:, 2] = 0
        return a
    return fn


def sym_touch(spec: ObsSpec, params=None):
    p = {**DEFAULTS["sym_touch"], **(params or {})}
    amap = ActionMap(spec)
    spec.check(need_fair=False)
    size = int(p["size"])
    q_max = _q_max(spec, p)

    def fn(obs):
        st = spec.decode(obs)
        sz = order_size(amap, st, size)
        cap = p["q_cap"] * st["min_size"]
        a = np.empty((st["bid_ref"].shape[0], 3), dtype=np.int32)
        over_b = st["q"] + sz > np.minimum(cap, q_max) + 1e-9
        over_a = st["q"] - sz < -np.minimum(cap, q_max) - 1e-9
        capped = amap.off if p["cap_mode"] == "pull" else amap.deepest
        a[:, 0] = np.where(~st["bid_ok"], amap.off, np.where(over_b, capped, amap.join))
        a[:, 1] = np.where(~st["ask_ok"], amap.off, np.where(over_a, capped, amap.join))
        if np.isfinite(q_max):                           # a hard limit always pulls
            a[:, 0] = np.where(st["q"] + sz > q_max + 1e-9, amap.off, a[:, 0])
            a[:, 1] = np.where(st["q"] - sz < -q_max - 1e-9, amap.off, a[:, 1])
        a[:, 2] = size
        return a
    return fn


def as_logit(spec: ObsSpec, params=None):
    p = {**DEFAULTS["as_logit"], **(params or {})}
    amap = ActionMap(spec)
    spec.check(need_fair=False)
    gamma, k, h, size = float(p["gamma"]), float(p["k"]), float(p["h_risk"]), int(p["size"])
    base = math.log1p(gamma / k) / gamma
    q_max = _q_max(spec, p)

    def fn(obs):
        st = spec.decode(obs)
        both, mid, _ = _book(st)
        m = np.clip(np.nan_to_num(mid, nan=0.5), P_CLIP, 1 - P_CLIP)
        v_x = digital_var(m, st["tau"], h) / (m * (1 - m)) ** 2
        x_r = logit(m) - (st["q"] / st["min_size"]) * gamma * v_x
        d = 0.5 * gamma * v_x + base
        bid = np.where(both, sigmoid(x_r - d), np.nan)
        ask = np.where(both, sigmoid(x_r + d), np.nan)
        bid, ask = inventory_limit(bid, ask, st["q"], order_size(amap, st, size), q_max)
        return to_actions(amap, st, bid, ask, size)
    return fn


def fv_quotes(st, p):
    """Centre, half-spread and gap of the REPORT section-6 quoter (prices in $)."""
    both, mid, spread = _book(st)
    pm = np.clip(st["p_model"], 0.001, 0.999)
    mm = np.clip(np.nan_to_num(mid, nan=0.5), 0.001, 0.999)
    c5 = sigmoid(0.6 * logit(pm) + 0.32 * logit(mm))
    c = np.where(np.abs(st["T"] - 300) < 1, c5, 0.5 * (pm + mm))
    c = np.clip(c, 0.01, 0.99)
    gap = st["p_model"] - mid
    h = p["spread_mult"] * (np.maximum(np.maximum(st["tick"], 0.01), 0.5 * spread) + 0.5 * np.abs(gap))
    cs = c - p["k_skew"] * st["q"] / st["min_size"]
    return both, cs, h, gap


def fair_value(spec: ObsSpec, params=None):
    p = {**DEFAULTS["fair_value"], **(params or {})}
    amap = ActionMap(spec)
    spec.check(need_fair=True)
    size = int(p["size"])
    q_max = _q_max(spec, p)

    def fn(obs):
        st = spec.decode(obs, need_fair=True)
        both, cs, h, gap = fv_quotes(st, p)
        live = both & (st["tau"] >= p["pull_s"]) & (np.abs(gap) <= p["gap_max"])
        bid = np.where(live, cs - h, np.nan)
        ask = np.where(live, cs + h, np.nan)
        bid, ask = inventory_limit(bid, ask, st["q"], order_size(amap, st, size), q_max)
        return to_actions(amap, st, bid, ask, size)
    return fn


def fv_smart(spec: ObsSpec, params=None):
    p = {**DEFAULTS["fv_smart"], **(params or {})}
    amap = ActionMap(spec)
    spec.check(need_fair=True)
    size = int(p["size"])
    q_max = _q_max(spec, p)

    def fn(obs):
        st = spec.decode(obs, need_fair=True)
        both, cs, h, gap = fv_quotes(st, p)
        live = both & (st["tau"] >= p["pull_s"]) & np.isfinite(gap)
        bid = np.where(live & (gap >= -p["g_adv"]), cs - h, np.nan)
        ask = np.where(live & (gap <= p["g_adv"]), cs + h, np.nan)
        bid, ask = inventory_limit(bid, ask, st["q"], order_size(amap, st, size), q_max)
        return to_actions(amap, st, bid, ask, size, inside_only=True)
    return fn


FACTORIES = {"no_quote": no_quote, "sym_touch": sym_touch, "as_logit": as_logit,
             "fair_value": fair_value, "fv_smart": fv_smart}


def make(name: str, spec: ObsSpec, params=None):
    return FACTORIES[name](spec, params)
