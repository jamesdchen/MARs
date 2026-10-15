"""Evaluation statistics for the Polymarket market-making project (numpy only).

Everything here exists to make results hard to fool. See EVALUATION.md for the
protocol these functions implement.

Conventions
-----------
* P&L is in USDC. Positive = profit.
* Fills are expressed in the **Up-token frame**: side=+1 means we bought Up
  (equivalently sold Down), price/mids are Up probabilities. Use
  `to_up_frame` to convert Down-token fills (Down BUY @ p == Up SELL @ 1-p).
* Time-indexed statistics resample **blocks of calendar time across all
  markets**, never individual markets: markets live at the same time on
  correlated coins share their shocks. Aggregate to a time grid with
  `time_bins` first (zero-filled, so idle hours count), then bootstrap.

Main entry points
-----------------
time_bins, block_bootstrap, optimal_block_length, paired_difference_ci,
sharpe, cvar, attribute_pnl, to_up_frame, flag_pnl_share, concentration,
assign_split, deflated_sharpe, sha256_file, sha256_array, sha256_json.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "time_bins", "block_bootstrap", "optimal_block_length", "paired_difference_ci",
    "sharpe", "cvar", "attribute_pnl", "to_up_frame", "flag_pnl_share",
    "concentration", "assign_split", "deflated_sharpe", "sha256_file",
    "sha256_array", "sha256_json",
]

# --------------------------------------------------------------------------
# time aggregation
# --------------------------------------------------------------------------


def time_bins(t, values, bin_seconds: float, t_start: float | None = None,
              t_end: float | None = None):
    """Sum `values` (shape (n,) or (n, k)) into a dense, zero-filled time grid.

    t: event times in seconds (e.g. unix). Bins are [t_start + i*w, t_start + (i+1)*w).
    Returns (bin_starts, sums) with sums shaped (n_bins,) or (n_bins, k).
    Empty bins are 0, so quiet periods are not silently dropped from a bootstrap.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(values, dtype=float)
    if v.shape[0] != t.shape[0]:
        raise ValueError("t and values must have the same length")
    t0 = float(np.min(t)) if t_start is None else float(t_start)
    t1 = float(np.max(t)) + 1e-9 if t_end is None else float(t_end)
    nb = int(math.ceil((t1 - t0) / bin_seconds))
    idx = np.floor((t - t0) / bin_seconds).astype(np.int64)
    if np.any(idx < 0) or np.any(idx >= nb):
        raise ValueError("event times outside [t_start, t_end)")
    out = np.zeros((nb,) + v.shape[1:])
    np.add.at(out, idx, v)
    return t0 + bin_seconds * np.arange(nb), out


# --------------------------------------------------------------------------
# block bootstrap
# --------------------------------------------------------------------------


def _resample_indices(n: int, block_len: float, n_rep: int, kind: str,
                      rng: np.random.Generator) -> np.ndarray:
    """(n_rep, n) integer index arrays for one bootstrap chunk."""
    if kind == "iid" or block_len <= 1 and kind != "stationary":
        return rng.integers(0, n, size=(n_rep, n))
    if kind in ("moving", "circular"):
        b = int(round(block_len))
        nblk = int(math.ceil(n / b))
        hi = n - b + 1 if kind == "moving" else n
        starts = rng.integers(0, hi, size=(n_rep, nblk))
        idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_rep, -1)[:, :n]
        return idx % n
    if kind == "stationary":  # Politis & Romano (1994): geometric block lengths, circular
        p = 1.0 / max(block_len, 1.0)
        new = rng.random((n_rep, n)) < p
        new[:, 0] = True
        starts = rng.integers(0, n, size=(n_rep, n))
        pos = np.arange(n)[None, :]
        last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
        return (np.take_along_axis(starts, last, axis=1) + (pos - last)) % n
    raise ValueError(f"unknown kind {kind!r}")


def optimal_block_length(x, kind: str = "stationary") -> float:
    """Politis & White (2004) automatic block length, with the Patton, Politis &
    White (2009) correction. Returns a float >= 1 (round it for moving blocks)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 8:
        return 1.0
    x = x - x.mean()
    var = x @ x / n
    if var == 0:
        return 1.0
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    m_max = int(math.ceil(math.sqrt(n))) + kn
    lags = min(m_max + kn, n - 1)
    acov = np.array([x[: n - k] @ x[k:] / n for k in range(lags + 1)])
    rho = acov / acov[0]
    thresh = 2.0 * math.sqrt(math.log10(n) / n)
    m_hat = None
    for m in range(1, lags - kn + 1):
        if np.all(np.abs(rho[m: m + kn]) < thresh):
            m_hat = m - 1
            break
    if m_hat is None:
        m_hat = m_max
    big_m = min(2 * max(m_hat, 1), m_max, lags)
    k = np.arange(-big_m, big_m + 1)
    lam = np.clip(2.0 * (1.0 - np.abs(k) / big_m), 0.0, 1.0)  # flat-top kernel
    r = acov[np.abs(k)]
    g = np.sum(lam * np.abs(k) * r)
    lrv = np.sum(lam * r)
    d = (2.0 if kind == "stationary" else 4.0 / 3.0) * lrv ** 2
    if d <= 0 or g == 0:
        return 1.0
    b = (2.0 * g ** 2 / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, min(3.0 * math.sqrt(n), n / 3.0)))


def block_bootstrap(x, stat: Callable[[np.ndarray], float] | str = "mean",
                    block_len: float | None = None, n_boot: int = 2000,
                    kind: str = "stationary", alpha: float = 0.05,
                    method: str = "percentile", seed: int | None = 0,
                    min_block_len: float = 1.0) -> dict:
    """Block-bootstrap CI for `stat` of a time-indexed series.

    x: shape (T,) or (T, k); rows are consecutive time bins (already summed across
       ALL markets, e.g. hourly mark-to-market P&L), resampled jointly so the
       cross-sectional correlation inside a bin is preserved.
    stat: callable on an array shaped like x, or "mean" / "sum".
    kind: "stationary" (default), "moving", "circular" or "iid" (for comparison only).
    block_len: None -> Politis-White on x (first column if 2-D), floored at min_block_len.
    method: "percentile" or "basic" (reflected).
    Returns dict(estimate, lo, hi, se, block_len, n, n_boot, kind, alpha, method).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if isinstance(stat, str):
        f = {"mean": lambda a: a.mean(axis=0), "sum": lambda a: a.sum(axis=0)}[stat]
    else:
        f = stat
    if block_len is None:
        block_len = optimal_block_length(x if x.ndim == 1 else x[:, 0], kind)
    block_len = max(float(block_len), float(min_block_len))
    rng = np.random.default_rng(seed)
    est = float(np.asarray(f(x)))
    reps = np.empty(n_boot)
    chunk = max(1, min(n_boot, 2_000_000 // max(n, 1)))
    done = 0
    vectorised = isinstance(stat, str) and x.ndim == 1
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = _resample_indices(n, block_len, m, kind, rng)
        if vectorised:
            xs = x[idx]
            reps[done: done + m] = xs.mean(axis=1) if stat == "mean" else xs.sum(axis=1)
        else:
            for j in range(m):
                reps[done + j] = float(np.asarray(f(x[idx[j]])))
        done += m
    ql, qh = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    lo, hi = (ql, qh) if method == "percentile" else (2 * est - qh, 2 * est - ql)
    return dict(estimate=est, lo=float(lo), hi=float(hi), se=float(reps.std(ddof=1)),
                block_len=block_len, n=n, n_boot=n_boot, kind=kind, alpha=alpha,
                method=method)


def paired_difference_ci(a, b, unit_time=None, bin_seconds: float = 3600.0,
                         statistic: str = "mean_per_unit", t_start=None, t_end=None,
                         **boot_kw) -> dict:
    """CI for method A minus method B evaluated on IDENTICAL markets/paths.

    Two input modes:
    * unit_time is None: a, b are aligned time series (e.g. hourly MTM P&L of
      both methods over the same hours). Bootstraps d_t = a_t - b_t; statistic
      "mean_per_unit" -> mean per bin, "total" -> sum over bins.
    * unit_time given: a, b are per-unit values (e.g. per-market P&L), unit_time
      the time each unit is assigned to (e.g. market close). Differences and unit
      counts are summed per time bin, and whole bins are resampled in blocks, so
      simultaneous markets on different coins move together. "mean_per_unit"
      is sum(d)/count (a ratio estimator), "total" is sum(d).
    Extra keyword args go to block_bootstrap. Also returns corr(a, b), which
    shows how much variance the pairing removed.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be evaluated on identical units")
    d = a - b
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 0 and b.std() > 0 else float("nan")
    if unit_time is None:
        res = block_bootstrap(d, "mean" if statistic == "mean_per_unit" else "sum", **boot_kw)
    else:
        _, s = time_bins(unit_time, np.column_stack([d, np.ones_like(d)]), bin_seconds,
                         t_start, t_end)
        if statistic == "mean_per_unit":
            def f(m):
                c = m[:, 1].sum()
                return m[:, 0].sum() / c if c > 0 else 0.0
        elif statistic == "total":
            def f(m):
                return m[:, 0].sum()
        else:
            raise ValueError(statistic)
        res = block_bootstrap(s, f, **boot_kw)
    res.update(mean_a=float(a.mean()), mean_b=float(b.mean()), corr_ab=corr,
               statistic=statistic)
    return res


# --------------------------------------------------------------------------
# risk / performance metrics
# --------------------------------------------------------------------------


def sharpe(x, periods_per_year: float | None = None) -> float:
    """mean/std (ddof=1) of a per-period P&L series (e.g. hourly blocks).
    Not annualised unless periods_per_year is given (sqrt scaling assumes iid)."""
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=1)
    if s == 0 or x.size < 2:
        return float("nan")
    sr = x.mean() / s
    return float(sr * math.sqrt(periods_per_year)) if periods_per_year else float(sr)


def cvar(x, level: float = 0.95) -> float:
    """Expected P&L in the worst (1-level) fraction of outcomes (lower tail).

    Exact for discrete samples (the boundary observation gets fractional
    weight). Negative = expected loss in the tail. E.g. per-market P&L,
    level=0.95 -> CVaR95."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    k = (1.0 - level) * n
    if n == 0 or k <= 0:
        return float("nan")
    full = int(math.floor(k + 1e-12))
    frac = k - full
    tail = x[:full].sum() + (x[full] * frac if full < n and frac > 1e-12 else 0.0)
    return float(tail / k)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def deflated_sharpe(x, n_trials: int, var_trial_sharpes: float) -> dict:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of the *selected*
    configuration's per-period P&L x, given how many configurations were tried
    on the same data (n_trials, from the trial ledger) and the variance of their
    per-period Sharpe ratios. Returns the probability that the true SR > 0 after
    accounting for selection. Assumes roughly serially independent periods;
    treat as a sanity check, not a test of record."""
    x = np.asarray(x, dtype=float)
    t = x.size
    sr = sharpe(x)
    g = 0.5772156649015329
    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(max(var_trial_sharpes, 0.0)) * (
            (1 - g) * _norm_ppf(1 - 1.0 / n_trials) + g * _norm_ppf(1 - 1.0 / (n_trials * math.e)))
    xc = x - x.mean()
    sd = xc.std()
    skew = float((xc ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    kurt = float((xc ** 4).mean() / sd ** 4) if sd > 0 else 3.0
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    z = (sr - sr0) * math.sqrt(max(t - 1, 1)) / math.sqrt(max(den, 1e-12))
    return dict(sharpe=sr, sr0=sr0, dsr=_norm_cdf(z), n_trials=n_trials, skew=skew, kurt=kurt)


# --------------------------------------------------------------------------
# P&L attribution
# --------------------------------------------------------------------------


def to_up_frame(fills: Mapping[str, np.ndarray], is_down) -> dict:
    """Map Down-token fills into the Up frame: side -> -side, every price-like
    column (price, mid*, fair*) -> 1 - value. Fees/rebates/qty unchanged."""
    is_down = np.asarray(is_down, dtype=bool)
    out = {}
    for k, v in fills.items():
        v = np.asarray(v)
        if k == "side":
            out[k] = np.where(is_down, -_side(v), _side(v))
        elif k == "price" or k.startswith("mid") or k.startswith("fair"):
            out[k] = np.where(is_down, 1.0 - v.astype(float), v.astype(float))
        else:
            out[k] = v
    return out


def _side(s) -> np.ndarray:
    s = np.asarray(s)
    if s.dtype.kind in "USO":
        low = np.char.lower(s.astype(str))
        out = np.where(low == "buy", 1, np.where(low == "sell", -1, 0))
    else:
        out = np.sign(s.astype(float)).astype(int)
    if np.any(out == 0):
        raise ValueError("side must be +1/-1 or 'buy'/'sell'")
    return out


def attribute_pnl(fills: Mapping[str, np.ndarray], markets: Mapping[str, np.ndarray],
                  horizon: int = 60, inv_tol: float = 1e-9) -> dict:
    """Decompose market-making P&L into components that sum exactly to the total.

    fills (equal-length arrays, Up frame, one row per fill of OUR resting order):
      market      market id (int or str)
      side        +1 buy / -1 sell (or 'buy'/'sell')
      qty         shares (> 0)
      price       fill price
      mid         mid at fill time
      mid_<h>     mid h seconds after the fill, e.g. mid_5, mid_30, mid_60. NaN
                  (window closed before t+h) is replaced by the settlement value.
      fee         fee paid (cost, >= 0), optional
      rebate      maker rebate received, optional
      fair        optional fair value at fill (diagnostic edge vs fair only)
    markets (equal-length arrays, one row per market we were active in):
      market      market id
      settlement  Up-token payoff (1.0 / 0.0; or a mark if unresolved)
      reward      liquidity-reward accrual (USDC), optional
      final_inventory  inventory reported by the env, optional (checked)
      mid_close   optional mid at a fixed pre-close reference time; splits
                  inventory P&L into carry (to mid_close) and resolution jump.

    Per fill with signed quantity sq = side*qty, mid m0, mid_h at the primary
    horizon and settlement S:
      spread_capture    sq*(m0 - price)        edge vs mid at the moment of fill
      adverse_selection sq*(mid_h - m0)        markout; < 0 when picked off
      inventory         sq*(S - mid_h)         holding the position to resolution
    These telescope to sq*(S - price), the trading P&L. Then
      total = spread_capture + adverse_selection + inventory
              + fees(-fee) + rebates + rewards
    which equals cash + final_inventory*S exactly (checked: `identity_error`).
    Markouts at every supplied horizon are reported too (`markouts`).
    """
    mk_f = np.asarray(fills["market"])
    nf = mk_f.size
    sq = _side(fills["side"]) * np.asarray(fills["qty"], dtype=float)
    price = np.asarray(fills["price"], dtype=float)
    m0 = np.asarray(fills["mid"], dtype=float)
    fee = np.asarray(fills.get("fee", np.zeros(nf)), dtype=float)
    reb = np.asarray(fills.get("rebate", np.zeros(nf)), dtype=float)

    mk_m = np.asarray(markets["market"])
    ids = np.unique(np.concatenate([mk_m, mk_f]))
    fi = np.searchsorted(ids, mk_f)
    mi = np.searchsorted(ids, mk_m)
    if mk_f.size and not np.all(ids[fi] == mk_f):
        raise ValueError("market id lookup failed")
    nm = ids.size
    settle = np.full(nm, np.nan)
    settle[mi] = np.asarray(markets["settlement"], dtype=float)
    if np.any(np.isnan(settle[np.unique(fi)])):
        raise ValueError("every market with fills needs a settlement value")
    reward = np.zeros(nm)
    if "reward" in markets:
        reward[mi] = np.asarray(markets["reward"], dtype=float)
    S = settle[fi]

    hz = sorted(int(k[4:]) for k in fills if k.startswith("mid_") and k[4:].isdigit())
    if horizon not in hz:
        raise ValueError(f"primary horizon mid_{horizon} missing; have {hz}")
    mid_h = {}
    for h in hz:
        v = np.asarray(fills[f"mid_{h}"], dtype=float)
        mid_h[h] = np.where(np.isnan(v), S, v)

    per_fill = dict(
        spread_capture=sq * (m0 - price),
        adverse_selection=sq * (mid_h[horizon] - m0),
        inventory=sq * (S - mid_h[horizon]),
        fees=-fee,
        rebates=reb,
    )
    per_fill["total"] = sum(per_fill.values())

    def by_market(v):
        out = np.zeros(nm)
        np.add.at(out, fi, v)
        return out

    per_market = {k: by_market(v) for k, v in per_fill.items() if k != "total"}
    inv = by_market(sq)
    if "mid_close" in markets:
        mc = np.full(nm, np.nan)
        mc[mi] = np.asarray(markets["mid_close"], dtype=float)
        mc = np.where(np.isnan(mc), settle, mc)
        carry = by_market(sq * (mc[fi] - mid_h[horizon]))
        per_market["inventory_carry"] = carry
        per_market["resolution"] = per_market["inventory"] - carry
    per_market["rewards"] = reward
    comp_keys = ["spread_capture", "adverse_selection", "inventory", "fees", "rebates", "rewards"]
    per_market["total"] = sum(per_market[k] for k in comp_keys)

    # independent cash accounting
    cash = by_market(-sq * price - fee + reb) + reward
    direct = cash + inv * np.nan_to_num(settle)
    identity_error = float(np.max(np.abs(direct - per_market["total"]))) if nm else 0.0

    mismatch = []
    if "final_inventory" in markets:
        rep = np.zeros(nm)
        rep[mi] = np.asarray(markets["final_inventory"], dtype=float)
        bad = np.abs(rep - inv) > inv_tol
        mismatch = ids[bad].tolist()

    totals = {k: float(v.sum()) for k, v in per_market.items()}
    markouts = {h: float(np.sum(sq * (mid_h[h] - m0))) for h in hz}
    if "fair" in fills:
        totals["edge_vs_fair"] = float(np.sum(sq * (np.asarray(fills["fair"], float) - price)))
    gross = float(np.sum(np.abs(sq) * price))
    return dict(totals=totals, markouts=markouts, per_market=per_market, market_ids=ids,
                per_fill=per_fill, inventory=inv, identity_error=identity_error,
                inventory_mismatch=mismatch, n_fills=int(nf),
                volume_shares=float(np.abs(sq).sum()), notional=gross,
                horizon=horizon)


def flag_pnl_share(fill_pnl, flags: Mapping[str, np.ndarray]) -> dict:
    """Exploit detector: how much P&L comes from fills made under suspicious
    simulator conditions (crossed book, stale book, one-sided book, last N
    seconds, quote far inside the touch, ...). flags: name -> bool per fill."""
    pnl = np.asarray(fill_pnl, dtype=float)
    tot = pnl.sum()
    out = {}
    for name, f in flags.items():
        f = np.asarray(f, dtype=bool)
        p = float(pnl[f].sum())
        out[name] = dict(n=int(f.sum()), frac_fills=float(f.mean()) if f.size else 0.0,
                         pnl=p, share_of_total=p / tot if tot > 0 else float("nan"),
                         pnl_excluding=float(tot - p))
    anyf = np.zeros(pnl.size, bool)
    for f in flags.values():
        anyf |= np.asarray(f, dtype=bool)
    out["_any"] = dict(n=int(anyf.sum()), pnl=float(pnl[anyf].sum()),
                       share_of_total=float(pnl[anyf].sum() / tot) if tot > 0 else float("nan"),
                       pnl_excluding=float(pnl[~anyf].sum()))
    return out


# --------------------------------------------------------------------------
# concentration
# --------------------------------------------------------------------------


def concentration(pnl, ks: Sequence[int] = (1, 5, 10)) -> dict:
    """How much of the result rests on a few markets / hours.

    pnl: P&L per group (market, hour, coin...). Returns total, gross profit,
    for each k: share of total and of gross profit from the top-k groups and the
    P&L left after removing them; Herfindahl of |P&L| shares and its inverse
    (effective number of groups); the number of groups needed to reach half the
    gross profit; and the fraction of groups with positive P&L."""
    x = np.asarray(pnl, dtype=float)
    tot = float(x.sum())
    srt = np.sort(x)[::-1]
    gp = float(x[x > 0].sum())
    out = dict(total=tot, gross_profit=gp, n=int(x.size),
               frac_positive=float((x > 0).mean()) if x.size else float("nan"))
    for k in ks:
        top = float(srt[:k].sum())
        out[f"top{k}_share_of_total"] = top / tot if tot > 0 else float("nan")
        out[f"top{k}_share_of_gross_profit"] = float(np.clip(srt[:k], 0, None).sum()) / gp if gp > 0 else float("nan")
        out[f"ex_top{k}"] = tot - top
    a = np.abs(x)
    if a.sum() > 0:
        w = a / a.sum()
        hhi = float((w ** 2).sum())
        out.update(hhi_abs=hhi, n_effective=1.0 / hhi)
    else:
        out.update(hhi_abs=float("nan"), n_effective=float("nan"))
    if gp > 0:
        c = np.cumsum(np.clip(srt, 0, None))
        out["n_for_half_profit"] = int(np.searchsorted(c, 0.5 * gp) + 1)
    else:
        out["n_for_half_profit"] = 0
    return out


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def assign_split(start, end, boundaries, labels, embargo: float = 0.0) -> np.ndarray:
    """Assign each market [start, end] to a split, purging straddlers.

    boundaries: sorted edges b0 < b1 < ... < bk (e.g. UTC midnights, seconds);
    labels: k names, interval i = [b_i, b_{i+1}). A market belongs to split i
    only if its whole life fits inside the interval, shrunk by `embargo` seconds
    at internal boundaries. Everything else gets None (purged)."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    b = np.asarray(boundaries, dtype=float)
    k = len(labels)
    if b.size != k + 1 or np.any(np.diff(b) <= 0):
        raise ValueError("need k+1 increasing boundaries for k labels")
    out = np.full(start.size, None, dtype=object)
    for i, lab in enumerate(labels):
        lo = b[i] + (embargo if i > 0 else 0.0)
        hi = b[i + 1] - (embargo if i < k - 1 else 0.0)
        out[(start >= lo) & (end <= hi)] = lab
    return out


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_array(a) -> str:
    """Hash of dtype, shape and C-order bytes (so a reshape or cast changes it)."""
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "O":
        raise TypeError("object arrays have no stable byte representation")
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_json(obj) -> str:
    """Hash of canonical JSON (sorted keys, no whitespace), e.g. for TEST_PLAN."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(s.encode()).hexdigest()
