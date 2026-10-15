#!/usr/bin/env python3
"""Fit every generator parameter on the TRAIN split only (431 markets, opens 2026-08-21..22 UTC) -> sim/params.json.

Data used (all train-only, see common.py loaders and their leakage asserts):
  * data/build/ticks.bin rows of split-0 markets (1 s book + trade aggregates, block-time stamped)
  * data/raw/trades.parquet legs of split-0 slugs (taker + maker legs)
  * data/raw/markets.parquet price_to_beat / final_price of split-0 slugs (settlement basis)
  * Binance 1 s closes 2026-08-20..22 (08-20 = EWMA warm-up), via fair_value.load_underlying()

Components (each simple, documented in README.md):
  settlement   open-level basis per coin, basis drift sd                     (moments)
  mid          logit(mid) = s*logit(p) + b(len,tau) + h + u                   (distributed lag on snapshot pairs,
               h: partial adjustment (a0 same-second, phi_h decay)            AR(1) of the residual)
  book         regime {normal, hollow, one-sided}: persistent Markov,        (empirical conditional tables)
               spread / level gaps / log-sizes / depth ratio / refresh rate
  flow         taker counts: linear Hawkes, baseline from a Poisson GLM with market effects (profile MLE);
               direction P(buy | gap bucket), copy prob; size quantiles; sweep depth; legs; tokens
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

import common as cm

MATCH_LAG = 3             # block time - match time used to align flow covariates (COVERAGE: 3-4 s)
EXCL_END = 30             # mid fits drop the last 30 s (p saturates, books one-sided)
P_LO, P_HI = 0.03, 0.97   # mid fits use rows with p and mid inside this band (logit well behaved)


def bucket_table(keys, states, n_states, shape, alpha=20.0, parent=None):
    """Smoothed conditional distribution table. keys: tuple of int arrays (cell index); states: int array."""
    cnt = np.zeros(shape + (n_states,))
    np.add.at(cnt, keys + (states,), 1.0)
    if parent is None:
        parent = cnt.reshape(-1, n_states).sum(0)
        parent = np.broadcast_to(parent / parent.sum(), cnt.shape)
    tab = cnt + alpha * parent
    tab /= tab.sum(-1, keepdims=True)
    return tab, cnt


def rounded(x, nd=6):
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return rounded(x.tolist(), nd)
    if isinstance(x, (float, np.floating)):
        return float(round(float(x), nd))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ============================================================================================ settlement
def fit_settlement(mk: pd.DataFrame) -> dict:
    raw = pd.read_parquet(cm.RAW / "markets.parquet", columns=["slug", "price_to_beat", "final_price", "winner_up"])
    d = mk.merge(raw, on="slug", how="left", suffixes=("", "_g"))
    ok = d.price_to_beat.notna() & d.final_price.notna() & (d.price_to_beat > 0)
    d = d[ok]
    b_open = 1e4 * np.log(d.price_to_beat / d.K)
    b_end = 1e4 * np.log(d.final_price / d.twap_end)
    drift = b_end - b_open
    agree = int(((d.twap_end >= d.K).astype(int) == d.winner_up).sum())
    out = {
        "open_basis_bp_by_coin": {c: float(np.median(b_open[d.coin == c])) for c in cm.COINS if (d.coin == c).any()},
        "sigma_basis_bp": float(np.std(drift)),
        "sigma_basis_bp_by_len": {cm.LEN_NAME[L]: float(np.std(drift[d.L == L])) for L in cm.LENS},
        "binance_twap_outcome_agrees": f"{agree}/{len(d)}",
        "n_markets": int(len(d)),
        "note": "price_to_beat (Chainlink) = K_binance * exp(open_basis); sim settles Up iff "
                "TWAP60_binance(end) * exp(eps) >= K_binance, eps ~ N(0, sigma_basis^2) (0 in the base sim)",
    }
    return out


# ============================================================================================ mid process
MID_H = {300: [1, 5, 30, 120], 900: [1, 5, 30, 120, 300], 14400: [1, 5, 30, 120, 300, 900]}
ACF_LAGS = [5, 30, 120]


def mid_moments(y, x, good, mi, fine_unused=None, H=(1, 5, 30), ok_y=None):
    """Moments of the displayed logit mid y vs model logit x on `good` rows (same market for lagged pairs)."""
    ok = good if ok_y is None else (good & ok_y)
    out = {}
    idx = np.flatnonzero(ok)
    for h in H:
        j = idx + h
        v = j < len(x)
        i_, j_ = idx[v], j[v]
        v = (mi[i_] == mi[j_]) & ok[j_]
        i_, j_ = i_[v], j_[v]
        dx, dy = x[j_] - x[i_], y[j_] - y[i_]
        out[f"slope_{h}"] = float(np.sum(dx * dy) / np.sum(dx * dx))
    xs, ys = x[ok], y[ok]
    out["level_slope"] = float(np.cov(xs, ys)[0, 1] / np.var(xs))
    g = ys - xs
    out["sd_gap"] = float(g.std())
    for k in ACF_LAGS:
        i_ = idx[(idx + k) < len(x)]
        j_ = i_ + k
        v = (mi[i_] == mi[j_]) & ok[j_]
        a, b = (y - x)[i_[v]], (y - x)[j_[v]]
        out[f"acf_{k}"] = float(np.corrcoef(a, b)[0, 1])
    i_ = idx[idx >= 1]
    v = (mi[i_ - 1] == mi[i_]) & ok[i_ - 1]
    out["p_change"] = float((np.abs(y[i_[v]] - y[i_[v] - 1]) > 1e-9).mean())
    # how the gap (logit p - logit mid) predicts later moves of p and of the mid (the real mid partly
    # anticipates p: fair_value REPORT section 3)
    for h in (30, 60):
        j = idx + h
        vv = j < len(x)
        i2, j2 = idx[vv], j[vv]
        vv = (mi[i2] == mi[j2]) & ok[j2]
        i2, j2 = i2[vv], j2[vv]
        gap = x[i2] - y[i2]
        out[f"gapdp_{h}"] = float(np.sum(gap * (x[j2] - x[i2])) / np.sum(gap * gap))
        out[f"gapdy_{h}"] = float(np.sum(gap * (y[j2] - y[i2])) / np.sum(gap * gap))
    return out


MOM_SCALE = dict(level_slope=0.03, sd_gap=0.03, p_change=0.02, **{f"acf_{k}": 0.03 for k in ACF_LAGS},
                 gapdp_30=0.03, gapdp_60=0.04, gapdy_30=0.04, gapdy_60=0.05)
ANT_H = {300: 60, 900: 120, 14400: 600}   # anticipation horizon of the mid (s)


def sim_mid_paths(X, TB, pi_prof, th, U, Z, b_tau=None):
    """Sticky mid on padded arrays [n_mkt, T]: target = s_f*x + (s_l-s_f)*ewma(x) + u (+ b[tau]);
    the displayed latent mid jumps to the target at update events (prob pi per second), else holds."""
    s_f, s_l, hl_slow, hl_u, sig_u, pi_mult = th[:6]
    psi, ant_h = (th[6], th[7]) if len(th) > 6 else (0.0, 60)
    d = 0.5 ** (1.0 / hl_slow)
    xb = lfilter([1 - d], [1, -d], X, axis=1, zi=(d * X[:, :1]))[0]
    phi = 0.5 ** (1.0 / hl_u)
    u = lfilter([sig_u], [1, -phi], Z, axis=1, zi=(phi * Z[:, :1] * sig_u / np.sqrt(1 - phi ** 2)))[0]
    fund = s_f * X + (s_l - s_f) * xb
    T = fund + psi * anticipation(fund, ant_h) + u
    if b_tau is not None:
        T = T + b_tau[TB]
    ev = U < np.clip(pi_mult * pi_prof[TB], 0, 1)
    ev[:, 0] = True
    last = np.where(ev, np.arange(X.shape[1])[None, :], 0)
    last = np.maximum.accumulate(last, axis=1)
    return np.take_along_axis(T, last, axis=1)


def anticipation(fund, H):
    """mean of fund over (t, t+H] (truncated at the market end) minus fund_t, per row of a padded [B, T] array."""
    B, T = fund.shape
    cs = np.concatenate([np.zeros((B, 1)), np.cumsum(fund, axis=1)], axis=1)
    t = np.arange(T)
    hi = np.minimum(t + int(H), T - 1)
    n = np.maximum(hi - t, 1)
    fut = (cs[:, hi + 1] - cs[:, t + 1]) / n[None, :]
    return np.where((hi > t)[None, :], fut - fund, 0.0)


def quantize_mid(ell, fine):
    m = cm.sigmoid(ell)
    q = np.where(fine, 0.0005, 0.005)
    return np.round(m / q) * q


def fit_mid(P: dict, mk: pd.DataFrame, reg, mid, fine, pi_prof, warm: dict | None = None) -> dict:
    """Simulated method of moments on the real train Binance paths (common random numbers).
    pi_prof[len, tau_b]: profile of the per-second book-update probability (scaled by pi_mult)."""
    off = np.concatenate([[0], np.cumsum(mk.n_rows.values)])
    tau = P["seconds_to_close"]
    x_all = cm.logit(P["p"], 1e-4)
    y_all = cm.logit(np.clip(mid, 1e-4, 1 - 1e-4), 1e-4)
    good = ((reg == cm.NORMAL) & (P["book_valid"] == 1) & (P["p"] > P_LO) & (P["p"] < P_HI)
            & (mid > P_LO) & (mid < P_HI) & (tau > EXCL_END))
    rng = np.random.default_rng(12345)
    out = {}
    for li, L in enumerate(cm.LENS):
        mis = np.flatnonzero(mk.L.values == L)
        n = L + 1
        rows = np.concatenate([np.arange(off[m], off[m + 1]) for m in mis])
        X = x_all[rows].reshape(len(mis), n)
        Y = y_all[rows].reshape(len(mis), n)
        G = good[rows].reshape(len(mis), n)
        FN = fine[rows].reshape(len(mis), n)
        TB = cm.tau_bucket(np.minimum(tau[rows], L - 1)).reshape(len(mis), n)
        MI = np.repeat(np.arange(len(mis)), n)
        H = MID_H[L]
        real = mid_moments(Y.ravel(), X.ravel(), G.ravel(), MI, H=H)
        U = rng.random(X.shape)
        Z = rng.standard_normal(X.shape)
        prof = np.asarray(pi_prof[li])

        def run(th, final=False):
            ell0 = sim_mid_paths(X, TB, prof, th, U, Z)
            r = np.where(G, Y - ell0, np.nan)
            b = np.array([np.nanmean(r[TB == k]) if np.isfinite(r[TB == k]).sum() > 100 else np.nan
                          for k in range(cm.NTAU)])
            okb = np.flatnonzero(np.isfinite(b))
            for k in range(cm.NTAU):
                if not np.isfinite(b[k]):
                    b[k] = b[okb[np.argmin(np.abs(okb - k))]]
            ell = sim_mid_paths(X, TB, prof, th, U, Z, b_tau=b)
            ms = quantize_mid(ell, FN)
            ys = cm.logit(np.clip(ms, 1e-4, 1 - 1e-4), 1e-4)
            okm = (ms > P_LO) & (ms < P_HI)
            mom = mid_moments(ys.ravel(), X.ravel(), G.ravel(), MI, H=H, ok_y=okm.ravel())
            return mom, b

        AH = ANT_H[L]

        def unpack(z):
            return (z[0], z[1], np.exp(z[2]), np.exp(z[3]), np.exp(z[4]), np.exp(z[5]), z[6], AH)

        def loss(z):
            th = unpack(z)
            if not (0 < th[0] < 2 and 0 < th[1] < 2 and 2 < th[2] < 5000 and 2 < th[3] < 50000 and th[5] < 5
                    and 0 <= th[6] < 2):
                return 1e6
            mom, _ = run(th)
            return sum(((mom[k] - real[k]) / MOM_SCALE.get(k, 0.03)) ** 2 for k in real)

        prev = (warm or {}).get(cm.LEN_NAME[L])
        if prev:
            z0 = np.array([prev["s_fast"], prev["s_slow"], np.log(prev["halflife_slow_s"]),
                           np.log(prev["halflife_u_s"]), np.log(prev["sigma_u"]), np.log(prev["update_prob_mult"]), 0.2])
            starts = (z0,)
        else:
            z0 = np.array([0.5, 0.9, np.log(120.0), np.log(300.0), np.log(0.03), 0.0, 0.2])
            starts = (z0, z0 + np.array([0.2, -0.1, 1.0, 1.5, 0.5, -0.5, 0.2]))
        best = None
        for start in starts:
            res = minimize(loss, start, method="Nelder-Mead",
                           options=dict(maxiter=400, xatol=1e-3, fatol=1e-3, adaptive=True))
            if best is None or res.fun < best.fun:
                best = res
        z = best.x
        th = unpack(z)
        mom, b = run(th)
        out[cm.LEN_NAME[L]] = dict(
            s_fast=th[0], s_slow=th[1], halflife_slow_s=th[2], halflife_u_s=th[3], sigma_u=th[4],
            update_prob_mult=th[5], update_prob=np.clip(th[5] * prof, 0, 1), b_tau=b, psi=th[6], ant_h_s=AH,
            moments_real=real, moments_sim=mom, smm_loss=float(best.fun), n_markets=int(len(mis)),
            note="target = fund + psi * (mean of fund over (t, t+ant_h] - fund_t) + b(tau) + u, "
                 "fund = s_fast*logit(p) + (s_slow-s_fast)*EWMA_hl(logit p), u AR(1) "
                 "(half-life halflife_u_s, innovation sd sigma_u); the book re-aligns to the target only at "
                 "update events (prob update_prob[tau_b] per second) and holds otherwise")
        print(f"  mid {cm.LEN_NAME[L]}: loss {best.fun:.2f} th={np.round(th, 4)}")
        print("    real", {k: round(v, 3) for k, v in real.items()})
        print("    sim ", {k: round(v, 3) for k, v in mom.items()})
    return out


# ============================================================================================ book
def fine_flags(P: dict, mk: pd.DataFrame):
    """Per row: market already on the 0.001 grid (any top-5 price off the 0.01 grid so far; sticky)."""
    pr = np.column_stack([P[f"{s}_px_{k}"] for s in ("bid", "ask") for k in range(cm.NLEV)])
    off = (np.abs(np.round(pr * 100) - pr * 100) > 1e-3) & (pr > 0) & (pr < 1)
    offr = off.any(1)
    fine = np.zeros(len(offr), bool)
    o = np.concatenate([[0], np.cumsum(mk.n_rows.values)])
    for m in range(len(mk)):
        fine[o[m]:o[m + 1]] = np.maximum.accumulate(offr[o[m]:o[m + 1]])
    return fine


def fit_book(P: dict, mk: pd.DataFrame, reg, mid, spread) -> dict:
    o = np.concatenate([[0], np.cumsum(mk.n_rows.values)])
    valid = P["book_valid"] == 1
    tau = P["seconds_to_close"]
    Lr = P["L"]
    li = np.searchsorted(cm.LENS, Lr)
    tb = cm.tau_bucket(np.minimum(tau, Lr - 1))
    eb = cm.ext_bucket(P["p"])
    fine = fine_flags(P, mk)
    same_mkt = np.r_[False, P["m"][1:] == P["m"][:-1]]
    out = {}

    # ---- tick switch: max |mid-0.5| on normal rows before the switch, markets that switched
    pre_max, switched = [], []
    for m in range(len(mk)):
        a, b = o[m], o[m + 1]
        f = fine[a:b]
        nr = (reg[a:b] == cm.NORMAL) & valid[a:b] & ~f
        if f.any():
            switched.append(m)
            if nr.any():
                pre_max.append(np.abs(mid[a:b][nr] - 0.5).max())
    out["tick_switch"] = dict(ext_threshold=float(np.median(pre_max)), n_switched=len(switched), n_markets=len(mk),
                              note="tick 0.01 -> 0.001 (sticky) once |latent mid - 0.5| exceeds ext_threshold")

    # ---- regime: P(one-sided | len, tau_b, ext_b) pooled over coins; P(hollow | not one-sided, len, coin group,
    # tau_b) (hollow books are a coin effect: ZEC); persistence via a Gaussian copula (below)
    v = valid
    cg = np.array([cm.COIN_GROUP[c] for c in mk.coin.values])[P["m"]]
    tab, cnt = bucket_table((li[v], tb[v], eb[v]), reg[v], 3, (3, cm.NTAU, cm.NEXT), alpha=20.0)
    vn = v & (reg != cm.ONESIDED)
    hc = np.zeros((3, cm.NCG, cm.NTAU, 2))
    np.add.at(hc, (li[vn], cg[vn], tb[vn], (reg[vn] == cm.HOLLOW).astype(int)), 1.0)
    hpar = hc.sum(2, keepdims=True)
    hpar = (hpar[..., 1] + 0.5) / (hpar.sum(-1) + 1.0)                        # (len, cg) hollow rate
    htab = (hc[..., 1] + 50.0 * hpar) / (hc.sum(-1) + 50.0)                   # [3, NCG, NTAU]
    keep = {}
    for k, name in ((cm.NORMAL, "normal"), (cm.HOLLOW, "hollow"), (cm.ONESIDED, "onesided")):
        prev = np.r_[-1, reg[:-1]]
        sel = same_mkt & v & (prev == k)
        keep[name] = [float((reg[sel & (li == j)] == k).mean()) if (sel & (li == j)).sum() > 50 else 0.99
                      for j in range(3)]
    # Persistence through a Gaussian copula: per market, z^O and z^H are stationary AR(1) N(0,1) processes;
    # one-sided iff Phi(z^O) < pi_O(cell_t), else hollow iff Phi(z^H) < pi_H / (1 - pi_O). The marginal equals
    # the table exactly and the state follows the conditions immediately; the AR half-lives are chosen so the
    # simulated P(S_t = S | S_{t-1} = S) on the real train p-paths matches the observed one.
    from scipy.special import ndtr
    hl_grid = [20, 80, 320, 1280, 5120, np.inf]           # copula AR half-life (inf = fixed per market)
    hp_grid = [0, 3, 10, 30, 100, 300]                      # EWMA half-life applied to pi(cell_t)
    rng = np.random.default_rng(3)
    cop = {}
    for j, L in enumerate(cm.LENS):
        mis = np.flatnonzero(mk.L.values == L)
        rows = np.concatenate([np.arange(o[m], o[m + 1]) for m in mis])
        sh = (len(mis), L + 1)
        tO = tab[j][tb[rows], eb[rows], cm.ONESIDED].reshape(sh)
        tH = htab[j][cg[rows], tb[rows]].reshape(sh)
        reps = 4
        Z = rng.standard_normal((reps * len(mis), L + 1))
        res_ = {}
        for name, pi0, target in (("onesided", tO, keep["onesided"][j]), ("hollow", tH, keep["hollow"][j])):
            real_frac = float((reg[rows] == (cm.ONESIDED if name == "onesided" else cm.HOLLOW)).mean())
            best = None
            for hp in hp_grid:
                pi = pi0 if hp == 0 else lfilter([1 - 0.5 ** (1 / hp)], [1, -0.5 ** (1 / hp)], pi0, axis=1,
                                                 zi=0.5 ** (1 / hp) * pi0[:, :1])[0]
                pi = np.tile(pi, (reps, 1))
                for hl in hl_grid:
                    if np.isinf(hl):
                        z = np.repeat(Z[:, :1], L + 1, 1)
                    else:
                        ph = 0.5 ** (1.0 / hl)
                        z = lfilter([np.sqrt(1 - ph ** 2)], [1, -ph], Z, axis=1, zi=ph * Z[:, :1])[0]
                    s = ndtr(z) < pi
                    pss = (s[:, 1:] & s[:, :-1]).sum() / max(s[:, :-1].sum(), 1)
                    err = abs(pss - target) + 0.5 * abs(s.mean() - real_frac)
                    if best is None or err < best[0]:
                        best = (err, hp, hl, float(pss), float(s.mean()))
            res_[name] = dict(pi_ewma_halflife_s=best[1], copula_halflife_s=(None if np.isinf(best[2]) else best[2]),
                              p_same_sim=best[3], frac_sim=best[4], p_same_real=target, frac_real=real_frac)
        cop[cm.LEN_NAME[L]] = res_
    # side that is empty when one-sided: the extreme side (ask empty when p > 0.5)?
    os_ = v & (reg == cm.ONESIDED)
    noask = P["ask_px_0"] >= 1
    agree_side = float(((P["p"] > 0.5) == noask)[os_].mean())
    hollow_open = [float((reg[o[:-1]][mk.L.values == L] == cm.HOLLOW).mean()) for L in cm.LENS]
    out["regime"] = dict(states=["normal", "hollow", "onesided"], table=tab, counts_total=cnt.sum((0, 1, 2)),
                         hollow_given_not_onesided=htab, coin_groups=cm.CG_NAMES,
                         p_same_state_observed=keep, copula=cop, onesided_extreme_side_agree=agree_side,
                         p_hollow_at_open=hollow_open,
                         note="P(one-sided) = table[len, tau_b, ext_b(p_model)][2]; P(hollow | not one-sided) = "
                              "hollow_given_not_onesided[len, coin_group, tau_b]; per market Gaussian-copula AR(1) "
                              "thresholds give the persistence (copula half-lives fitted to P(S_t=S|S_t-1=S)); "
                              "one-sided removes the side beyond the extreme (ask if p_model > 0.5)")

    # ---- spread (normal rows) by (len, tau_b, ext_b, fine) over pruned support (mils)
    nr = v & (reg == cm.NORMAL)
    smils = np.round(spread * 1000).astype(int)
    vals, cts = np.unique(smils[nr], return_counts=True)
    keepv = vals[(cts / cts.sum() > 2e-4)]
    keepv = keepv[keepv >= 1]
    sidx = np.abs(smils[:, None] - keepv[None, :]).argmin(1)
    spr = {}
    for fflag in (0, 1):
        sel = nr & (fine == fflag)
        if fflag == 0:   # coarse grid: only multiples of 10 mils are reachable
            allowed = (keepv % 10) == 0
        else:
            allowed = np.ones(len(keepv), bool)
        sup = keepv[allowed]
        idxmap = -np.ones(len(keepv), int)
        idxmap[allowed] = np.arange(allowed.sum())
        si = idxmap[sidx[sel]]
        okk = si >= 0
        par_cnt = np.zeros((3, cm.NEXT, len(sup)))
        np.add.at(par_cnt, (li[sel][okk], eb[sel][okk], si[okk]), 1.0)
        par = (par_cnt + 1.0) / (par_cnt + 1.0).sum(-1, keepdims=True)              # (len, ext)
        pc_cnt = np.zeros((3, cm.NCG, cm.NEXT, len(sup)))
        np.add.at(pc_cnt, (li[sel][okk], cg[sel][okk], eb[sel][okk], si[okk]), 1.0)
        parc = (pc_cnt + 30.0 * par[:, None]) / (pc_cnt.sum(-1, keepdims=True) + 30.0)   # (len, cg, ext)
        parc = np.broadcast_to(parc[:, :, None], (3, cm.NCG, cm.NTAU, cm.NEXT, len(sup)))
        t_, c_ = bucket_table((li[sel][okk], cg[sel][okk], tb[sel][okk], eb[sel][okk]), si[okk], len(sup),
                              (3, cm.NCG, cm.NTAU, cm.NEXT), alpha=30.0, parent=parc)
        spr["fine" if fflag else "coarse"] = dict(support_mils=sup, table=t_, n=int(okk.sum()))
    out["spread"] = spr

    # ---- level gaps (normal rows, both sides pooled) by (fine, level)
    gaps = {}
    for fflag in (0, 1):
        sel = nr & (fine == fflag)
        per = []
        for k in range(1, cm.NLEV):
            g = []
            for s_, sign in (("bid", 1), ("ask", -1)):
                p0, p1 = P[f"{s_}_px_{k-1}"], P[f"{s_}_px_{k}"]
                pres = (p1 > 0) & (p1 < 1) & (p0 > 0) & (p0 < 1)
                g.append(np.round(sign * (p0 - p1) * 1000)[sel & pres])
            g = np.concatenate(g).astype(int)
            g = g[g >= 1]
            vv, cc = np.unique(g, return_counts=True)
            m_ = cc / cc.sum() > 0.002
            per.append(dict(support_mils=vv[m_], p=cc[m_] / cc[m_].sum()))
        gaps["fine" if fflag else "coarse"] = per
    out["gaps"] = gaps

    # ---- sizes: log-normal per level and length (normal rows); persistence at refresh
    sizes = {}
    for j, L in enumerate(cm.LENS):
        mu, sd = [], []
        for k in range(cm.NLEV):
            z = np.concatenate([np.log(P[f"{s_}_sz_{k}"][nr & (li == j) & (P[f"{s_}_sz_{k}"] > 0)])
                                for s_ in ("bid", "ask")])
            mu.append(z.mean())
            sd.append(z.std())
        sizes[cm.LEN_NAME[L]] = dict(mu=mu, sd=sd)
    # book-change profile: P(book_age == 0) by (len, tau_b) -> profile of the update probability (the mid fit
    # scales it); per-second autocorrelation of log size at level 0 while best prices are unchanged
    fresh = P["book_age"] == 0
    age0 = np.zeros((3, cm.NTAU))
    for j in range(3):
        for k in range(cm.NTAU):
            sel = v & (li == j) & (tb == k)
            age0[j, k] = fresh[sel].mean() if sel.sum() > 100 else np.nan
        good = np.isfinite(age0[j])
        age0[j, ~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), age0[j, good])
    unchanged = same_mkt & v & np.r_[False, (P["bid_px_0"][1:] == P["bid_px_0"][:-1])
                                     & (P["ask_px_0"][1:] == P["ask_px_0"][:-1])]
    r1 = []
    for j in range(3):
        sel = unchanged & (li == j) & nr & np.r_[False, nr[:-1]]
        z1 = np.log(np.maximum(P["bid_sz_0"][sel], 1e-3))
        z0 = np.log(np.maximum(np.r_[0, P["bid_sz_0"][:-1]][sel], 1e-3))
        r1.append(float(np.corrcoef(z0, z1)[0, 1]))
    sizes["acf1_log_size_per_second"] = r1
    sizes["note"] = ("log size ~ N(mu[level], sd[level]); redrawn at update events as AR(1) with "
                     "rho = (acf1 - (1 - pi_update)) / pi_update so the per-second autocorrelation matches")
    out["size"] = sizes
    out["book_change_profile"] = dict(p_age0=age0, note="P(book_age == 0) per (len, tau_b) in train ticks")
    # state dependence of book updates: P(age0 | len, |dp_model over 1 s| bucket) / P(age0 | len) (normal rows,
    # tau > 30 s). Flat for 5m/15m; on 4h snapshots follow fair-value moves (0.14 -> 0.36).
    dp1 = np.abs(P["p"] - np.r_[P["p"][0], P["p"][:-1]]) * 100
    dp1 = np.where(same_mkt, dp1, 0.0)
    edges = np.array([0.0, 0.1, 0.3, 1.0, 2.0, 5.0, np.inf])
    db = np.clip(np.searchsorted(edges, dp1, side="right") - 1, 0, len(edges) - 2)
    gain = np.ones((3, len(edges) - 1))
    for j in range(3):
        sel = v & (li == j) & (reg == cm.NORMAL) & (tau > 30)
        base = fresh[sel].mean()
        for k in range(len(edges) - 1):
            s2 = sel & (db == k)
            if s2.sum() > 200:
                gain[j, k] = fresh[s2].mean() / base
    out["update_dp_gain"] = dict(edges_cents=edges[:-1], gain=gain,
                                 note="update probability multiplier by |p_model change over the last second| "
                                      "(cents, bucket lower edges); normalised so the mean rate is kept")
    # empirical size library: joint top-5 bid/ask size vectors of real normal rows by (len, tau_b) plus the
    # log ratio of total depth to top-5 depth (resampled whole at update events: keeps round-lot clustering,
    # level correlation and missing levels); hollow rows separately
    lib = {}
    top5b = sum(P[f"bid_sz_{k}"] for k in range(cm.NLEV))
    top5a = sum(P[f"ask_sz_{k}"] for k in range(cm.NLEV))
    rb = np.log(np.maximum(P["bid_depth_total"], 1e-6) / np.maximum(top5b, 1e-6))
    ra = np.log(np.maximum(P["ask_depth_total"], 1e-6) / np.maximum(top5a, 1e-6))
    vec = np.column_stack([P[f"bid_sz_{k}"] for k in range(cm.NLEV)] + [P[f"ask_sz_{k}"] for k in range(cm.NLEV)]
                          + [np.maximum(rb, 0), np.maximum(ra, 0)]).astype(np.float32)
    rng_l = np.random.default_rng(11)
    for j, L in enumerate(cm.LENS):
        for k in range(cm.NTAU):
            pooled = np.flatnonzero(nr & (li == j) & (tb == k) & (top5b > 0) & (top5a > 0))
            if len(pooled) == 0:
                continue
            for g in range(cm.NCG):
                sel = np.flatnonzero(nr & (li == j) & (tb == k) & (cg == g) & (top5b > 0) & (top5a > 0))
                if len(sel) < 300:                       # thin cell -> pooled over coin groups
                    sel = pooled
                pick = rng_l.choice(sel, min(3000, len(sel)), replace=False)
                lib[f"size_{j}_{g}_{k}"] = vec[pick]
        hsel = np.flatnonzero((reg == cm.HOLLOW) & v & (li == j))
        if len(hsel):
            lib[f"hollow_{j}"] = vec[rng_l.choice(hsel, min(2000, len(hsel)), replace=False)]
    np.savez_compressed(cm.HERE / "params_book.npz", **lib)
    out["size_library"] = dict(file="params_book.npz", vectors_per_cell=3000, n_cells=len(lib),
                               key="size_<len_idx>_<coin_group>_<tau_b>, hollow_<len_idx>",
                               keep_prob_note="at an update event the previous vector is kept with prob "
                                              "1 - (1 - acf1)/pi_update, else a fresh one is drawn")
    # depth total / top-5 ratio
    top5 = sum(P[f"bid_sz_{k}"] for k in range(cm.NLEV))
    ratio = np.log(np.maximum(P["bid_depth_total"], 1e-6) / np.maximum(top5, 1e-6))
    out["depth_ratio"] = {cm.LEN_NAME[L]: dict(mu=float(ratio[nr & (li == j) & (top5 > 0)].mean()),
                                               sd=float(ratio[nr & (li == j) & (top5 > 0)].std()))
                          for j, L in enumerate(cm.LENS)}
    # ---- hollow configurations (absolute top-5 prices), sizes
    hr = v & (reg == cm.HOLLOW)
    cfg = np.round(np.column_stack([P[f"bid_px_{k}"] for k in range(cm.NLEV)]
                                   + [P[f"ask_px_{k}"] for k in range(cm.NLEV)])[hr], 3)
    u_, inv, cc = np.unique(cfg, axis=0, return_inverse=True, return_counts=True)
    top = np.argsort(-cc)[:12]
    hs = np.concatenate([np.log(P[f"{s_}_sz_{k}"][hr & (P[f"{s_}_sz_{k}"] > 0)]) for s_ in ("bid", "ask")
                         for k in range(cm.NLEV)])
    out["hollow"] = dict(bid_px=u_[top, :cm.NLEV], ask_px=u_[top, cm.NLEV:], p=cc[top] / cc[top].sum(),
                         coverage=float(cc[top].sum() / cc.sum()), log_size_mu=float(hs.mean()),
                         log_size_sd=float(hs.std()))
    return out, fine


# ============================================================================================ flow
def flow_covariates(P: dict, mk: pd.DataFrame, mid, reg):
    """Match-time covariates aligned to block-time rows (row t uses state at row t - MATCH_LAG)."""
    o = np.concatenate([[0], np.cumsum(mk.n_rows.values)])
    n = len(P["p"])
    src = np.arange(n) - MATCH_LAG
    first = o[P["m"]]
    src = np.maximum(src, first)
    gap = (P["p"] - mid)[src]
    S = P["S"]
    i5 = np.maximum(np.arange(n) - 5, first)
    z5 = np.abs(np.log(S / S[i5])) / (np.maximum(P["sigma"], 1e-7) * np.sqrt(5))
    z5 = z5[src]
    return dict(gap=gap, z5=z5, reg=reg[src], tau=P["seconds_to_close"][src])


def glm_design(cov, Lr, tau):
    li = np.searchsorted(cm.LENS, Lr)
    tb = cm.tau_bucket(np.minimum(tau, Lr - 1))
    cell = li * cm.NTAU + tb
    X = np.column_stack([np.minimum(np.abs(cov["gap"]), 0.3), np.minimum(cov["z5"], 6.0),
                         (cov["reg"] == cm.ONESIDED).astype(float)])
    return cell, X


def fit_flow(P: dict, mk: pd.DataFrame, mid, reg, pre: pd.DataFrame | None = None) -> dict:
    """pre: optional pre-period on-chain trades (cm.pre_period_trades), pooled only into book-free parameters
    (taker size quantiles, token mix, maker legs per level, sweep depth); everything that needs the book or
    Binance state (arrival covariates, P(buy | gap), markouts) stays train-only."""
    o = np.concatenate([[0], np.cumsum(mk.n_rows.values)])
    n_t = P["trade_count"]
    cov = flow_covariates(P, mk, mid, reg)
    cell, Xc = glm_design(cov, P["L"], cov["tau"])
    ncell = 3 * cm.NTAU
    # reference cell per length (bucket of the opening second) is fixed at 0 (market effects absorb level)
    ref = [li * cm.NTAU + cm.tau_bucket(L - 1) for li, L in enumerate(cm.LENS)]
    free_cells = np.array([c for c in range(ncell) if c not in ref and (cell == c).any()])
    cpos = -np.ones(ncell, int)
    cpos[free_cells] = np.arange(len(free_cells))
    m_idx = P["m"]
    # hollow seconds (stub quotes only) carry 0.0065 trades/s vs 0.245 in normal books and are concentrated in
    # a few dead markets: they are excluded from the likelihood and the sim generates no flow in that state
    w_row = (cov["reg"] != cm.HOLLOW).astype(float)
    rate_hollow = float(n_t[w_row == 0].mean()) if (w_row == 0).any() else 0.0
    rate_other = float(n_t[w_row == 1].mean())
    N_m = np.bincount(m_idx, weights=n_t * w_row, minlength=len(mk))

    def negll(theta):
        k = len(free_cells)
        th_c = np.zeros(ncell)
        th_c[free_cells] = theta[:k]
        beta = theta[k:]
        eta = np.minimum(th_c[cell] + Xc @ beta, 30)
        w = np.exp(eta) * w_row
        Sm = np.bincount(m_idx, weights=w, minlength=len(mk))
        ll = np.sum(w_row * n_t * eta) - np.sum(N_m[N_m > 0] * np.log(Sm[N_m > 0]))
        coefm = np.where(Sm > 0, N_m / np.maximum(Sm, 1e-300), 0.0)
        r = w_row * n_t - coefm[m_idx] * w
        return -ll, -np.r_[np.bincount(cell, weights=r, minlength=ncell)[free_cells], Xc.T @ r]

    # (1) baseline shape: Poisson GLM with market fixed effects (profiled), no excitation
    res = minimize(negll, np.zeros(len(free_cells) + Xc.shape[1]), jac=True, method="L-BFGS-B")
    k = len(free_cells)
    th_c = np.zeros(ncell)
    th_c[free_cells] = res.x[:k]
    beta = res.x[k:]
    base = np.exp(th_c[cell] + Xc @ beta) * w_row
    Sb = np.bincount(m_idx, weights=base, minlength=len(mk))
    use = w_row > 0

    # (2) linear Hawkes on top: lambda_t = A_m * base_t + eta * H_t, H_t = sum_{s<t} n_s (1-d) d^(t-1-s)
    def hawkes_H(tau_e):
        d = np.exp(-1.0 / tau_e)
        H = np.zeros_like(n_t)
        for m in range(len(mk)):
            a, b = o[m], o[m + 1]
            H[a:b] = lfilter([0.0, 1.0 - d], [1.0, -d], n_t[a:b])
        return H

    def profile(eta, H):
        A = np.maximum(N_m * (1 - eta), 0.5 * (1 - eta)) / np.maximum(Sb, 1e-9)
        nb = n_t * base
        for _ in range(40):
            lam = np.maximum(A[m_idx] * base + eta * H, 1e-300)
            g = np.bincount(m_idx, weights=np.where(use, nb / lam, 0.0), minlength=len(mk)) - Sb
            h = -np.bincount(m_idx, weights=np.where(use, nb * base / lam ** 2, 0.0), minlength=len(mk))
            A = np.where(h < 0, np.maximum(A - g / np.minimum(h, -1e-300), 1e-12), A)
        lam = np.maximum(A[m_idx] * base + eta * H, 1e-300)
        ll = np.sum(np.where(use, n_t * np.log(lam) - lam, 0.0))
        return ll, A

    from scipy.optimize import minimize_scalar
    best = None
    for tau_e in (1.0, 2.0, 5.0, 10.0, 20.0, 60.0):
        H = hawkes_H(tau_e)
        r = minimize_scalar(lambda e: -profile(e, H)[0], bounds=(0.0, 0.97), method="bounded",
                            options=dict(xatol=1e-3))
        if best is None or r.fun < best[1]:
            best = (tau_e, r.fun, r.x, H)
    tau_e, nll_h, eta_h, H = best
    ll_h, A = profile(eta_h, H)
    ll_0, A0 = profile(0.0, H)
    okm = Sb > 1.0                                   # markets with >= ~1 s-equivalent of non-hollow exposure
    floor = 0.5 * (1 - eta_h) / np.maximum(Sb, 1e-9)
    logA = np.log(np.maximum(A, floor))[okm]
    D = np.column_stack([(mk.L.values[okm] == L).astype(float) for L in cm.LENS]
                        + [(mk.coin.values[okm] == c).astype(float) for c in cm.COINS[1:]])
    coef, *_ = np.linalg.lstsq(D, logA, rcond=None)
    resid = logA - D @ coef
    glm = dict(tau_cell=th_c.reshape(3, cm.NTAU), b_absgap=beta[0], b_absz5=beta[1], b_onesided=beta[2],
               rate_hollow_per_s=rate_hollow, rate_non_hollow_per_s=rate_other,
               hawkes_eta=float(eta_h), hawkes_tau_s=float(tau_e), loglik_hawkes=float(ll_h),
               loglik_poisson=float(ll_0), lr_stat_hawkes=float(2 * (ll_h - ll_0)),
               note="lambda_t = A_m * exp(tau_cell[len,tau_b] + b_absgap*min(|p-mid|,0.3) + b_absz5*min(|z5|,6) "
                    "+ b_onesided*O) + hawkes_eta * H_t,  H_t = d*H_{t-1} + (1-d)*n_{t-1}, d = exp(-1/hawkes_tau); "
                    "baseline shape from a Poisson GLM with market fixed effects, (eta, tau) and A_m by profile "
                    "MLE; covariates at match time (block time - 3 s); lambda = 0 in the hollow state")
    meff = dict(a_len={cm.LEN_NAME[L]: coef[j] for j, L in enumerate(cm.LENS)},
                a_coin={"btc": 0.0, **{c: coef[3 + j] for j, c in enumerate(cm.COINS[1:])}},
                sd=float(resid.std()), n_markets=int(okm.sum()),
                note="log A_m = a_len + a_coin + N(0, sd^2) (Hawkes baseline scale per market)")

    # ---- taker transactions (direction, size, sweep, legs, tokens)
    tr = cm.train_trades(mk.slug)
    mk_by_slug = mk.set_index("slug")
    tk = tr[tr.is_taker].copy()
    mkl = tr[~tr.is_taker].copy()
    tk["o"] = mk_by_slug.loc[tk.slug, "open_ts"].values
    tk["e"] = mk_by_slug.loc[tk.slug, "end_ts"].values
    tk["mpos"] = mk_by_slug.index.get_indexer(tk.slug)
    tk = tk[(tk.timestamp >= tk.o) & (tk.timestamp <= tk.e)]
    tk["L"] = mk.L.values[tk.mpos.values]
    row = o[tk.mpos.values] + np.maximum(tk.timestamp.values - tk.o.values - MATCH_LAG, 0)
    tk["gap"] = (P["p"] - mid)[row]
    tk["gb"] = cm.gap_bucket(tk.gap.values)
    buy = tk.aggressor_up.values == 1
    dirn = {}
    for j, L in enumerate(cm.LENS):
        sel = tk.L.values == L
        nb = np.bincount(tk.gb.values[sel], weights=buy[sel], minlength=cm.NGAP)
        nn = np.bincount(tk.gb.values[sel], minlength=cm.NGAP)
        dirn[cm.LEN_NAME[L]] = dict(p_buy=(nb + 5) / (nn + 10), n=nn)
    # copy probability from sign persistence of consecutive txs within a market
    tk = tk.sort_values(["mpos", "timestamp"], kind="stable")
    q = np.array([dirn[cm.LEN_NAME[L]]["p_buy"][g] for L, g in zip(tk.L.values, tk.gb.values)])
    sg = tk.aggressor_up.values
    same_m = tk.mpos.values[1:] == tk.mpos.values[:-1]
    p_same = float((sg[1:] == sg[:-1])[same_m].mean())
    p_ind = float((q[1:] * q[:-1] + (1 - q[1:]) * (1 - q[:-1]))[same_m].mean())
    rho_copy = (p_same - p_ind) / (1 - p_ind)
    rho_copy_len = {}
    for L in cm.LENS:
        s2 = same_m & (tk.L.values[1:] == L)
        ps = (sg[1:] == sg[:-1])[s2].mean()
        pi_ = (q[1:] * q[:-1] + (1 - q[1:]) * (1 - q[:-1]))[s2].mean()
        rho_copy_len[cm.LEN_NAME[L]] = float((ps - pi_) / (1 - pi_))
    # sizes (pooled with pre-period on-chain taker txs when given: no book needed)
    qs = np.linspace(0, 1, 101)
    n_pre = 0
    if pre is not None and len(pre):
        ptk = pre[pre.is_taker]
        n_pre = int(len(ptk))
        ext = pd.DataFrame({"size": ptk["size"].values.astype(float), "L": ptk.L.values,
                            "coin": ptk.coin.values})
        tk_pool = pd.concat([pd.DataFrame({"size": tk["size"].values.astype(float), "L": tk.L.values,
                                           "coin": mk.coin.values[tk.mpos.values]}), ext], ignore_index=True)
    else:
        tk_pool = pd.DataFrame({"size": tk["size"].values.astype(float), "L": tk.L.values,
                                "coin": mk.coin.values[tk.mpos.values]})
    size_q = {cm.LEN_NAME[L]: np.quantile(np.log(np.maximum(tk_pool["size"].values[tk_pool.L.values == L], 1e-3)), qs)
              for L in cm.LENS}
    # by coin group too: BTC prints are much larger than alt prints (volume mix across coins)
    tk_cg = np.array([cm.COIN_GROUP[c] for c in tk_pool.coin.values])
    size_q_cg = {}
    for L in cm.LENS:
        for g_ in range(cm.NCG):
            s_ = (tk_pool.L.values == L) & (tk_cg == g_)
            src = s_ if s_.sum() >= 300 else (tk_pool.L.values == L)
            size_q_cg[f"{cm.LEN_NAME[L]}|{cm.CG_NAMES[g_]}"] = np.quantile(
                np.log(np.maximum(tk_pool["size"].values[src], 1e-3)), qs)
    # sweep depth (distinct maker price levels per tx) and legs per level (pooled with pre-period legs)
    if pre is not None and len(pre):
        pm = pre[~pre.is_taker]
        mkl = pd.concat([mkl[["tx", "px_up"]], pm[["tx", "px_up"]]], ignore_index=True)
        tr = pd.concat([tr[["is_taker", "outcome_idx", "aggressor_up"]],
                        pre[["is_taker", "outcome_idx", "aggressor_up"]]], ignore_index=True)
    mkl["pxr"] = mkl.px_up.round(4)
    g = mkl.groupby("tx", sort=False)
    nlev = g.pxr.nunique().values
    legs_per_level = float((g.size() / g.pxr.nunique()).mean())
    jlev = np.minimum(nlev - 1, cm.NLEV - 1)
    p_lim = np.bincount(jlev, minlength=cm.NLEV) / len(jlev)
    # token mix
    t_all = tr[tr.is_taker]
    m_all = tr[~tr.is_taker]
    tok = dict(
        taker_up_given_buy=float((t_all.outcome_idx[t_all.aggressor_up == 1] == 0).mean()),
        taker_up_given_sell=float((t_all.outcome_idx[t_all.aggressor_up == -1] == 0).mean()),
        maker_up_given_buy=float((m_all.outcome_idx[m_all.aggressor_up == 1] == 0).mean()),
        maker_up_given_sell=float((m_all.outcome_idx[m_all.aggressor_up == -1] == 0).mean()),
        note="probability that a leg is on the Up token, by aggressor direction (Up frame); Down legs carry "
             "price 1-p and the opposite side")
    out = dict(glm=glm, market_effect=meff,
               direction=dict(gap_edges=cm.GAP_EDGES, **dirn, rho_copy=rho_copy, rho_copy_by_len=rho_copy_len,
                              p_same_sign_observed=p_same, p_same_sign_independent=p_ind,
                              note="P(taker buys Up | p_model - mid at match time); with prob rho_copy a trade "
                                   "repeats the previous trade's direction in that market"),
               size_log_quantiles=dict(q=qs, **size_q, by_coin_group=size_q_cg,
                                       note="log taker size quantiles by length and by 'length|coin group' "
                                            "(the engine uses the coin-group version)"),
               sweep=dict(p_extra_levels=p_lim, legs_per_level_mean=legs_per_level,
                          note="taker limit allows j extra levels beyond the best (j ~ p_extra_levels); size is "
                               "capped at the displayed depth within the limit; each level touched yields "
                               "a geometric number of maker legs"),
               tokens=tok,
               pre_period=dict(n_taker_tx=n_pre, used_for="size quantiles, token mix, legs per level, sweep depth"
                               if n_pre else "not used"),
               stamp_delay=dict(values=[3, 4], p=[0.5, 0.5],
                                note="block time - match time (COVERAGE.md: alignment with the book peaks at 3-4 s)"),
               n_taker_tx=int(len(tk)))
    return out


def calibrate_cluster(P: dict, mk: pd.DataFrame, params: dict, reps: int = 3, iters: int = 2) -> dict:
    """Within-second bunching of taker txs: arrivals are compound Poisson (bursts of geometric size with mean mu
    per (length, coin group); the mean rate is unchanged). mu is set so that the simulated mean number of txs
    per second-with-a-tx matches train (seconds with > 30 s to close)."""
    import copy
    import metrics as mt
    import engine as en
    ul = cm.train_underlying()
    prices = [cm.market_pricing(ul[r.coin][1], ul[r.coin][2], ul[r.coin][0], r.open_ts, r.end_ts)
              for r in mk.itertuples()]
    cg_m = np.array([cm.COIN_GROUP[c] for c in mk.coin.values])

    def stat(tc, Lr, cgr, tau):
        out = {}
        for L in cm.LENS:
            for g in range(cm.NCG):
                s = (Lr == L) & (cgr == g) & (tau > 30)
                x = tc[s]
                out[(L, g)] = (float(x[x > 0].mean()) if (x > 0).sum() > 50 else np.nan, float((x > 0).mean()) if s.any() else np.nan)
        return out
    real = stat(P["trade_count"], P["L"], cg_m[P["m"]], P["seconds_to_close"])
    mu = params["flow"].get("cluster_mean") or {cm.LEN_NAME[L]: {n: 1.0 for n in cm.CG_NAMES} for L in cm.LENS}
    mu = copy.deepcopy(mu)
    hist = []
    for it in range(iters):
        p2 = copy.deepcopy(params)
        p2["flow"]["cluster_mean"] = mu
        model = en.Model(p2)
        acc = {}
        for d in mt.sim_datasets(model, mk, prices, reps, 41 + it, reps):
            s = stat(d.X[:, cm.C["trade_count"]], d.L, cg_m[d.m], d.tau)
            for k, v in s.items():
                acc.setdefault(k, []).append(v[0])
        for (L, g), v in acc.items():
            sv = np.nanmean(v)
            rv = real[(L, g)][0]
            if np.isfinite(sv) and np.isfinite(rv) and sv > 0:
                n = cm.LEN_NAME[L]
                mu[n][cm.CG_NAMES[g]] = float(np.clip(mu[n][cm.CG_NAMES[g]] * rv / sv, 1.0, 20.0))
                hist.append(dict(iter=it, len=n, cg=cm.CG_NAMES[g], real=rv, sim=float(sv)))
        print(f"  cluster iter {it}:", {f"{h['len']}|{h['cg']}": (round(h['real'], 2), round(h['sim'], 2))
                                         for h in hist if h['iter'] == it}, flush=True)
    return dict(mu=mu, history=hist)


def calibrate_cell_rate(P: dict, mk: pd.DataFrame, params: dict, reps: int = 10) -> np.ndarray:
    """Per-(length, tau bucket) log shift of the arrival baseline so the simulated mean taker-tx rate per cell
    matches train (cells with >= 15 real txs; clipped to +-1.5). Adds to flow.glm.tau_cell_shift."""
    import copy
    import metrics as mt
    import engine as en
    ul = cm.train_underlying()
    prices = [cm.market_pricing(ul[r.coin][1], ul[r.coin][2], ul[r.coin][0], r.open_ts, r.end_ts)
              for r in mk.itertuples()]
    tb_r = cm.tau_bucket(np.minimum(P["seconds_to_close"], P["L"] - 1))
    li_r = np.searchsorted(cm.LENS, P["L"])
    real = np.zeros((3, cm.NTAU))
    np.add.at(real, (li_r, tb_r), P["trade_count"])
    model = en.Model(copy.deepcopy(params))
    sim = np.zeros((3, cm.NTAU))
    nrep = 0
    for d in mt.sim_datasets(model, mk, prices, reps, 31, reps):
        tb_s = cm.tau_bucket(np.minimum(d.tau, d.L - 1))
        np.add.at(sim, (d.li, tb_s), d.X[:, cm.C["trade_count"]])
        nrep += 1
    sim /= max(nrep, 1)
    old = np.asarray(params["flow"]["glm"].get("tau_cell_shift", np.zeros((3, cm.NTAU))), float)
    ratio = np.where((real >= 15) & (sim > 0), real / np.maximum(sim, 1e-9), 1.0)
    new = np.clip(old + np.log(ratio), -1.5, 1.5)
    print("  cell rate ratios real/sim:\n", np.round(ratio, 2))
    return new


def calibrate_rate(P: dict, mk: pd.DataFrame, params: dict, reps: int = 2, iters: int = 2) -> dict:
    """Per-length level shift of log A_m so the simulated total taker-tx count on the train schedule equals the
    real one (the lognormal market effect is fitted on log scale, so its mean differs from the data's)."""
    import copy
    import metrics as mt
    import engine as en
    ul = cm.train_underlying()
    prices = [cm.market_pricing(ul[r.coin][1], ul[r.coin][2], ul[r.coin][0], r.open_ts, r.end_ts)
              for r in mk.itertuples()]
    real_n = {L: float(P["trade_count"][P["L"] == L].sum()) for L in cm.LENS}
    shift = dict(params["flow"]["market_effect"].get("a_len_shift") or {cm.LEN_NAME[L]: 0.0 for L in cm.LENS})
    hist = []
    for it in range(iters):
        p2 = copy.deepcopy(params)
        p2["flow"]["market_effect"]["a_len_shift"] = dict(shift)
        model = en.Model(p2)
        sim_n = {L: [] for L in cm.LENS}
        for d in mt.sim_datasets(model, mk, prices, reps, 21 + it, reps):
            for L in cm.LENS:
                sim_n[L].append(float(d.X[d.L == L, cm.C["trade_count"]].sum()))
        for L in cm.LENS:
            ratio = real_n[L] / max(np.mean(sim_n[L]), 1.0)
            shift[cm.LEN_NAME[L]] += float(np.log(ratio))
            hist.append(dict(iter=it, len=cm.LEN_NAME[L], real=real_n[L], sim=float(np.mean(sim_n[L])), ratio=ratio))
        print("  rate calibration", it, {h["len"]: round(h["ratio"], 3) for h in hist[-3:]}, flush=True)
    return dict(a_len_shift=shift, history=hist,
                note="added to a_len so the simulated mean taker-tx count per length matches train")


def fit_delay(P: dict, mk: pd.DataFrame, params: dict, dmax: int = 10) -> dict:
    """Block-time delay distribution p(d), d = 0..dmax s, from the real alignment curve.

    Real: A(k) = share of taker prints at-or-through (and through) the best opposite quote of the book at
    block-time row T - k, k = 0..dmax. Sim with zero delay gives c(j) = the same shares against the book at
    match + j. With delay d ~ p, A(k) = sum_d p(d) c(d - k): linear in p -> non-negative least squares with
    sum(p) = 1, both curves stacked. Train data only."""
    import copy
    import metrics as mt
    import engine as en
    from scipy.optimize import nnls
    real = mt.real_dataset(P, mk)
    ks_r, a_r, t_r = mt.alignment_curve(real, mt.taker_table(real), ks=range(0, dmax + 1))
    p2 = copy.deepcopy(params)
    p2["flow"]["stamp_delay"] = dict(values=[0], p=[1.0])
    model = en.Model(p2)
    ul = cm.train_underlying()
    prices = [cm.market_pricing(ul[r.coin][1], ul[r.coin][2], ul[r.coin][0], r.open_ts, r.end_ts)
              for r in mk.itertuples()]
    cs_a, cs_t = [], []
    for dset in mt.sim_datasets(model, mk, prices, 2, 9, 2):
        js, ca, ct = mt.alignment_curve(dset, mt.taker_table(dset), ks=range(-dmax, dmax + 1))
        cs_a.append(ca)
        cs_t.append(ct)
    ca, ct = np.mean(cs_a, 0), np.mean(cs_t, 0)
    c_at = dict(zip(js, ca))
    c_th = dict(zip(js, ct))
    ds = np.arange(dmax + 1)
    A = np.array([[c_at[-(int(d) - k)] for d in ds] for k in ks_r] + [[c_th[-(int(d) - k)] for d in ds] for k in ks_r])
    y = np.r_[a_r, t_r]
    w = 100.0
    p, _ = nnls(np.vstack([A, w * np.ones(len(ds))]), np.r_[y, w])
    p = p / p.sum()
    fit_a = A[:len(ks_r)] @ p
    fit_t = A[len(ks_r):] @ p
    print("  delay p(d):", dict(zip(ds.tolist(), np.round(p, 3).tolist())))
    print("  real at|through:", np.round(a_r, 3), np.round(t_r, 3))
    print("  fit  at|through:", np.round(fit_a, 3), np.round(fit_t, 3))
    keep = p > 1e-4
    return dict(values=ds[keep].tolist(), p=(p[keep] / p[keep].sum()).tolist(),
                mean_s=float((ds * p).sum()),
                real_curve=dict(k=ks_r.tolist(), at_or_through=a_r.tolist(), through=t_r.tolist()),
                fitted_curve=dict(at_or_through=fit_a.tolist(), through=fit_t.tolist()),
                note="block time - match time (s), fitted to the real alignment of prints with the book at T-k "
                     "(convolution of the zero-delay sim curve, NNLS); COVERAGE.md: alignment peaks at 3-4 s")


def calibrate_informed(P: dict, mk: pd.DataFrame, params: dict) -> dict:
    """Informed-taker share w and print impact kappa per length (impact half-life fixed at IMPACT_HL_S), by
    simulated moments on the train schedule with the full engine: the post-trade markout drift (maker markout of
    all taker prints at 5/30/60 s minus at 0 s), the P(buy | p_model - mid) curve and the alignment of prints
    with the captured book (at-or-through / through shares at k = 0 and 4 s). Train data only."""
    import copy
    import metrics as mt
    import engine as en
    real = mt.real_dataset(P, mk)
    rs = mt.trade_stats(real, mt.taker_table(real))
    ul = cm.train_underlying()
    prices = [cm.market_pricing(ul[r.coin][1], ul[r.coin][2], ul[r.coin][0], r.open_ts, r.end_ts)
              for r in mk.itertuples()]
    out_inf, out_imp = {}, {}
    grids = {300: ([0.0, 0.15, 0.3], [0.0, 0.005, 0.01, 0.02], 2),
             900: ([0.0, 0.15, 0.3], [0.0, 0.005, 0.01, 0.02], 2),
             14400: ([0.0, 0.15, 0.3], [0.0, 0.02, 0.04, 0.08], 1)}
    for L in cm.LENS:
        name = cm.LEN_NAME[L]
        sel = np.flatnonzero(mk.L.values == L)
        sub = mk.iloc[sel].reset_index(drop=True)
        psub = [prices[i] for i in sel]
        keys_pb = [k for k, v in rs.items() if k.startswith(f"pbuy|{name}|") and v[1] >= 300]
        res = []
        ws, kis, reps = grids[L]
        for w in ws:
            for ki in kis:
                p2 = copy.deepcopy(params)
                p2["flow"].setdefault("informed", {})[name] = dict(w=w, kappa_noise=1.0)
                p2["flow"].setdefault("impact", {})[name] = dict(kappa=ki, halflife_s=IMPACT_HL_S,
                                                                 rise_halflife_s=IMPACT_RISE_S)
                model = en.Model(p2)
                st = [mt.trade_stats(d, mt.taker_table(d)) for d in mt.sim_datasets(model, sub, psub, reps, 5, reps)]
                loss, mom = 0.0, {}
                for h in (5, 30, 60):
                    r_ = rs[f"markout|{name}|{h}"] - rs[f"markout|{name}|0"]
                    s_ = np.mean([s[f"markout|{name}|{h}"] - s[f"markout|{name}|0"] for s in st])
                    mom[f"drift{h}"] = (r_, float(s_))
                    loss += ((s_ - r_) / 0.25) ** 2
                pbl = 0.0
                for k in keys_pb:
                    s_ = np.nanmean([s[k][0] for s in st])
                    pbl += ((s_ - rs[k][0]) / 0.03) ** 2
                loss += pbl
                for kk in (0, 4):                       # alignment of prints with the captured book
                    for nm_ in ("align_at_or_through", "align_through"):
                        key = f"{nm_}|{name}|k{kk}"
                        s_ = np.nanmean([s[key] for s in st])
                        mom[f"{nm_[6:]}{kk}"] = (rs[key], float(s_))
                        loss += ((s_ - rs[key]) / 0.03) ** 2
                res.append((loss, w, ki, mom, pbl))
                print(f"  {name}: w={w:.2f} kappa_imp={ki:.3f} loss={loss:.1f} (pbuy {pbl:.1f}) "
                      + " ".join(f"{k}={v[0]:+.2f}/{v[1]:+.2f}" for k, v in mom.items()), flush=True)
        best = min(res, key=lambda x: x[0])
        out_inf[name] = dict(w=best[1], kappa_noise=1.0, loss=float(best[0]), loss_pbuy=float(best[4]),
                             markout_drift_real_sim={k: list(v) for k, v in best[3].items()},
                             grid=[dict(w=r[1], kappa_imp=r[2], loss=float(r[0])) for r in res])
        out_imp[name] = dict(kappa=best[2], halflife_s=IMPACT_HL_S, rise_halflife_s=IMPACT_RISE_S)
    out_inf["note"] = ("P(buy) = (1-w) * P_fit(buy | gap) + w * 1[staleness > 0], staleness = sigmoid(target + v) - "
                       "displayed latent mid (what the next book update reveals); w and the impact kappa by grid "
                       "search on simulated markout drift + P(buy|gap) curve (train)")
    out_imp["note"] = ("each print adds kappa * sign * sqrt(min(size, 500)/10) to an impact accumulator a (logit "
                       "units, decays with half-life halflife_s); the mid target carries v, which approaches a with "
                       "half-life rise_halflife_s")
    return out_inf, out_imp


IMPACT_HL_S = 300.0     # impact decay half-life (s)
IMPACT_RISE_S = 15.0    # the mid target absorbs a print's impact with this half-life (real drift builds ~20-30 s)


def fit_fees(mk: pd.DataFrame) -> dict:
    rw = {}
    for L in cm.LENS:
        d = mk[mk.L == L]
        pairs, cnt = np.unique(np.column_stack([d.rewards_min_size, d.rewards_max_spread]), axis=0,
                               return_counts=True)
        rw[cm.LEN_NAME[L]] = dict(pairs=pairs, p=cnt / cnt.sum())
    return dict(taker_fee_rate=float(np.median(mk.taker_fee_rate)), fee_exponent=1.0,
                maker_rebate_rate=float(np.median(mk.maker_rebate_rate)), min_order_size=5.0, rewards=rw)


RANDOMIZATION = {
    "note": "per market (episode), applied only with --randomize; uniform unless 'log' (log-uniform)",
    "mid_extra_lag_s": {"kind": "int", "lo": 0, "hi": 3},
    "passthrough_a0_mult": {"kind": "uniform", "lo": 0.7, "hi": 1.15, "clip_hi_a0": 1.0},
    "lag_halflife_mult": {"kind": "log", "lo": 0.5, "hi": 2.0},
    "sigma_u_mult": {"kind": "log", "lo": 0.5, "hi": 2.0},
    "refresh_rate_mult": {"kind": "log", "lo": 0.5, "hi": 2.0},
    "taker_intensity_mult": {"kind": "log", "lo": 0.5, "hi": 2.0},
    "taker_size_mult": {"kind": "log", "lo": 0.5, "hi": 2.0},
    "informed_share": {"kind": "uniform", "lo": 0.1, "hi": 0.6,
                       "doc": "share of takers trading on the sign of the mid's staleness (fitted value per length "
                              "in flow.informed.w)"},
    "impact_mult": {"kind": "log", "lo": 0.5, "hi": 2.0},
    "lookahead_mult": {"kind": "log", "lo": 0.5, "hi": 2.0,
                       "doc": "scales the look-ahead informed-taker sensitivity kappa (flow.lookahead)"},
    "hawkes_eta": {"kind": "uniform", "lo": 0.0, "hi": 0.8,
                   "doc": "branching ratio; the baseline is rescaled by (1-eta)/(1-eta_fit) to keep the mean rate"},
    "depth_mult": {"kind": "log", "lo": 0.3, "hi": 3.0},
    "spread_mult": {"kind": "log", "lo": 0.5, "hi": 2.0},
    "onesided_logit_shift": {"kind": "uniform", "lo": -0.5, "hi": 0.5},
    "hollow_logit_shift": {"kind": "uniform", "lo": -0.5, "hi": 0.5},
    "fee_rate_mult": {"kind": "uniform", "lo": 0.5, "hi": 1.5, "p_fee_free": 0.1},
    "rebate_mult": {"kind": "uniform", "lo": 0.5, "hi": 1.5},
    "sigma_basis_mult": {"kind": "uniform", "lo": 0.0, "hi": 2.0,
                         "doc": "base sim uses 0 (pure Binance settlement proxy)"},
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["settlement", "book", "mid", "flow", "informed", "rate", "fees"],
                    help="refit only these components (others are kept from the existing params.json)")
    ap.add_argument("--pre-trades", nargs=2, metavar=("START", "END"), default=None,
                    help="also pool on-chain pre-period trades (onchain/, days START..END <= 2026-08-20) into the "
                         "book-free flow parameters (sizes, tokens, legs, sweep depth)")
    ap.add_argument("--pre-frac", type=float, default=0.03,
                    help="share of pre-period markets used (deterministic slug-hash sample; ~1.2 M taker txs/day)")
    a = ap.parse_args()
    t0 = time.time()
    P, mk = cm.train_panel()
    reg, mid, spread = cm.book_state(P)
    print(f"train panel: {len(P['p'])} rows, {len(mk)} markets  {time.time()-t0:.1f}s")
    params = cm.load_params() if cm.PARAMS.exists() and set(a.parts) != {"settlement", "book", "mid", "flow", "informed", "rate", "fees"} else {}
    params["meta"] = dict(
        fitted_on="train split 0 only: %d markets (%s), opens 2026-08-21..22 UTC; Binance < 2026-08-23"
                  % (len(mk), ", ".join(f"{cm.LEN_NAME[L]} {int((mk.L == L).sum())}" for L in cm.LENS)),
        created_utc=pd.Timestamp.now("UTC").isoformat(), n_rows=int(len(P["p"])),
        pricer=cm.PRICER, tau_edges=cm.TAU_EDGES, ext_edges=cm.EXT_EDGES, gap_edges=cm.GAP_EDGES,
        hollow_spread=cm.HOLLOW_SPREAD, match_lag_s=MATCH_LAG)
    if "settlement" in a.parts:
        params["settlement"] = fit_settlement(mk)
    book, fine = fit_book(P, mk, reg, mid, spread)
    if "book" in a.parts:
        params["book"] = book
        print(f"book done {time.time()-t0:.1f}s")
    if "mid" in a.parts:
        params["mid"] = fit_mid(P, mk, reg, mid, fine, np.asarray(params["book"]["book_change_profile"]["p_age0"]),
                                warm=params.get("mid"))
        print(f"mid done {time.time()-t0:.1f}s")
    if "flow" in a.parts:
        keep = {k: params.get("flow", {}).get(k) for k in ("informed", "impact", "stamp_delay", "lookahead",
                                                           "refill_prob", "actual_book_tracking",
                                                           "lookahead_calibration", "cluster_mean", "cluster_note")}
        keep_shift = params.get("flow", {}).get("market_effect", {}).get("a_len_shift")
        keep_cell = params.get("flow", {}).get("glm", {}).get("tau_cell_shift")
        pre = cm.pre_period_trades(*a.pre_trades, frac=a.pre_frac) if a.pre_trades else None
        if pre is not None:
            print(f"pre-period trades: {len(pre)} legs, {int(pre.is_taker.sum())} taker txs, "
                  f"{pre.slug.nunique()} markets")
        params["flow"] = fit_flow(P, mk, mid, reg, pre)
        params["flow"].update({k: v for k, v in keep.items() if v is not None
                               and not (k in ("informed", "impact") and "informed" in a.parts)
                               and not (k in ("lookahead", "refill_prob", "lookahead_calibration")
                                        and "lookahead" in a.parts)
                               and not (k == "stamp_delay" and "delay" in a.parts)})
        if keep_shift is not None and "rate" not in a.parts:
            params["flow"]["market_effect"]["a_len_shift"] = keep_shift
        if keep_cell is not None and "cellrate" not in a.parts:
            params["flow"]["glm"]["tau_cell_shift"] = keep_cell
        print(f"flow done {time.time()-t0:.1f}s")
    if "cluster" in a.parts:
        cl = calibrate_cluster(P, mk, params)
        params["flow"]["cluster_mean"] = cl["mu"]
        params["flow"]["cluster_note"] = ("taker arrivals are compound Poisson per second: Poisson(lambda/mu) bursts, "
                                          "each of geometric size with mean mu[len][coin group] (fit.py calibrate_cluster)")
        print(f"cluster calibration done {time.time()-t0:.1f}s")
    if "cellrate" in a.parts:
        params["flow"]["glm"]["tau_cell_shift"] = calibrate_cell_rate(P, mk, params)
        print(f"cell rate calibration done {time.time()-t0:.1f}s")
    if "rate" in a.parts:
        params["flow"]["market_effect"]["a_len_shift"] = calibrate_rate(P, mk, params, reps=10, iters=1)["a_len_shift"]
        print(f"rate calibration done {time.time()-t0:.1f}s")
    if "delay" in a.parts:
        params["flow"]["stamp_delay"] = fit_delay(P, mk, params)
        print(f"delay fit done {time.time()-t0:.1f}s")
    if "informed" in a.parts:
        params["flow"]["informed"], params["flow"]["impact"] = calibrate_informed(P, mk, params)
        print(f"informed calibration done {time.time()-t0:.1f}s")
    if "fees" in a.parts:
        params["fees"] = fit_fees(mk)
    params["randomization"] = RANDOMIZATION
    cm.PARAMS.write_text(json.dumps(rounded(params), indent=1))
    print(f"wrote {cm.PARAMS} ({cm.PARAMS.stat().st_size/1e3:.0f} kB)  {time.time()-t0:.1f}s")
    summary(params)


def summary(p):
    print("settlement:", {k: v for k, v in p["settlement"].items() if k != "note"})
    for L, d in p["mid"].items():
        print(f"mid {L}: psi={d.get('psi', 0):.3f} s_fast={d['s_fast']:.3f} s_slow={d['s_slow']:.3f} hl_slow={d['halflife_slow_s']:.0f}s "
              f"hl_u={d['halflife_u_s']:.0f}s sigma_u={d['sigma_u']:.4f} upd_mult={d['update_prob_mult']:.2f} "
              f"b_tau={np.round(d['b_tau'], 3)} loss={d['smm_loss']:.2f}")
    b = p["book"]
    print("tick switch", b["tick_switch"]["ext_threshold"], "copula", b["regime"]["copula"],
          "extreme-side agree", b["regime"]["onesided_extreme_side_agree"])
    print("p_age0", np.round(np.array(b["book_change_profile"]["p_age0"]), 3))
    print("size", {k: (np.round(v["mu"], 2) if isinstance(v, dict) else v) for k, v in b["size"].items()})
    f = p["flow"]
    g = f["glm"]
    print("glm:", {k: (np.round(v, 3) if not isinstance(v, str) else None) for k, v in g.items() if k != "note"})
    print("market effect:", f["market_effect"])
    print("direction rho_copy", f["direction"]["rho_copy"], f["direction"]["rho_copy_by_len"])
    for L in ("5m", "15m", "4h"):
        print(" p_buy", L, np.round(f["direction"][L]["p_buy"], 3))
    print("sweep", f["sweep"]["p_extra_levels"], f["sweep"]["legs_per_level_mean"], "tokens", f["tokens"])


if __name__ == "__main__":
    main()
