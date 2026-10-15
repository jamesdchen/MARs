"""Vectorised batch engine: simulate B markets of one contract length in lockstep, second by second.

Inputs are real (or any) Binance-derived paths per market: model p, spot S, EWMA sigma, strike K, TWAP at end.
Everything else (mid, book, taker flow, maker fills, settlement noise) is drawn from params.json.

Timing inside second t (row t = unix second open+t, interval (t-1, t]):
  1. book regime / tick grid / update event; on an update the book re-aligns to the latent mid and redraws
     spread, gaps and sizes;
  2. taker arrivals matched against that book (direction from gap = p_t - mid), each sweeping the opposite
     side and depleting it; maker legs at resting prices;
  3. row t = book after the trades (build.py: latest snapshot <= t). Trades are stamped (block time)
     t + d, d in {3, 4}, and aggregated into the trade columns of row t + d (dropped from ticks if > end,
     like real settlement-lag prints, but kept in the fills table).
Prices are handled in integer mils (0.001); the coarse grid is 10 mils.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

import common as cm

C = cm.C
EPS = 1e-9


# --------------------------------------------------------------------------------------------- params
class Model:
    """params.json converted to arrays for fast lookups (one per contract length)."""

    def __init__(self, params: dict):
        self.raw = params
        b, f, mid = params["book"], params["flow"], params["mid"]
        self.thr_fine = b["tick_switch"]["ext_threshold"]
        self.reg_tab = np.asarray(b["regime"]["table"])                     # [3, NTAU, NEXT, 3]
        self.keep = {i: np.array([b["regime"]["p_keep"][s][i] for s in ("normal", "hollow", "onesided")])
                     for i in range(3)}
        self.spread = {}
        for g in ("coarse", "fine"):
            sp = b["spread"][g]
            self.spread[g] = (np.asarray(sp["support_mils"], int), np.cumsum(np.asarray(sp["table"]), -1))
        self.gaps = {g: [(np.asarray(d["support_mils"], int), np.cumsum(d["p"])) for d in b["gaps"][g]]
                     for g in ("coarse", "fine")}
        self.size = {i: (np.asarray(b["size"][cm.LEN_NAME[L]]["mu"]), np.asarray(b["size"][cm.LEN_NAME[L]]["sd"]))
                     for i, L in enumerate(cm.LENS)}
        self.size_acf1 = b["size"]["acf1_log_size_per_second"]
        self.depth_ratio = {i: (b["depth_ratio"][cm.LEN_NAME[L]]["mu"], b["depth_ratio"][cm.LEN_NAME[L]]["sd"])
                            for i, L in enumerate(cm.LENS)}
        h = b["hollow"]
        self.hollow_bid = np.round(np.asarray(h["bid_px"]) * 1000).astype(int)
        self.hollow_ask = np.round(np.asarray(h["ask_px"]) * 1000).astype(int)
        self.hollow_cum = np.cumsum(h["p"])
        self.hollow_size = (h["log_size_mu"], h["log_size_sd"])
        self.mid = {i: mid[cm.LEN_NAME[L]] for i, L in enumerate(cm.LENS)}
        g = f["glm"]
        self.tau_cell = np.asarray(g["tau_cell"])
        self.b_gap, self.b_z5, self.b_os = g["b_absgap"], g["b_absz5"], g["b_onesided"]
        self.gamma, self.decay = g["gamma_excite"], np.exp(-1.0 / g["tau_excite_s"])
        me = f["market_effect"]
        self.a_len = {i: me["a_len"][cm.LEN_NAME[L]] for i, L in enumerate(cm.LENS)}
        self.a_coin = np.array([me["a_coin"][c] for c in cm.COINS])
        self.a_sd = me["sd"]
        d = f["direction"]
        self.p_buy = {i: np.asarray(d[cm.LEN_NAME[L]]["p_buy"]) for i, L in enumerate(cm.LENS)}
        self.rho_copy = {i: d["rho_copy_by_len"][cm.LEN_NAME[L]] for i, L in enumerate(cm.LENS)}
        sq = f["size_log_quantiles"]
        self.size_q = np.asarray(sq["q"])
        self.size_v = {i: np.asarray(sq[cm.LEN_NAME[L]]) for i, L in enumerate(cm.LENS)}
        self.lim_cum = np.cumsum(f["sweep"]["p_extra_levels"])
        self.legs_mean = f["sweep"]["legs_per_level_mean"]
        self.tok = f["tokens"]
        self.delay_v = np.asarray(f["stamp_delay"]["values"])
        self.delay_cum = np.cumsum(f["stamp_delay"]["p"])
        s = params["settlement"]
        self.open_basis = np.array([s["open_basis_bp_by_coin"].get(c, -2.4) for c in cm.COINS]) / 1e4
        self.sigma_basis = s["sigma_basis_bp"] / 1e4
        self.fees = params["fees"]
        self.rand = params["randomization"]


# --------------------------------------------------------------------------------------------- randomisation
DEFAULT_KNOBS = dict(mid_extra_lag_s=0, passthrough_a0_mult=1.0, lag_halflife_mult=1.0, sigma_u_mult=1.0,
                     refresh_rate_mult=1.0, taker_intensity_mult=1.0, taker_size_mult=1.0, informed_mult=1.0,
                     excitation_mult=1.0, depth_mult=1.0, spread_mult=1.0, onesided_logit_shift=0.0,
                     hollow_logit_shift=0.0, fee_rate_mult=1.0, fee_free=0, rebate_mult=1.0, sigma_basis_mult=0.0)


def draw_knobs(model: Model, B: int, rng, randomize: bool) -> dict:
    k = {n: np.full(B, v, dtype=float) for n, v in DEFAULT_KNOBS.items()}
    if not randomize:
        return k
    for name, spec in model.rand.items():
        if not isinstance(spec, dict) or name not in k:
            continue
        lo, hi = spec["lo"], spec["hi"]
        if spec["kind"] == "int":
            k[name] = rng.integers(lo, hi + 1, B).astype(float)
        elif spec["kind"] == "log":
            k[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), B))
        else:
            k[name] = rng.uniform(lo, hi, B)
    k["fee_free"] = (rng.random(B) < model.rand["fee_rate_mult"].get("p_fee_free", 0.0)).astype(float)
    return k


# --------------------------------------------------------------------------------------------- helpers
def _cat(cum, u):
    """Categorical draw: cum [..., S] cumulative probs (broadcast to u's shape + [S])."""
    return np.minimum((u[..., None] > cum).sum(-1), cum.shape[-1] - 1)


def _ffill_idx(ev):
    idx = np.where(ev, np.arange(ev.shape[1])[None, :], 0)
    return np.maximum.accumulate(idx, axis=1)


# --------------------------------------------------------------------------------------------- engine
def simulate(model: Model, L: int, inp: dict, rng, knobs: dict) -> dict:
    """Simulate B markets of length L. inp: p, S, S_prev, sigma [B, n]; K, twap_end, coin_id [B]."""
    li = cm.LEN_IDX[L]
    p = np.asarray(inp["p"], float)
    B, n = p.shape
    assert n == L + 1
    S = np.asarray(inp["S"], float)
    tau = L - np.arange(n)
    tb_row = cm.tau_bucket(np.minimum(tau, L - 1))                         # [n]
    eb = cm.ext_bucket(p)                                                  # [B, n]
    kn = knobs

    # ---------------- latent mid (sticky, re-aligns at update events) ----------------
    mp = model.mid[li]
    x = cm.logit(p, 1e-4)
    lag = kn["mid_extra_lag_s"].astype(int)
    if lag.any():
        cols = np.maximum(np.arange(n)[None, :] - lag[:, None], 0)
        x = np.take_along_axis(x, cols, axis=1)
    s_f = mp["s_fast"] * np.minimum(kn["passthrough_a0_mult"], 1.0 / max(mp["s_fast"], 1e-3))
    s_l = mp["s_slow"]
    d_slow = 0.5 ** (1.0 / (mp["halflife_slow_s"] * kn["lag_halflife_mult"]))
    xb = np.empty_like(x)
    for dv in np.unique(d_slow):                                           # few distinct values unless randomised
        r = d_slow == dv
        xb[r] = lfilter([1 - dv], [1, -dv], x[r], axis=1, zi=dv * x[r, :1])[0]
    phi = 0.5 ** (1.0 / mp["halflife_u_s"])
    sig_u = mp["sigma_u"] * kn["sigma_u_mult"]
    z = rng.standard_normal((B, n))
    u = lfilter([1.0], [1, -phi], z, axis=1, zi=phi * z[:, :1] / np.sqrt(1 - phi ** 2))[0] * sig_u[:, None]
    target = s_f[:, None] * x + (s_l - s_f)[:, None] * xb + u + np.asarray(mp["b_tau"])[tb_row][None, :]
    pi_upd = np.clip(np.asarray(mp["update_prob"])[tb_row][None, :] * kn["refresh_rate_mult"][:, None], 0, 1)
    ev = rng.random((B, n)) < pi_upd
    ev[:, 0] = True
    m_lat = cm.sigmoid(np.take_along_axis(target, _ffill_idx(ev), axis=1))   # displayed latent mid
    fine = np.maximum.accumulate(np.abs(m_lat - 0.5) > model.thr_fine, axis=1)

    # ---------------- book regime chain (normal / hollow / one-sided) ----------------
    tab = model.reg_tab[li][tb_row[None, :], eb]                            # [B, n, 3]
    shift = np.stack([np.zeros(B), kn["hollow_logit_shift"], kn["onesided_logit_shift"]], 1)[:, None, :]
    tab = tab * np.exp(shift)
    tab /= tab.sum(-1, keepdims=True)
    keep = model.keep[li]
    tabp = tab * (1 - keep)[None, None, :]
    cump = np.cumsum(tabp / tabp.sum(-1, keepdims=True), -1)
    reg = np.empty((B, n), np.int8)
    reg[:, 0] = _cat(np.cumsum(tab[:, 0], -1), rng.random(B))
    U1, U2 = rng.random((B, n)), rng.random((B, n))
    for t in range(1, n):
        prev = reg[:, t - 1]
        stay = (U1[:, t] < keep[prev]) & ~((prev == cm.ONESIDED) & (tab[:, t, cm.ONESIDED] < 0.02))
        reg[:, t] = np.where(stay, prev, _cat(cump[:, t], U2[:, t]))
    rebuild = ev.copy()
    rebuild[:, 1:] |= (reg[:, 1:] != reg[:, :-1]) | (fine[:, 1:] != fine[:, :-1])

    # ---------------- per-market flow constants ----------------
    coin = np.asarray(inp["coin_id"], int)
    A = np.exp(model.a_len[li] + model.a_coin[coin] + model.a_sd * rng.standard_normal(B)) * kn["taker_intensity_mult"]
    gam = model.gamma * kn["excitation_mult"]
    b_gap = model.b_gap * kn["informed_mult"]
    pbuy = model.p_buy[li]
    rho_copy = model.rho_copy[li]
    lnS = np.log(S)
    lnS5 = np.concatenate([np.repeat(lnS[:, :1], 5, 1), lnS[:, :-5]], 1)
    z5 = np.minimum(np.abs(lnS - lnS5) / (np.maximum(np.asarray(inp["sigma"]), 1e-7) * np.sqrt(5)), 6.0)
    cell = model.tau_cell[li][tb_row]                                      # [n]
    mu_sz, sd_sz = model.size[li]
    pi_bar = float(np.mean(np.asarray(mp["update_prob"])))
    rho_sz = float(np.clip((model.size_acf1[li] - (1 - pi_bar)) / max(pi_bar, 1e-3), 0.0, 0.98))
    dr_mu, dr_sd = model.depth_ratio[li]
    depth_mult = kn["depth_mult"]
    size_v = model.size_v[li]

    # ---------------- state ----------------
    bid_px = np.zeros((B, 5), np.int64)
    ask_px = np.full((B, 5), 1000, np.int64)
    bid_sz = np.zeros((B, 5))
    ask_sz = np.zeros((B, 5))
    zsz = rng.standard_normal((B, 2, 5))
    dratio = np.zeros((B, 2))
    age = np.zeros(B)
    E = np.zeros(B)
    prev_sign = np.zeros(B, np.int8)
    ticks = np.zeros((B, n, cm.F), np.float32)
    tr_b, tr_t, tr_sign, tr_fill, tr_px = [], [], [], [], []

    def draw_gap(R, fine_r, level):
        out = np.empty(len(R), np.int64)
        for g, fl in (("coarse", False), ("fine", True)):
            sel = fine_r == fl
            if sel.any():
                sup, cum = model.gaps[g][level - 1]
                out[sel] = sup[_cat(cum, rng.random(sel.sum()))]
        return out

    def layout_normal(R, t):
        fr = fine[R, t]
        tick = np.where(fr, 1, 10)
        spr = np.empty(len(R), np.int64)
        for g, fl in (("coarse", False), ("fine", True)):
            sel = fr == fl
            if sel.any():
                sup, cum = model.spread[g]
                c = cum[li][tb_row[t], eb[R[sel], t]]
                spr[sel] = sup[_cat(c, rng.random(sel.sum()))]
        s = np.maximum(np.round(spr * kn["spread_mult"][R] / tick) * tick, tick).astype(np.int64)
        c = m_lat[R, t] * 1000.0
        b0 = np.round((c - s / 2.0) / tick).astype(np.int64) * tick
        a0 = b0 + s
        hi = 1000 - tick
        over = a0 > hi
        a0 = np.where(over, hi, a0)
        b0 = np.where(over, np.maximum(a0 - s, tick), b0)
        under = b0 < tick
        b0 = np.where(under, tick, b0)
        a0 = np.where(under, np.minimum(b0 + s, hi), a0)
        a0 = np.maximum(a0, b0 + tick)
        bp = np.empty((len(R), 5), np.int64)
        ap = np.empty((len(R), 5), np.int64)
        bp[:, 0], ap[:, 0] = b0, a0
        for k in range(1, 5):
            bp[:, k] = bp[:, k - 1] - draw_gap(R, fr, k)
            ap[:, k] = ap[:, k - 1] + draw_gap(R, fr, k)
        bp = np.where(bp >= tick[:, None], bp, 0)
        ap = np.where(ap <= hi[:, None], ap, 1000)
        return bp, ap

    def redraw_sizes(R):
        zsz[R] = rho_sz * zsz[R] + np.sqrt(1 - rho_sz ** 2) * rng.standard_normal((len(R), 2, 5))
        sz = np.exp(mu_sz[None, None, :] + sd_sz[None, None, :] * zsz[R]) * depth_mult[R, None, None]
        dratio[R] = dr_mu + dr_sd * rng.standard_normal((len(R), 2))
        return sz[:, 0], sz[:, 1]

    def compact(px, sz, side, R, t):
        """Remove exhausted levels (shift left) and append fresh levels behind the last one."""
        empty = sz <= EPS
        order = np.argsort(empty, axis=1, kind="stable")
        px = np.take_along_axis(px, order, 1)
        sz = np.take_along_axis(sz, order, 1)
        empty = np.take_along_axis(empty, order, 1)
        fr = fine[R, t]
        tick = np.where(fr, 1, 10)
        for k in range(1, 5):
            need = empty[:, k] & ~empty[:, k - 1]
            if not need.any():
                continue
            rr = np.flatnonzero(need)
            gap = draw_gap(R[rr], fr[rr], k)
            newp = px[rr, k - 1] - gap if side == 0 else px[rr, k - 1] + gap
            ok = (newp >= tick[rr]) if side == 0 else (newp <= 1000 - tick[rr])
            px[rr, k] = np.where(ok, newp, 0 if side == 0 else 1000)
            sz[rr, k] = np.where(ok, np.exp(mu_sz[k] + sd_sz[k] * rng.standard_normal(len(rr)))
                                 * depth_mult[R[rr]], 0.0)
            empty[rr, k] = ~ok
        px = np.where(empty, 0 if side == 0 else 1000, px)
        sz = np.where(empty, 0.0, sz)
        return px, sz

    hcfg_n = len(model.hollow_cum)
    for t in range(n):
        # ---- 1. book update ----
        R = np.flatnonzero(rebuild[:, t])
        if len(R):
            rg = reg[R, t]
            bsz, asz = redraw_sizes(R)
            bp, ap = layout_normal(R, t)
            hol = rg == cm.HOLLOW
            if hol.any():
                ci = np.minimum((rng.random(hol.sum())[:, None] > model.hollow_cum).sum(1), hcfg_n - 1)
                bp[hol] = model.hollow_bid[ci]
                ap[hol] = model.hollow_ask[ci]
                hs = np.exp(model.hollow_size[0] + model.hollow_size[1] * rng.standard_normal((hol.sum(), 2, 5)))
                bsz[hol] = hs[:, 0] * depth_mult[R[hol], None]
                asz[hol] = hs[:, 1] * depth_mult[R[hol], None]
            one = rg == cm.ONESIDED
            if one.any():
                up = p[R, t] > 0.5
                kill_ask = one & up
                kill_bid = one & ~up
                ap[kill_ask] = 1000
                bp[kill_bid] = 0
            bid_px[R], ask_px[R] = bp, ap
            bid_sz[R] = np.where(bp > 0, bsz, 0.0)
            ask_sz[R] = np.where(ap < 1000, asz, 0.0)
            age[R] = 0
        nr = ~rebuild[:, t]
        age[nr] += 1

        # ---- 2. taker flow ----
        hb, ha = bid_sz[:, 0] > EPS, ask_sz[:, 0] > EPS
        mid = np.where(hb & ha, (bid_px[:, 0] + ask_px[:, 0]) / 2000.0,
                       np.where(hb, (bid_px[:, 0] / 1000.0 + 1) / 2, np.where(ha, ask_px[:, 0] / 2000.0, 0.5)))
        gap = p[:, t] - mid
        lam = (A * np.exp(cell[t] + b_gap * np.minimum(np.abs(gap), 0.3) + model.b_z5 * z5[:, t]
                          + model.b_os * (reg[:, t] == cm.ONESIDED)) * (1.0 + E) ** gam)
        lam = np.where(reg[:, t] == cm.HOLLOW, 0.0, np.minimum(lam, 50.0))
        N = rng.poisson(lam)
        q = np.clip(0.5 + kn["informed_mult"] * (pbuy[cm.gap_bucket(gap)] - 0.5), 0.02, 0.98)
        executed = np.zeros(B)
        for k in range(int(N.max()) if N.size else 0):
            R = np.flatnonzero(N > k)
            copy = (rng.random(len(R)) < rho_copy) & (prev_sign[R] != 0)
            sgn = np.where(copy, prev_sign[R], np.where(rng.random(len(R)) < q[R], 1, -1)).astype(np.int8)
            Q = np.exp(np.interp(rng.random(len(R)), model.size_q, size_v)) * kn["taker_size_mult"][R]
            jlim = np.minimum((rng.random(len(R))[:, None] > model.lim_cum).sum(1), 4)
            buy = sgn == 1
            szs = np.where(buy[:, None], ask_sz[R], bid_sz[R])
            pxs = np.where(buy[:, None], ask_px[R], bid_px[R])
            szs = np.where(np.arange(5)[None, :] <= jlim[:, None], szs, 0.0)
            cum = np.cumsum(szs, 1)
            prevc = cum - szs
            fill = np.clip(Q[:, None] - prevc, 0.0, szs)
            tot = fill.sum(1)
            ok = tot > 1e-6
            if not ok.any():
                continue
            R, sgn, fill, pxs, buy = R[ok], sgn[ok], fill[ok], pxs[ok], buy[ok]
            tr_b.append(R)
            tr_t.append(np.full(len(R), t, np.int32))
            tr_sign.append(sgn)
            tr_fill.append(fill)
            tr_px.append(pxs)
            prev_sign[R] = sgn
            executed[R] += 1
            age[R] = 0
            for side, Rs in ((1, R[buy]), (0, R[~buy])):
                if not len(Rs):
                    continue
                f = fill[buy] if side == 1 else fill[~buy]
                if side == 1:
                    ask_px[Rs], ask_sz[Rs] = compact(ask_px[Rs], ask_sz[Rs] - f, 1, Rs, t)
                else:
                    bid_px[Rs], bid_sz[Rs] = compact(bid_px[Rs], bid_sz[Rs] - f, 0, Rs, t)
        E = model.decay * E + executed

        # ---- 3. write row t (book after trades) ----
        row = ticks[:, t]
        row[:, C["bid_px_0"]:C["bid_px_0"] + 5] = bid_px / 1000.0
        row[:, C["ask_px_0"]:C["ask_px_0"] + 5] = ask_px / 1000.0
        row[:, C["bid_sz_0"]:C["bid_sz_0"] + 5] = bid_sz
        row[:, C["ask_sz_0"]:C["ask_sz_0"] + 5] = ask_sz
        row[:, C["bid_depth_total"]] = bid_sz.sum(1) * np.exp(np.maximum(dratio[:, 0], 0.0))
        row[:, C["ask_depth_total"]] = ask_sz.sum(1) * np.exp(np.maximum(dratio[:, 1], 0.0))
        row[:, C["book_age"]] = age

    # ---------------- assemble ticks ----------------
    ticks[:, :, C["seconds_to_close"]] = tau[None, :]
    ticks[:, :, C["book_valid"]] = 1.0
    ticks[:, :, C["const_one"]] = 1.0
    coin_ob = model.open_basis[coin]
    ptb = np.asarray(inp["K"]) * np.exp(coin_ob)
    ticks[:, :, C["ul_logp_rel_open"]] = lnS - lnS[:, :1]
    ticks[:, :, C["ul_logret_1s"]] = lnS - np.log(np.asarray(inp["S_prev"]))
    ticks[:, :, C["ul_logp_vs_strike"]] = lnS - np.log(ptb)[:, None]

    if tr_b:
        tb_ = np.concatenate(tr_b)
        tt_ = np.concatenate(tr_t)
        sg_ = np.concatenate(tr_sign)
        fl_ = np.concatenate(tr_fill)
        px_ = np.concatenate(tr_px) / 1000.0
    else:
        tb_ = tt_ = sg_ = np.zeros(0, int)
        fl_ = px_ = np.zeros((0, 5))
    d_ = model.delay_v[_cat(model.delay_cum, rng.random(len(tb_)))]
    stamp = tt_ + d_
    vol = fl_.sum(1)
    vwap = (fl_ * px_).sum(1) / np.maximum(vol, 1e-12)
    filled = fl_ > 0
    deep_buy = np.where(filled, px_, -np.inf).max(1)
    deep_sell = np.where(filled, px_, np.inf).min(1)
    inwin = stamp <= L
    for sgn, vcol, wcol, xcol, ext in ((1, "buy_vol", "buy_vwap", "buy_max_px", deep_buy),
                                       (-1, "sell_vol", "sell_vwap", "sell_min_px", deep_sell)):
        m = inwin & (sg_ == sgn)
        flat = tb_[m] * n + stamp[m]
        v = np.bincount(flat, weights=vol[m], minlength=B * n).reshape(B, n)
        pv = np.bincount(flat, weights=(vol * vwap)[m], minlength=B * n).reshape(B, n)
        xx = np.full(B * n, -np.inf if sgn == 1 else np.inf)
        (np.maximum if sgn == 1 else np.minimum).at(xx, flat, ext[m])
        xx = xx.reshape(B, n)
        has = v > 0
        ticks[:, :, C[vcol]] = v
        ticks[:, :, C[wcol]] = np.where(has, pv / np.where(has, v, 1), 0)
        ticks[:, :, C[xcol]] = np.where(has & np.isfinite(xx), xx, 0)
    ticks[:, :, C["trade_count"]] = np.bincount(tb_[inwin] * n + stamp[inwin], minlength=B * n).reshape(B, n)

    # ---------------- settlement and market fields ----------------
    eps = rng.standard_normal(B) * model.sigma_basis * kn["sigma_basis_mult"]
    winner = (np.asarray(inp["twap_end"]) * np.exp(eps) >= np.asarray(inp["K"])).astype(np.int32)
    n_in = np.bincount(tb_[inwin], minlength=B)
    return dict(ticks=ticks, winner_up=winner, fine_final=fine[:, -1], ptb=ptb,
                log_strike_vs_ul_open=np.log(S[:, 0] / ptb), n_trades_in_window=n_in,
                trades=dict(b=tb_, t=tt_, stamp=stamp, sign=sg_, fill=fl_, px=px_),
                regime=reg, m_lat=m_lat)


# --------------------------------------------------------------------------------------------- fill legs
def trade_legs(model: Model, trades: dict, rng):
    """Expand simulated taker trades into fill legs in data/raw/trades.parquet's schema (without slug):
    one taker leg per trade (size-weighted avg price) + a geometric number of maker legs per level touched,
    each leg on the Up or Down token (complementary matching) with fitted probabilities."""
    b, stamp, sgn, fill, px = trades["b"], trades["stamp"], trades["sign"], trades["fill"], trades["px"]
    T = len(b)
    if T == 0:
        return {k: np.zeros(0) for k in ("trade", "b", "stamp", "outcome_idx", "side", "price", "size",
                                         "is_taker", "aggressor_up")}
    vol = fill.sum(1)
    vwap = (fill * px).sum(1) / vol
    tk = model.tok
    # taker legs
    p_up_t = np.where(sgn == 1, tk["taker_up_given_buy"], tk["taker_up_given_sell"])
    t_up = rng.random(T) < p_up_t
    t_out = np.where(t_up, 0, 1).astype(np.int8)
    t_side = np.where(t_up, sgn, -sgn).astype(np.int8)
    t_price = np.where(t_up, vwap, 1 - vwap)
    # maker legs: per (trade, level) with fill > 0 -> k legs, sizes split by uniform stick-breaking
    ti, lv = np.nonzero(fill > 0)
    q = 1 - 1.0 / max(model.legs_mean, 1.0 + 1e-6)
    k = rng.geometric(1 - q, len(ti)) if q > 0 else np.ones(len(ti), int)
    k = np.minimum(k, 20)
    rep = np.repeat(np.arange(len(ti)), k)
    w = rng.random(len(rep)) + 0.05
    wsum = np.bincount(rep, weights=w)
    msz = fill[ti, lv][rep] * w / wsum[rep]
    mt = ti[rep]
    mpx = px[ti, lv][rep]
    msg = sgn[mt]
    p_up_m = np.where(msg == 1, tk["maker_up_given_buy"], tk["maker_up_given_sell"])
    m_up = rng.random(len(rep)) < p_up_m
    m_out = np.where(m_up, 0, 1).astype(np.int8)
    m_side = np.where(m_up, -msg, msg).astype(np.int8)       # maker is opposite the aggressor in the Up frame
    m_price = np.where(m_up, mpx, 1 - mpx)
    out = dict(
        trade=np.r_[np.arange(T), mt], b=np.r_[b, b[mt]], stamp=np.r_[stamp, stamp[mt]],
        outcome_idx=np.r_[t_out, m_out], side=np.r_[t_side, m_side],
        price=np.r_[t_price, m_price].astype(np.float32), size=np.r_[vol, msz].astype(np.float32),
        is_taker=np.r_[np.ones(T, bool), np.zeros(len(rep), bool)], aggressor_up=np.r_[sgn, msg].astype(np.int8))
    return out
