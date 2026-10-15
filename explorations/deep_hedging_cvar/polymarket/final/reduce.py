"""MarketTrace (adapters / cenv) -> result tables and per-run aggregates.

Per (method, variant) run, over all markets of the split:

markets  one row per market: P&L (with and without rewards+rebates), components, fills,
         volume, max |inventory|, final inventory, quote uptime (two-sided / one-sided / any;
         the C Log's exact values when the trace carries one), quoted market-minutes, fill rate,
         taker volume and our share of it, the C Log's P&L.
minutes  one row per (market, UTC minute): MTM P&L increment (inventory marked to the mark
         each second and to the settlement value on the closing row, so the increments sum to
         the market's P&L), the same without rewards+rebates, fills, volume, live seconds,
         max |inventory|. Summing over markets and minutes gives the zero-filled 15-min and
         1-h grids of EVALUATION.md section 1 exactly (minutes align with UTC hours).
fills    one row per fill (Up frame): side/qty/price/fee/rebate, the mark at the fill and at
         +1/+5/+10/+30/+60 s (NaN after the close: stats.attribute_pnl then uses the settlement),
         `edge60` = sq (mid +60 s - price) (the env's per-fill P&L of a flag: spread capture +
         60 s markout), the book at the match second, and the T5 flags exactly as the env judges
         them on book[tau] (ENV.md section 7; book.bin in mils when present).
portfolio  per UTC minute: max |sum over live markets of inventory x dp/dlnS| (USD of
         underlying), the combined portfolio delta.

Aggregates (JSON): stats.attribute_pnl totals and markouts (identity error; max difference from
the trace's cash accounting), stats.flag_pnl_share on edge60 for every flag, the T5 verdict on
the artifact flags of DECISIONS.md D2, the T6 directional numbers, the D1.3 market-making
classification, P&L by Up/Down winner, corr(inventory, next 60 s Binance log return), and,
with the C env, the per-market agreement of these flags with the C Log (vol_X, pnl_X).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import LENGTH_NAMES, MARKOUT_HORIZONS, MID_CLOSE_STC, STALE_S, Dataset, S

# the env's flag names (mm.h MM_FL_*, ENV.md section 7), plus a union used by D2
FLAG_NAMES = ["crossed", "stale", "onesided", "last30", "last10", "inside2", "extreme", "outside",
              "inspread"]
FLAG_DOCS = {
    "crossed": "book crossed in the raw data (last uncrossed book forward-filled)",
    "stale": f"book_age > {STALE_S:g} s",
    "onesided": "either side of the book empty",
    "last30": "fewer than 30 s to the close (end - tau < 30)",
    "last10": "fewer than 10 s to the close",
    "inside2": "our quote more than 2 ticks inside the touch",
    "extreme": "our quote <= 0.02 or >= 0.98",
    "outside": "the print is outside [best bid, best ask] (includes every sweep through the touch; "
               "reported, not an exploit detector: DECISIONS.md D2)",
    "inspread": "the print is on the wrong side of the hit touch (a sell above the best bid or a buy "
                "below the best ask): the snapshot cannot explain it",
    "crossed_or_stale": "crossed or stale book (union)",
}
# DECISIONS.md D2: the 25% rule applies to these artifact flags only
RULE_FLAGS = ["inspread", "crossed_or_stale", "onesided", "last30"]
ALL_FLAGS = FLAG_NAMES + ["crossed_or_stale"]
MAX_FLAG_SHARE = 0.25
MAX_INVENTORY_SHARE = 0.50
MIN_FILLS_PER_MARKET = 5.0          # DECISIONS.md D1.3


def _book(ds: Dataset, rows, mirror: bool, tick_rule: int = 0):
    """Top of book at `rows` in integer mils (bb 0 = no bid, ba 1000 = no ask), the env's book
    flags and tick; from book.bin when present (exactly what the env reads), else from ticks.bin."""
    c = ds.ci
    t = np.asarray(ds.ticks[rows], dtype=np.float64)
    out = dict(stc=t[:, c["seconds_to_close"]], taker_vol=t[:, c["buy_vol"]] + t[:, c["sell_vol"]],
               ul=t[:, c["ul_logp_rel_open"]] * (-1.0 if mirror else 1.0))
    if ds.book is not None:
        b = ds.book[rows]
        bb = b["bid_px"][:, 0].astype(np.int64)
        ba = b["ask_px"][:, 0].astype(np.int64)
        fl = b["flags"].astype(np.int64)
        out.update(crossed=(fl & 2) != 0, stale=(fl & 4) != 0, valid=(fl & 1) != 0,
                   tick=(b["tick_alt"] if tick_rule else b["tick"]).astype(np.int64))
    else:
        bid, ask = t[:, c["bid_px_0"]], t[:, c["ask_px_0"]]
        bb = np.where(bid > 1e-9, np.round(bid * 1000), 0).astype(np.int64)
        ba = np.where(ask < 1 - 1e-9, np.round(ask * 1000), 1000).astype(np.int64)
        j = ds.fcol("tick_fine")
        fine = (np.asarray(ds.feat[rows, j]) > 0.5) if j is not None else \
            ((bb > 0) & (bb < 40)) | ((ba < 1000) & (ba > 960))
        out.update(crossed=t[:, c["crossed_raw"]] > 0.5, stale=t[:, c["book_age"]] > STALE_S,
                   valid=t[:, c["book_valid"]] > 0.5, tick=np.where(fine, 1, 10))
    if mirror:
        bb, ba = 1000 - ba, 1000 - bb
    out.update(bb=bb, ba=ba)
    return out


class RunAccumulator:
    """mirror_in_data: the dataset itself is the mirrored copy (mirror.py), so marks and books are
    read as they are; otherwise a mirror variant is mirrored here (stand-in env flag)."""

    def __init__(self, ds: Dataset, split: int, method: str, variant: dict, t_range,
                 mirror_in_data: bool = False, tick_rule: int = 0):
        self.ds, self.split, self.method, self.variant = ds, split, method, variant
        self.mirror = bool(variant.get("mirror", False)) and not mirror_in_data
        self.tick_rule = int(tick_rule)
        self.checks, self.flag_check = [], []
        self.t0, self.t1 = t_range
        self.mk_rows, self.min_parts, self.fill_parts = [], [], []
        self.port = np.zeros(self.t1 - self.t0)
        self.corr = np.zeros(6)        # n, sx, sy, sxx, syy, sxy

    # ---------------------------------------------------------------- one market
    def add(self, tr: dict):
        ds, m = self.ds, tr["market"]
        rec = ds.markets[m]
        sl = ds.market_rows(m)
        n = int(rec["n_rows"])
        t = int(rec["open_ts"]) + np.arange(n)
        mark = ds.mark(m)
        if self.mirror:
            mark = 1.0 - mark
        Sm = float(tr["settlement"])
        markS = mark.copy()
        markS[-1] = Sm
        cash, inv = tr["cash"], tr["inv"]
        rew, reb, fee = tr["reward_cum"], tr["rebate_cum"], tr["fee_cum"]
        for name, v in (("cash", cash), ("inv", inv), ("reward_cum", rew), ("rebate_cum", reb)):
            if not np.all(np.isfinite(v)):
                raise ValueError(f"market {m}: non-finite {name} in trace")
        mtm = cash + inv * markS
        mtm_ex = mtm - rew - reb
        d = np.diff(mtm, prepend=0.0)
        d_ex = np.diff(mtm_ex, prepend=0.0)
        bl = np.asarray(tr["bid_live"]) > 0.5
        al = np.asarray(tr["ask_live"]) > 0.5
        bl[0] = al[0] = False                              # row 0: nothing can be live yet
        two, one, anyq = bl & al, bl ^ al, bl | al
        bk = _book(ds, sl, self.mirror, self.tick_rule)

        # fills
        f = tr["fills"]
        nf = len(f["row"])
        k = (np.asarray(f["row"], np.int64) - sl.start) if nf else np.zeros(0, np.int64)
        if nf and (k.min() < 0 or k.max() >= n):
            raise ValueError(f"market {m}: fill rows outside the market")
        side = np.asarray(f["side"], np.int64)
        qty = np.asarray(f["qty"], np.float64)
        price = np.asarray(f["price"], np.float64)

        # minute grid
        mi = (t // 60) - (t[0] // 60)
        nm = int(mi[-1]) + 1

        def bc(w, idx=mi, size=nm):
            return np.bincount(idx, weights=w, minlength=size)
        fm = mi[k] if nf else np.zeros(0, np.int64)
        live = np.zeros(n, bool)
        live[1:] = True
        absinv = np.abs(inv)
        max_inv_min = np.zeros(nm)
        np.maximum.at(max_inv_min, mi, absinv)
        last_in_min = np.r_[np.flatnonzero(np.diff(mi)), n - 1]
        self.min_parts.append(pd.DataFrame(dict(
            market=m, minute_ts=(t[0] // 60 + np.arange(nm)) * 60,
            dpnl=bc(d), dpnl_ex_rr=bc(d_ex), fills=bc(np.ones(nf), fm), volume=bc(qty, fm),
            live_s=bc(live.astype(float)), two_sided_s=bc(two.astype(float)),
            any_quote_s=bc(anyq.astype(float)), max_abs_inv=max_inv_min,
            inv_end=inv[last_in_min])))

        # portfolio delta
        dl = ds.delta_usd(m)
        if dl is not None:
            dd = inv * dl * (-1.0 if self.mirror else 1.0)
            self.port[t[0] - self.t0: t[-1] + 1 - self.t0] += dd

        # inventory vs next-60 s underlying return
        if n > 61:
            x, y = inv[:-60], bk["ul"][60:] - bk["ul"][:-60]
            self.corr += [x.size, x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]

        # fill table with the env's flags on book[tau]
        lg = tr.get("log")
        if nf:
            mids = {h: np.where(k + h <= n - 1, mark[np.minimum(k + h, n - 1)], np.nan)
                    for h in MARKOUT_HORIZONS}
            edge = side * qty * (np.where(np.isnan(mids[60]), Sm, mids[60]) - price)
            bb, ba, tick = bk["bb"][k], bk["ba"][k], bk["tick"][k]
            pm = np.round(price * 1000).astype(np.int64)
            ppx = np.asarray(f["print_px"], np.float64)
            known = np.isfinite(ppx)
            fp = np.where(known, np.round(np.nan_to_num(ppx) * 1000), 0).astype(np.int64)
            buy = side > 0                                 # our bid (env side 0) was hit
            flags = dict(
                crossed=bk["crossed"][k], stale=bk["stale"][k], onesided=(bb == 0) | (ba == 1000),
                last30=(n - 1 - k) < 30, last10=(n - 1 - k) < 10,
                inside2=np.where(buy, (bb > 0) & (pm > bb + 2 * tick), (ba < 1000) & (pm < ba - 2 * tick)),
                extreme=(pm <= 20) | (pm >= 980),
                outside=known & ((fp < bb) | (fp > ba)),
                inspread=known & np.where(buy, fp > bb, fp < ba))
            flags["crossed_or_stale"] = flags["crossed"] | flags["stale"]
            self.fill_parts.append(pd.DataFrame(dict(
                market=m, t=t[k], seconds_to_close=bk["stc"][k], side=side, qty=qty, price=price,
                fee=np.asarray(f["fee"], float), rebate=np.asarray(f["rebate"], float),
                print_px=ppx, mid=mark[k], **{f"mid_{h}": v for h, v in mids.items()}, edge60=edge,
                book_bid=bb / 1000.0, book_ask=ba / 1000.0, tick=tick / 1000.0,
                print_px_known=known, **flags)))
        if lg:                                    # the env's own flag sums for this market
            cmp = {}
            for x in FLAG_NAMES:
                if f"vol_{x}" not in lg:
                    continue
                hit = self.fill_parts[-1][x].to_numpy(bool) if nf else np.zeros(0, bool)
                v = float(qty[hit].sum()) if nf else 0.0
                p = float(edge[hit].sum()) if nf else 0.0
                cmp[x] = (abs(v - lg[f"vol_{x}"]), abs(p - lg.get(f"pnl_{x}", p)))
            if cmp:
                self.flag_check.append(dict(market=m, vol=max(a for a, _ in cmp.values()),
                                            pnl=max(b for _, b in cmp.values())))

        pnl = float(mtm[-1])
        ref = max(n - 1 - MID_CLOSE_STC, 0)
        qmin = int((bc(anyq.astype(float)) > 0).sum())
        tv = float(bk["taker_vol"].sum())
        vol = float(qty.sum())
        L = int(rec["contract_len_s"])
        up2, up1, upa = float(two[1:].mean()), float(one[1:].mean()), float(anyq[1:].mean())
        maxinv = float(absinv.max())
        cpnl = float("nan")
        if lg:                                   # the C Log of this market: exact per-second numbers
            up2, upa = float(lg["uptime_two_sided"]), float(lg["uptime_any"])
            up1, maxinv, cpnl = upa - up2, max(maxinv, float(lg["max_abs_inv"])), float(lg["pnl"])
        if tr.get("checks"):
            self.checks.append(dict(market=m, **tr["checks"]))
        self.mk_rows.append(dict(
            market=m, slug=ds.slug(m), coin=ds.coin(m), length=LENGTH_NAMES.get(L, str(L)),
            contract_len_s=L, open_ts=int(rec["open_ts"]), end_ts=int(rec["end_ts"]),
            settlement=Sm, winner_up=int(rec["winner_up"]), pnl=pnl, pnl_ex_rr=float(mtm_ex[-1]),
            rewards=float(rew[-1]), rebates=float(reb[-1]), fees=float(fee[-1]), n_fills=nf,
            buy_fills=int((side > 0).sum()), sell_fills=int((side < 0).sum()), volume=vol,
            notional=float((qty * price).sum()), max_abs_inv=maxinv,
            final_inv=float(inv[-1]), final_cash=float(cash[-1]), mid_close=float(mark[ref]),
            uptime_two=up2, uptime_one=up1, uptime_any=upa, quoted_minutes=qmin, c_log_pnl=cpnl,
            fill_rate=nf / qmin if qmin else np.nan, taker_volume=tv,
            share_taker_volume=vol / tv if tv > 0 else np.nan))

    # ---------------------------------------------------------------- finish
    def finish(self):
        mk = pd.DataFrame(self.mk_rows)
        mins = pd.concat(self.min_parts, ignore_index=True) if self.min_parts else pd.DataFrame()
        fl = pd.concat(self.fill_parts, ignore_index=True) if self.fill_parts else None
        if fl is None:
            cols = {"market": "int64", "t": "int64", "seconds_to_close": "float64", "side": "int64"}
            cols.update({c: "float64" for c in ["qty", "price", "fee", "rebate", "print_px", "mid"]
                         + [f"mid_{h}" for h in MARKOUT_HORIZONS] + ["edge60", "book_bid", "book_ask", "tick"]})
            cols.update({c: "bool" for c in ["print_px_known"] + ALL_FLAGS})
            fl = pd.DataFrame({c: pd.Series(dtype=d) for c, d in cols.items()})
        for frame in (mk, mins, fl):
            frame.insert(0, "variant", self.variant["id"])
            frame.insert(0, "method", self.method)
        pm = self.port.reshape(-1, 60) if self.port.size % 60 == 0 else \
            np.pad(self.port, (0, -self.port.size % 60)).reshape(-1, 60)
        port = pd.DataFrame(dict(method=self.method, variant=self.variant["id"],
                                 minute_ts=self.t0 + 60 * np.arange(pm.shape[0]),
                                 max_abs_delta_usd=np.abs(pm).max(1), mean_delta_usd=pm.mean(1)))
        agg = self.aggregates(mk, fl)
        if self.checks:
            ck = pd.DataFrame(self.checks)
            bad = ~ck.replay_ok.astype(bool)
            agg["replay"] = dict(n_markets=int(len(ck)), n_not_ok=int(bad.sum()),
                                 markets_not_ok=[int(x) for x in ck.market[bad]],
                                 state_max_abs_diff=float(ck.state_max_abs_diff.max()),
                                 log_pnl_max_abs_diff=float(ck.log_pnl_abs_diff.max()),
                                 obs_bit_identical=bool(ck.obs_bit_identical.all()))
        if self.flag_check:
            fc = pd.DataFrame(self.flag_check)
            agg["flag_check_vs_c_log"] = dict(n_markets=int(len(fc)), vol_max_abs_diff=float(fc.vol.max()),
                                              pnl_max_abs_diff=float(fc.pnl.max()))
        return dict(markets=mk, minutes=mins, fills=fl, portfolio=port, agg=agg)

    def aggregates(self, mk: pd.DataFrame, fl: pd.DataFrame) -> dict:
        fills = {"market": fl["market"].to_numpy(np.int64), "side": fl["side"].to_numpy(np.int64),
                 "qty": fl["qty"].to_numpy(float), "price": fl["price"].to_numpy(float),
                 "mid": fl["mid"].to_numpy(float), "fee": fl["fee"].to_numpy(float),
                 "rebate": fl["rebate"].to_numpy(float)}
        for h in MARKOUT_HORIZONS:
            fills[f"mid_{h}"] = fl[f"mid_{h}"].to_numpy(float)
        markets = {"market": mk["market"].to_numpy(np.int64),
                   "settlement": mk["settlement"].to_numpy(float),
                   "reward": mk["rewards"].to_numpy(float),
                   "final_inventory": mk["final_inv"].to_numpy(float),
                   "mid_close": mk["mid_close"].to_numpy(float)}
        att = S.attribute_pnl(fills, markets, horizon=60)
        pm = dict(zip(att["market_ids"].tolist(), att["per_market"]["total"].tolist()))
        diff = [abs(pm.get(int(a), 0.0) - b) for a, b in zip(mk["market"], mk["pnl"])]
        tot = att["totals"]
        ex = exploit_tables(fl)
        cls = market_making(mk, tot, ex["verdict"])
        n, sx, sy, sxx, syy, sxy = self.corr
        cov = sxy / n - sx * sy / n ** 2 if n else float("nan")
        vx = sxx / n - (sx / n) ** 2 if n else float("nan")
        vy = syy / n - (sy / n) ** 2 if n else float("nan")
        corr = cov / np.sqrt(vx * vy) if n and vx > 0 and vy > 0 else float("nan")
        up = mk["winner_up"] == 1
        return dict(
            method=self.method, variant=self.variant["id"], n_markets=int(len(mk)),
            pnl=float(mk["pnl"].sum()), pnl_ex_rr=float(mk["pnl_ex_rr"].sum()),
            n_fills=int(mk["n_fills"].sum()), volume=float(mk["volume"].sum()),
            attribution=dict(totals=tot, markouts={str(h): v for h, v in att["markouts"].items()},
                             identity_error=att["identity_error"],
                             cash_check_max_abs=float(max(diff)) if diff else 0.0,
                             inventory_mismatch=[int(x) for x in att["inventory_mismatch"]]),
            exploit=ex, market_making=cls,
            directional=dict(**directional_verdict(tot),
                             pnl_up_won=float(mk.loc[up, "pnl"].sum()),
                             pnl_down_won=float(mk.loc[~up, "pnl"].sum()),
                             n_up_won=int(up.sum()), n_down_won=int((~up).sum()),
                             corr_inventory_next60s_return=float(corr), corr_n=int(n)))


def exploit_tables(fl: pd.DataFrame) -> dict:
    """T5 on the per-fill edge (spread capture + 60 s markout, as the env's pnl_X): shares for
    every flag (reported) and the verdict on the D2 artifact flags."""
    edge = fl["edge60"].to_numpy(float) if len(fl) else np.zeros(0)
    allf = S.flag_pnl_share(edge, {k: fl[k].to_numpy(bool) for k in ALL_FLAGS}) if len(fl) else \
        {k: dict(n=0, frac_fills=0.0, pnl=0.0, share_of_total=float("nan"), pnl_excluding=0.0)
         for k in ALL_FLAGS + ["_any"]}
    rule = S.flag_pnl_share(edge, {k: fl[k].to_numpy(bool) for k in RULE_FLAGS}) if len(fl) else \
        {k: dict(n=0, share_of_total=float("nan"), pnl=0.0, pnl_excluding=0.0) for k in RULE_FLAGS + ["_any"]}
    v = exploit_verdict(rule, float(edge.sum()))
    vol = fl["qty"].to_numpy(float) if len(fl) else np.zeros(0)
    vshare = {k: (float(vol[fl[k].to_numpy(bool)].sum() / vol.sum()) if vol.sum() > 0 else float("nan"))
              for k in ALL_FLAGS}
    return dict(flags=allf, rule_flags=rule, volume_share=vshare, verdict=v,
                n_fills_print_px_unknown=int((~fl["print_px_known"].astype(bool)).sum()) if len(fl) else 0)


def exploit_verdict(flags: dict, edge_total: float, rule_flags=None) -> dict:
    """T5 + D2: invalid if any artifact flag carries > 25% of the total fill edge, or the edge
    excluding all artifact-flagged fills is <= 0 while the total is > 0 (sign change)."""
    names = [k for k in (rule_flags or RULE_FLAGS) if k in flags]
    single = {k: flags[k]["share_of_total"] for k in names}
    ok = {k: v for k, v in single.items() if v is not None and v == v}
    worst_k = max(ok, key=ok.get) if ok else None
    worst = ok[worst_k] if ok else float("nan")
    excl = flags["_any"]["pnl_excluding"]
    inv_share = bool(ok and worst > MAX_FLAG_SHARE)
    sign = bool(edge_total > 0 and excl <= 0)
    return dict(edge_total=float(edge_total), rule_flags=names, worst_single_flag=worst_k,
                worst_single_share=worst, edge_excluding_flagged=float(excl),
                invalid_single_flag=inv_share, invalid_sign_change=sign, invalid=inv_share or sign)


def directional_verdict(tot: dict) -> dict:
    """T6 from attribution totals."""
    total = float(tot["total"])
    inv_pnl = float(tot["inventory"])
    pos = sum(max(float(tot[c]), 0.0) for c in
              ("spread_capture", "adverse_selection", "inventory", "rebates", "rewards"))
    share_total = inv_pnl / total if total > 0 else float("nan")
    share_pos = inv_pnl / pos if pos > 0 else float("nan")
    directional = bool(inv_pnl > 0 and ((total > 0 and share_total > MAX_INVENTORY_SHARE)
                                        or (pos > 0 and share_pos > MAX_INVENTORY_SHARE)))
    return dict(inventory_pnl=inv_pnl, total=total, share_of_total=share_total,
                share_of_positive=share_pos, directional=directional)


def market_making(mk: pd.DataFrame, tot: dict, ex_verdict: dict) -> dict:
    """DECISIONS.md D1.3: a policy counts as making markets only with >= 5 fills per market, a T6
    directional share <= 50% and a T5 artifact share <= 25%; otherwise describe what it does."""
    w = mk["contract_len_s"].to_numpy(float)
    fpm = float(mk["n_fills"].mean()) if len(mk) else 0.0
    up2 = float((mk["uptime_two"] * w).sum() / w.sum()) if w.sum() else float("nan")
    upa = float((mk["uptime_any"] * w).sum() / w.sum()) if w.sum() else float("nan")
    dv = directional_verdict(tot)
    ok = dict(fills=fpm >= MIN_FILLS_PER_MARKET, t6=not dv["directional"], t5=not ex_verdict["invalid_single_flag"])
    counts = all(ok.values())
    frac_quiet = float((mk["n_fills"] == 0).mean()) if len(mk) else float("nan")
    if counts:
        what = "makes markets"
    elif upa < 0.05 or fpm < 1:
        what = "mostly does not quote"
    elif not ok["t6"]:
        what = "holds directional inventory (T6)"
    elif not ok["t5"]:
        what = "earns from replay artifacts (T5)"
    elif up2 < 0.25:
        what = "quotes mostly one side"
    else:
        what = "quotes but rarely trades"
    return dict(counts_as_market_making=counts, criteria=ok, fills_per_market=fpm,
                uptime_two_sided=up2, uptime_any=upa, frac_markets_without_fills=frac_quiet,
                description=what)
