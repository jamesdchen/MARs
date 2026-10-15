"""MarketTrace (adapters.drive) -> result tables and per-run aggregates.

Per (method, variant) run, over all markets of the split:

markets  one row per market: P&L (with and without rewards+rebates), components, fills,
         volume, max |inventory|, final inventory, quote uptime (two-sided / one-sided / any,
         over the market's live seconds open+1..end), quoted market-minutes, fill rate,
         taker volume and our share of it.
minutes  one row per (market, UTC minute): MTM P&L increment (inventory marked to the mark
         each second and to the settlement value on the closing row, so the increments sum to
         the market's P&L), the same without rewards+rebates, fills, volume, live seconds,
         max |inventory|. Summing over markets and minutes gives the zero-filled 15-min and
         1-h grids of EVALUATION.md section 1 exactly (minutes align with UTC hours).
fills    one row per fill: Up-frame side/qty/price/fee/rebate, mark at the fill and at
         +5/+30/+60 s (NaN after the close: stats.attribute_pnl then uses the settlement), the
         book at the fill second and the exploit-detector flags of T5.
portfolio  per UTC minute: max |sum over live markets of inventory x dp/dlnS| (USD of
         underlying), the combined portfolio delta.

Aggregates (JSON): stats.attribute_pnl totals and markouts (identity error, and the max
difference between attribution and the trace's cash accounting), stats.flag_pnl_share with
the T5 verdict, the T6 directional numbers, P&L by Up/Down winner, corr(inventory, next 60 s
Binance log return).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (LENGTH_NAMES, MARKOUT_HORIZONS, MID_CLOSE_STC, STALE_S, Dataset, S)

FLAG_NAMES = ["crossed", "stale_gt10s", "one_sided", "last_30s", "last_10s", "inside_gt2ticks",
              "extreme_price", "print_outside_book"]
FLAG_DOCS = {
    "crossed": "book crossed in the raw data (last uncrossed book forward-filled)",
    "stale_gt10s": f"book_age > {STALE_S:g} s",
    "one_sided": "book invalid or missing a side",
    "last_30s": "seconds_to_close <= 30",
    "last_10s": "seconds_to_close <= 10",
    "inside_gt2ticks": "our quote more than 2 ticks inside the replay touch",
    "extreme_price": "fill price <= 0.02 or >= 0.98",
    "print_outside_book": "the print that filled us is outside the snapshot's [bid, ask]",
}
MAX_FLAG_SHARE = 0.25
MAX_INVENTORY_SHARE = 0.50


def _book(ds: Dataset, rows: slice | np.ndarray, mirror: bool):
    c = ds.ci
    t = np.asarray(ds.ticks[rows], dtype=np.float64)
    bid, ask = t[:, c["bid_px_0"]], t[:, c["ask_px_0"]]
    if mirror:
        bid, ask = 1.0 - ask, 1.0 - bid
    valid = t[:, c["book_valid"]] > 0.5
    return dict(bid=bid, ask=ask, valid=valid, crossed=t[:, c["crossed_raw"]] > 0.5,
                age=t[:, c["book_age"]], stc=t[:, c["seconds_to_close"]],
                taker_vol=t[:, c["buy_vol"]] + t[:, c["sell_vol"]],
                ul=t[:, c["ul_logp_rel_open"]] * (-1.0 if mirror else 1.0))


def _tick(ds, rows, bk):
    j = ds.fcol("tick_fine")
    if j is not None:
        fine = np.asarray(ds.feat[rows, j]) > 0.5
    else:
        fine = ((bk["bid"] > 0) & (bk["bid"] < 0.04)) | ((bk["ask"] < 1) & (bk["ask"] > 0.96))
    return np.where(fine, 0.001, 0.01)


class RunAccumulator:
    def __init__(self, ds: Dataset, split: int, method: str, variant: dict, t_range):
        self.ds, self.split, self.method, self.variant = ds, split, method, variant
        self.mirror = bool(variant.get("mirror", False))
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
        bk = _book(ds, sl, self.mirror)

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

        # fill table
        if nf:
            mids = {h: np.where(k + h <= n - 1, mark[np.minimum(k + h, n - 1)], np.nan)
                    for h in MARKOUT_HORIZONS}
            fb = {kk: v[k] for kk, v in bk.items() if kk not in ("ul", "taker_vol")}
            tick = _tick(ds, np.arange(sl.start, sl.stop)[k], fb)
            bid_ok = fb["valid"] & (fb["bid"] > 1e-9)
            ask_ok = fb["valid"] & (fb["ask"] < 1 - 1e-9)
            ppx = np.asarray(f["print_px"], np.float64)
            inside = np.where(side > 0, bid_ok & (price > fb["bid"] + 2 * tick + 1e-9),
                              ask_ok & (price < fb["ask"] - 2 * tick - 1e-9))
            outside = np.isfinite(ppx) & bid_ok & ask_ok & ((ppx < fb["bid"] - 1e-9) | (ppx > fb["ask"] + 1e-9))
            flags = dict(crossed=fb["crossed"], stale_gt10s=fb["age"] > STALE_S,
                         one_sided=~(bid_ok & ask_ok), last_30s=fb["stc"] <= 30,
                         last_10s=fb["stc"] <= 10, inside_gt2ticks=inside,
                         extreme_price=(price <= 0.02 + 1e-12) | (price >= 0.98 - 1e-12),
                         print_outside_book=outside)
            self.fill_parts.append(pd.DataFrame(dict(
                market=m, t=t[k], seconds_to_close=fb["stc"], side=side, qty=qty, price=price,
                fee=np.asarray(f["fee"], float), rebate=np.asarray(f["rebate"], float),
                print_px=ppx, mid=mark[k], **{f"mid_{h}": v for h, v in mids.items()},
                book_bid=fb["bid"], book_ask=fb["ask"], tick=tick,
                print_px_known=np.isfinite(ppx), **flags)))

        pnl = float(mtm[-1])
        ref = max(n - 1 - MID_CLOSE_STC, 0)
        qmin = int((bc(anyq.astype(float)) > 0).sum())
        tv = float(bk["taker_vol"].sum())
        vol = float(qty.sum())
        L = int(rec["contract_len_s"])
        self.mk_rows.append(dict(
            market=m, slug=ds.slug(m), coin=ds.coin(m), length=LENGTH_NAMES.get(L, str(L)),
            contract_len_s=L, open_ts=int(rec["open_ts"]), end_ts=int(rec["end_ts"]),
            settlement=Sm, winner_up=int(rec["winner_up"]), pnl=pnl, pnl_ex_rr=float(mtm_ex[-1]),
            rewards=float(rew[-1]), rebates=float(reb[-1]), fees=float(fee[-1]), n_fills=nf,
            buy_fills=int((side > 0).sum()), sell_fills=int((side < 0).sum()), volume=vol,
            notional=float((qty * price).sum()), max_abs_inv=float(absinv.max()),
            final_inv=float(inv[-1]), final_cash=float(cash[-1]), mid_close=float(mark[ref]),
            uptime_two=float(two[1:].mean()), uptime_one=float(one[1:].mean()),
            uptime_any=float(anyq[1:].mean()), quoted_minutes=qmin,
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
                         + [f"mid_{h}" for h in MARKOUT_HORIZONS] + ["book_bid", "book_ask", "tick"]})
            cols.update({c: "bool" for c in ["print_px_known"] + FLAG_NAMES})
            fl = pd.DataFrame({c: pd.Series(dtype=d) for c, d in cols.items()})
        for frame in (mk, mins, fl):
            frame.insert(0, "variant", self.variant["id"])
            frame.insert(0, "method", self.method)
        # portfolio per minute
        pm = self.port.reshape(-1, 60) if self.port.size % 60 == 0 else \
            np.pad(self.port, (0, -self.port.size % 60)).reshape(-1, 60)
        port = pd.DataFrame(dict(method=self.method, variant=self.variant["id"],
                                 minute_ts=self.t0 + 60 * np.arange(pm.shape[0]),
                                 max_abs_delta_usd=np.abs(pm).max(1), mean_delta_usd=pm.mean(1)))
        agg = self.aggregates(mk, fl)
        return dict(markets=mk, minutes=mins, fills=fl, portfolio=port, agg=agg)

    def aggregates(self, mk: pd.DataFrame, fl: pd.DataFrame) -> dict:
        fills = {"market": fl["market"].to_numpy(np.int64), "side": fl["side"].to_numpy(np.int64),
                 "qty": fl["qty"].to_numpy(float), "price": fl["price"].to_numpy(float),
                 "mid": fl["mid"].to_numpy(float), "fee": fl["fee"].to_numpy(float),
                 "rebate": fl["rebate"].to_numpy(float)}
        for h in MARKOUT_HORIZONS:
            fills[f"mid_{h}"] = fl[f"mid_{h}"].to_numpy(float)
        if len(fills["side"]) == 0:
            fills["side"] = np.zeros(0, np.int64)
        markets = {"market": mk["market"].to_numpy(np.int64),
                   "settlement": mk["settlement"].to_numpy(float),
                   "reward": mk["rewards"].to_numpy(float),
                   "final_inventory": mk["final_inv"].to_numpy(float),
                   "mid_close": mk["mid_close"].to_numpy(float)}
        att = _attribute(fills, markets)
        # attribution vs the env's own cash accounting, per market
        pm = dict(zip(att["market_ids"].tolist(), att["per_market"]["total"].tolist()))
        diff = [abs(pm.get(int(a), 0.0) - b) for a, b in zip(mk["market"], mk["pnl"])]
        cash_err = float(max(diff)) if diff else 0.0
        tot = att["totals"]
        flags = S.flag_pnl_share(att["per_fill"]["total"],
                                 {k: fl[k].to_numpy(bool) for k in FLAG_NAMES}) if len(fl) else \
            {k: dict(n=0, frac_fills=0.0, pnl=0.0, share_of_total=float("nan"), pnl_excluding=0.0)
             for k in FLAG_NAMES + ["_any"]}
        fill_tot = float(np.sum(att["per_fill"]["total"]))
        verdict = exploit_verdict(flags, fill_tot)
        verdict.update(
            n_fills_print_px_unknown=int((~fl["print_px_known"].astype(bool)).sum()) if len(fl) else 0)
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
                             identity_error=att["identity_error"], cash_check_max_abs=cash_err,
                             inventory_mismatch=[int(x) for x in att["inventory_mismatch"]]),
            exploit=dict(flags=flags, verdict=verdict),
            directional=dict(**directional_verdict(tot),
                             pnl_up_won=float(mk.loc[up, "pnl"].sum()),
                             pnl_down_won=float(mk.loc[~up, "pnl"].sum()),
                             n_up_won=int(up.sum()), n_down_won=int((~up).sum()),
                             corr_inventory_next60s_return=float(corr), corr_n=int(n)))


def exploit_verdict(flags: dict, fill_total: float) -> dict:
    """T5: invalid if any single flag carries > 25% of total fill P&L, or the fill P&L
    excluding all flagged fills is <= 0 while the total is > 0 (sign change)."""
    single = {k: flags[k]["share_of_total"] for k in FLAG_NAMES if k in flags}
    ok = {k: v for k, v in single.items() if v is not None and v == v}
    worst_k = max(ok, key=ok.get) if ok else None
    worst = ok[worst_k] if ok else float("nan")
    excl = flags["_any"]["pnl_excluding"]
    inv_share = bool(ok and worst > MAX_FLAG_SHARE)
    sign = bool(fill_total > 0 and excl <= 0)
    return dict(fill_pnl=float(fill_total), worst_single_flag=worst_k, worst_single_share=worst,
                pnl_excluding_flagged=float(excl), invalid_single_flag=inv_share,
                invalid_sign_change=sign, invalid=inv_share or sign)


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


def _attribute(fills, markets):
    return S.attribute_pnl(fills, markets, horizon=60)
