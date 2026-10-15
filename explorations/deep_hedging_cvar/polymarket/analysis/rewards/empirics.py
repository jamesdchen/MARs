#!/usr/bin/env python3
"""Train-only empirics: taker fees / maker rebates, spread capture, and the
liquidity-reward share a hypothetical maker would earn, per contract length.

TRAIN = markets whose whole window [event_start, end] is inside
2026-08-21..22 UTC (see common.py).  No Aug 23/24 market is ever opened.

Outputs (out/):
  markets.csv            one row per train market: fees, rebate pool, maker spread capture,
                         assumed reward pool
  fills_window.csv.gz    maker legs inside the L2 coverage with mid / half-spread at fill
  reward_samples.csv.gz  one row per (sampled minute, reward config): book Q_one/Q_two,
                         adjusted mid, distances of our hypothetical level-k quotes
  reward_marginal.csv    per (length, coin group, config, X, k, competitor bound, pool basis):
                         our share and $/hour
  summary_by_length.csv  the per-share comparison table used in REPORT.md
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from common import OUT, load_l2, load_market_cache, slug_window, train_slugs

FEE_RATE = 0.07          # feeSchedule.rate (crypto_fees_v2, exponent 1)
REBATE_BASE = 0.20       # feeSchedule.rebateRate; verified 0.2000 via /rebates/current
REBATE_BONUS = 0.35      # rate observed for a subset of large makers, ~Aug 12 - Sep 10
C_SCALE = 3.0
# Reward configs. "aug": rewardsMinSize=50 / rewardsMaxSpread=1.5c, the value still
# readable on ~20% of the incentivised train markets (the rest read 0/0 after the
# program ended); "live": 50 / 4.5c, today's default on every crypto up/down market
# and on the never-incentivised ZEC markets.
CONFIGS = {"aug": (50.0, 1.5), "live": (50.0, 4.5)}
# Crypto TWAP reward caps (docs.polymarket.com/programs/liquidity-rewards), USD per month
POOL = {
    "5m": {"btc": 300e3, "sol": 50e3, "eth": 50e3, "hype": 50e3, "xrp": 50e3, "bnb": 25e3, "doge": 25e3},
    "15m": {"btc": 225e3, "sol": 25e3, "eth": 25e3, "hype": 25e3, "xrp": 25e3, "bnb": 12.5e3, "doge": 12.5e3},
    "4h": {"btc": 50e3, "sol": 10e3, "eth": 10e3, "hype": 10e3, "xrp": 10e3, "bnb": 5e3, "doge": 5e3},
}
MARKETS_PER_DAY = {"5m": 288, "15m": 96, "4h": 6}
# pool spread over 31 calendar days vs the 28 epochs that actually paid REWARD
# (Aug 7-12 + Aug 15 - Sep 5, see out/payouts_daily.csv)
POOL_DAYS = {"calendar": 31, "observed": 28}
X_GRID = [50.0, 200.0]
K_GRID = [1, 2, 3]       # our quote at the k-th 1c-grid price on each side of the adjusted mid
RNG = np.random.default_rng(0)


def fee(p, c):
    return c * FEE_RATE * p * (1 - p)


def S(dist_c, v):
    dist_c = np.asarray(dist_c, dtype=float)
    return np.where(dist_c < v, ((v - dist_c) / v) ** 2, 0.0)


def adj_price(prices, sizes, m):
    """Price at which cumulative depth first reaches m (size-cutoff-adjusted touch)."""
    cs = np.cumsum(sizes)
    k = np.searchsorted(cs, m)
    return prices[min(k, len(prices) - 1)]


def tick_for(p):
    return 0.001 if (p < 0.04 or p > 0.96) else 0.01


def our_levels(mid, best_bid, best_ask, k):
    """k-th non-crossing grid price on each side of mid (k=1 is the closest)."""
    t = tick_for(mid)
    b1 = np.floor(mid / t + 1e-9) * t
    if b1 >= best_ask:
        b1 -= t
    a1 = np.ceil(mid / t - 1e-9) * t
    if a1 <= best_bid or a1 <= b1 + 1e-12:
        a1 = max(a1, b1 + t)
        while a1 <= best_bid:
            a1 += t
    return b1 - (k - 1) * t, a1 + (k - 1) * t


def book_q(bp, bs, ap, as_, mid, min_size, v):
    qb = float((S(np.abs(mid - bp) * 100, v) * bs * (bs >= min_size)).sum())
    qa = float((S(np.abs(ap - mid) * 100, v) * as_ * (as_ >= min_size)).sum())
    return qb, qa


def maker_legs(d):
    tk = collections.Counter(tuple(r[:7]) for r in d["taker"])
    for r in d["all"]:
        k = tuple(r[:7])
        if tk[k] > 0:
            tk[k] -= 1
            continue
        yield r


def up_frame(outcome_idx, side_str, price):
    """(+1 bought Up / -1 sold Up, Up-frame price)."""
    s = 1 if side_str == "BUY" else -1
    return (s, price) if outcome_idx == 0 else (-s, 1.0 - price)


def main():
    OUT.mkdir(exist_ok=True)
    slugs = train_slugs()
    mrows, frows, qrows = [], [], []
    for cl, sl in slugs.items():
        l2 = load_l2(cl, columns=["market_slug", "outcome", "captured_at", "bid_prices", "bid_sizes",
                                  "ask_prices", "ask_sizes", "best_bid", "best_ask", "crossed"])
        # Up and Down books are exact mirrors in this dataset -> use the Up book only (no double count)
        l2 = l2[l2.outcome == "Up"].sort_values(["market_slug", "captured_at"])
        by = dict(tuple(l2.groupby("market_slug")))
        for slug in sl:
            d = load_market_cache(slug)
            g = d["gamma"]
            coin = slug.split("-")[0]
            _, start, end = slug_window(slug)
            t_start, t_end = start.timestamp(), end.timestamp()
            aot = pd.Timestamp(g["acceptingOrdersTimestamp"]).timestamp()
            life_min = (t_end - aot) / 60
            book = by.get(slug)
            # ---------------- fees / rebates -----------------
            tk = np.array([(r[1], r[5], r[6]) for r in d["taker"]], dtype=float).reshape(-1, 3)
            fees_all = fee(tk[:, 1], tk[:, 2]).sum()
            inwin = (tk[:, 0] >= t_start) & (tk[:, 0] < t_end + 1)
            fees_win = fee(tk[inwin, 1], tk[inwin, 2]).sum()
            sh_all = tk[:, 2].sum()
            # --------------- maker legs at fill: edge vs mid ---------------
            ts_book = book.captured_at.astype("int64").values / 1e6 if book is not None else np.array([])
            edge_sum = edge_sh = 0.0
            for r in maker_legs(d):
                if book is None or not (t_start - 60 <= r[1] < t_end + 1):
                    continue
                k = np.searchsorted(ts_book, r[1], side="left") - 1     # strictly before the fill second
                if k < 0:
                    continue
                b = book.iloc[k]
                if b.crossed or pd.isna(b.best_bid) or pd.isna(b.best_ask):
                    continue
                mid = 0.5 * (b.best_bid + b.best_ask)
                s, pu = up_frame(r[4], r[3], r[5])
                edge = s * (mid - pu)
                frows.append((slug, cl, coin, r[1], r[1] - t_end, r[2], s, pu, r[6], mid,
                              0.5 * (b.best_ask - b.best_bid), edge, fee(r[5], 1.0)))
                edge_sum += edge * r[6]
                edge_sh += r[6]
            # --------------- liquidity reward samples (1/min, random offset) ---------------
            pool_mkt = {k: POOL[cl].get(coin, 0.0) / (MARKETS_PER_DAY[cl] * dd) for k, dd in POOL_DAYS.items()}
            if book is not None:
                for m0 in np.arange(t_start, t_end, 60.0):
                    t = m0 + RNG.uniform(0, 60)
                    k = np.searchsorted(ts_book, t, side="right") - 1
                    if k < 0:
                        continue
                    b = book.iloc[k]
                    base = (slug, cl, coin, t - t_end)
                    if b.crossed or len(b.bid_prices) == 0 or len(b.ask_prices) == 0:
                        for cfg in CONFIGS:
                            qrows.append(base + (cfg, np.nan, np.nan, np.nan, np.nan, 0) + (np.nan,) * (2 * len(K_GRID)))
                        continue
                    bp, bs, ap, as_ = (np.asarray(b.bid_prices), np.asarray(b.bid_sizes),
                                       np.asarray(b.ask_prices), np.asarray(b.ask_sizes))
                    for cfg, (msz, v) in CONFIGS.items():
                        mid = 0.5 * (adj_price(bp, bs, msz) + adj_price(ap, as_, msz))
                        q1, q2 = book_q(bp, bs, ap, as_, mid, msz, v)
                        dists = []
                        for kk in K_GRID:
                            ob, oa = our_levels(mid, b.best_bid, b.best_ask, kk)
                            dists += [abs(mid - ob) * 100, abs(oa - mid) * 100]
                        qrows.append(base + (cfg, mid, 0.5 * (b.best_ask - b.best_bid), q1, q2, 1) + tuple(dists))
            mrows.append(dict(slug=slug, contract_length=cl, coin=coin, life_min=life_min,
                              taker_fees=fees_all, taker_fees_window=fees_win, taker_shares=sh_all,
                              rebate_pool_base=REBATE_BASE * fees_all, rebate_pool_bonus=REBATE_BONUS * fees_all,
                              maker_edge_usd_window=edge_sum, maker_shares_window=edge_sh,
                              reward_pool_calendar=pool_mkt["calendar"], reward_pool_observed=pool_mkt["observed"],
                              rewards_min_size_now=g.get("rewardsMinSize"), rewards_max_spread_now=g.get("rewardsMaxSpread")))
        print(cl, "done", flush=True)

    mk = pd.DataFrame(mrows)
    fl = pd.DataFrame(frows, columns=["slug", "contract_length", "coin", "ts", "t_rel_end", "maker", "upside",
                                      "p_up", "size", "mid", "half_spread", "edge", "fee_per_share"])
    dcols = [f"d{s}{k}" for k in K_GRID for s in ("b", "a")]
    q = pd.DataFrame(qrows, columns=["slug", "contract_length", "coin", "t_rel_end", "cfg", "adj_mid", "half_spread",
                                     "q_bid", "q_ask", "scorable"] + dcols)
    mk.to_csv(OUT / "markets.csv", index=False)
    fl.to_csv(OUT / "fills_window.csv.gz", index=False)
    q.to_csv(OUT / "reward_samples.csv.gz", index=False)

    # ---------------- marginal reward for a hypothetical two-sided maker ----------------
    q = q.merge(mk[["slug", "life_min", "reward_pool_calendar", "reward_pool_observed"]], on="slug")
    inside = (q.adj_mid >= 0.10) & (q.adj_mid <= 0.90)
    qmin_, qmax_ = np.minimum(q.q_bid, q.q_ask), np.maximum(q.q_bid, q.q_ask)
    # competitors' sum of Q_min is unobservable (no order ownership); bracket it
    q["comp_hi"] = np.where(inside, qmin_ + (qmax_ - qmin_) / C_SCALE, qmin_)   # makers as two-sided as possible
    q["comp_lo"] = np.where(inside, (q.q_bid + q.q_ask) / C_SCALE, 0.0)        # every maker one-sided
    rrows = []
    for cfg, (msz, v) in CONFIGS.items():
        qc = q[q.cfg == cfg]
        ins = (qc.adj_mid >= 0.10) & (qc.adj_mid <= 0.90)
        for X in X_GRID:
            for kk in K_GRID:
                o1 = X * S(qc[f"db{kk}"], v)
                o2 = X * S(qc[f"da{kk}"], v)
                qo = np.where(ins, np.maximum(np.minimum(o1, o2), np.maximum(o1, o2) / C_SCALE), np.minimum(o1, o2))
                for comp in ("comp_hi", "comp_lo"):
                    den = qc[comp].values + qo
                    share = np.where(qc.scorable == 1, np.divide(qo, den, out=np.zeros_like(qo), where=den > 0), 0.0)
                    for basis in ("calendar", "observed"):
                        usd_h = 60 * qc[f"reward_pool_{basis}"].values / qc.life_min.values * share
                        tmp = pd.DataFrame({"cl": qc.contract_length.values, "grp": np.where(
                            qc.coin == "btc", "btc", np.where(qc.coin == "zec", "zec", "alt")),
                            "usd_h": usd_h, "share": share, "S_mean": qo / X})
                        for (cl, grp), gg in list(tmp.groupby(["cl", "grp"])) + [((c, "all_incentivised"), x) for c, x in
                                                                               tmp[tmp.grp != "zec"].groupby("cl")]:
                            rrows.append(dict(contract_length=cl, coins=grp, cfg=cfg, X=X, k=kk, competitor=comp,
                                              pool_basis=basis, our_Qmin_per_share=gg.S_mean.mean(),
                                              share_mean=gg.share.mean(), usd_per_hour=gg.usd_h.mean(),
                                              usd_per_quoted_share_hour=gg.usd_h.mean() / (2 * X)))
    rm = pd.DataFrame(rrows)
    rm.to_csv(OUT / "reward_marginal.csv", index=False)

    # ---------------- per-length comparison ----------------
    rows = []
    for cl in ("5m", "15m", "4h"):
        m = mk[mk.contract_length == cl]
        mi = m[m.coin != "zec"]
        f = fl[fl.contract_length == cl]
        qq = q[(q.contract_length == cl) & (q.cfg == "aug")]
        w = f["size"]

        def r(X, k, comp="comp_hi", basis="calendar"):
            x = rm[(rm.contract_length == cl) & (rm.coins == "all_incentivised") & (rm.cfg == "aug") & (rm.X == X)
                   & (rm.k == k) & (rm.competitor == comp) & (rm.pool_basis == basis)]
            return x.iloc[0]
        rows.append(dict(
            contract_length=cl, n_markets=len(m), n_incentivised=len(mi),
            taker_fees_per_mkt_mean=m.taker_fees.mean(), taker_fees_per_mkt_median=m.taker_fees.median(),
            frac_fees_in_window=m.taker_fees_window.sum() / m.taker_fees.sum(),
            rebate_pool_per_mkt_mean=m.rebate_pool_base.mean(),
            taker_fee_per_share=m.taker_fees.sum() / m.taker_shares.sum(),
            rebate_per_filled_share=REBATE_BASE * (f.fee_per_share * w).sum() / w.sum(),
            rebate_per_filled_share_bonus=REBATE_BONUS * (f.fee_per_share * w).sum() / w.sum(),
            quoted_half_spread_median=qq.half_spread.median(),
            half_spread_at_fill=(f.half_spread * w).sum() / w.sum(),
            maker_edge_per_filled_share=(f.edge * w).sum() / w.sum(),
            maker_edge_per_filled_share_median=f.edge.median(),
            adverse_selection_per_filled_share=np.nan,     # placeholder: separate agent
            maker_edge_usd_per_mkt=m.maker_edge_usd_window.mean(),
            reward_pool_per_mkt_calendar=mi.reward_pool_calendar.mean(),
            reward_pool_per_mkt_observed=mi.reward_pool_observed.mean(),
            frac_minutes_scorable=qq.scorable.mean(),
            book_qmin_median_aug=np.nanmedian(np.minimum(qq.q_bid, qq.q_ask)),
            share_X50_k1=r(50, 1).share_mean,
            reward_usd_h_X50_k1=r(50, 1).usd_per_hour, reward_usd_h_X50_k2=r(50, 2).usd_per_hour,
            reward_usd_h_X200_k1=r(200, 1).usd_per_hour,
            reward_usd_h_X50_k1_optimistic=r(50, 1, "comp_lo", "observed").usd_per_hour,
            reward_per_quoted_share_hour_X50_k1=r(50, 1).usd_per_quoted_share_hour,
        ))
    sm = pd.DataFrame(rows)
    sm.to_csv(OUT / "summary_by_length.csv", index=False)
    pd.set_option("display.width", 250)
    print(sm.T.to_string())


if __name__ == "__main__":
    main()
