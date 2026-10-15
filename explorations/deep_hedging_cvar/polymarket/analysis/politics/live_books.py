"""Live-book statistics from the polyrec recording (politics universe) and, for comparison, the crypto up/down L2
(data/raw, train days Aug 21-22 only).

  python live_books.py politics   -> out/books_politics.csv, out/rewards_live.csv, out/books_politics_summary.json
  python live_books.py crypto     -> out/books_crypto.csv

Per market (outcome-0 token book; the recorder writes one row per second in which the book changed):
  spread_c (time-weighted median and mean), share of time at one tick, touch depth (shares and $),
  depth within 1c / within rewardsMaxSpread of the mid, touch changes per hour (best bid or ask price moved),
  touch-size changes per hour, book updates per hour.
Rewards (politics): every minute at a random offset, the Q scores of the recorded book (competitors, bracketed
hi/lo as in ../rewards/reward_terms.py), and the share / $ per hour of a hypothetical maker that quotes
`rewardsMinSize` x {1, 5} shares on both sides at the touch (k=0) or one tick behind it (k=1). Pool = the current
daily rate from /rewards/markets/current.
"""
import glob, json, sys
from pathlib import Path
import numpy as np
from dataclasses import replace
import pandas as pd

HERE = Path(__file__).resolve().parent
PM = HERE.parents[1]
sys.path.insert(0, str(PM / "analysis/rewards"))
from reward_terms import RewardConfig, adjusted_mid, side_scores, q_min, competitor_q  # noqa: E402

SET = sys.argv[1] if len(sys.argv) > 1 else "politics"
OUT = HERE / "out"


def load_politics():
    sys.path.insert(0, str(PM / "recorder"))
    from polyrec.sinks import read_dataset
    b = read_dataset(HERE / "rec_out", "books").to_pandas()
    return b[b.outcome.isin(["Yes"])]


def load_crypto():
    fr = []
    for cl in ("5m", "15m", "4h"):
        b = pd.read_parquet(PM / f"data/raw/updown_{cl}.parquet",
                            columns=["market_slug", "outcome", "captured_at", "market_end_at", "bid_prices", "bid_sizes",
                                     "ask_prices", "ask_sizes", "best_bid", "best_ask"],
                            filters=[("outcome", "=", "Up"), ("captured_at", ">=", pd.Timestamp("2026-08-21", tz="UTC")),
                                     ("captured_at", "<", pd.Timestamp("2026-08-23", tz="UTC"))])
        b = b[b.market_end_at < pd.Timestamp("2026-08-23", tz="UTC")]
        b["contract_length"] = cl
        fr.append(b)
    return pd.concat(fr, ignore_index=True)


def depth_within(prices, sizes, ref, c, side):
    p, s = np.asarray(prices, float), np.asarray(sizes, float)
    m = (ref - p <= c + 1e-9) if side == "b" else (p - ref <= c + 1e-9)
    return float(s[m].sum()), float((s[m] * p[m]).sum())


def market_stats(g, maxspread_c=None):
    g = g.sort_values("captured_at").drop_duplicates("captured_at", keep="last")
    t = g.captured_at.astype("int64").values / 1e6
    ok = np.isfinite(g.best_bid.values) & np.isfinite(g.best_ask.values)
    g, t = g[ok], t[ok]
    if len(g) < 3:
        return None
    dur = np.diff(np.append(t, t[-1] + 1.0))                  # each row holds until the next change
    dur = np.minimum(dur, 3600)
    spr = 100 * (g.best_ask.values - g.best_bid.values)
    mid = (g.best_ask.values + g.best_bid.values) / 2
    o = np.argsort(spr)
    cw = np.cumsum(dur[o]) / dur.sum()
    r = dict(hours=dur.sum() / 3600, rows=len(g), mid_mean=float(np.average(mid, weights=dur)),
             spread_c_med=float(spr[o][np.searchsorted(cw, 0.5)]), spread_c_mean=float(np.average(spr, weights=dur)),
             one_tick_share=float(np.average(spr <= 1.0 + 1e-6, weights=dur)))
    bs = np.array([s[0] if len(s) else np.nan for s in g.bid_sizes.values])
    as_ = np.array([s[0] if len(s) else np.nan for s in g.ask_sizes.values])
    r["touch_shares"] = float(np.average(np.nan_to_num(bs + as_) / 2, weights=dur))
    r["touch_usd"] = float(np.average(np.nan_to_num(bs * g.best_bid.values + as_ * g.best_ask.values) / 2, weights=dur))
    # depth within 1c and within max spread of the mid, sampled at <= 2000 rows
    idx = np.linspace(0, len(g) - 1, min(len(g), 2000)).astype(int)
    for lab, c in (("1c", 0.01), ("maxspread", (maxspread_c or 4.5) / 100)):
        d = [depth_within(g.bid_prices.values[i], g.bid_sizes.values[i], mid[i], c, "b")[1]
             + depth_within(g.ask_prices.values[i], g.ask_sizes.values[i], mid[i], c, "a")[1] for i in idx]
        r[f"depth_usd_{lab}"] = float(np.median(d))
    bb, ba = g.best_bid.values, g.best_ask.values
    ch = (np.diff(bb) != 0) | (np.diff(ba) != 0)
    chs = (np.diff(bs) != 0) | (np.diff(as_) != 0)
    h = r["hours"]
    r["touch_px_changes_per_h"] = float(ch.sum() / h)
    r["touch_sz_changes_per_h"] = float(chs.sum() / h)
    r["updates_per_h"] = float(len(g) / h)
    r["mid_abs_move_c_per_h"] = float(np.abs(np.diff(mid)).sum() * 100 / h)
    return r


def rewards_live(g, pool_day, min_size, max_spread, rng, tick=0.01):
    cfg = RewardConfig(name="live", reward_enabled=True, reward_min_size=float(min_size), reward_max_spread_c=float(max_spread))
    g = g.sort_values("captured_at")
    t = g.captured_at.astype("int64").values / 1e6
    t0, t1 = t[0], t[-1]
    m0 = np.arange(np.ceil(t0 / 60) * 60, t1, 60)
    ts = m0 + rng.uniform(0, 60, len(m0))
    ts = ts[ts < t1]
    rows = []
    for s in ts:
        i = np.searchsorted(t, s, side="right") - 1
        if i < 0:
            continue
        x = g.iloc[i]
        bp, bsz, ap, asz = (np.asarray(x[c], float) for c in ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"))
        if not len(bp) or not len(ap):
            continue
        mid = adjusted_mid(bp, bsz, ap, asz, cfg.reward_min_size)
        if not np.isfinite(mid):
            continue
        c1, c2 = side_scores(bp, bsz, ap, asz, mid, cfg)
        r = dict(t=s, mid=mid, q1=c1, q2=c2)
        for bound in ("hi", "lo"):
            comp = competitor_q(c1, c2, mid, replace(cfg, competitor_bound=bound))
            for k in (0, 1):
                for mult in (1, 5):
                    X = mult * cfg.reward_min_size
                    ob, oa = bp[0] - k * tick, ap[0] + k * tick
                    o1, o2 = side_scores([ob], [X], [oa], [X], mid, cfg)
                    mine = q_min(o1, o2, mid, cfg)
                    r[f"share_{bound}_k{k}_x{mult}"] = mine / (mine + comp) if mine > 0 else 0.0
        rows.append(r)
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    out = dict(samples=len(d), pool_day=pool_day, min_size=min_size, max_spread=max_spread, mid=float(d.mid.mean()))
    for c in [c for c in d.columns if c.startswith("share_")]:
        out[c] = float(d[c].mean())
        out[c.replace("share_", "usd_h_")] = float(d[c].mean() * pool_day / 24)
    # $ per hour per $1k of capital for a two-sided quote (k=0, x1, hi bound)
    notional = min_size          # bid X at p costs pX, ask X (= buy X of outcome 1 at 1-p) costs (1-p)X: X dollars
    out["usd_h_per_1k_quoted_k0_x1_hi"] = out["usd_h_hi_k0_x1"] / max(notional, 1e-9) * 1000
    return out


def main():
    if SET == "politics":
        b = load_politics()
        rw = json.load(open(OUT / "rewards.json"))["current"]
        rwd = {x["slug"]: x for x in rw}
        U = {r["slug"]: r for r in json.load(open(OUT / "universe.json"))}
        rows, rrows = [], []
        rng = np.random.default_rng(0)
        for slug, g in b.groupby("market_slug"):
            x = rwd.get(slug, {})
            ms = x.get("rewards_max_spread") or U.get(slug, {}).get("rewards_max_spread")
            r = market_stats(g, ms)
            if r is None:
                continue
            r.update(slug=slug, category=U.get(slug, {}).get("category"), pool_day=x.get("total_daily_rate", 0.0))
            rows.append(r)
            if x:
                rr = rewards_live(g, float(x["total_daily_rate"]), float(x["rewards_min_size"]), float(x["rewards_max_spread"]), rng)
                if rr:
                    rr.update(slug=slug, category=r["category"])
                    rrows.append(rr)
        S = pd.DataFrame(rows).set_index("slug").sort_values("hours", ascending=False)
        S.to_csv(OUT / "books_politics.csv")
        R = pd.DataFrame(rrows).set_index("slug")
        R.to_csv(OUT / "rewards_live.csv")
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
            print(S.drop(columns=["rows"]).head(45))
            print(R[["samples", "pool_day", "min_size", "max_spread", "mid", "share_hi_k0_x1", "share_lo_k0_x1",
                     "usd_h_hi_k0_x1", "usd_h_hi_k1_x1", "usd_h_hi_k0_x5", "usd_h_per_1k_quoted_k0_x1_hi"]])
        summ = dict(hours_recorded=float(S.hours.max()), markets=len(S),
                    median=S.drop(columns=["category"]).median(numeric_only=True).to_dict(),
                    by_category=S.groupby("category").median(numeric_only=True).to_dict(orient="index"),
                    rewards_median=R.median(numeric_only=True).to_dict(),
                    rewards_sum_usd_h_hi_k0_x1=float(R.usd_h_hi_k0_x1.sum()))
        json.dump(summ, open(OUT / "books_politics_summary.json", "w"), indent=1, default=float)
    else:
        b = load_crypto()
        rows = []
        for slug, g in b.groupby("market_slug"):
            r = market_stats(g, 4.5)
            if r:
                r.update(slug=slug, contract_length=g.contract_length.iloc[0])
                rows.append(r)
        S = pd.DataFrame(rows).set_index("slug")
        S.to_csv(OUT / "books_crypto.csv")
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
            print(S.groupby("contract_length").median(numeric_only=True).T)


if __name__ == "__main__":
    main()
