"""Summarise the polyrec sports recording (books at 1 Hz on change) into compact tables.

Per market, on the outcome-0 token, with the book forward-filled to a 1 s grid:
  spread (cents), depth within 1/2/5 cents of mid (shares, both sides), mid / touch changes per minute;
  split into pre-game (t < gameStartTime) and live (t >= gameStartTime).
Game start: total resting depth (both sides, all levels) and spread at gs-60 s, gs-5 s, gs+5 s, gs+60 s,
gs+300 s, to see the clearBookOnStart wipe and how fast the book refills.
In-play jumps: seconds where |mid(t) - mid(t-30 s)| >= 5 cents; spread and depth in the 30 s before vs
the 60 s after each jump.
Usage: live_books.py <rec_out_dir>
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
REC = Path(sys.argv[1])
up = pd.read_parquet(HERE / "out" / "upcoming.parquet").drop_duplicates("slug").set_index("slug")


def load_books():
    fs = sorted(glob.glob(str(REC / "books" / "*" / "*.parquet")) + glob.glob(str(REC / "books" / "*" / "_parts" / "*.parquet")))
    b = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return b.drop_duplicates(["token_id", "captured_at"])


def depth(prices, sizes, mid, w):
    if prices is None or len(prices) == 0:
        return 0.0
    p = np.asarray(prices, float); s = np.asarray(sizes, float)
    return float(s[np.abs(p - mid) <= w + 1e-9].sum())


def per_market(g, gs):
    g = g.sort_values("captured_at")
    t = (g.captured_at.astype("int64") // 10**6).to_numpy()  # seconds
    bb, ba = g.best_bid.to_numpy(), g.best_ask.to_numpy()
    mid = (bb + ba) / 2
    d = {"t": t, "bb": bb, "ba": ba, "mid": mid, "spread": (ba - bb) * 100}
    for w in (0.01, 0.02, 0.05):
        d[f"dep{int(w*100)}"] = [depth(bp, bs, m, w) + depth(ap, as_, m, w) for bp, bs, ap, as_, m in
                                 zip(g.bid_prices, g.bid_sizes, g.ask_prices, g.ask_sizes, mid)]
    d["dep_all"] = [float(np.sum(bs if bs is not None else 0)) + float(np.sum(as_ if as_ is not None else 0))
                    for bs, as_ in zip(g.bid_sizes, g.ask_sizes)]
    x = pd.DataFrame(d).set_index("t")
    grid = np.arange(x.index.min(), x.index.max() + 1)
    x = x[~x.index.duplicated(keep="last")].reindex(grid).ffill()
    x["phase"] = np.where(x.index < gs, "pre", "live")
    return x


def summarise(x):
    out = {}
    for ph, y in x.groupby("phase"):
        if len(y) < 30:
            continue
        ok = y.spread.notna() & (y.spread > 0) & (y.spread < 50)
        y = y[ok]
        mins = len(y) / 60
        out[ph] = dict(minutes=round(mins, 1), spread_med=y.spread.median(), spread_mean=y.spread.mean(),
                       one_tick_share=(y.spread <= 1.0001).mean(), mid_med=y.mid.median(),
                       dep1_med=y.dep1.median(), dep2_med=y.dep2.median(), dep5_med=y.dep5.median(),
                       mid_chg_per_min=(y.mid.diff().abs() > 1e-9).sum() / mins,
                       touch_chg_per_min=((y.bb.diff().abs() > 1e-9) | (y.ba.diff().abs() > 1e-9)).sum() / mins)
    return out


def at_start(x, gs):
    r = {}
    for off in (-60, -5, 5, 60, 300, 900):
        tt = gs + off
        if x.index.min() <= tt <= x.index.max():
            row = x.loc[tt]
            r[f"dep_all@{off:+d}"] = row.dep_all; r[f"spread@{off:+d}"] = row.spread
    return r


def jumps(x, gs, thr=0.05):
    y = x[x.index >= gs]
    if len(y) < 120:
        return []
    m = y.mid
    dm = (m - m.shift(30)).abs()
    J, last = [], -1e9
    for t in dm.index[dm >= thr]:
        if t - last < 120:
            continue
        last = t
        pre = y.loc[t - 60:t - 31]; post = y.loc[t:t + 60]
        if len(pre) < 10 or len(post) < 10:
            continue
        J.append(dict(t=int(t), move=float(m.loc[t] - m.loc[t - 30]), spread_pre=pre.spread.median(), spread_post=post.spread.median(),
                      spread_post_max=post.spread.max(), dep2_pre=pre.dep2.median(), dep2_post=post.dep2.median()))
    return J


def main():
    b = load_books()
    rows, starts, J = [], [], []
    for slug, g in b.groupby("market_slug"):
        if slug not in up.index:
            continue
        gs = pd.Timestamp(up.loc[slug].gs).timestamp()
        tok0 = json.loads(json.dumps(None)) if False else None
        g0 = g[g.token_id == sorted(g.token_id.unique(), key=lambda tk: (g[g.token_id == tk].outcome.iloc[0] not in ("Yes",), tk))[0]]
        x = per_market(g0, gs)
        for ph, s in summarise(x).items():
            rows.append(dict(slug=slug, league=slug.split("-")[0], phase=ph, **s))
        starts.append(dict(slug=slug, **at_start(x, gs)))
        for j in jumps(x, gs):
            J.append(dict(slug=slug, **j))
    S = pd.DataFrame(rows).round(3); ST = pd.DataFrame(starts).round(1); JJ = pd.DataFrame(J).round(3)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    txt = [__doc__, f"recording: {REC}", f"rows: {len(b)}, markets: {b.market_slug.nunique()}",
           "\n## per market and phase (outcome-0 token; spread in cents; depth in shares within w of mid, both sides)",
           S.to_string(index=False),
           "\n## medians by phase", S.groupby("phase")[["spread_med", "one_tick_share", "dep1_med", "dep2_med", "dep5_med", "mid_chg_per_min", "touch_chg_per_min"]].median().round(3).to_string(),
           "\n## around game start (total resting shares both sides, spread)", ST.to_string(index=False),
           f"\n## in-play jumps (|dmid| >= 5c within 30 s): {len(JJ)}", JJ.to_string(index=False) if len(JJ) else "",
           ("\n## jump medians\n" + JJ[["spread_pre", "spread_post", "spread_post_max", "dep2_pre", "dep2_post"]].median().round(3).to_string()) if len(JJ) else ""]
    (HERE / "out" / "live_books.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
