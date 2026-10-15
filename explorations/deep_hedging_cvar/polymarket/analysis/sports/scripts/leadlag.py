"""Polymarket (polyrec books, 1 Hz) vs Smarkets (exchange quotes, ~3 s polls) lead/lag for tonight's games.

For each matched outcome: Polymarket mid of that outcome's token and Smarkets mid (best bid/offer, bps/1e4).
Both on a 3 s grid (last value), pre-game and live separately:
  level gap (Polymarket - Smarkets, cents): mean |gap|, share of time the Polymarket touch excludes Smarkets mid;
  cross-correlation of 3 s mid changes at lags -60..+60 s (positive lag = Polymarket leads);
  big moves: for every Polymarket 30 s move >= 4 cents in-play, the time until Smarkets has moved half as much in
  the same direction (negative = Smarkets moved first).
Note Smarkets polling (3 s) and their own in-play bet delay make this a coarse test (resolution ~3 s).
Usage: leadlag.py <rec_out_dir> <smarkets.jsonl>
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
PAIRS = {  # polymarket slug: (polymarket outcome, smarkets contract name)
    "nfl-sea-den-2026-10-16": ("Seahawks", "Seahawks"), "nhl-utah-edm-2026-10-15": ("Oilers", "Oilers"),
    "nhl-nyr-nj-2026-10-15": ("Devils", "Devils"), "nhl-nyi-ott-2026-10-15": ("Senators", "Senators"),
    "nhl-buf-tor-2026-10-15": ("Maple Leafs", "Maple Leafs"), "nba-okc-hou-2026-10-15": ("Thunder", "Thunder"),
    "nba-tor-nyk-2026-10-15": ("Knicks", "Knicks"), "col-bha-kau-2026-10-15-kau": ("Yes", "Kauno Žalgiris"),
    "uel-olm-oly-2026-10-15-oly": ("Yes", "Olympiacos"), "uel-cel1-juv-2026-10-15-juv": ("Yes", "Juventus"),
    "uel-hof-bes-2026-10-15-hof": ("Yes", "Hoffenheim"), "uel-din-and-2026-10-15-din": ("Yes", "Dinamo Zagreb"),
    "uel-ben-cel-2026-10-15-ben": ("Yes", "Benfica")}
G = 3


def smarkets(path):
    names, rows = {}, []
    for line in open(path):
        d = json.loads(line)
        if d.get("meta"):
            names.update(d["contracts"]); continue
        for cid, q in d["quotes"].items():
            b = q.get("bids") or []; o = q.get("offers") or []
            if b and o:
                rows.append((d["t"], names.get(cid), b[0]["price"] / 1e4, o[0]["price"] / 1e4))
    return pd.DataFrame(rows, columns=["t", "name", "bid", "ask"])


def main():
    rec, smk = Path(sys.argv[1]), sys.argv[2]
    up = pd.read_parquet(HERE / "out" / "upcoming.parquet").drop_duplicates("slug").set_index("slug")
    fs = glob.glob(str(rec / "books" / "*" / "*.parquet")) + glob.glob(str(rec / "books" / "*" / "_parts" / "*.parquet"))
    B = pd.concat([pd.read_parquet(f, columns=["market_slug", "outcome", "captured_at", "best_bid", "best_ask"]) for f in fs])
    B = B[B.market_slug.isin(PAIRS)]
    S = smarkets(smk)
    out, xc, bm = [], [], []
    for slug, (po, so) in PAIRS.items():
        gs = pd.Timestamp(up.loc[slug].gs).timestamp()
        b = B[(B.market_slug == slug) & (B.outcome == po)].copy()
        s = S[S.name == so].copy()
        if len(b) < 5 or len(s) < 20:
            continue
        b["t"] = b.captured_at.astype("int64") / 1e6
        lo, hi = max(b.t.min(), s.t.min()), min(b.t.max(), s.t.max())
        grid = np.arange(np.ceil(lo), hi, G)
        pb = b.sort_values("t"); ss = s.sort_values("t")
        ib = np.searchsorted(pb.t.values, grid, "right") - 1; isx = np.searchsorted(ss.t.values, grid, "right") - 1
        x = pd.DataFrame({"t": grid, "pbid": pb.best_bid.values[ib], "pask": pb.best_ask.values[ib],
                          "sbid": ss.bid.values[isx], "sask": ss.ask.values[isx]})
        x = x[(x.pask - x.pbid < 0.1) & (x.sask - x.sbid < 0.1)]
        x["pm"] = (x.pbid + x.pask) / 2; x["sm"] = (x.sbid + x.sask) / 2
        x["phase"] = np.where(x.t < gs, "pre", "live")
        for ph, y in x.groupby("phase"):
            if len(y) < 40:
                continue
            gap = (y.pm - y.sm) * 100
            r = dict(slug=slug, phase=ph, minutes=len(y) * G / 60, mean_gap_c=gap.mean(), mean_abs_gap_c=gap.abs().mean(),
                     pm_spread_c=((y.pask - y.pbid) * 100).median(), sm_spread_c=((y.sask - y.sbid) * 100).median(),
                     sm_outside_pm_touch=((y.sm < y.pbid) | (y.sm > y.pask)).mean())
            dp, ds = y.pm.diff().fillna(0).values, y.sm.diff().fillna(0).values
            for L in range(-20, 21):  # lag in grid steps
                if L >= 0:
                    c = np.corrcoef(dp[:len(dp) - L], ds[L:])[0, 1] if L < len(dp) - 5 else np.nan
                else:
                    c = np.corrcoef(dp[-L:], ds[:len(ds) + L])[0, 1]
                xc.append(dict(slug=slug, phase=ph, lag_s=L * G, corr=c, w=len(y)))
            out.append(r)
            if ph == "live":
                pm, sm, t = y.pm.values, y.sm.values, y.t.values
                last = -1e9
                for i in range(10, len(y)):
                    mv = pm[i] - pm[i - 10]
                    if abs(mv) >= 0.04 and t[i] - last > 120:
                        last = t[i]; i0 = i - 10; sgn = np.sign(mv)
                        # first time Polymarket moved half of it, and same for Smarkets (search -60 s .. +90 s)
                        def first(arr):
                            for k in range(max(0, i0 - 20), min(len(arr), i + 30)):
                                if sgn * (arr[k] - arr[i0 - 1 if i0 > 0 else 0]) >= abs(mv) / 2:
                                    return t[k]
                            return np.nan
                        tp, ts = first(pm), first(sm)
                        bm.append(dict(slug=slug, t=int(t[i]), move_c=mv * 100, smk_minus_pm_s=ts - tp))
    O = pd.DataFrame(out).round(3); X = pd.DataFrame(xc); BM = pd.DataFrame(bm)
    txt = [__doc__, "## level agreement", O.to_string(index=False)]
    if len(X):
        X = X.dropna()
        agg = X.groupby(["phase", "lag_s"]).apply(lambda d: np.average(d["corr"], weights=d.w), include_groups=False).unstack(0).round(3)
        txt += ["\n## cross-correlation of 3 s mid changes (weighted over markets); lag>0: Polymarket leads", agg.to_string()]
    if len(BM):
        txt += [f"\n## in-play big moves (>= 4c in 30 s on Polymarket): {len(BM)}", BM.round(1).to_string(index=False),
                f"median Smarkets-minus-Polymarket half-move time: {BM.smk_minus_pm_s.median():.1f} s"]
    (HERE / "out" / "leadlag.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
