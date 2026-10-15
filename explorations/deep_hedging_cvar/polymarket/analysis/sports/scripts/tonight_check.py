"""Out-of-sample recheck on tonight's recorded games, with true book mids instead of the touch proxy.

Maker legs come from data-api /trades (takerOnly=false rows that are not taker rows, see proxy_check.py) for the
markets polyrec recorded. Markouts use the recorded outcome-0 book mid at (block time - 3 s) + h, h = 10 s, 1 min,
5 min, and the 0/1 payoff once gamma reports the market resolved. The pre/live split uses gameStartTime. Wallet x
market pairs are classed two-/one-sided as in maker_econ.py. Cents/share, size-weighted; market-clustered SE.
Usage: tonight_check.py <rec_out_dir> slug [slug ...]
"""
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

HERE = Path(__file__).resolve().parent.parent
D = "https://data-api.polymarket.com/trades"
HS = {"10s": 10, "1m": 60, "5m": 300}


def trades(cond, taker_only):
    out, off = [], 0
    while off <= 20000:
        r = requests.get(D, params={"market": cond, "limit": 500, "offset": off, "takerOnly": str(taker_only).lower()}, timeout=30)
        j = r.json() if r.ok else []
        out += j
        if len(j) < 500:
            break
        off += 500
    return pd.DataFrame(out)


def main():
    rec = Path(sys.argv[1]); slugs = sys.argv[2:]
    fs = glob.glob(str(rec / "books" / "*" / "*.parquet")) + glob.glob(str(rec / "books" / "*" / "_parts" / "*.parquet"))
    B = pd.concat([pd.read_parquet(f, columns=["market_slug", "token_id", "captured_at", "best_bid", "best_ask"]) for f in fs])
    legs = []
    for slug in slugs:
        g = requests.get("https://gamma-api.polymarket.com/markets", params={"slug": slug}, timeout=30).json()
        if not g:
            g = requests.get("https://gamma-api.polymarket.com/markets", params={"slug": slug, "closed": "true"}, timeout=30).json()
        g = g[0]
        toks = json.loads(g["clobTokenIds"]); gs = pd.Timestamp(g.get("gameStartTime") or g["endDate"]).timestamp()
        op = [float(x) for x in json.loads(g.get("outcomePrices") or "[]")]
        y = (1.0 if op[0] > 0.5 else 0.0) if g.get("closed") and op and max(op) > 0.99 else np.nan
        A = trades(g["conditionId"], False); T = trades(g["conditionId"], True)
        if A.empty or T.empty:
            continue
        key = lambda d: d.transactionHash + d.proxyWallet + d.side + d["size"].astype(str)
        A = A[~key(A).isin(set(key(T)))].copy()          # maker legs
        A["oi"] = A.outcomeIndex.astype(int)
        A["p_yes"] = np.where(A.oi == 0, A.price, 1 - A.price)
        A["dir"] = np.where((A.side == "BUY") == (A.oi == 0), 1, -1)
        A["t"] = A.timestamp.astype(int) - 3
        b = B[(B.market_slug == slug) & (B.token_id == toks[0])].sort_values("captured_at")
        if b.empty:
            continue
        bt = (b.captured_at.astype("int64") // 10**6).to_numpy(); bm = ((b.best_bid + b.best_ask) / 2).to_numpy()
        A = A[(A.t >= bt.min() + 5) & (A.t <= bt.max())]
        def mid(q):
            i = np.searchsorted(bt, q, "right") - 1
            return np.where((i >= 0) & (q <= bt.max()), bm[np.clip(i, 0, None)], np.nan)
        A["sc"] = A["dir"] * (mid(A.t.to_numpy()) - A.p_yes)
        for k, h in HS.items():
            A["mo_" + k] = A["dir"] * (mid(A.t.to_numpy() + h) - A.p_yes)
        A["mo_res"] = A["dir"] * (y - A.p_yes)
        A["phase"] = np.where(A.t < gs, "pre", "live"); A["slug"] = slug
        legs.append(A[["slug", "proxyWallet", "phase", "dir", "size", "p_yes", "sc", "mo_10s", "mo_1m", "mo_5m", "mo_res"]])
        time.sleep(0.3)
    L = pd.concat(legs, ignore_index=True)
    L["sd"] = L["dir"] * L["size"]
    wm = L.groupby(["proxyWallet", "slug"]).agg(n=("size", "size"), gross=("size", "sum"), net=("sd", "sum"))
    wm["kind"] = np.where((wm.net.abs() / wm.gross < 0.3) & (wm.n >= 20), "two-sided", np.where(wm.net.abs() / wm.gross > 0.7, "one-sided", "mixed"))
    L = L.join(wm["kind"], on=["proxyWallet", "slug"])
    C = ["sc", "mo_10s", "mo_1m", "mo_5m", "mo_res"]
    rows = []
    for key_, d in list(L.groupby("phase")) + list(L.groupby(["kind", "phase"])):
        per = d.groupby("slug").apply(lambda x: pd.Series({**{c: np.nansum(x[c] * x["size"]) for c in C},
                                                           **{c + "_w": x["size"][x[c].notna()].sum() for c in C}}), include_groups=False)
        r = {"group": key_ if isinstance(key_, str) else "/".join(key_), "markets": len(per), "maker_shares_k": d["size"].sum() / 1e3}
        for c in C:
            w = per[c + "_w"].sum()
            if w <= 0:
                r[c] = np.nan; r[c + "_se"] = np.nan; continue
            mu = per[c].sum() / w; n = len(per)
            r[c] = 100 * mu; r[c + "_se"] = 100 * np.sqrt(n / max(n - 1, 1) * (((per[c] - mu * per[c + "_w"]) / w) ** 2).sum())
        rows.append(r)
    R = pd.DataFrame(rows).set_index("group").round(2)
    txt = [__doc__, f"markets: {L.slug.nunique()}, maker legs: {len(L)}", R.T.to_string()]
    (HERE / "out" / "tonight_check.txt").write_text("\n".join(txt) + "\n"); print("\n".join(txt))


if __name__ == "__main__":
    main()
