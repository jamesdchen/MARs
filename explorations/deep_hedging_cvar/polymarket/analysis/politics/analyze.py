"""Book-free maker-edge analysis on on-chain fills (politics universe or crypto up/down train days).

  python analyze.py politics      # onchain/data/politics + out/universe.json + data/prices_history.parquet
  python analyze.py crypto        # onchain/data/crypto_train (Aug 21-22, train only) + data/raw/markets.parquet

Every price is put in the frame of outcome 0 (Yes / Up): a leg on outcome 1 at price p is a leg on
outcome 0 at 1 - p with the opposite side. Per maker fill:
  edge_c   = side0 * (mid_ref(t) - price0)                    spread capture vs the pre-fill reference mid
  mo_h_c   = side0 * (mid(t + h) - price0)                    markout, h in 1m/5m/30m/2h, and to resolution
  as_h_c   = edge_c - mo_h_c                                  adverse selection at h
(cents per share). mid_ref(t) is the trade-implied mid: the latest maker BUY price (bid evidence) and latest maker
SELL price (ask evidence) strictly before t, both younger than QUOTE_AGE and not crossed; else the CLOB
prices-history point at or before t (politics only); else missing. mid(t+h) is the same estimator at t+h, or
the payout when t+h is past the market's end and the market has resolved.
Rebate per maker fill = rebateRate * rate * p(1-p) * size on days the market charged taker fees (on-chain fee>0).
Wallet P&L: every leg of the wallet (maker and taker) in the window, cash + inventory marked at the end of the
window (payout if resolved, else the last reference mid), minus taker fees paid, plus modelled rebates.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PM = HERE.parents[1]
OUT = HERE / "out"
SET = sys.argv[1] if len(sys.argv) > 1 else "politics"
QUOTE_AGE = 600.0                                     # s
H = {"1m": 60, "5m": 300, "30m": 1800, "2h": 7200}


def load_fills(src):
    fs = sorted((PM / "onchain/data" / src / "fills").glob("date=*/day.parquet"))
    f = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
    return f.drop_duplicates(["tx_hash", "log_index"])


def event_end(slug, _c={}):
    if not _c:
        p = OUT / "event_end.json"
        _c.update(json.load(open(p)) if p.exists() else {})
    if slug not in _c:
        import urllib.request
        req = urllib.request.Request(f"https://gamma-api.polymarket.com/events?slug={slug}", headers={"User-Agent": "research"})
        e = json.load(urllib.request.urlopen(req, timeout=30))
        _c[slug] = pd.Timestamp(e[0]["endDate"]).timestamp() if e and e[0].get("endDate") else 0.0
        json.dump(_c, open(OUT / "event_end.json", "w"))
    return _c[slug]


def setup():
    if SET == "politics":
        U = json.load(open(OUT / "universe.json"))
        res = json.load(open(OUT / "resolution.json"))
        rows = []
        for r in U:
            g = res.get(r["slug"], {})
            pay0 = np.nan
            if g.get("closed"):
                op = [float(x) for x in json.loads(g["outcomePrices"])]
                if max(op) > 0.99:
                    pay0 = op[0]
            end = pd.Timestamp(g.get("closedTime") or g.get("endDate") or r["end"])
            end = (end.tz_localize("UTC") if end.tzinfo is None else end).timestamp()
            if not g.get("closed"):          # market endDate can be stale (e.g. Brazil after round 1): use the event's
                end = max(end, event_end(r["event"]))
            fs = r.get("fee_schedule") or {}
            for i, t in enumerate(r["tokens"]):
                rows.append(dict(token_id=str(t), slug=r["slug"], outcome_idx=i, category=r["category"], end=end,
                                 pay0=pay0, fee_rate=fs.get("rate", 0.0) if r.get("fees_enabled") else 0.0,
                                 rebate_rate=fs.get("rebateRate", 0.0) if r.get("fees_enabled") else 0.0,
                                 resolved_flag=g.get("closed", False)))
        tm = pd.DataFrame(rows)
        ph = pd.read_parquet(HERE / "data/prices_history.parquet")
        # prices-history in the outcome-0 frame, per slug (prefer the outcome-0 token's own series)
        ph = ph.merge(tm[["token_id", "slug", "outcome_idx"]], on="token_id")
        ph["p0"] = np.where(ph.outcome_idx == 0, ph.p, 1 - ph.p)
        ph = ph[ph.outcome_idx == 0][["slug", "t", "p0"]].sort_values(["slug", "t"])
        fills = load_fills("politics")
    else:
        mk = pd.read_parquet(PM / "data/raw/markets.parquet")
        mk = mk[(mk.event_start >= pd.Timestamp("2026-08-21", tz="UTC")) & (mk.end < pd.Timestamp("2026-08-23", tz="UTC"))]
        rows = []
        for r in mk.itertuples():
            for i, t in enumerate((r.token_up, r.token_down)):
                rows.append(dict(token_id=str(t), slug=r.slug, outcome_idx=i, category=f"crypto-{r.contract_length}",
                                 end=r.end.timestamp(), pay0=float(r.winner_up) if pd.notna(r.winner_up) else np.nan,
                                 fee_rate=float(r.fee_rate or 0), rebate_rate=float(r.fee_rebate_rate or 0),
                                 resolved_flag=True, start=r.event_start.timestamp()))
        tm = pd.DataFrame(rows)
        ph = None
        fills = load_fills("crypto_train")
    fills["token_id"] = fills.token_id.astype(str)
    fills = fills.merge(tm, on="token_id")
    if SET == "crypto":   # train markets only, fills inside the market window
        fills = fills[(fills.block_time >= fills.start - 3600) & (fills.block_time <= fills.end + 600)]
    return fills, tm, ph


def quote_mid(times_b, px_b, times_a, px_a, t):
    """Trade-implied mid at each t (strictly before t). Returns mid, valid mask."""
    ib = np.searchsorted(times_b, t, side="left") - 1
    ia = np.searchsorted(times_a, t, side="left") - 1
    ok = (ib >= 0) & (ia >= 0)
    ibc, iac = np.clip(ib, 0, None), np.clip(ia, 0, None)
    b = np.where(ok, px_b[ibc] if len(px_b) else np.nan, np.nan)
    a = np.where(ok, px_a[iac] if len(px_a) else np.nan, np.nan)
    tb = np.where(ok, times_b[ibc] if len(times_b) else -1e18, -1e18)
    ta = np.where(ok, times_a[iac] if len(times_a) else -1e18, -1e18)
    ok &= (t - tb <= QUOTE_AGE) & (t - ta <= QUOTE_AGE) & (a > b)
    return np.where(ok, (a + b) / 2, np.nan), ok, np.where(ok, a - b, np.nan)


def ph_at(ph_t, ph_p, t):
    i = np.searchsorted(ph_t, t, side="right") - 1
    ok = (i >= 0) & (t - ph_t[np.clip(i, 0, None)] <= 1800) if len(ph_t) else np.zeros(len(t), bool)
    return np.where(ok, ph_p[np.clip(i, 0, None)] if len(ph_p) else np.nan, np.nan)


MATCH_LAG = 3


def l2_books():
    if SET == "crypto":
        fr = []
        for cl in ("5m", "15m", "4h"):
            b = pd.read_parquet(PM / f"data/raw/updown_{cl}.parquet", columns=["market_slug", "outcome", "captured_at", "best_bid", "best_ask"],
                                filters=[("outcome", "=", "Up")])
            fr.append(b)
    else:
        rec = HERE / "rec_out"
        if not rec.exists():
            return None
        sys.path.insert(0, str(PM / "recorder"))
        from polyrec.sinks import read_dataset
        fr = [read_dataset(rec, "books").to_pandas()[["market_slug", "outcome", "captured_at", "best_bid", "best_ask"]]]
        fr[0] = fr[0][fr[0].outcome == "Yes"]
    b = pd.concat(fr, ignore_index=True)
    b["t"] = (b.captured_at - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    b["mid"] = (b.best_bid + b.best_ask) / 2
    return {s: (g.t.values.astype(float), g.mid.values) for s, g in b.dropna(subset=["mid"]).sort_values("t").groupby("market_slug")}


_L2 = {}


def l2_mid(M, h=0):
    if "b" not in _L2:
        _L2["b"] = l2_books()
    B = _L2["b"]
    if not B:
        return None
    out = np.full(len(M), np.nan)
    for slug, idx in M.groupby("slug").indices.items():
        if slug not in B:
            continue
        t, m = B[slug]
        q = M.block_time.values[idx].astype(float) - MATCH_LAG + h
        i = np.searchsorted(t, q, side="right") - 1
        ok = (i >= 0) & (q - t[np.clip(i, 0, None)] <= 3600)
        out[idx] = np.where(ok, m[np.clip(i, 0, None)], np.nan)
    return out


def main():
    fills, tm, ph = setup()
    f = fills
    f["side_n"] = np.where(f.side == "BUY", 1, -1)
    f["side0"] = np.where(f.outcome_idx == 0, f.side_n, -f.side_n)
    f["p0"] = np.where(f.outcome_idx == 0, f.price, 1 - f.price)
    f["fee_usd"] = f.fee / 1e6
    # fee charged on taker legs this day -> rebates paid this day (per market)
    f["day"] = pd.to_datetime(f.block_time, unit="s").dt.date
    fee_on = f[f.role == "taker"].groupby(["slug", "day"]).fee_usd.sum() > 0
    f["fee_on"] = pd.MultiIndex.from_frame(f[["slug", "day"]]).map(fee_on).fillna(False).astype(bool).values
    mk_all = []
    for slug, g in f.groupby("slug", sort=False):
        g = g.sort_values(["block_time", "log_index"])
        mm = g[g.role == "maker"]
        bids, asks = mm[mm.side0 == 1], mm[mm.side0 == -1]
        tb, pb = bids.block_time.values.astype(float), bids.p0.values
        ta, pa = asks.block_time.values.astype(float), asks.p0.values
        if ph is not None:
            pg = ph[ph.slug == slug]; pht, php = pg.t.values.astype(float), pg.p0.values
        else:
            pht, php = np.array([]), np.array([])
        end, pay0 = g.end.iloc[0], g.pay0.iloc[0]

        def mid(t):
            m, ok, spr = quote_mid(tb, pb, ta, pa, t)
            m_ph = ph_at(pht, php, t)
            src = np.where(ok, 1, np.where(np.isfinite(m_ph), 2, 0))
            return np.where(ok, m, m_ph), src, spr, m_ph

        t = mm.block_time.values.astype(float)
        m0, src0, spr0, mph0 = mid(t)
        out = mm[["slug", "category", "block_time", "tx_hash", "maker", "side0", "p0", "size", "price", "fee_on",
                  "fee_rate", "rebate_rate", "end", "pay0", "exchange"]].copy()
        out["mid0"], out["mid_src"], out["qspread"], out["mid_ph"] = m0, src0, spr0, mph0
        for k, h in H.items():
            th = t + h
            mh, _, _, mphh = mid(th)
            past = th >= end
            if np.isfinite(pay0):
                mh = np.where(past, pay0, mh)
                mphh = np.where(past, pay0, mphh)
            out[f"mid_{k}"] = mh
            out[f"midph_{k}"] = mphh
        out["mid_res"] = pay0
        mk_all.append(out)
    M = pd.concat(mk_all, ignore_index=True)
    M["edge_c"] = 100 * M.side0 * (M.mid0 - M.p0)
    l2 = l2_mid(M)
    if l2 is not None:          # validation of the book-free proxy against the recorded L2 mid 3 s before the block
        M["mid_l2"] = l2
        M["edge_l2_c"] = 100 * M.side0 * (M.mid_l2 - M.p0)
        for k, h in H.items():
            M[f"mo_l2_{k}_c"] = 100 * M.side0 * (np.where(M.block_time + h >= M.end, M.pay0, l2_mid(M, h)) - M.p0)
    for k in list(H) + ["res"]:
        M[f"mo_{k}_c"] = 100 * M.side0 * (M[f"mid_{k}"] - M.p0)
    for k in H:
        M[f"moph_{k}_c"] = 100 * M.side0 * (M[f"midph_{k}"] - M.p0)
    M["edge_ph_c"] = 100 * M.side0 * (M.mid_ph - M.p0)
    M["rebate_c"] = 100 * np.where(M.fee_on, M.rebate_rate * M.fee_rate * M.price * (1 - M.price), 0.0)
    M["notional"] = M.price * M["size"]
    M["ttr_d"] = (M.end - M.block_time) / 86400
    M["plevel"] = pd.cut(M.mid0.fillna(M.p0), [0, .05, .2, .4, .6, .8, .95, 1.0], include_lowest=True).astype(str)
    M["ttr"] = pd.cut(M.ttr_d, [-1e9, 1 / 24, 1, 7, 30, 1e9], labels=["<1h", "1h-1d", "1-7d", "7-30d", ">30d"]).astype(str)
    M.to_parquet(HERE / f"data/maker_fills_{SET}.parquet")

    # ------------------------------------------------------------------ summaries (size-weighted, cents/share)
    cols = ["edge_c", "edge_ph_c"] + [c for c in M.columns if c.startswith(("edge_l2", "mo_l2"))] +  [f"mo_{k}_c" for k in list(H) + ["res"]] + [f"moph_{k}_c" for k in H] + ["rebate_c"]

    def summ(d):
        w = d["size"]
        r = {"maker_fills": len(d), "shares": w.sum(), "notional_usd": d.notional.sum(),
             "mid_from_trades": (d.mid_src == 1).mean(), "mid_from_ph": (d.mid_src == 2).mean()}
        for c in cols:
            ok = np.isfinite(d[c])
            r[c] = np.average(d[c][ok], weights=w[ok]) if ok.any() else np.nan
            if c in ("edge_c", "mo_5m_c", "mo_2h_c", "mo_res_c"):
                r[c + "_n"] = ok.mean()
        r["qspread_c"] = 100 * np.nanmedian(d.qspread) if d.qspread.notna().any() else np.nan
        return pd.Series(r)

    tabs = {"all": M.groupby(lambda _: "all").apply(summ),
            "category": M.groupby("category").apply(summ),
            "plevel": M.groupby("plevel").apply(summ),
            "ttr": M.groupby("ttr").apply(summ),
            "slug": M.groupby("slug").apply(summ).sort_values("shares", ascending=False)}
    for k, v in tabs.items():
        v.to_csv(OUT / f"{SET}_edge_by_{k}.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3g}".format):
        for k in ("all", "category", "plevel", "ttr"):
            print(f"== {SET} by {k}\n", tabs[k][["maker_fills", "shares", "notional_usd", "mid_from_trades", "edge_c", "edge_ph_c",
                                                   "mo_1m_c", "mo_5m_c", "mo_30m_c", "mo_2h_c", "mo_res_c", "moph_2h_c", "rebate_c"]
                                                  + [c for c in tabs[k].columns if c.startswith(("edge_l2", "mo_l2"))]])

    # ------------------------------------------------------------------ wallet P&L
    end_mark = {}
    for slug, g in M.groupby("slug"):
        pay0 = g.pay0.iloc[0]
        if np.isfinite(pay0):
            end_mark[slug] = pay0
        else:
            last = g.sort_values("block_time").mid0.dropna()
            end_mark[slug] = last.iloc[-1] if len(last) else g.p0.iloc[-1]
            if ph is not None:
                pg = ph[ph.slug == slug]
                if len(pg):
                    end_mark[slug] = pg.p0.iloc[-1]
    f["mark0"] = f.slug.map(end_mark)
    f["pnl_mark"] = f.side0 * f["size"] * (f.mark0 - f.p0)
    f["is_maker"] = f.role == "maker"
    f["maker_sz"] = np.where(f.is_maker, f["size"], 0.0)
    f["taker_sz"] = np.where(f.is_maker, 0.0, f["size"])
    f["notional"] = f.price * f["size"]
    W = f.groupby("maker").agg(legs=("size", "size"), maker_shares=("maker_sz", "sum"), taker_shares=("taker_sz", "sum"),
                               notional=("notional", "sum"), pnl_mark=("pnl_mark", "sum"), fees_paid=("fee_usd", "sum"),
                               n_markets=("slug", "nunique"))
    mw = M.assign(spread_usd=M.edge_c * M["size"] / 100, rebate_usd=M.rebate_c * M["size"] / 100,
                  mo5_usd=M.mo_5m_c * M["size"] / 100, mo2h_usd=M.mo_2h_c * M["size"] / 100)
    W = W.join(mw.groupby("maker")[["spread_usd", "rebate_usd", "mo5_usd", "mo2h_usd"]].sum())
    tk = f[~f.is_maker].copy()
    W["maker_pnl_mark"] = f[f.is_maker].groupby("maker").pnl_mark.sum()
    W["taker_pnl_mark"] = tk.groupby("maker").pnl_mark.sum()
    W = W.fillna(0.0)
    W["total_pnl"] = W.pnl_mark - W.fees_paid + W.rebate_usd
    W["maker_share"] = W.maker_shares / (W.maker_shares + W.taker_shares)
    W["direction_usd"] = W.maker_pnl_mark - W.spread_usd       # maker legs: mark-to-end minus spread at fill
    W = W.sort_values("maker_shares", ascending=False)
    W.to_csv(OUT / f"{SET}_wallets.csv")
    top = W.head(25)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        print(f"== {SET}: top 25 makers by maker shares")
        print(top[["n_markets", "maker_shares", "taker_shares", "maker_share", "spread_usd", "rebate_usd", "mo5_usd",
                   "mo2h_usd", "maker_pnl_mark", "taker_pnl_mark", "fees_paid", "direction_usd", "total_pnl"]])
        print("top25 sums\n", top[["maker_shares", "spread_usd", "rebate_usd", "mo5_usd", "mo2h_usd", "maker_pnl_mark",
                                   "taker_pnl_mark", "fees_paid", "direction_usd", "total_pnl"]].sum())
    summary = dict(set=SET, n_fills=len(f), n_maker_fills=len(M), markets=int(M.slug.nunique()),
                   days=sorted(map(str, f.day.unique())),
                   taker_fees_usd=float(f[f.role == "taker"].fee_usd.sum()),
                   maker_shares=float(M["size"].sum()), maker_notional=float(M.notional.sum()),
                   top25=top[["maker_shares", "spread_usd", "rebate_usd", "mo5_usd", "mo2h_usd", "maker_pnl_mark",
                              "taker_pnl_mark", "fees_paid", "direction_usd", "total_pnl"]].sum().to_dict(),
                   n_wallets_maker=int((W.maker_shares > 0).sum()))
    json.dump(summary, open(OUT / f"{SET}_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
