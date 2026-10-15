"""Maker economics of the on-chain sports sample (onchain/data/sports).

Everything is put in the YES frame of each market (outcome 0 = YES):
  p_yes = p for outcome-0 tokens, 1 - p for outcome-1 tokens;
  dir   = +1 when the leg ends long YES (BUY outcome 0 or SELL outcome 1), else -1.
Mid proxy (no book on chain): every taker tx reveals one touch. A taker that buys YES lifts makers
selling YES, so the best maker price of the tx is the ask; a taker selling YES hits the bid.
mid(t) = (last ask + last bid) / 2 using touches strictly before t (both seen within 15 min, else the
last touch). Markout of a maker leg at horizon h: dir * (mid(t + h) - p_yes) per share.
"spread capture" uses mid just before the leg's block. Resolution markout uses the 0/1 payoff.
Rebate = rebateRate * rate * p(1 - p) per maker share (feeSchedule from gamma).
Phases: pre (t < game start), live (t >= game start).
Jump fills: a mid move >= JUMP within [t, t + 60 s] (scoring events and big in-play swings).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
SRC = HERE.parent.parent / "onchain" / "data" / "sports"
OUT = HERE / "out"
H = {"10s": 10, "1m": 60, "5m": 300, "30m": 1800}
JUMP = 0.08
EXCH = {"0xe2222d279d744050d28e00520010520000310f59", "0xc5d563a36ae78145c45a50134d48a1215220f80a",
        "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e", "0xe111180000d2663c0091e4f400237545b87b996b"}


def load():
    f = pd.read_parquet(SRC / "fills")
    mk = pd.concat([pd.read_parquet(p) for p in sorted((SRC / "markets").glob("*.parquet"))]).drop_duplicates("token_id")
    cen = pd.read_parquet(OUT / "census_markets.parquet").drop_duplicates("slug").set_index("slug")
    f = f.merge(mk[["token_id", "slug", "outcome_idx"]], on="token_id", how="inner")
    f = f[f["size"] > 0]
    f["t"] = f.block_time.astype(np.int64)
    f["p_yes"] = np.where(f.outcome_idx == 0, f.price, 1 - f.price)
    buy = f.side.eq("BUY")
    f["dir"] = np.where(buy == (f.outcome_idx == 0), 1, -1)
    meta = {}
    for s in f.slug.unique():
        c = cen.loc[s]
        fee = json.loads(c.fee) if c.fee else {}
        op = json.loads(c.outcome_prices) if c.outcome_prices else None
        y = None
        if op and len(op) == 2 and {round(float(x), 3) for x in op} <= {0.0, 1.0}:
            y = 1.0 if float(op[0]) > 0.5 else 0.0
        meta[s] = dict(league=c.league, smt=c.smt, gs=pd.Timestamp(c.game_start).timestamp(),
                       rate=float(fee.get("rate", 0)), rebate=float(fee.get("rebateRate", 0)), y=y,
                       volume=c.volume)
    return f, meta


def mid_series(g):
    """g: one market's legs sorted by (block, log_index). Returns (times, mids) of touch updates."""
    tk = g[g.role.eq("taker")][["tx_hash", "dir"]].drop_duplicates("tx_hash")
    mkr = g[g.role.eq("maker")]
    best = mkr.groupby("tx_hash").agg(t=("t", "first"), pmin=("p_yes", "min"), pmax=("p_yes", "max"))
    best = best.join(tk.set_index("tx_hash"), how="inner").sort_values("t")
    t = best.t.to_numpy(); ask = np.where(best["dir"] > 0, best.pmin, np.nan); bid = np.where(best["dir"] < 0, best.pmax, np.nan)
    la, lb, ta, tb, mids = np.nan, np.nan, -1e18, -1e18, []
    for i in range(len(t)):
        if not np.isnan(ask[i]): la, ta = ask[i], t[i]
        if not np.isnan(bid[i]): lb, tb = bid[i], t[i]
        if abs(ta - tb) <= 900 and not (np.isnan(la) or np.isnan(lb)) and lb <= la + 0.02:
            mids.append((la + lb) / 2)
        else:
            mids.append(la if ta > tb else lb)
    return t, np.array(mids)


def mid_at(t, m, q, strict=False):
    i = np.searchsorted(t, q, side="left" if strict else "right") - 1
    out = np.where(i >= 0, m[np.clip(i, 0, None)], np.nan)
    return out


def main():
    f, meta = load()
    f = f.sort_values(["block", "log_index"]).reset_index(drop=True)
    rows = []
    for s, g in f.groupby("slug", sort=False):
        md = meta[s]
        t, m = mid_series(g)
        if len(t) < 5:
            continue
        mk = g[g.role.eq("maker") & ~g.maker.isin(EXCH)].copy()
        q = mk.t.to_numpy()
        mk["mid0"] = mid_at(t, m, q, strict=True)
        for k, h in H.items():
            mk["mid_" + k] = mid_at(t, m, q + h)
        # max |mid move| within [t, t+60]
        jmp = np.zeros(len(q))
        for j, tq in enumerate(q):
            a, b = np.searchsorted(t, tq, "left"), np.searchsorted(t, tq + 60, "right")
            if b > a and not np.isnan(mk.mid0.iloc[j]):
                jmp[j] = np.nanmax(np.abs(m[a:b] - mk.mid0.iloc[j]))
        mk["jump"] = jmp
        mk["slug"] = s
        for k in ("league", "smt", "gs", "rate", "rebate", "y"):
            mk[k] = md[k]
        rows.append(mk)
    M = pd.concat(rows, ignore_index=True)
    A = f.copy(); A["y"] = A.slug.map({k: v["y"] for k, v in meta.items()})
    A[["maker", "slug", "role", "dir", "p_yes", "size", "fee", "y", "t"]].to_parquet(OUT / "all_legs.parquet")
    M["phase"] = np.where(M.t < M.gs, "pre", "live")
    M["sc"] = M["dir"] * (M.mid0 - M.p_yes)
    for k in H:
        M["mo_" + k] = M["dir"] * (M["mid_" + k] - M.p_yes)
    M["mo_res"] = M["dir"] * (M.y - M.p_yes)
    M["rebate_ps"] = M.rebate * M.rate * M.price * (1 - M.price)
    M["fee_eq_ps"] = M.rate * M.price * (1 - M.price)
    M["fav"] = np.maximum(M.p_yes, 1 - M.p_yes)
    M["pbin"] = pd.cut(M.fav, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001], right=False)
    M["is_jump"] = M.jump >= JUMP
    M["hrs_to_start"] = (M.gs - M.t) / 3600
    M.drop(columns="pbin").to_parquet(OUT / "maker_legs.parquet")
    return M


def wavg(d, cols, w="size"):
    ww = d[w]
    return pd.Series({c: np.nansum(d[c] * ww) / ww[d[c].notna()].sum() * 100 for c in cols})  # cents/share


COLS = ["sc", "mo_10s", "mo_1m", "mo_5m", "mo_30m", "mo_res", "rebate_ps"]


def table(M, by):
    g = M.groupby(by, observed=True)
    t = g.apply(lambda d: wavg(d, COLS), include_groups=False)
    t.insert(0, "maker_shares_k", g["size"].sum() / 1e3)
    t.insert(1, "legs", g.size())
    t["net_res+reb"] = t.mo_res + t.rebate_ps
    t["net_30m+reb"] = t.mo_30m + t.rebate_ps
    return t.round(2)


if __name__ == "__main__":
    M = main() if "--reuse" not in sys.argv else pd.read_parquet(OUT / "maker_legs.parquet")
    M["pbin"] = pd.cut(M.fav, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001], right=False)
    pd.set_option("display.width", 250)
    res = {}
    res["all"] = table(M.assign(all="all"), "all")
    res["phase"] = table(M, "phase")
    res["league_phase"] = table(M, ["league", "phase"])
    res["smt"] = table(M, ["smt", "phase"])
    res["pbin_phase"] = table(M, ["phase", "pbin"])
    M["hbin"] = pd.cut(M.hrs_to_start, [-10, -3, -1, 0, 0.5, 2, 6, 24, 1e4], right=False)
    res["time_to_start"] = table(M, "hbin")
    res["jump_live"] = table(M[M.phase == "live"], "is_jump")
    # market-clustered standard errors of the size-weighted means (cents/share), by phase
    se = []
    for ph, d in M.groupby("phase"):
        per = d.groupby("slug").apply(lambda x: pd.Series({c: np.nansum(x[c] * x["size"]) for c in COLS} | {"w": x["size"].sum()}), include_groups=False)
        n = len(per); tot = per.w.sum()
        r = {"phase": ph, "markets": n}
        for c in COLS:
            mu = per[c].sum() / tot
            r[c] = mu * 100
            r[c + "_se"] = np.sqrt(n / max(n - 1, 1) * (((per[c] - mu * per.w) / tot) ** 2).sum()) * 100
        se.append(r)
    res["clustered_se"] = pd.DataFrame(se).set_index("phase").round(2).T
    # wallets: maker-leg P&L to resolution and rebates, plus all legs (maker + taker) to resolution
    W = M[M.y.notna()].assign(pnl_res=lambda d: d.mo_res * d["size"], reb=lambda d: d.rebate_ps * d["size"],
                               sc_usd=lambda d: d.sc * d["size"], m30=lambda d: d.mo_30m * d["size"])
    w = W.groupby("maker").agg(maker_shares=("size", "sum"), markets=("slug", "nunique"), sc_usd=("sc_usd", "sum"),
                               mo30_usd=("m30", "sum"), pnl_res_usd=("pnl_res", "sum"), rebate_usd=("reb", "sum"))
    F = pd.read_parquet(OUT / "all_legs.parquet")
    F = F[F.y.notna()]
    F["cash"] = -F["dir"] * F.p_yes * F["size"]; F["pos"] = F["dir"] * F["size"]
    F["fee_usd"] = np.where(F.role.eq("taker"), F.fee / 1e6, 0)
    a = F.groupby(["maker", "slug"]).agg(cash=("cash", "sum"), pos=("pos", "sum"), y=("y", "first"), fee=("fee_usd", "sum"),
                                         taker_sh=("size", lambda x: x[F.loc[x.index, "role"].eq("taker")].sum()))
    a["pnl"] = a.cash + a.pos * a.y - a.fee
    aw = a.groupby("maker").agg(all_legs_pnl_usd=("pnl", "sum"), taker_shares=("taker_sh", "sum"), fees_paid=("fee", "sum"))
    w = w.join(aw, how="left")
    w["net_maker_usd"] = w.pnl_res_usd + w.rebate_usd
    w["net_c_per_sh"] = 100 * w.net_maker_usd / w.maker_shares
    w = w.sort_values("maker_shares", ascending=False)
    top = w.head(25).copy(); top.index = [i[:10] for i in top.index]
    res["top25_maker_wallets"] = top.round(1)
    tot = w.maker_shares.sum()
    res["maker_concentration"] = pd.Series({"wallets": len(w), "top5_share": w.maker_shares.head(5).sum() / tot,
        "top25_share": w.maker_shares.head(25).sum() / tot,
        "top25_net_usd": w.net_maker_usd.head(25).sum(), "top25_all_legs_usd": w.all_legs_pnl_usd.head(25).sum(),
        "all_net_usd": w.net_maker_usd.sum()}).round(3).to_frame("value")
    with open(OUT / "maker_econ.txt", "w") as fh:
        fh.write(__doc__ + "\nUnits: cents per maker share, size-weighted. net = markout + rebate.\n")
        for k, v in res.items():
            fh.write(f"\n## {k}\n{v.to_string()}\n")
    print(open(OUT / "maker_econ.txt").read())
