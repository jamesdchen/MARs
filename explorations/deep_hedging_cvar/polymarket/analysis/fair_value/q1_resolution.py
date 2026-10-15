"""Q1: resolution rule checks on TRAIN markets (2026-08-21/22 only).

Outputs cache/markets_train.parquet (per-market strike/final, Chainlink vs Binance) and prints stats.
"""
import numpy as np
import pandas as pd

import fair_value as fv

m = fv.train_markets()
u = fv.load_underlying()
rows = []
for r in m.itertuples():
    d = fv.load_market_meta(r.slug)
    g = d["gamma"]
    em = g.get("eventMetadata") or {}
    rows.append(dict(slug=r.slug, desc=g.get("description"), res_src=g.get("resolutionSource"),
                     ptb=float(em.get("priceToBeat", np.nan)), final=float(em.get("finalPrice", np.nan)),
                     fee=json_fee if (json_fee := g.get("feeSchedule")) is None else str(json_fee)))
meta = pd.DataFrame(rows)
m = m.merge(meta, on="slug")
nmiss = int(m.ptb.isna().sum() | 0)
print("markets missing priceToBeat/finalPrice in gamma eventMetadata:", int(m.ptb.isna().sum()), int(m.final.isna().sum()))
m = m[m.ptb.notna() & m.final.notna()].reset_index(drop=True)

# Binance proxies
c = m.coin.values
m["k_bin"] = [fv.twap_binance(u[cc], t) for cc, t in zip(c, m.t_open)]
m["f_bin"] = [fv.twap_binance(u[cc], t) for cc, t in zip(c, m.t_end)]
m["s_open"] = [fv.spot_at(u[cc], t) for cc, t in zip(c, m.t_open)]
m["s_end"] = [fv.spot_at(u[cc], t) for cc, t in zip(c, m.t_end)]
m["basis_open_bp"] = 1e4 * np.log(m.ptb / m.k_bin)
m["basis_end_bp"] = 1e4 * np.log(m.final / m.f_bin)
m["dbasis_bp"] = m.basis_end_bp - m.basis_open_bp
m["ret_cl_bp"] = 1e4 * np.log(m.final / m.ptb)
m["ret_bin_bp"] = 1e4 * np.log(m.f_bin / m.k_bin)
m["win_rule"] = (m.final >= m.ptb).astype(int)
m["win_bin"] = (m.f_bin >= m.k_bin).astype(int)
m.drop(columns=["desc"]).to_parquet(fv.CACHE / "markets_train.parquet", index=False)

print("TRAIN split only. markets:", len(m), m.groupby("cl").size().to_dict())
print("resolution sources:", sorted(m.res_src.unique()))
print("fee schedules:", m.fee.unique())
print("\nwinner == (final >= priceToBeat):", (m.win_rule == m.winner_up).mean(), " n mismatches",
      int((m.win_rule != m.winner_up).sum()))
print("exact ties final==ptb:", int((m.final == m.ptb).sum()))
# chain check: ptb of market starting at T == final of market ending at T (same coin)
fin = {(r.coin, r.t_end): r.final for r in m.itertuples()}
ch = [(r.ptb, fin[(r.coin, r.t_open)]) for r in m.itertuples() if (r.coin, r.t_open) in fin]
ch = np.array(ch)
print(f"chain pairs (ptb(start=T) vs final(end=T), same coin): n={len(ch)}, "
      f"max |rel diff|={np.max(np.abs(ch[:,0]/ch[:,1]-1)) if len(ch) else np.nan:.2e}")

# which Binance window best matches the Chainlink reference?  shift s: mean over [T-60+s, T+s]
print("\nAlignment of Chainlink reference vs Binance windows (std of log diff, bp), pooled over open+end refs:")
refs = [(cc, t, v) for cc, t, v in zip(c, m.t_open, m.ptb)] + [(cc, t, v) for cc, t, v in zip(c, m.t_end, m.final)]
best = []
for name, fn in [("spot@T", lambda a, T: fv.spot_at(a, T)),
                 *[(f"twap60 shift {s:+d}s", (lambda s: lambda a, T: fv.twap_binance(a, T + s))(s))
                   for s in (-30, -15, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 5, 15, 30)],
                 ("twap30", lambda a, T: fv.twap_binance(a, T, 30)), ("twap120", lambda a, T: fv.twap_binance(a, T, 120))]:
    dd = np.array([1e4 * np.log(v / fn(u[cc], t)) for cc, t, v in refs])
    # remove per-coin mean level basis
    dfc = pd.DataFrame({"coin": [x[0] for x in refs], "d": dd})
    dm = dfc.d - dfc.groupby("coin").d.transform("median")
    best.append((name, np.std(dm), np.median(np.abs(dm))))
    print(f"  {name:22s} std={np.std(dm):6.2f}bp  MAD={np.median(np.abs(dm)):5.2f}bp")

print("\nLevel basis ln(Chainlink/Binance TWAP60) at open, by coin (bp): median / IQR")
print(m.groupby("coin").basis_open_bp.describe()[["count", "50%", "25%", "75%"]].round(2))
print("\nBasis CHANGE over the window (end-open), bp; and |Chainlink return| for scale")
g = m.groupby("cl")
print(pd.DataFrame({"sd_dbasis_bp": g.dbasis_bp.std(), "mad_dbasis_bp": g.dbasis_bp.apply(lambda x: np.median(np.abs(x - x.median()))),
                    "sd_ret_bp": g.ret_cl_bp.std(), "corr_ret_bin_vs_cl": g.apply(lambda x: np.corrcoef(x.ret_bin_bp, x.ret_cl_bp)[0, 1]),
                    "outcome_agree_bin_vs_cl": g.apply(lambda x: (x.win_bin == x.winner_up).mean()),
                    "n": g.size()}).round(3))
print("disagreements:")
print(m.loc[m.win_bin != m.winner_up, ["slug", "ret_cl_bp", "ret_bin_bp", "dbasis_bp"]].round(2).to_string())
