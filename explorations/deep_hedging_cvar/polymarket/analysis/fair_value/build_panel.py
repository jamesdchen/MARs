"""Build the TRAIN-only snapshot panel (Up token, snapshots inside [open, end)).

Writes cache/panel.parquet (one row per Up snapshot) and cache/sigma_<coin>.npz (EWMA vol grids).
"""
import numpy as np
import pandas as pd

import fair_value as fv

HALFLIVES = [60, 300, 900, 3600]
SAMPLES = [1, 5, 30, 60]

mk = pd.read_parquet(fv.CACHE / "markets_train.parquet")[["slug", "ptb", "final"]]
m = fv.train_markets().merge(mk, on="slug", how="left")
u = fv.load_underlying()

b = fv.load_books()
b = b[(b.captured_at >= b.market_open) & (b.captured_at < b.market_end_at)].copy()
b = b.rename(columns={"market_slug": "slug", "contract_length": "cl"})
b["t"] = (b.captured_at.astype("int64") // 10**6).astype(np.int64)
bb, ba = b.best_bid, b.best_ask
b["mid"] = np.where(bb.notna() & ba.notna(), (bb + ba) / 2,
                    np.where(bb.notna(), (bb + 1) / 2, np.where(ba.notna(), ba / 2, np.nan)))
b["two_sided"] = bb.notna() & ba.notna()
b["spread"] = ba - bb
b = b.merge(m[["slug", "t_open", "t_end", "winner_up", "ptb", "final"]], on="slug")
b["tau"] = b.t_end - b.t
b["T"] = b.cl.map(fv.LEN_S)
b["frac"] = b.tau / b["T"]

# EWMA vol grids
sig = {}
for c, a in u.items():
    d = {f"h{h}_k{k}": fv.ewma_sigma(a, h, k).astype(np.float32) for h in HALFLIVES for k in SAMPLES}
    np.savez_compressed(fv.CACHE / f"sigma_{c}.npz", **d)
    sig[c] = d

out = []
for c, g in b.groupby("coin"):
    g = g.copy()
    a = u[c]
    for sh in (0, -4):
        K = np.array([fv.twap_binance(a, t + sh) for t in g.t_open])
        g[f"k_bin{sh}"] = K
    i = g.t.values - 1 - fv.T0
    g["S"] = a[i]
    for key, s in sig[c].items():
        g[f"sig_{key}"] = s[i]
    out.append(g)
p = pd.concat(out, ignore_index=True)
p = p.drop(columns=["outcome", "captured_at", "market_open"]).sort_values(["slug", "t"]).reset_index(drop=True)
p.to_parquet(fv.CACHE / "panel.parquet", index=False)
print(p.shape, p.groupby("cl").slug.nunique().to_dict(), "two-sided frac", p.two_sided.mean().round(3))
print(p.groupby("cl").apply(lambda x: x.groupby("slug").size().median()))
