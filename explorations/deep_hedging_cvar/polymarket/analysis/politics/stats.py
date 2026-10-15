"""Cluster-bootstrap CIs (clusters = market x UTC day) for size-weighted maker markouts, politics vs crypto.

Net per filled share at horizon h = markout_h + rebate (both cents/share). Also per $100 of notional.
  -> out/compare_ci.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
rows = []
for SET in ("politics", "crypto"):
    M = pd.read_parquet(HERE / f"data/maker_fills_{SET}.parquet")
    M["day"] = (M.block_time // 86400).astype(int)
    M["cl"] = M.slug + "|" + M.day.astype(str)
    groups = [("all", M)] + list(M.groupby("category"))
    for name, d in groups:
        for col in ["edge_c", "edge_l2_c", "mo_1m_c", "mo_5m_c", "mo_30m_c", "mo_2h_c", "mo_res_c", "rebate_c", "net_5m_c", "net_2h_c"]:
            if col.startswith("net"):
                v = d[f"mo_{col.split('_')[1]}_c"] + d.rebate_c
            elif col in d:
                v = d[col]
            else:
                continue
            ok = np.isfinite(v)
            if ok.sum() < 50:
                continue
            x = pd.DataFrame({"cl": d.cl[ok], "wv": (v * d["size"])[ok], "w": d["size"][ok], "wn": (v * d["size"] / d.notional)[ok] * 0 + (v * d["size"])[ok], "n": d.notional[ok]})
            g = x.groupby("cl")[["wv", "w", "n"]].sum()
            est = g.wv.sum() / g.w.sum()
            est_n = g.wv.sum() / g.n.sum() * 100 / 100      # cents per $1 notional == % of notional
            idx = rng.integers(0, len(g), (1000, len(g)))
            wv, w, n = g.wv.values[idx].sum(1), g.w.values[idx].sum(1), g.n.values[idx].sum(1)
            bs, bsn = wv / w, wv / n
            rows.append(dict(set=SET, group=name, metric=col, clusters=len(g), cents_per_share=est,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             bps_of_notional=100 * g.wv.sum() / g.n.sum(),
                             bps_lo=100 * np.percentile(bsn, 2.5), bps_hi=100 * np.percentile(bsn, 97.5),
                             coverage=ok.mean()))
R = pd.DataFrame(rows)
R.to_csv(HERE / "out/compare_ci.csv", index=False)
with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3g}".format):
    print(R[R.group == "all"])
    print(R[(R.group != "all") & R.metric.isin(["edge_l2_c", "mo_5m_c", "mo_2h_c", "net_5m_c", "net_2h_c", "mo_res_c"])])
