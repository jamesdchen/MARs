"""Pre-game maker edge by time to kickoff (D14 buckets) on the keyless on-chain yardstick (maker_legs_big.parquet).
Edge per bucket in cents/share and in bps of maker notional (sum markout$ / sum price*size), size-weighted, with a
market-clustered 95% CI (cluster bootstrap over markets, 2000 draws). Ex-rewards; rebate shown separately.
Usage: pregame_buckets.py [maker_legs parquet]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
M = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else HERE / "out" / "maker_legs_big.parquet")
M = M[M.phase == "pre"].copy()
M["bucket"] = pd.cut(M.hrs_to_start, [0, 1, 6, 24, 1e5], right=False, labels=["last 1h", "1-6h", "6-24h", ">24h"])
M["notional"] = M.price * M["size"]
H = ["sc", "mo_1m", "mo_5m", "mo_30m", "mo_res"]
rng = np.random.default_rng(0)


def stats(d):
    per = d.groupby("slug").apply(lambda x: pd.Series({**{h: np.nansum(x[h] * x["size"]) for h in H},
                                                       "w": x["size"].sum(), "n": x.notional.sum(), "reb": (x.rebate_ps * x["size"]).sum()}), include_groups=False)
    r = {"markets": len(per), "maker_shares_M": per.w.sum() / 1e6, "notional_$M": per.n.sum() / 1e6}
    for h in H:
        r[h + "_c"] = 100 * per[h].sum() / per.w.sum()
        r[h + "_bps"] = 1e4 * per[h].sum() / per.n.sum()
    idx = rng.integers(0, len(per), (2000, len(per)))
    bs = 1e4 * per["mo_30m"].to_numpy()[idx].sum(1) / per.n.to_numpy()[idx].sum(1)
    r["mo_30m_bps_lo"], r["mo_30m_bps_hi"] = np.percentile(bs, [2.5, 97.5])
    r["rebate_bps"] = 1e4 * per.reb.sum() / per.n.sum()
    return pd.Series(r)


T = M.groupby("bucket", observed=True).apply(stats, include_groups=False)
M["win"] = np.where(M.hrs_to_start >= 1, ">1h", "last 1h")
L = M.groupby(["league", "win"]).apply(stats, include_groups=False)
L = L[L.markets >= 5]
cols = ["markets", "maker_shares_M", "sc_c", "mo_1m_c", "mo_5m_c", "mo_30m_c", "mo_30m_bps", "mo_30m_bps_lo", "mo_30m_bps_hi", "rebate_bps"]
txt = [__doc__, "## all pre-game legs by bucket", T.round(2).to_string(),
       "\n## by league: >1h vs last hour (leagues with >= 5 markets in the cell)", L[cols].round(1).to_string()]
out = HERE / "out" / ("pregame_buckets.txt" if len(sys.argv) < 2 else "pregame_buckets_" + Path(sys.argv[1]).stem + ".txt")
out.write_text("\n".join(txt) + "\n"); print("\n".join(txt))
