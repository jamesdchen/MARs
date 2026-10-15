#!/usr/bin/env python3
"""Event-bootstrap CIs for sub-cells of the recommended categories (pre-game sports by time to kickoff and by
league x kickoff; elections and geopolitics by price band), ex rewards, per $ of maker notional (bps).
Uses the period from period.py (CENSUS_TAGS ...). Writes <OUT>/boot_cells.csv."""
import duckdb
import numpy as np
import pandas as pd

from period import HERE, LEGS, OUT

KEY = """CASE WHEN category = 'sports_game' THEN (CASE WHEN inplay THEN 'sports_inplay' ELSE 'sports_pregame' END)
              ELSE category END"""
KO = """CASE WHEN to_game_h < 1 THEN 'a<1h' WHEN to_game_h < 6 THEN 'b1-6h'
             WHEN to_game_h < 24 THEN 'c6-24h' ELSE 'd>24h' END"""
PB = """CASE WHEN q < .15 THEN 'p<.15' WHEN q < .85 THEN 'p.15-.85' ELSE 'p>.85' END"""
H = ["30m", "2h", "1d"]


def cell_sums(con, where, cell):
    f = lambda h: f"mid_{h} IS NOT NULL AND mid_pre IS NOT NULL"
    cols = ",\n".join(
        f"sum(s*(mid_{h}-q)*size + rebate_ps*size) FILTER (WHERE {f(h)}) AS num_{h}, "
        f"sum(q*size) FILTER (WHERE {f(h)}) AS den_{h}" for h in H)
    return con.execute(f"""
      SELECT {cell} AS cell, cond, sum(q*size) usd,
        sum(s*(final-q)*size + rebate_ps*size) FILTER (WHERE final IS NOT NULL) AS num_res,
        sum(q*size) FILTER (WHERE final IS NOT NULL) AS den_res, {cols}
      FROM legs WHERE {where} GROUP BY ALL""").df()


def boot(df, nb=1000, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for c, g0 in df.groupby("cell"):
        g = g0.groupby("event").sum(numeric_only=True)
        w = rng.poisson(1.0, (nb, len(g)))
        r = {"cell": c, "n_events": len(g), "n_markets": g0.cond.nunique(), "usd": g.usd.sum()}
        for h in H + ["res"]:
            num, den = g[f"num_{h}"].fillna(0).to_numpy(), g[f"den_{h}"].fillna(0).to_numpy()
            e = 1e4 * (w @ num) / np.maximum(w @ den, 1e-9)
            r[f"xr_{h}"] = 1e4 * num.sum() / max(den.sum(), 1e-9)
            r[f"xr_{h}_lo"], r[f"xr_{h}_hi"] = np.nanpercentile(e, [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='10GB'")
    con.execute(f"CREATE VIEW legs AS SELECT *, {KEY} AS ckey FROM {LEGS}")
    ev = pd.read_parquet(HERE / "cache" / "markets.parquet", columns=["condition_id", "ev_slug"]).drop_duplicates(
        "condition_id").rename(columns={"condition_id": "cond", "ev_slug": "event"})
    parts = [("pregame by kickoff", "ckey = 'sports_pregame'", f"'pregame ' || {KO}"),
             ("pregame by league", "ckey = 'sports_pregame'", "'pregame ' || subcat"),
             ("pregame league x kickoff", "ckey = 'sports_pregame'", f"'pregame ' || subcat || ' ' || {KO}"),
             ("elections by price", "ckey = 'elections'", f"'elections ' || {PB}"),
             ("geopolitics by price", "ckey = 'geopolitics'", f"'geopolitics ' || {PB}"),
             ("in-play by league", "ckey = 'sports_inplay'", "'inplay ' || subcat")]
    out = []
    for name, where, cell in parts:
        d = cell_sums(con, where, cell).merge(ev, on="cond", how="left")
        d["event"] = d.event.fillna(d.cond)
        b = boot(d)
        b.insert(0, "cut", name)
        out.append(b)
        print(name, "done", flush=True)
    res = pd.concat(out)
    res.to_csv(OUT / "boot_cells.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(res[res.usd > 1e5].round(0).to_string())


if __name__ == "__main__":
    main()
