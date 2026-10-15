"""Rewards config, prices-history (10-min) and resolution for the universe.

  out/rewards.json          /rewards/markets/current rows for universe condition ids (+ per-market endpoint)
  out/rewards_all_summary.json  size of the whole current rewards programme (count, sum of daily rates)
  data/prices_history.parquet   token_id, t, p (CLOB prices-history, fidelity 10 min, Sep 1 -> now)
  out/resolution.json       gamma closed/outcomePrices/umaResolutionStatuses per slug (refreshed)
"""
import json, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT, DATA = HERE / "out", HERE / "data"; DATA.mkdir(exist_ok=True)
CLOB, G = "https://clob.polymarket.com", "https://gamma-api.polymarket.com"


def get(url):
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 ** i)
    return None


U = json.load(open(OUT / "universe.json"))
cids = {r["condition_id"]: r["slug"] for r in U}

# ---- rewards
rows, cur, n, total = [], "", 0, 0.0
while True:
    d = get(f"{CLOB}/rewards/markets/current?next_cursor={urllib.parse.quote(cur)}")
    if not d:
        break
    for x in d["data"]:
        n += 1; total += float(x.get("total_daily_rate") or 0)
        if x["condition_id"] in cids:
            x["slug"] = cids[x["condition_id"]]; rows.append(x)
    cur = d.get("next_cursor")
    if not cur or cur == "LTE=":
        break
json.dump(dict(n_markets=n, sum_daily_rate=total), open(OUT / "rewards_all_summary.json", "w"))
per = {}
for c, s in cids.items():
    per[s] = get(f"{CLOB}/rewards/markets/{c}")
json.dump(dict(current=rows, per_market=per), open(OUT / "rewards.json", "w"), indent=1)
print("rewards programme:", n, "markets, $", round(total), "/day; universe rows", len(rows))

# ---- resolution
res = {}
for r in U:
    g = get(f"{G}/markets?slug={r['slug']}") or get(f"{G}/markets?slug={r['slug']}&closed=true")
    g = g[0] if g else {}
    res[r["slug"]] = {k: g.get(k) for k in ("closed", "closedTime", "outcomePrices", "umaResolutionStatuses", "endDate",
                                            "lastTradePrice", "bestBid", "bestAsk", "spread", "liquidityNum", "volume24hrClob")}
json.dump(res, open(OUT / "resolution.json", "w"), indent=1)

# ---- prices history
t0, t1 = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp()), int(time.time())
ph = []
for r in U:
    for tok in r["tokens"]:
        a = t0
        while a < t1:
            b = min(a + 6 * 86400, t1)
            d = get(f"{CLOB}/prices-history?market={tok}&startTs={a}&endTs={b}&fidelity=10") or {}
            for h in d.get("history", []):
                ph.append((tok, int(h["t"]), float(h["p"])))
            a = b
ph = pd.DataFrame(ph, columns=["token_id", "t", "p"]).drop_duplicates(["token_id", "t"]).sort_values(["token_id", "t"])
ph.to_parquet(DATA / "prices_history.parquet")
print("prices-history rows", len(ph), "tokens", ph.token_id.nunique())
