"""Gamma census of sports game events (tag 100639 'Games') closed in a window, plus open upcoming ones.
Writes out/census_markets.parquet (one row per market) and prints league / market-type volume shares."""
import requests, json, sys, time, pandas as pd
S = requests.Session()
G = "https://gamma-api.polymarket.com"
def events(**params):
    out, off = [], 0
    while True:
        r = S.get(f"{G}/events", params={**params, "limit": 100, "offset": off}, timeout=60); r.raise_for_status()
        j = r.json(); out += j
        if len(j) < 100: return out
        off += 100
LEAGUE_SKIP = {"Sports", "Games", "Esports"}
def rows(evs, status):
    R = []
    for e in evs:
        tags = [t["label"] for t in e.get("tags", [])]
        slugs = [t.get("slug") for t in e.get("tags", [])]
        league = e["slug"].split("-")[0]
        for m in e.get("markets", []):
            R.append(dict(status=status, event_slug=e["slug"], league=league, tags="|".join(tags),
                esports="Esports" in tags, event_volume=float(e.get("volume") or 0),
                slug=m["slug"], question=m.get("question"), smt=m.get("sportsMarketType"),
                game_start=m.get("gameStartTime") or e.get("startTime"), end=m.get("endDate"),
                closed_time=m.get("closedTime"), created=m.get("createdAt"),
                volume=float(m.get("volumeNum") or m.get("volume") or 0),
                fee_type=m.get("feeType"), fee=json.dumps(m.get("feeSchedule")),
                rew_min=m.get("rewardsMinSize"), rew_spread=m.get("rewardsMaxSpread"),
                clob_rewards=json.dumps(m.get("clobRewards")), tick=m.get("orderPriceMinTickSize"),
                delay=m.get("secondsDelay"), clear_on_start=m.get("clearBookOnStart"),
                neg_risk=m.get("negRisk"), tokens=m.get("clobTokenIds"), outcomes=m.get("outcomes"),
                outcome_prices=m.get("outcomePrices"), condition_id=m.get("conditionId")))
    return R
lo = sys.argv[1] if len(sys.argv) > 1 else "2026-10-01T00:00:00Z"
import datetime as dt
closed, d = [], dt.datetime.fromisoformat(lo.replace("Z","+00:00"))
now = dt.datetime.now(dt.timezone.utc)
while d < now:
    for h in range(0, 24, 6):
        a = d + dt.timedelta(hours=h); b = a + dt.timedelta(hours=6)
        closed += events(tag_id=100639, closed="true", end_date_min=a.isoformat(), end_date_max=b.isoformat())
    d += dt.timedelta(days=1)
seen=set(); closed=[e for e in closed if not (e["id"] in seen or seen.add(e["id"]))]
print("closed events", len(closed), file=sys.stderr)
openev = []
for h in range(-24, 96, 6):
    a = now + dt.timedelta(hours=h); b = a + dt.timedelta(hours=6)
    openev += events(tag_id=100639, closed="false", end_date_min=a.isoformat(), end_date_max=b.isoformat())
print("open events", len(openev), file=sys.stderr)
df = pd.DataFrame(rows(closed, "closed") + rows(openev, "open"))
df.to_parquet("out/census_markets.parquet")
print(df.shape)
