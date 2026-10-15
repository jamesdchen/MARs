"""Pick the politics / economics (Fed) / geopolitics universe from gamma.

Active events (by 1-month volume) and events that resolved in the last 30 days, from several real
category tags. Events carrying sports / crypto / pop-culture tags are dropped (tag filters leak).
Writes out/universe.json (full gamma market records + event tags) and out/universe.csv.
"""
import json, sys, time, urllib.request
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
G = "https://gamma-api.polymarket.com"
TAGS = ["politics", "elections", "geopolitics", "economy", "fed", "fed-rates", "world", "midterms",
        "us-politics", "trump", "iran", "middle-east"]
BAD = {"sports", "crypto", "pop-culture", "tweets-markets", "esports", "nba", "nfl", "soccer", "mlb",
       "nhl", "tennis", "golf", "ufc", "crypto-prices", "bitcoin", "ethereum", "up-or-down", "culture",
       "mentions", "movies", "music", "awards", "weather", "tech", "ai", "big-tech", "stocks", "equities"}
NOW = pd.Timestamp.now(tz="UTC")
N_ACTIVE, N_CLOSED, PER_EVENT = 40, 20, 4


def get(url):
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=30) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(2 ** i)
    raise RuntimeError(url)


def category(tags):
    t = set(tags)
    if t & {"fed", "fed-rates", "fomc", "economy", "ecb", "treasuries", "economic-policy", "inflation"}:
        return "economics"
    if t & {"geopolitics", "iran", "israel", "russia", "ukraine", "china", "middle-east", "military", "nato"}:
        return "geopolitics"
    return "politics"


events = {}
for tag in TAGS:
    for closed in (False, True):
        q = f"{G}/events?tag_slug={tag}&limit=100&"
        q += ("closed=true&order=volume&ascending=false&end_date_min=" + (NOW - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
              if closed else "active=true&closed=false&order=volume1mo&ascending=false")
        for off in (0, 100):
            for e in get(q + f"&offset={off}"):
                events.setdefault(e["id"], e)
print("events", len(events), file=sys.stderr)

rows = []
for e in events.values():
    tags = [t["slug"] for t in e.get("tags", [])]
    if set(tags) & BAD or not set(tags) & {"politics", "elections", "geopolitics", "economy", "fed", "fed-rates", "world"}:
        continue
    for m in e.get("markets", []):
        if not m.get("clobTokenIds") or not m.get("enableOrderBook"):
            continue
        end = pd.Timestamp(m.get("closedTime") or m.get("endDate") or e.get("endDate"))
        end = end.tz_localize("UTC") if end.tzinfo is None else end
        closed = bool(m.get("closed"))
        if closed and end < NOW - pd.Timedelta(days=30):
            continue
        rows.append(dict(event=e["slug"], event_id=e["id"], slug=m["slug"], question=m["question"], tags=tags,
                         category=category(tags), closed=closed, end=end.isoformat(),
                         vol1mo=float(m.get("volume1moClob") or m.get("volume1mo") or 0),
                         vol=float(m.get("volumeClob") or m.get("volume") or 0), m=m))
df = pd.DataFrame(rows).drop_duplicates("slug")
# recent volume proxy: 1-month volume for active; total volume for markets that closed in the last 30 d
# (gamma zeroes rolling volumes after close).
df["vol_rank"] = df.vol1mo.where(~df.closed, df.vol)
pick = []
for closed, n in ((False, N_ACTIVE), (True, N_CLOSED)):
    d = df[df.closed == closed].sort_values("vol_rank", ascending=False)
    d = d.groupby("event", sort=False).head(PER_EVENT).head(n)
    pick.append(d)
u = pd.concat(pick)
recs = []
for r in u.itertuples():
    m = r.m
    toks = json.loads(m["clobTokenIds"]); outs = json.loads(m["outcomes"])
    recs.append(dict(slug=r.slug, event=r.event, question=r.question, category=r.category, tags=r.tags,
                     closed=r.closed, end=r.end, vol_rank=r.vol_rank, condition_id=m["conditionId"],
                     tokens=toks, outcomes=outs, outcome_prices=m.get("outcomePrices"),
                     neg_risk=m.get("negRisk"), tick=m.get("orderPriceMinTickSize"),
                     fees_enabled=m.get("feesEnabled"), fee_type=m.get("feeType"), fee_schedule=m.get("feeSchedule"),
                     maker_base_fee=m.get("makerBaseFee"), taker_base_fee=m.get("takerBaseFee"),
                     rewards_min_size=m.get("rewardsMinSize"), rewards_max_spread=m.get("rewardsMaxSpread"),
                     holding_rewards=m.get("holdingRewardsEnabled"), last_price=m.get("lastTradePrice"),
                     best_bid=m.get("bestBid"), best_ask=m.get("bestAsk"), start=m.get("startDate"),
                     created=m.get("createdAt"), uma_status=m.get("umaResolutionStatuses")))
json.dump(recs, open(OUT / "universe.json", "w"), indent=1, default=str)
pd.DataFrame(recs).drop(columns=["tags", "tokens", "outcomes"]).to_csv(OUT / "universe.csv", index=False)
print(pd.DataFrame(recs)[["slug", "category", "closed", "vol_rank", "fee_type", "rewards_max_spread", "holding_rewards"]].to_string())
