"""Poll Smarkets (keyless betting exchange) quotes for the match-winner markets of selected events and log JSONL.
Usage: smarkets_poll.py OUT.jsonl DURATION_S EVENT_ID[,EVENT_ID...]
Each line: {"t": unix_s, "market": id, "quotes": {contract_id: {"bids": [...], "offers": [...]}}}; a first block of
lines with "meta" maps market/contract ids to names. Prices are in basis points of probability (5650 = 0.565)."""
import json, sys, time, requests
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0 research"
A = "https://api.smarkets.com/v3"
out, dur, evs = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(",")
fh = open(out, "a")
mids = []
for e in evs:
    ev = S.get(f"{A}/events/{e}/", timeout=20).json()["events"][0]
    ms = S.get(f"{A}/events/{e}/markets/", timeout=20).json()["markets"]
    pick = [m for m in ms if m["name"].lower() in ("winner", "full-time result", "match odds", "money line", "moneyline", "winner (incl. overtime)", "match winner")] or ms[:1]
    for m in pick[:2]:
        cs = S.get(f"{A}/markets/{m['id']}/contracts/", timeout=20).json()["contracts"]
        fh.write(json.dumps({"meta": True, "event": e, "event_name": ev["name"], "start": ev["start_datetime"], "market": m["id"],
                             "market_name": m["name"], "contracts": {c["id"]: c["name"] for c in cs}}) + "\n")
        mids.append(m["id"])
fh.flush(); print("markets", mids, flush=True)
t_end = time.time() + dur
while time.time() < t_end:
    t0 = time.time()
    try:
        q = S.get(f"{A}/markets/{','.join(mids)}/quotes/", timeout=10).json()
        for cid, v in q.items():
            pass
        fh.write(json.dumps({"t": round(t0, 2), "quotes": q}) + "\n"); fh.flush()
    except Exception as ex:  # noqa: BLE001
        print("err", ex, flush=True); time.sleep(5)
    time.sleep(max(0, 3 - (time.time() - t0)))
