"""Pull all CTF Exchange V2 (+NegRisk V2) OrderFilled/OrdersMatched/FeeCharged logs in a block window,
keep only txs touching the given token ids, decode, save. Usage: RPC=... python3 pull_market.py"""
import rpc, json, time, gzip, sys
EX = ["0xE111180000d2663C0091e4f400237545B87B996B", "0xe2222d279d744050d28e00520010520000310F59"]
OF = "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
OM = "0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c"
FC = "0x55bb3cade9d43b798a4fe5ffdd05024b2d7870df53920673bfc7e68047cd0ab1"
TOK = {64793885128357825834423179738523532073824858947614514127086033867692990103600: "Up",
       76829508296972834187696916736387841390568398120722272795376069210348175677477: "Down"}
T0, T1 = 1787277000, 1787278500   # 01:50 .. 02:15 UTC 2026-08-21
CH = int(sys.argv[1]) if len(sys.argv) > 1 else 250
a = rpc.block_at_or_after(T0); b = rpc.block_at_or_after(T1)
print("blocks", a, b, file=sys.stderr)
def words(d): d = d[2:]; return [int(d[i:i+64], 16) for i in range(0, len(d), 64)]
def addr(t): return "0x" + t[-40:]
stats = {"chunks": 0, "logs_total": 0, "t": time.time()}
keep = []
blk = a
while blk <= b:
    hi = min(b, blk + CH - 1)
    r = rpc.call("eth_getLogs", [{"fromBlock": hex(blk), "toBlock": hex(hi), "address": EX, "topics": [[OF, OM, FC]]}])
    if isinstance(r, dict):
        print("err", blk, hi, r, file=sys.stderr); CH = max(10, CH // 2); continue
    stats["chunks"] += 1; stats["logs_total"] += len(r)
    bytx = {}
    for L in r: bytx.setdefault(L["transactionHash"], []).append(L)
    for tx, ls in bytx.items():
        hit = False
        for L in ls:
            if L["topics"][0] in (OF, OM) and words(L["data"])[1] in TOK: hit = True
        if hit: keep.extend(ls)
    blk = hi + 1
stats["elapsed"] = time.time() - stats["t"]; stats["rpc"] = rpc.STATS; stats["blocks"] = [a, b]
out = []
for L in keep:
    t0 = L["topics"][0]; w = words(L["data"])
    base = {"block": int(L["blockNumber"], 16), "tx": L["transactionHash"], "logIndex": int(L["logIndex"], 16), "exchange": L["address"]}
    if t0 == OF:
        base.update(ev="OrderFilled", orderHash=L["topics"][1], maker=addr(L["topics"][2]), taker=addr(L["topics"][3]),
                    side=["BUY", "SELL"][w[0]], tokenId=str(w[1]), outcome=TOK.get(w[1], "other"),
                    makerAmountFilled=w[2], takerAmountFilled=w[3], fee=w[4], builder=hex(w[5]), metadata=hex(w[6]))
    elif t0 == OM:
        base.update(ev="OrdersMatched", orderHash=L["topics"][1], maker=addr(L["topics"][2]),
                    side=["BUY", "SELL"][w[0]], tokenId=str(w[1]), outcome=TOK.get(w[1], "other"),
                    makerAmountFilled=w[2], takerAmountFilled=w[3])
    else:
        base.update(ev="FeeCharged", receiver=addr(L["topics"][1]), amount=w[0])
    out.append(base)
ts = {}
for o in out:
    if o["block"] not in ts: ts[o["block"]] = rpc.block_ts(o["block"])
    o["ts"] = ts[o["block"]]
json.dump(out, open("onchain_events.json", "w"), indent=0)
json.dump(stats, open("pull_stats.json", "w"), indent=1)
print(json.dumps(stats), len(out))
