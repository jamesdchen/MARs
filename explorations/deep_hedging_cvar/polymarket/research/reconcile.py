import json,collections
ev=json.load(open("onchain_events.json"))
EXA={"0xe111180000d2663c0091e4f400237545b87b996b","0xe2222d279d744050d28e00520010520000310f59"}
TOK={"64793885128357825834423179738523532073824858947614514127086033867692990103600":"Up","76829508296972834187696916736387841390568398120722272795376069210348175677477":"Down"}
of=[e for e in ev if e["ev"]=="OrderFilled"]
def norm(e):
    # side BUY: maker gives USDC (makerAmountFilled), receives tokens (takerAmountFilled)
    if e["side"]=="BUY": usdc,sh=e["makerAmountFilled"],e["takerAmountFilled"]
    else: sh,usdc=e["makerAmountFilled"],e["takerAmountFilled"]
    return dict(wallet=e["maker"].lower(),side=e["side"],asset=e["tokenId"],size=round(sh/1e6,6),price=round(usdc/sh,6),tx=e["tx"].lower(),
                role="taker" if e["taker"].lower() in EXA else "maker",fee=e["fee"]/1e6,ts=e["ts"])
oc=[norm(e) for e in of]
api=json.load(open("api_trades_takerfalse_0.json")); apiT=json.load(open("api_trades_takertrue_0.json"))
print("api keys",sorted(api[0].keys()))
def key(d,wk): return (d[wk].lower(),d["side"],d["asset"],round(float(d["size"]),4),d["transactionHash"].lower() if "transactionHash" in d else d["tx"])
ak=collections.Counter((a["proxyWallet"].lower(),a["side"],a["asset"],round(a["size"],4),a["transactionHash"].lower()) for a in api)
ok=collections.Counter((o["wallet"],o["side"],o["asset"],round(o["size"],4),o["tx"]) for o in oc)
print("api rows",len(api),"onchain fills",len(oc),"exact key matches",sum((ak&ok).values()))
print("api-only",list((ak-ok).elements())[:5]); print("onchain-only",list((ok-ak).elements())[:5])
akT=collections.Counter((a["proxyWallet"].lower(),a["side"],a["asset"],round(a["size"],4),a["transactionHash"].lower()) for a in apiT)
okT=collections.Counter((o["wallet"],o["side"],o["asset"],round(o["size"],4),o["tx"]) for o in oc if o["role"]=="taker")
print("takerOnly api",len(apiT),"onchain taker",sum(okT.values()),"matches",sum((akT&okT).values()))
# price compare
am={(a["proxyWallet"].lower(),a["transactionHash"].lower(),a["asset"],round(a["size"],4)):a for a in api}
dp=[abs(am[k]["price"]-o["price"]) for o in oc if (k:=(o["wallet"],o["tx"],o["asset"],round(o["size"],4))) in am]
print("max |price diff|",max(dp) if dp else None)
# timestamps
dt=[am[k]["timestamp"]-o["ts"] for o in oc if (k:=(o["wallet"],o["tx"],o["asset"],round(o["size"],4))) in am]
print("ts diff api-block", collections.Counter(dt))
# taker orders matched against several makers
bytx=collections.defaultdict(list)
for o in oc: bytx[o["tx"]].append(o)
nm=collections.Counter(sum(x["role"]=="maker" for x in v) for v in bytx.values()); print("makers per taker tx",sorted(nm.items()))
cross=sum(1 for v in bytx.values() for x in v if x["role"]=="maker" and x["asset"]!=[t for t in v if t["role"]=="taker"][0]["asset"])
print("maker fills on opposite token (MINT/MERGE)",cross)
print("taker fees USDC total",round(sum(o["fee"] for o in oc if o["role"]=="taker"),4),"maker fees",round(sum(o["fee"] for o in oc if o["role"]=="maker"),4))
tv=sum(o["size"]*o["price"] for o in oc if o["role"]=="taker"); print("taker notional",round(tv,2),"taker fee/notional",round(sum(o["fee"] for o in oc if o["role"]=="taker")/tv,5))
print("distinct makers",len({o["wallet"] for o in oc if o["role"]=="maker"}),"distinct takers",len({o["wallet"] for o in oc if o["role"]=="taker"}))
mv=collections.Counter()
for o in oc:
    if o["role"]=="maker": mv[o["wallet"]]+=o["size"]
print("top makers by shares",mv.most_common(5))
print("unique orderHashes maker",len({e["orderHash"] for e in of if e["taker"].lower() not in EXA}))
print("builders",collections.Counter(e["builder"] for e in of).most_common(4))
print("metadata",collections.Counter(e["metadata"] for e in of).most_common(3))
print("--- price diffs")
for o in oc:
    k=(o["wallet"],o["tx"],o["asset"],round(o["size"],4))
    if abs(am[k]["price"]-o["price"])>1e-6: print(o["role"],o["side"],TOK[o["asset"]],"onchain",o["price"],"api",am[k]["price"],"size",o["size"],am[k]["size"])
# same tx: does api mark role? check takerOnly=false rows of tx with multiple makers
