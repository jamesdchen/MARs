import json, rpc, collections
from eth_abi import decode
from Crypto.Hash import keccak
ORDER="(uint256,address,address,uint256,uint256,uint256,uint8,uint8,uint256,bytes32,bytes32,bytes)"
sig=f"matchOrders(bytes32,{ORDER},{ORDER}[],uint256,uint256[],uint256,uint256[])"
h=keccak.new(digest_bits=256); h.update(sig.encode()); sel="0x"+h.hexdigest()[:8]
ev=json.load(open("onchain_events.json"))
txs=sorted({e["tx"] for e in ev})
out=[]; sels=collections.Counter()
for tx in txs:
    t=rpc.call("eth_getTransactionByHash",[tx]); inp=t["input"]; sels[inp[:10]]+=1
    if inp[:10]!=sel: continue
    cid,tko,mkos,tfa,mfa,tfee,mfees=decode(["bytes32",ORDER,f"{ORDER}[]","uint256","uint256[]","uint256","uint256[]"],bytes.fromhex(inp[10:]))
    blk_ts=rpc.block_ts(int(t["blockNumber"],16))
    def od(o,role,fill,fee):
        side=["BUY","SELL"][o[6]]; px=(o[4]/o[5]) if side=="BUY" else (o[5]/o[4])
        return dict(tx=tx,role=role,maker=o[1],tokenId=str(o[3]),side=side,orig_makerAmount=o[4],orig_takerAmount=o[5],limit_px=round(px,6),
                    sigType=o[7],created_ms=o[8],age_s=round(blk_ts-o[8]/1000,3),fill=fill,fee=fee,from_=t["from"])
    out.append(od(tko,"taker",tfa,tfee))
    for o,f,fe in zip(mkos,mfa,mfees): out.append(od(o,"maker",f,fe))
print("selectors",sels,"expected",sel)
json.dump(out,open("calldata_orders.json","w"),indent=0)
import statistics as st
for role in ["taker","maker"]:
    a=[o["age_s"] for o in out if o["role"]==role]
    print(role,len(a),"order age at fill (s): min %.2f median %.2f max %.1f"%(min(a),st.median(a),max(a)))
mk=[o for o in out if o["role"]=="maker"]
print("maker partial fills (fill < orig making):",sum(o["fill"]<o["orig_makerAmount"] for o in mk),"/",len(mk))
print("tx senders",collections.Counter(o["from_"] for o in out).most_common(3))
print("sigTypes",collections.Counter(o["sigType"] for o in out))
print(json.dumps(out[:3],indent=0)[:1500])
