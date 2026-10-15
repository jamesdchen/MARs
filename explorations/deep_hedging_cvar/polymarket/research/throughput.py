import requests,time,json,sys,concurrent.futures as cf
EX=["0xE111180000d2663C0091e4f400237545B87B996B","0xe2222d279d744050d28e00520010520000310F59"]
OF="0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
u=sys.argv[1]; span=int(sys.argv[2]); n=int(sys.argv[3]); conc=int(sys.argv[4]); start=int(sys.argv[5])
def one(i):
    a=start+i*span; t=time.time()
    try:
        r=requests.post(u,json={"jsonrpc":"2.0","id":1,"method":"eth_getLogs","params":[{"fromBlock":hex(a),"toBlock":hex(a+span-1),"address":EX,"topics":[OF]}]},timeout=120)
        j=r.json()
        if "result" in j: return ("ok",len(j["result"]),len(r.content),time.time()-t)
        return ("err",str(j.get("error"))[:100],r.status_code,time.time()-t)
    except Exception as e: return ("exc",repr(e)[:100],0,time.time()-t)
t=time.time()
with cf.ThreadPoolExecutor(conc) as ex: res=list(ex.map(one,range(n)))
el=time.time()-t
oks=[r for r in res if r[0]=="ok"]
print(u.split('/')[2],"span",span,"n",n,"conc",conc,"ok",len(oks),"errs",[r[:3] for r in res if r[0]!="ok"][:3])
if oks: print(" logs/block %.1f  bytes/log %.0f  wall %.1fs  blocks/s %.0f"%(sum(r[1] for r in oks)/(len(oks)*span), sum(r[2] for r in oks)/max(1,sum(r[1] for r in oks)), el, len(oks)*span/el))
