import requests,time,json,sys
EX="0xE111180000d2663C0091e4f400237545B87B996B"
OF="0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
a=92381315
for u in sys.argv[1:]:
    for span in [10,50,100,250,500,1000]:
        t=time.time()
        try:
            r=requests.post(u,json={"jsonrpc":"2.0","id":1,"method":"eth_getLogs","params":[{"fromBlock":hex(a),"toBlock":hex(a+span-1),"address":EX,"topics":[OF]}]},timeout=90)
            j=r.json(); res = len(j["result"]) if "result" in j else str(j.get("error"))[:150]
        except Exception as e: res=f"EXC {r.status_code if 'r' in dir() else ''} {repr(e)[:100]}"
        print(u.split('/')[2],span,res,round(time.time()-t,2),"s",flush=True)
        if not isinstance(res,int): break
