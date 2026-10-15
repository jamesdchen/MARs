import requests, time, json
import os
RPC = os.environ.get("RPC", "https://polygon-bor-rpc.publicnode.com")
S = requests.Session()
STATS = {"calls": 0, "errors": 0, "retries": 0}
def call(method, params, retries=6):
    for i in range(retries):
        STATS["calls"] += 1
        try:
            r = S.post(RPC, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=60)
            if r.status_code != 200:
                STATS["errors"] += 1; STATS["retries"] += 1
                time.sleep(1.5 * (i + 1)); last = f"HTTP {r.status_code} {r.text[:200]}"; continue
            j = r.json()
            if "error" in j:
                STATS["errors"] += 1
                return {"__error__": j["error"]}
            return j["result"]
        except Exception as e:
            STATS["errors"] += 1; STATS["retries"] += 1; last = repr(e); time.sleep(1.5 * (i + 1))
    raise RuntimeError(last)
_ts = {}
def block_ts(n):
    if n not in _ts:
        b = call("eth_getBlockByNumber", [hex(n), False]); _ts[n] = int(b["timestamp"], 16)
    return _ts[n]
def latest():
    return int(call("eth_blockNumber", []), 16)
def block_at_or_after(ts, lo=None, hi=None):
    hi = hi or latest(); lo = lo or 1
    while lo < hi:
        mid = (lo + hi) // 2
        if block_ts(mid) < ts: lo = mid + 1
        else: hi = mid
    return lo
