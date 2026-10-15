"""Multi-endpoint JSON-RPC client with failover, per-endpoint concurrency limits, cooldown/backoff and
error classification, plus a cached block -> timestamp clock.

Error classes
  RangeError      the request asked for too much (result cap / range cap / response size); the caller
                  should shrink the block range. Carries the endpoint's suggested upper block if given.
  TransientError  429 / 5xx / timeouts / truncated bodies / unknown JSON-RPC errors; retry elsewhere.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests

import common as C


class RpcError(Exception):
    pass


class RangeError(RpcError):
    def __init__(self, msg, suggest_hi=None):
        super().__init__(msg)
        self.suggest_hi = suggest_hi


class TransientError(RpcError):
    pass


_RATE = ("rate limit", "too many requests", "429", "capacity", "throttl", "exceeded the quota", "try again later")
_RANGE = ("more than", "too many", "limit", "range", "exceed", "too large", "response size", "block range",
          "query timeout", "timed out", "returned more", "max results", "size limit")
_SUGGEST = re.compile(r"\[\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*\]")


def classify(err, method: str) -> RpcError:
    msg = (err.get("message", "") + " " + str(err.get("data", ""))) if isinstance(err, dict) else str(err)
    low = msg.lower()
    if any(k in low for k in _RATE):
        return TransientError(msg[:300])
    if method == "eth_getLogs" and any(k in low for k in _RANGE):
        m = _SUGGEST.search(msg)
        return RangeError(msg[:300], int(m.group(2), 16) if m else None)
    return TransientError(msg[:300])


class Endpoint:
    def __init__(self, url: str, conc: int = 2, span: int = 100, span_max: int = 100, rps: float = 10.0,
                 max_batch: int = 10, span_min: int = 5):
        self.url, self.conc, self.rps, self.max_batch = url, conc, float(rps), int(max_batch)
        self._next = 0.0
        self.name = url.split("/")[2]
        self.span, self.span_cap, self.span_max, self.span_min = span, span_max, span_max, span_min
        self.inflight, self.fails, self.cool_until = 0, 0, 0.0
        self._tl = threading.local()
        self.stats = dict(calls=0, errors=0, range_errors=0, logs=0, bytes=0, secs=0.0)
        self._lock = threading.Lock()

    def _session(self) -> requests.Session:
        s = getattr(self._tl, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update(C.HTTP_HEADERS)
            self._tl.s = s
        return s

    def _throttle(self, n: int = 1):
        """Pace requests to `rps` (each batch element counts as one request)."""
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + n / self.rps
        if start > now:
            time.sleep(start - now)

    def post(self, payload, timeout: float = 90):
        self._throttle(len(payload) if isinstance(payload, list) else 1)
        t0 = time.time()
        try:
            r = self._session().post(self.url, data=json.dumps(payload), timeout=timeout)
            body = r.content
        except (requests.RequestException, OSError) as e:
            self._stat(errors=1, secs=time.time() - t0)
            raise TransientError(f"{self.name}: {type(e).__name__}: {str(e)[:200]}")
        self._stat(calls=1, bytes=len(body), secs=time.time() - t0)
        try:
            j = json.loads(body)
        except ValueError:
            self._stat(errors=1)
            if r.status_code == 413:
                raise RangeError(f"{self.name}: HTTP 413")
            raise TransientError(f"{self.name}: HTTP {r.status_code} non-JSON body {body[:120]!r}")
        if r.status_code != 200 and not (isinstance(j, dict) and "error" in j):
            self._stat(errors=1)
            raise TransientError(f"{self.name}: HTTP {r.status_code}")
        return j

    def call(self, method: str, params, timeout: float = 90):
        j = self.post({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout)
        if not isinstance(j, dict):
            raise TransientError(f"{self.name}: unexpected response type")
        if "error" in j:
            e = classify(j["error"], method)
            self._stat(errors=1, range_errors=int(isinstance(e, RangeError)))
            raise e
        if "result" not in j:
            raise TransientError(f"{self.name}: no result")
        return j["result"]

    def batch(self, calls: list[tuple[str, list]], timeout: float = 90) -> list:
        """JSON-RPC batch, split into max_batch-sized posts; results in order (raises if any element errors)."""
        if len(calls) > self.max_batch:
            return [r for i in range(0, len(calls), self.max_batch)
                    for r in self.batch(calls[i:i + self.max_batch], timeout)]
        j = self.post([{"jsonrpc": "2.0", "id": i, "method": m, "params": p} for i, (m, p) in enumerate(calls)], timeout)
        if isinstance(j, dict):
            raise classify(j.get("error", j), calls[0][0]) if "error" in j else TransientError("batch: dict reply")
        byid = {x.get("id"): x for x in j}
        out = []
        for i, (m, _) in enumerate(calls):
            x = byid.get(i)
            if x is None or "error" in x or x.get("result") is None:
                self._stat(errors=1)
                raise TransientError(f"{self.name}: batch element {i} failed: {str((x or {}).get('error'))[:150]}")
            out.append(x["result"])
        return out

    def _stat(self, **kw):
        with self._lock:
            for k, v in kw.items():
                self.stats[k] += v

    # adaptive block span for eth_getLogs: the cap is learned from failures (a failing span lowers it to span-1,
    # or to the endpoint's suggested range) and relaxed by 10% after every 20 consecutive successes at the cap,
    # so it tracks the log density through the day instead of ratcheting down.
    def on_range_error(self, span: int, suggest: int | None):
        with self._lock:
            self._streak = 0
            new = suggest if suggest and suggest < span else span // 2
            self.span_cap = max(self.span_min, min(self.span_cap, span - 1, max(new, int(span * 0.8))))
            self.span = max(self.span_min, min(self.span_cap, new))

    def on_ok(self, span: int):
        with self._lock:
            self._streak = getattr(self, "_streak", 0) + 1
            if self._streak >= 20 and self.span_cap < self.span_max:
                self.span_cap = min(self.span_max, int(self.span_cap * 1.1) + 1)
                self._streak = 0
            if span >= self.span:
                self.span = min(self.span_cap, max(self.span + 1, int(self.span * 1.2)))


class Pool:
    """Hands out endpoint slots. acquire() blocks until an endpoint with a free slot is out of cooldown."""

    def __init__(self, endpoints: list[Endpoint], max_consec_fail: int = 6):
        self.eps = endpoints
        self.cv = threading.Condition()
        self.max_consec_fail = max_consec_fail

    @classmethod
    def from_specs(cls, specs=None):
        eps = []
        for s in specs or C.DEFAULT_ENDPOINTS:
            if isinstance(s, str):           # "url[,conc[,span[,span_max[,rps[,max_batch]]]]]"
                parts = s.split(",")
                url, nums = parts[0], [int(x) for x in parts[1:]]
            else:
                url, nums = s[0], list(s[1:])
            conc = nums[0] if len(nums) > 0 else 2
            span = nums[1] if len(nums) > 1 else 100
            smax = nums[2] if len(nums) > 2 else span
            rps = nums[3] if len(nums) > 3 else 10
            mb = nums[4] if len(nums) > 4 else 10
            eps.append(Endpoint(url, conc, span, smax, rps, mb))
        return cls(eps)

    def acquire(self, avoid: Endpoint | None = None) -> Endpoint:
        with self.cv:
            while True:
                now = time.monotonic()
                free = [e for e in self.eps if e.inflight < e.conc and e.cool_until <= now]
                pref = [e for e in free if e is not avoid] or free
                if pref:
                    e = min(pref, key=lambda e: (e.inflight / e.conc, -e.conc))
                    e.inflight += 1
                    return e
                nxt = min([e.cool_until for e in self.eps if e.cool_until > now] + [now + 1.0])
                self.cv.wait(timeout=max(0.05, min(1.0, nxt - now)))

    def release(self, e: Endpoint, ok: bool = True):
        with self.cv:
            e.inflight -= 1
            if ok:
                e.fails = 0
            else:
                e.fails += 1
                e.cool_until = time.monotonic() + min(120.0, 1.5 * 2 ** min(e.fails, 7)) * (0.75 + 0.5 * random.random())
            self.cv.notify_all()

    def request(self, fn, tries: int = 12):
        """Run fn(endpoint) with failover across endpoints and exponential backoff."""
        last, avoid = None, None
        for k in range(tries):
            e = self.acquire(avoid)
            try:
                out = fn(e)
            except RpcError as ex:
                self.release(e, ok=False)
                last, avoid = ex, e
                time.sleep(min(10.0, 0.25 * 2 ** k) * random.random())
                continue
            self.release(e, ok=True)
            return out
        raise RuntimeError(f"request failed after {tries} tries: {last}")

    def call(self, method: str, params, tries: int = 12):
        return self.request(lambda e: e.call(method, params), tries)

    def stats(self) -> dict:
        return {e.name: dict(e.stats, span=e.span, span_cap=e.span_cap) for e in self.eps}


class BlockClock:
    """Cached block -> unix-time map from sparse headers, with linear interpolation in between.

    Logs from all three default endpoints carry `blockTimestamp`, which the extractor uses as the exact block
    time; the clock serves date -> block boundary searches, the fallback for endpoints that omit
    `blockTimestamp`, and the interpolation-error check reported per day.
    """

    def __init__(self, pool: Pool, path: Path | None = None):
        self.pool = pool
        self.path = path or (C.CACHE / "blocktime.json")
        self.lock = threading.Lock()
        self.map: dict[int, int] = {}
        if self.path.exists():
            self.map = {int(k): int(v) for k, v in json.loads(self.path.read_text()).items()}
        self._dirty = False

    def save(self):
        with self.lock:
            if not self._dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({str(k): v for k, v in sorted(self.map.items())}))
            tmp.replace(self.path)
            self._dirty = False

    def latest(self) -> int:
        return int(self.pool.call("eth_blockNumber", []), 16)

    def ts(self, b: int) -> int:
        if b not in self.map:
            h = self.pool.call("eth_getBlockByNumber", [hex(b), False])
            if h is None:
                raise RuntimeError(f"block {b} not available")
            with self.lock:
                self.map[b] = int(h["timestamp"], 16)
                self._dirty = True
        return self.map[b]

    def ts_many(self, blocks, batch: int = 50):
        miss = sorted({int(b) for b in blocks} - self.map.keys())
        for i in range(0, len(miss), batch):
            chunk = miss[i:i + batch]
            res = self.pool.request(lambda e: e.batch([("eth_getBlockByNumber", [hex(b), False]) for b in chunk]))
            with self.lock:
                for b, h in zip(chunk, res):
                    self.map[b] = int(h["timestamp"], 16)
                self._dirty = True
        return [self.map[int(b)] for b in blocks]

    def block_at_or_after(self, t: int, hi: int | None = None) -> int:
        """First block with timestamp >= t (interpolation search mixed with bisection)."""
        lo = C.DEPLOY_BLOCK
        if self.ts(lo) >= t:
            return lo
        hi = hi or self.latest()
        if self.ts(hi) < t:
            raise ValueError(f"time {t} is after the chain head ({hi} @ {self.ts(hi)})")
        # tighten with cached anchors first
        for b, bt in self.map.items():
            if lo < b < hi:
                if bt < t:
                    lo = b
                else:
                    hi = b
        k = 0
        while hi - lo > 1:
            tl, th = self.ts(lo), self.ts(hi)
            if k % 2 == 0 and th > tl:
                g = lo + int(round((t - tl) * (hi - lo) / (th - tl)))
                g = min(max(g, lo + 1), hi - 1)
            else:
                g = (lo + hi) // 2
            if self.ts(g) < t:
                lo = g
            else:
                hi = g
            k += 1
        return hi

    def ensure_anchors(self, lo: int, hi: int, stride: int = 2000):
        bl = list(range(lo - lo % stride, hi + stride, stride)) + [lo, hi]
        self.ts_many(bl)

    def interp(self, blocks: np.ndarray, lo: int | None = None, hi: int | None = None, stride: int | None = None):
        """Linear interpolation from anchors. With stride, only anchors on that grid (+lo, hi) are used."""
        keys = np.array(sorted(self.map), dtype=np.int64)
        if stride:
            keep = (keys % stride == 0)
            if lo is not None:
                keep |= (keys == lo) | (keys == hi)
            keys = keys[keep]
        vals = np.array([self.map[int(k)] for k in keys], dtype=np.float64)
        return np.interp(np.asarray(blocks, dtype=np.float64), keys.astype(np.float64), vals)
