"""Shared constants and small helpers for the on-chain Polymarket extractor."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"            # extraction outputs (git-ignored)
CACHE = HERE / "cache"          # block clock, gamma cache (git-ignored)
REPO_DATA = HERE.parent / "data" / "raw"   # existing pipeline inputs (read-only for us)

# CTF Exchange V2 contracts (Polygon). Both deployed at DEPLOY_BLOCK.
EXCHANGES = {
    "0xe111180000d2663c0091e4f400237545b87b996b": "ctf",
    "0xe2222d279d744050d28e00520010520000310f59": "negrisk",
}
EXCHANGE_ADDRS = list(EXCHANGES)
DEPLOY_BLOCK = 84_902_353

# topic0 = keccak(signature); checked against abi.keccak256 in abi.selftest()
EVENT_SIGS = {
    "OrderFilled": "OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)",
    "OrdersMatched": "OrdersMatched(bytes32,address,uint8,uint256,uint256,uint256)",
    "FeeCharged": "FeeCharged(address,uint256)",
}
TOPIC_OF = "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
TOPIC_OM = "0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c"
TOPIC_FC = "0x55bb3cade9d43b798a4fe5ffdd05024b2d7870df53920673bfc7e68047cd0ab1"
TOPICS = {"OrderFilled": TOPIC_OF, "OrdersMatched": TOPIC_OM, "FeeCharged": TOPIC_FC}

# Keyless archive endpoints: url, max in-flight requests, initial span, max span (blocks per eth_getLogs).
# tenderly caps a response at 20k results (~100 blocks at ~190 logs/block) and suggests a range;
# drpc's free tier rejects ~>100-block ranges; quiknode had no cap up to 1000 blocks (146k logs, 133 MB).
DEFAULT_ENDPOINTS = [
    ("https://rpc-mainnet.matic.quiknode.pro", 3, 200, 400),
    ("https://gateway.tenderly.co/public/polygon", 2, 100, 100),
    ("https://polygon.drpc.org", 1, 100, 100),
]

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (research data pipeline; polite)",
                "Accept": "application/json", "Content-Type": "application/json"}

BLOCK_TIME_S = 1.5   # Polygon PoS target, only used for coarse estimates


def utc_ts(s: str) -> int:
    """'2026-08-21' or '2026-08-21T02:05[:00]' (UTC) -> unix seconds."""
    if len(s) == 10:
        d = dt.datetime.strptime(s, "%Y-%m-%d")
    else:
        d = dt.datetime.fromisoformat(s.replace("Z", ""))
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp())


def ts_date(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def date_range(start: str, end: str) -> list[str]:
    """Inclusive list of UTC dates."""
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def write_parquet_atomic(table: pa.Table, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    pq.write_table(table, tmp, compression=kw.pop("compression", "zstd"), **kw)
    os.replace(tmp, path)
