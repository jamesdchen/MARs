"""Async HTTP helpers (aiohttp) with a browser-like User-Agent, retries and exponential backoff.

gamma-api and data-api reject Python's default User-Agent with HTTP 403 (see data/fetch_trades.py).
`trust_env=True` honours HTTPS_PROXY / SSL_CERT_FILE when present and is a no-op on a plain VPS.
"""
from __future__ import annotations

import asyncio
import logging
import random

import aiohttp
import orjson

from .config import UA

log = logging.getLogger("polyrec.http")


class HttpError(Exception):
    def __init__(self, status: int, url: str, body: bytes = b""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]!r}")
        self.status, self.url, self.body = status, url, body


class Http:
    RETRY_STATUS = {403, 408, 425, 429, 500, 502, 503, 504, 520, 522, 524}

    def __init__(self, timeout_s: float = 20.0, limit: int = 32):
        self.timeout = aiohttp.ClientTimeout(total=timeout_s, connect=10)
        self.limit = limit
        self.session: aiohttp.ClientSession | None = None
        self.n_requests = 0
        self.n_errors = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=self.timeout, trust_env=True,
            headers={"User-Agent": UA, "Accept": "application/json"},
            connector=aiohttp.TCPConnector(limit=self.limit, ttl_dns_cache=300, enable_cleanup_closed=True),
            json_serialize=lambda o: orjson.dumps(o).decode())
        return self

    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()

    async def request(self, method: str, url: str, *, params=None, json=None, tries: int = 4,
                      no_retry_status=(400, 404)):
        delay = 0.5
        for k in range(tries):
            self.n_requests += 1
            try:
                async with self.session.request(method, url, params=params, json=json) as r:
                    body = await r.read()
                    if r.status == 200:
                        return orjson.loads(body)
                    err = HttpError(r.status, str(r.url), body)
                    if r.status in no_retry_status or r.status not in self.RETRY_STATUS:
                        self.n_errors += 1
                        raise err
            except HttpError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, orjson.JSONDecodeError) as e:
                err = e
            self.n_errors += 1
            if k < tries - 1:
                await asyncio.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 30)
        raise err

    async def get(self, url: str, params=None, **kw):
        return await self.request("GET", url, params=params, **kw)

    async def post(self, url: str, json=None, **kw):
        return await self.request("POST", url, json=json, **kw)
