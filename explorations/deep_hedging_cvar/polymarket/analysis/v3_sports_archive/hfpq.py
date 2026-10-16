"""Partial download of one Joseph3222/polymarket-orderbook day parquet (sorted by condition_id within each hour,
~100k-row row groups with min/max stats): fetch the footer, choose row groups, fetch only their byte ranges into
a SPARSE local file of the full size (holes cost no disk), then read them with pyarrow as a normal parquet.
Downloaded bytes are untrusted data: they are only parsed by pyarrow, never executed."""
from __future__ import annotations

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

REPO = "https://huggingface.co/datasets/Joseph3222/polymarket-orderbook/resolve/main/"


def _get(url, a=None, b=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"} if a is not None else {})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
                if a is not None and len(data) != b - a + 1:
                    raise IOError(f"short read {len(data)} != {b - a + 1}")
                return data, r.headers
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def size_of(url):
    req = urllib.request.Request(url, method="HEAD")
    for k in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"]), r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
        except Exception:  # noqa: BLE001
            if k == 5:
                raise
            time.sleep(2 ** k)


class SparseParquet:
    def __init__(self, relpath: str, local: str, reuse: bool = False):
        """reuse (dev only): keep an existing local sparse file of the right size and skip fetch()."""
        self.url = REPO + relpath
        self.local = local
        self.size, self.etag = size_of(self.url)
        self.bytes_fetched = 0
        self.reused = reuse and os.path.exists(local) and os.path.getsize(local) == self.size
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if self.reused:
            self.meta = pq.read_metadata(local)
            return
        with open(local, "wb") as f:
            f.truncate(self.size)
        tail, _ = _get(self.url, self.size - 8, self.size - 1)
        assert tail[4:] == b"PAR1", "not a parquet file"
        flen = int.from_bytes(tail[:4], "little")
        foot, _ = _get(self.url, self.size - 8 - flen, self.size - 1)
        self._write(self.size - 8 - flen, foot)
        self.meta = pq.read_metadata(local)

    def _write(self, off, data):
        with open(self.local, "r+b") as f:
            f.seek(off)
            f.write(data)
        self.bytes_fetched += len(data)

    def rg_span(self, i):
        rg = self.meta.row_group(i)
        lo, hi = None, 0
        for j in range(rg.num_columns):
            c = rg.column(j)
            st = c.dictionary_page_offset if c.has_dictionary_page and c.dictionary_page_offset else c.data_page_offset
            lo = st if lo is None else min(lo, st)
            hi = max(hi, st + c.total_compressed_size)
        return lo, hi

    def rg_stats(self, i, names=("event_ts", "condition_id")):
        rg = self.meta.row_group(i)
        out = {}
        for j in range(rg.num_columns):
            c = rg.column(j)
            if c.path_in_schema in names and c.statistics is not None and c.statistics.has_min_max:
                out[c.path_in_schema] = (c.statistics.min, c.statistics.max)
        return out

    def fetch(self, rgs, workers=8, max_gap=1 << 20, max_chunk=64 << 20):
        if self.reused:
            return 0
        spans = sorted(self.rg_span(i) for i in rgs)
        merged = []
        for a, b in spans:                       # coalesce nearby ranges into fewer requests
            if merged and a - merged[-1][1] <= max_gap and b - merged[-1][0] <= max_chunk:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])

        def one(ab):
            data, _ = _get(self.url, ab[0], ab[1] - 1)
            return ab[0], data
        with ThreadPoolExecutor(workers) as ex:
            for off, data in ex.map(one, merged):
                self._write(off, data)
        return len(merged)

    def read(self, rgs, columns=None):
        return pq.ParquetFile(self.local).read_row_groups(sorted(rgs), columns=columns)

    def close(self, remove=True):
        if remove and os.path.exists(self.local):
            os.remove(self.local)
