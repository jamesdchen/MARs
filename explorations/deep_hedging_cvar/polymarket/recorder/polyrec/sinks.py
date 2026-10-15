"""Output sinks: hourly-rotated parquet datasets (written as crash-safe part files, compacted per hour)
and the zstd-compressed JSONL raw event log.

Layout under out_dir:
    <name>/<YYYY-MM-DD>/<name>_<YYYY-MM-DD>T<HH>.parquet              compacted hour (final)
    <name>/<YYYY-MM-DD>/_parts/<name>_<YYYY-MM-DD>T<HH>__<run>_<seq>.parquet   flushed parts (in progress)
    raw/<YYYY-MM-DD>/events_<YYYY-MM-DD>T<HH>_<run>.jsonl.zst

Part files are complete parquet files written to a temp name and renamed, so a crash loses at most
one flush interval. An hour is compacted once it is over (or at startup for leftovers); the final
file records the part names it absorbed in its schema metadata, so a crash between writing the final
file and deleting the parts never duplicates rows. Readers use `dataset_files()`, which returns the
final files plus any parts not yet absorbed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from array import array
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .schema import BOOK_SCHEMA

log = logging.getLogger("polyrec.sinks")
MANIFEST_KEY = b"polyrec_parts"


def hour_key(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H", time.gmtime(t))


def day_of(hour: str) -> str:
    return hour[:10]


def _fsync_rename(tmp: Path, final: Path):
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)


class PartSink:
    """Hour-partitioned parquet dataset written as parts and compacted per hour."""

    def __init__(self, root: Path, name: str, schema: pa.Schema, run_id: str, level: int = 9,
                 sort_by: list | None = None, dictionary_cols: list | None = None):
        self.root, self.name, self.schema, self.run_id = Path(root), name, schema, run_id
        self.level, self.sort_by = level, sort_by
        self.dictionary_cols = dictionary_cols
        self.seq = 0
        self.rows_written = 0
        self.bytes_written = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths
    def day_dir(self, hour: str) -> Path:
        return self.root / self.name / day_of(hour)

    def final_path(self, hour: str) -> Path:
        return self.day_dir(hour) / f"{self.name}_{hour}.parquet"

    def _write(self, table: pa.Table, path: Path, level: int, metadata: dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if metadata:
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        pq.write_table(table, tmp, compression="zstd", compression_level=level,
                       use_dictionary=self.dictionary_cols if self.dictionary_cols is not None else True,
                       row_group_size=1 << 20)
        _fsync_rename(tmp, path)
        return path.stat().st_size

    def write_part(self, hour: str, table: pa.Table):
        """Thread-safe; called from a worker thread."""
        if table.num_rows == 0:
            return
        with self._lock:
            self.seq += 1
            p = self.day_dir(hour) / "_parts" / f"{self.name}_{hour}__{self.run_id}_{self.seq:05d}.parquet"
            n = self._write(table, p, level=3)
            self.rows_written += table.num_rows
            self.bytes_written += n

    # ------------------------------------------------------------------ compaction
    def part_hours(self) -> list[str]:
        hours = set()
        base = self.root / self.name
        if not base.exists():
            return []
        for p in base.glob("*/_parts/*.parquet"):
            h = p.name[len(self.name) + 1:].split("__", 1)[0]
            hours.add(h)
        return sorted(hours)

    def compact(self, hour: str) -> Path | None:
        with self._lock:
            parts_dir = self.day_dir(hour) / "_parts"
            parts = sorted(parts_dir.glob(f"{self.name}_{hour}__*.parquet")) if parts_dir.exists() else []
            final = self.final_path(hour)
            absorbed: list[str] = []
            tables = []
            if final.exists():
                md = pq.read_schema(final).metadata or {}
                absorbed = json.loads(md.get(MANIFEST_KEY, b"[]"))
                tables.append(pq.read_table(final))
            new = [p for p in parts if p.name not in absorbed]
            stale = [p for p in parts if p.name in absorbed]
            for p in stale:                      # already inside the final file (crash after rename)
                p.unlink(missing_ok=True)
            if not new:
                return final if final.exists() else None
            for p in new:
                try:
                    tables.append(pq.read_table(p))
                except Exception as e:  # noqa: BLE001  (truncated part from a hard crash)
                    log.error("unreadable part %s skipped: %s", p, e)
            t = pa.concat_tables([x.replace_schema_metadata(None).cast(self.schema)
                                  if x.schema.remove_metadata() != self.schema else x.replace_schema_metadata(None)
                                  for x in tables])
            if len(tables) > 1:
                t = t.unify_dictionaries()
            if self.sort_by and t.num_rows:
                t = t.take(pc.sort_indices(t, sort_keys=[(c, "ascending") for c in self.sort_by]))
            names = absorbed + [p.name for p in new]
            self._write(t, final, self.level, {MANIFEST_KEY: json.dumps(names).encode()})
            for p in new:
                p.unlink(missing_ok=True)
            try:
                parts_dir.rmdir()
            except OSError:
                pass
            return final


def dataset_files(root: Path, name: str, days: list[str] | None = None) -> list[Path]:
    """Final hourly files plus parts not yet absorbed into a final file."""
    base = Path(root) / name
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or (days is not None and d.name not in days):
            continue
        absorbed = set()
        for f in sorted(d.glob(f"{name}_*.parquet")):
            md = pq.read_schema(f).metadata or {}
            absorbed |= set(json.loads(md.get(MANIFEST_KEY, b"[]")))
            out.append(f)
        pdir = d / "_parts"
        if pdir.exists():
            out += [p for p in sorted(pdir.glob("*.parquet")) if p.name not in absorbed]
    return out


def read_dataset(root: Path, name: str, schema: pa.Schema | None = None, columns=None,
                 days: list[str] | None = None) -> pa.Table | None:
    files = dataset_files(root, name, days)
    if not files:
        return None
    ts = []
    for f in files:
        try:
            t = pq.read_table(f, columns=columns)
        except Exception as e:  # noqa: BLE001
            log.error("skip unreadable %s: %s", f, e)
            continue
        ts.append(t)
    if not ts:
        return None
    if schema is not None and columns is None:
        ts = [t.cast(schema) if t.schema.remove_metadata() != schema else t for t in ts]
    return pa.concat_tables([t.replace_schema_metadata(None) for t in ts], promote_options="default")


class TableSink(PartSink):
    """Low-volume sink for trades / markets / quality logs: dict rows or ready-made tables."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.buf: dict = {}          # hour -> list of dict rows or pa.Table

    def append(self, row: dict, t: float):
        self.buf.setdefault(hour_key(t), []).append(row)

    def append_table(self, table: pa.Table, t: float):
        if table.num_rows:
            self.buf.setdefault(hour_key(t), []).append(table)

    def take(self) -> list:
        """Swap out the buffer (event-loop thread). Write the result with write_items in a thread."""
        out, self.buf = list(self.buf.items()), {}
        return out

    def write_items(self, items: list):
        for hour, objs in items:
            rows = [o for o in objs if isinstance(o, dict)]
            tabs = [o for o in objs if isinstance(o, pa.Table)]
            if rows:
                tabs.insert(0, pa.Table.from_pylist(rows, schema=self.schema))
            if tabs:
                tabs = [x if x.schema == self.schema else x.cast(self.schema) for x in tabs]
                t = pa.concat_tables(tabs).unify_dictionaries() if len(tabs) > 1 else tabs[0]
                self.write_part(hour, t)

    def flush(self):
        self.write_items(self.take())


class BookRowBuffer(PartSink):
    """Columnar row buffer for book snapshots (8 bytes per ladder value, no per-row objects)."""

    STR_COLS = ("market_slug", "coin", "contract_length", "token_id", "outcome")

    def __init__(self, root: Path, run_id: str, level: int = 9):
        super().__init__(root, "books", BOOK_SCHEMA, run_id, level,
                         sort_by=["market_slug", "token_id", "captured_at"],
                         dictionary_cols=list(self.STR_COLS))
        self.hour: str | None = None
        self.pending: list = []          # full buffers of earlier hours awaiting a write
        self._reset()

    def _reset(self):
        self.s = {c: [] for c in self.STR_COLS}
        self.end_us = array("q")
        self.cap_us = array("q")
        self.stc = array("d")
        self.lv = {k: array("d") for k in ("bp", "bs", "ap", "as")}
        self.boff = array("i", [0])
        self.aoff = array("i", [0])
        self.bb = array("d")
        self.ba = array("d")
        self.crossed = array("b")
        self.recon = array("b")
        self.n = 0
        self.nvals = 0

    def append(self, meta: tuple, end_s: int, cap_s: int, bp, bs, ap, asz, bb, ba, crossed: bool, recon: bool):
        """meta = (market_slug, coin, contract_length, token_id, outcome)."""
        h = hour_key(cap_s)
        if self.hour is not None and h != self.hour and self.n:
            self.pending.append(self.take())
        self.hour = h
        for c, v in zip(self.STR_COLS, meta):
            self.s[c].append(v)
        self.end_us.append(end_s * 1_000_000)
        self.cap_us.append(cap_s * 1_000_000)
        self.stc.append(float(end_s - cap_s))
        self.lv["bp"].extend(bp)
        self.lv["bs"].extend(bs)
        self.lv["ap"].extend(ap)
        self.lv["as"].extend(asz)
        self.boff.append(self.boff[-1] + len(bp))
        self.aoff.append(self.aoff[-1] + len(ap))
        self.bb.append(bb if bb is not None else float("nan"))
        self.ba.append(ba if ba is not None else float("nan"))
        self.crossed.append(crossed)
        self.recon.append(recon)
        self.n += 1
        self.nvals += 2 * (len(bp) + len(ap))

    def take_all(self) -> list:
        """Swap out every buffered hour (event-loop thread); write them with write_cols in a thread."""
        out, self.pending = self.pending, []
        cur = self.take()
        if cur:
            out.append(cur)
        return out

    def take(self):
        if not self.n:
            return None
        cols = (self.hour, self.s, self.end_us, self.cap_us, self.stc, self.lv, self.boff, self.aoff,
                self.bb, self.ba, self.crossed, self.recon, self.n)
        self._reset()
        return cols

    @staticmethod
    def to_table(cols) -> tuple[str, pa.Table]:
        hour, s, end_us, cap_us, stc, lv, boff, aoff, bb, ba, crossed, recon, n = cols
        lt = BOOK_SCHEMA.field("bid_prices").type
        bo = pa.array(boff, pa.int32())
        ao = pa.array(aoff, pa.int32())

        def lst(off, vals):
            return pa.ListArray.from_arrays(off, pa.array(vals, pa.float64()), type=lt)

        bba = pa.array(bb, pa.float64())
        baa = pa.array(ba, pa.float64())
        arrays = [pa.array(s[c], pa.string()) for c in BookRowBuffer.STR_COLS] + [
            pa.nulls(n, pa.string()), pa.nulls(n, pa.bool_()),
            pa.array(end_us, pa.int64()).cast(BOOK_SCHEMA.field("market_end_at").type),
            pa.array(cap_us, pa.int64()).cast(BOOK_SCHEMA.field("captured_at").type),
            pa.array(stc, pa.float64()),
            lst(bo, lv["bp"]), lst(bo, lv["bs"]), lst(ao, lv["ap"]), lst(ao, lv["as"]),
            pc.if_else(pc.is_nan(bba), pa.scalar(None, pa.float64()), bba),
            pc.if_else(pc.is_nan(baa), pa.scalar(None, pa.float64()), baa),
            pa.array(crossed, pa.int8()).cast(pa.bool_()),
            pa.array(recon, pa.int8()).cast(pa.bool_()),
        ]
        return hour, pa.Table.from_arrays(arrays, schema=BOOK_SCHEMA)

    def write_cols(self, cols):
        hour, t = self.to_table(cols)
        self.write_part(hour, t)


class RawLog:
    """Hourly zstd-compressed JSONL of every stream event (+ REST snapshots and control events).

    Lines are buffered and compressed in 256 KiB chunks; every `flush_interval_s` the zstd block is
    flushed so a crash leaves a decodable prefix. One file per (hour, run): concatenating runs into one
    file would put a truncated frame in the middle after a hard crash.
    """

    CHUNK = 256 * 1024

    def __init__(self, root: Path, run_id: str, level: int = 6, enabled: bool = True):
        self.root, self.run_id, self.level, self.enabled = Path(root) / "raw", run_id, level, enabled
        self.hour = None
        self.fh = None
        self.writer = None
        self.buf: list = []
        self.buf_bytes = 0
        self.bytes_in = 0
        self.bytes_out_closed = 0
        self.lines = 0
        self.paused = False   # set by the disk guard

    def _open(self, hour: str):
        import zstandard as zstd
        d = self.root / day_of(hour)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"events_{hour}_{self.run_id}.jsonl.zst"
        self.fh = open(self.path, "ab")
        self.writer = zstd.ZstdCompressor(level=self.level).stream_writer(self.fh, closefd=False)
        self.hour = hour

    def write(self, line: bytes, t: float):
        if not self.enabled or self.paused:
            return
        h = hour_key(t)
        if h != self.hour:
            self.close()
            self._open(h)
        self.buf.append(line)
        self.buf_bytes += len(line) + 1
        self.lines += 1
        if self.buf_bytes >= self.CHUNK:
            self._drain()

    def _drain(self):
        if self.buf and self.writer is not None:
            data = b"\n".join(self.buf) + b"\n"
            self.writer.write(data)
            self.bytes_in += len(data)
        self.buf, self.buf_bytes = [], 0

    def flush(self):
        import zstandard as zstd
        if self.writer is None:
            return
        self._drain()
        self.writer.flush(zstd.FLUSH_BLOCK)
        self.fh.flush()

    def close(self):
        import zstandard as zstd
        if self.writer is None:
            return
        self._drain()
        self.writer.flush(zstd.FLUSH_FRAME)
        self.writer.close()
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        try:
            self.bytes_out_closed += self.path.stat().st_size
        except OSError:
            pass
        self.writer = self.fh = None
        self.hour = None

    def bytes_out(self) -> int:
        cur = 0
        if self.fh is not None:
            try:
                cur = self.path.stat().st_size
            except OSError:
                pass
        return self.bytes_out_closed + cur


def iter_raw(path: Path):
    """Yield decoded JSON objects from a raw log file (tolerates a truncated tail)."""
    import orjson
    import zstandard as zstd
    with open(path, "rb") as fh:
        rd = zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=True)
        pending = b""
        while True:
            try:
                chunk = rd.read(1 << 20)
            except zstd.ZstdError:
                break
            if not chunk:
                break
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for ln in lines:
                if ln:
                    yield orjson.loads(ln)
