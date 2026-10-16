#!/usr/bin/env python3
"""Free historical pre-game sports L2 from Hugging Face `Joseph3222/polymarket-orderbook` (pmxt archive, CC BY 4.0)
-> the recorder / backfill raw layout that `analysis/v3_sports/build_sports.py --rec` reads (Track S, D11/D15).

    raw_hf/books/<day>/books_<day>T<HH>.parquet     data/raw updown schema (BOOK_SCHEMA_RAW), outcome-0 token only
                                                    (build_sports mirrors outcome 1), full depth, 1 row per changed second
    raw_hf/trades/<day>/..., raw_hf/markets/<day>/...  data-api legs + gamma rows (backfill_pob.py --stage meta code)
    raw_hf/coverage/coverage_<day>.jsonl            recorder-style liveness log: live only where the archive is
                                                    present and the replayed book is anchored (see README)
    raw_hf/_state/                                  per-day done markers, carried book state, QC
    daily/<day>/                                    compact git copy: top 10 levels, zstd 19 (export stage)

Stages (each resumable; days are processed oldest first and the book state is carried day to day):
    python convert.py select --start 2026-04-13 --end 2026-08-10           # gamma selection, cached per kickoff day
    python convert.py books  --start 2026-08-08 --end 2026-08-08           # archive days -> raw_hf/books
    python convert.py meta   --start 2026-08-08 --end 2026-08-09           # kickoff days -> raw_hf/trades, markets
    python convert.py export --start 2026-08-08 --end 2026-08-08           # raw_hf -> daily/
    python convert.py run    --start 2026-04-13 --end 2026-08-10           # books + meta + export, day by day
    python convert.py status

Selection = backfill_pob.select_markets: gamma sports "Games" tag, sportsMarketType moneyline, closed, kickoff
(gameStartTime) in the day, two tokens, volume >= --min-volume (default 10000), all leagues. Book window per
market = [kickoff - --hours-before (48 h), kickoff + --post-min (5 min)), the recorder's window.

Downloaded parquet bytes are untrusted data (parsed by pyarrow only). Run with `python -I` if the working
directory holds downloads; this script keeps downloads under --dl (default: dl/, git-ignored).
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import gzip
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
pa.set_memory_pool(pa.system_memory_pool())
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v3_sports"))
import backfill_pob as bp  # noqa: E402  (also puts recorder/ on sys.path)
import hfpq  # noqa: E402
import replay as rp  # noqa: E402
from polyrec.schema import BOOK_SCHEMA_RAW  # noqa: E402
from polyrec.sinks import PartSink  # noqa: E402

log = logging.getLogger("convert")
ARCHIVE_FIRST, ARCHIVE_LAST = "2026-02-22", "2026-08-10"
ARCHIVE_MISSING = {f"2026-06-{d:02d}" for d in range(12, 18)}
COLS = ["event_ts_ms", "condition_id", "asset_id", "event_type", "bids_json", "asks_json", "price", "size", "side",
        "best_bid", "best_ask", "transaction_hash", "new_tick_size"]
RG_BATCH = 120


def days(a: str, b: str):
    d, e = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    while d <= e:
        yield d.isoformat()
        d += dt.timedelta(days=1)


def d0(day: str) -> int:
    return bp.day_start(day)


def atomic_json(path: Path, obj, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if gz:
        with gzip.open(tmp, "wt") as f:
            json.dump(obj, f, separators=(",", ":"))
    else:
        tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def read_json(path: Path, gz=False):
    if gz:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------------------- selection
def sel_args(a, day):
    return argparse.Namespace(start=day, end=day, out=str(a.cache / "pob_sel"), leagues=a.leagues.split(","),
                              market_types=a.market_types.split(","), hours_before=a.hours_before,
                              post_min=a.post_min, min_volume=a.min_volume, order="kickoff", max_markets=0)


def _ts(x):
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return int(x)
    return int(bp.parse_ts(str(x)))


def load_market_file(path: Path, a) -> dict:
    """--markets FILE (JSON array or JSON lines) -> {UTC day of kickoff (or window end): [market dicts]}.

    Per entry: `slug`; `condition_id` (or `cond`); `token_ids` (or `tokens`; the FIRST token is replayed and
    written as outcome 0); and either `window_start` + `window_end` or `kickoff` (window = kickoff -
    --hours-before .. kickoff + --post-min). Times are unix seconds or ISO strings. Optional: `outcomes`,
    `league` (default: slug prefix), `market_type`, `volume` (export floor; default kept), `listing` (replay
    starts there when within --track-days of the window), `gamma` (a gamma market dict; used for resolution,
    endDate and by the meta stage, which needs it)."""
    txt = Path(path).read_text().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(x) for x in txt.splitlines() if x.strip()]
    out = defaultdict(list)
    for r in rows:
        toks = [str(t) for t in (r.get("token_ids") or r.get("tokens") or [])]
        cond = r.get("condition_id") or r.get("cond")
        if not r.get("slug") or not cond or not toks:
            raise SystemExit(f"--markets: entry needs slug, condition_id and token_ids: {r}")
        ko = _ts(r.get("kickoff"))
        ws = _ts(r.get("window_start"))
        we = _ts(r.get("window_end"))
        if ws is None or we is None:
            if ko is None:
                raise SystemExit(f"--markets: {r['slug']}: give window_start/window_end or kickoff")
            ws = ko - int(a.hours_before * 3600) if ws is None else ws
            we = ko + int(a.post_min * 60) if we is None else we
        g = dict(r.get("gamma") or {})
        g.setdefault("slug", r["slug"])
        g.setdefault("conditionId", cond)
        if r.get("listing") is not None and not g.get("acceptingOrdersTimestamp"):
            g["acceptingOrdersTimestamp"] = dt.datetime.fromtimestamp(_ts(r["listing"]), dt.timezone.utc).isoformat()
        g.setdefault("endDate", dt.datetime.fromtimestamp(ko or we, dt.timezone.utc).isoformat())
        m = {"slug": r["slug"], "league": r.get("league") or r["slug"].split("-")[0],
             "smt": r.get("market_type") or g.get("sportsMarketType") or "", "kickoff": ko or we, "ws": ws, "we": we,
             "tokens": toks, "outcomes": list(r.get("outcomes") or [""] * len(toks)), "cond": cond,
             "volume": float(r.get("volume") if r.get("volume") is not None else float("inf")), "g": g}
        out[dt.datetime.fromtimestamp(m["kickoff"], dt.timezone.utc).date().isoformat()].append(m)
    return out


def selection(a, kday: str) -> list[dict]:
    """Markets with kickoff on UTC day kday: from --markets FILE if given, else gamma (cached)."""
    if a.markets:
        if getattr(a, "_mfile", None) is None:
            a._mfile = load_market_file(a.markets, a)
        return a._mfile.get(kday, [])
    p = a.cache / "selection" / f"{kday}.json.gz"
    if p.exists():
        return read_json(p, gz=True)
    ms = bp.select_markets(sel_args(a, kday))
    keep = []
    for m in ms:
        g = m["g"]
        keep.append({k: m[k] for k in ("slug", "league", "smt", "kickoff", "ws", "we", "tokens", "outcomes", "cond",
                                       "volume")} | {"g": g})
    atomic_json(p, keep, gz=True)
    return keep


def targets_for_day(a, day: str) -> list[dict]:
    """Markets whose replay window [listing or ws - track_days, we) may intersect archive day `day`."""
    lo, hi = d0(day), d0(day) + 86400
    if a.markets:                                   # any window length: scan the whole list
        if getattr(a, "_mfile", None) is None:
            a._mfile = load_market_file(a.markets, a)
        return [m for v in a._mfile.values() for m in v
                if m["we"] > lo and m["ws"] - a.track_days * 86400 < hi]
    span = int((a.hours_before * 3600 + a.post_min * 60) // 86400 + a.track_days) + 2
    out = []
    for k in range(-1, span + 1):
        kd = (dt.date.fromisoformat(day) + dt.timedelta(days=k)).isoformat()
        for m in selection(a, kd):
            if m["we"] > lo and m["ws"] - a.track_days * 86400 < hi:
                out.append(m)
    return out


# ------------------------------------------------------------------------------------------- books stage
def book_rows_table(m: dict, rows: list) -> pa.Table:
    cols = bp.new_cols()
    win, _ = bp.resolution(m["g"])
    winner = None if win is None else ("yes" if win == 0 else "no")
    is_win = None if win is None else (win == 0)
    end_us = int(round((bp.parse_ts(m["g"].get("endDate")) or m["kickoff"]) * 1e6))
    for cap, b, a_ in rows:
        bp_, bs = rp.to_lists(b)
        ap, as_ = rp.to_lists(a_)
        cap_us = cap * 1_000_000
        bb = bp_[0] if bp_ else None
        ba = ap[0] if ap else None
        cols["market_slug"].append(m["slug"])
        cols["coin"].append(m["league"])
        cols["contract_length"].append(m["smt"])
        cols["token_id"].append(m["tokens"][0])
        cols["outcome"].append(str(m["outcomes"][0]))
        cols["winner"].append(winner)
        cols["is_winning_outcome"].append(is_win)
        cols["market_end_at"].append(end_us)
        cols["captured_at"].append(cap_us)
        cols["seconds_to_close"].append((end_us - cap_us) / 1e6)
        cols["bid_prices"].append(bp_)
        cols["bid_sizes"].append(bs)
        cols["ask_prices"].append(ap)
        cols["ask_sizes"].append(as_)
        cols["best_bid"].append(bb)
        cols["best_ask"].append(ba)
        cols["crossed"].append(bool(bb is not None and ba is not None and bb >= ba))
    return bp.columns_to_table(cols, BOOK_SCHEMA_RAW)


def rg_hour_stats(sp):
    """Per row group: (hour start, (t_lo, t_hi), (cid_lo, cid_hi)) from the footer statistics."""
    out = []
    for i in range(sp.meta.num_row_groups):
        s = sp.rg_stats(i)
        t = s.get("event_ts")
        out.append((None if t is None else int(t[0]) // 3600 * 3600, t, s.get("condition_id")))
    return out


_W = {}


def _winit(local, toks, we_ms, t0_ms):
    _W.update(pf=pq.ParquetFile(local), toks=pa.array(toks), we=pa.array(we_ms, pa.int64()),
              t0=pa.array(t0_ms, pa.int64()))


def _filter_hour(idx):
    """Worker: rows of the target tokens inside their replay windows, from row groups idx (one archive hour)."""
    pf, toks, we_ms, t0_ms = _W["pf"], _W["toks"], _W["we"], _W["t0"]
    parts, nread, src = [], 0, set()
    for i in idx:
        key = pf.read_row_group(i, columns=["asset_id", "event_ts_ms"])
        nread += key.num_rows
        pos = pc.index_in(key["asset_id"], value_set=toks)
        ok = pc.is_valid(pos)
        if not pc.any(ok).as_py():
            continue
        ts_ = key["event_ts_ms"].cast(pa.int64())
        pz = pc.fill_null(pos, 0)
        ok = pc.and_(ok, pc.and_(pc.less(ts_, pc.take(we_ms, pz)), pc.greater_equal(ts_, pc.take(t0_ms, pz))))
        if not pc.any(ok).as_py():
            continue
        tab = pf.read_row_group(i, columns=COLS + ["source_archive"]).filter(ok)
        src.update(x for x in pc.unique(tab["source_archive"]).to_pylist() if x)
        parts.append(tab.drop_columns(["source_archive", "condition_id"]))
    return (pa.concat_tables(parts) if parts else None), nread, src


def track_window(m: dict, a) -> tuple[int, int, bool]:
    """[t0, we) over which a market's book is replayed, and whether t0 is its listing (book empty at t0)."""
    g = m["g"]
    lst = bp.parse_ts(g.get("acceptingOrdersTimestamp")) or bp.parse_ts(g.get("startDate"))
    floor = m["ws"] - int(a.track_days * 86400)
    if lst is not None and lst >= floor:
        return int(lst), m["we"], True
    return floor, m["we"], False


def run_books_day(a, day: str) -> dict:
    out = a.raw
    done = out / "_state" / "books" / f"{day}.json"
    if done.exists():
        return read_json(done)
    t0 = time.time()
    lo, hi = d0(day), d0(day) + 86400
    if day in ARCHIVE_MISSING or not (ARCHIVE_FIRST <= day <= ARCHIVE_LAST):
        absent_day = True
    else:
        absent_day = False
    ms = [m for m in targets_for_day(a, day)]
    tw = {m["tokens"][0]: track_window(m, a) for m in ms}
    by_tok = {m["tokens"][0]: m for m in ms if tw[m["tokens"][0]][0] < hi}
    prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    sp_ = out / "_state" / "bookstate" / f"{prev}.json.gz"
    carried = read_json(sp_, gz=True) if sp_.exists() else {}
    books = {}
    for tok in by_tok:
        if tok in carried:
            books[tok] = rp.TokenBook.from_state(carried[tok], lo)
        else:
            books[tok] = rp.TokenBook()
            t_start, _, is_listing = tw[tok]
            if is_listing and t_start >= lo and not absent_day:   # listed today: empty book at creation
                books[tok].has_book = True
                books[tok].set_verified(True, t_start)
    if absent_day:
        for tok, tb in books.items():
            tb.set_verified(False, lo)
        st = {"day": day, "status": "absent", "targets": len(by_tok)}
        nxt_state = {tok: books[tok].state() for tok, m in by_tok.items() if m["we"] > hi and books[tok].has_book}
        atomic_json(out / "_state" / "bookstate" / f"{day}.json.gz", nxt_state, gz=True)
        atomic_json(done, st)
        log.info("books %s: archive day absent", day)
        return st
    sp = hfpq.SparseParquet(f"orderbook/date={day}/data_0.parquet", str(a.dl / f"{day}.parquet"), reuse=a.reuse_dl)
    md = sp.meta
    rgs = rg_hour_stats(sp)
    present = sorted({h for h, _, _ in rgs if h is not None and lo <= h < hi})
    win = sorted((by_tok[t]["cond"], tw[t][0], tw[t][1]) for t in by_tok)
    conds_sorted = [w[0] for w in win]
    want = defaultdict(list)
    for i, (h, t, c) in enumerate(rgs):
        if t is None or c is None:
            want[h].append(i)
            continue
        k = bisect.bisect_left(conds_sorted, c[0])
        while k < len(win) and win[k][0] <= c[1]:
            if win[k][1] <= t[1] and win[k][2] > t[0]:
                want[h].append(i)
                break
            k += 1
    allwant = [i for v in want.values() for i in v]
    nreq = sp.fetch(allwant, workers=a.workers)
    t_dl = time.time() - t0
    toks_sorted = sorted(by_tok)
    toks = pa.array(toks_sorted)
    we_ms = pa.array([tw[t][1] * 1000 for t in toks_sorted], pa.int64())
    t0_ms = pa.array([tw[t][0] * 1000 for t in toks_sorted], pa.int64())
    pf = pq.ParquetFile(sp.local)
    sink = PartSink(out, "books", BOOK_SCHEMA_RAW, f"hf-{day}", level=3, sort_by=bp.BOOK_SORT,
                    dictionary_cols=bp.BOOK_DICT)
    nrows, nread, nkept, src, trades_all = 0, 0, 0, set(), []
    rows_by_tok = defaultdict(list)
    nrows_tok = defaultdict(int)

    def flush_rows():
        by_hour = defaultdict(list)
        for tok, rows in rows_by_tok.items():
            if not rows:
                continue
            t = book_rows_table(by_tok[tok], rows)
            hrs = np.asarray([r[0] for r in rows]) // 3600
            for hh in np.unique(hrs):
                by_hour[int(hh)].append(t.take(pa.array(np.flatnonzero(hrs == hh))))
            nrows_tok[tok] += len(rows)
            rows.clear()
        for hh, ts_ in by_hour.items():
            hk = dt.datetime.fromtimestamp(hh * 3600, dt.timezone.utc).strftime("%Y-%m-%dT%H")
            sink.write_part(hk, pa.concat_tables(ts_))
    hours = sorted(set(want) | set(range(lo, hi, 3600)), key=lambda x: (x is None, x))
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(a.procs, initializer=_winit,
                                       initargs=(sp.local, toks_sorted, [tw[t][1] * 1000 for t in toks_sorted],
                                                 [tw[t][0] * 1000 for t in toks_sorted]))
    jobs = [(h, pool.apply_async(_filter_hour, (want.get(h, []),)) if want.get(h) else None) for h in hours]
    for h, job in jobs:
        if h is not None and h not in present and lo <= h < hi:     # archive hour missing: books unknown
            for tb in books.values():
                tb.set_verified(False, h)
            continue
        if job is None:
            continue
        tab, nr, sr = job.get()
        nread += nr
        src |= sr
        if tab is None:
            continue
        nkept += tab.num_rows
        tab = tab.take(pc.sort_indices(tab, [("asset_id", "ascending"), ("event_ts_ms", "ascending")]))
        aid = tab["asset_id"].to_numpy(zero_copy_only=False)
        if len(aid) == 0:
            continue
        cut = np.flatnonzero(aid[1:] != aid[:-1]) + 1
        cols = rp.np_cols(tab.drop_columns(["asset_id"]))
        del tab
        for s0, e0 in zip(np.r_[0, cut], np.r_[cut, len(aid)]):
            tok = aid[s0]
            m = by_tok[tok]
            ev = {c: v[s0:e0] for c, v in cols.items()}
            trades = []
            rp.replay_token(ev, books[tok], max(lo, m["ws"]), min(hi, m["we"]) + 1, rows_by_tok[tok], trades,
                            a.recover_matches)
            trades_all += [(m["slug"],) + tr for tr in trades]
        del cols, aid
        flush_rows()
    flush_rows()
    pool.close()
    pool.join()
    sp.close(remove=not a.reuse_dl)
    t_read = time.time() - t0 - t_dl
    for hk in sink.part_hours():
        sink.compact(hk)
    # coverage: live while the replayed book is verified, inside the emit window [ws, we) of the day
    cov = [{"t": float(lo), "run": f"hf-{day}", "ev": "start"}]
    qc = {}
    for tok, m in by_tok.items():
        tb = books[tok]
        elo, ehi = max(lo, m["ws"]), min(hi, m["we"])
        live = []
        state = tok in carried and bool(carried[tok].get("v"))
        cur = lo if state else None
        for t, v in tb.trans:
            if v and cur is None:
                cur = t
            elif not v and cur is not None:
                live.append((cur, t))
                cur = None
        if cur is not None:
            live.append((cur, hi))
        live = [(max(x0, elo), min(x1, ehi)) for x0, x1 in live if min(x1, ehi) > max(x0, elo)]
        for x0, x1 in live:
            cov.append({"t": float(x0), "run": f"hf-{day}", "ev": "live", "slug": m["slug"]})
            cov.append({"t": float(x1), "run": f"hf-{day}", "ev": "down", "slug": m["slug"]})
        if ehi > elo:
            qc[m["slug"]] = {"n_book": tb.n_book, "pc_checked": tb.pc_checked, "pc_match": tb.pc_match,
                             "ghosts": tb.ghosts, "recovered": tb.recovered, "rows": nrows_tok.get(tok, 0),
                             "live_s": round(sum(x1 - x0 for x0, x1 in live)), "window_s": ehi - elo,
                             "verified_end": tb.verified}
    for h in present:                                  # archive heartbeat: silent over missing archive hours
        for t in range(max(h, lo), min(h + 3600, hi), 10):
            cov.append({"t": float(t), "run": f"hf-{day}", "ev": "hb"})
    cov.append({"t": float(hi), "run": f"hf-{day}", "ev": "stop"})
    cov.sort(key=lambda e: e["t"])
    cp = out / "coverage" / f"coverage_{day}.jsonl"
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w") as f:
        for e in cov:
            f.write(json.dumps(e) + "\n")
    nxt_state = {tok: books[tok].state() for tok, m in by_tok.items() if m["we"] > hi and books[tok].has_book}
    atomic_json(out / "_state" / "bookstate" / f"{day}.json.gz", nxt_state, gz=True)
    if trades_all:
        cols_ = list(zip(*trades_all))
        tt = pa.table({"slug": cols_[0], "ts_ms": cols_[1], "price": cols_[2], "size": cols_[3], "side": cols_[4],
                       "tx": cols_[5]})
        (out / "ltp").mkdir(parents=True, exist_ok=True)
        pq.write_table(tt, out / "ltp" / f"ltp_{day}.parquet", compression="zstd")
    atomic_json(out / "_state" / "qc" / f"{day}.json.gz", qc, gz=True)
    pcc = sum(q["pc_checked"] for q in qc.values())
    pcm = sum(q["pc_match"] for q in qc.values())
    ls = sum(q["live_s"] for q in qc.values())
    ws_ = sum(q["window_s"] for q in qc.values())
    absent = sorted(set(range(lo, hi, 3600)) - set(present))
    st = {"day": day, "status": "ok", "source_archive": sorted(src), "tracked": len(by_tok),
          "in_window": len(qc), "rows": sum(nrows_tok.values()), "rg_wanted": len(allwant), "rg_total": md.num_row_groups,
          "requests": nreq, "mb_fetched": round(sp.bytes_fetched / 1e6, 1), "rows_read": nread, "rows_kept": nkept,
          "hours_absent": [dt.datetime.fromtimestamp(h, dt.timezone.utc).strftime("%H") for h in absent],
          "pc_checked": pcc, "pc_match_frac": round(pcm / pcc, 5) if pcc else None,
          "live_frac": round(ls / ws_, 4) if ws_ else None, "carried_in": len(carried),
          "s_download": round(t_dl, 1), "s_read": round(t_read, 1), "s_total": round(time.time() - t0, 1)}
    atomic_json(done, st)
    log.info("books %s: %s", day, json.dumps(st))
    return st


# ------------------------------------------------------------------------------------------- meta stage
def run_meta_days(a, start: str, end: str) -> dict:
    ms = []
    for kd in days(start, end):
        ms += selection(a, kd)
    ns = argparse.Namespace(out=str(a.raw), trades_full=False, meta_workers=a.meta_workers)
    st = bp.run_meta(ns, ms)
    bp.compact(a.raw, names=("trades", "markets"))
    return st


# ------------------------------------------------------------------------------------------- export
def export_volumes(a, day: str) -> dict:
    vol = {}
    for k in range(-1, int(a.hours_before // 24) + 3):
        kd = (dt.date.fromisoformat(day) + dt.timedelta(days=k)).isoformat()
        if a.markets:
            vol.update({m["slug"]: m["volume"] for m in selection(a, kd)})
            continue
        p = a.cache / "selection" / f"{kd}.json.gz"
        if p.exists():
            vol.update({m["slug"]: m["volume"] for m in read_json(p, gz=True)})
    return vol


def export_day(a, day: str) -> dict:
    od = a.daily / day
    od.mkdir(parents=True, exist_ok=True)
    s = {}
    bf = sorted((a.raw / "books" / day).glob("*.parquet"))
    if bf:
        tabs = []
        for f in bf:
            t = pq.read_table(f)
            for col in ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"):
                i = t.schema.get_field_index(col)
                t = t.set_column(i, col, pc.list_slice(t[col], 0, a.levels))
            tabs.append(t)
        t = pa.concat_tables(tabs)
        del tabs
        vol = export_volumes(a, day)
        keep_slugs = pa.array(sorted(s_ for s_, v in vol.items() if v >= a.export_min_volume))
        t = t.filter(pc.is_in(pc.cast(t["market_slug"], pa.string()), value_set=keep_slugs))
        t = t.take(pc.sort_indices(t, [("market_slug", "ascending"), ("captured_at", "ascending")]))
        # rows whose top --levels ladder equals the previous row of the same market carry no information here
        sig = pc.binary_join_element_wise(
            pc.cast(t["market_slug"], pa.string()),
            *[pc.binary_join(pc.cast(t[c_], pa.list_(pa.string())), ",") for c_ in
              ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes")], "|").to_numpy(zero_copy_only=False)
        t = t.filter(pa.array(np.r_[True, sig[1:] != sig[:-1]] if len(sig) else np.zeros(0, bool)))
        del sig
        t = t.append_column("reconciled", pa.array(np.zeros(t.num_rows, bool)))
        p = od / f"books_{day}.parquet"
        pq.write_table(t, p, compression="zstd", compression_level=19, row_group_size=256 * 1024)
        s["books_rows"] = t.num_rows
        s["books_markets"] = len(pc.unique(t["market_slug"]))
        s["books_mb"] = round(p.stat().st_size / 1e6, 2)
    for kind in ("trades", "markets"):
        files = sorted((a.raw / kind / day).glob("*.parquet"))
        if files:
            t = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")
            p = od / f"{kind}_{day}.parquet"
            pq.write_table(t, p, compression="zstd", compression_level=19)
            s[f"{kind}_rows"] = t.num_rows
            s[f"{kind}_mb"] = round(p.stat().st_size / 1e6, 2)
    cov = a.raw / "coverage" / f"coverage_{day}.jsonl"
    if cov.exists():
        shutil.copy(cov, od / cov.name)
    bs = a.raw / "_state" / "books" / f"{day}.json"
    if bs.exists():
        s["books_stage"] = read_json(bs)
    atomic_json(od / "summary.json", s)
    big = [f.name for f in od.iterdir() if f.stat().st_size > 50e6]
    if big:
        raise SystemExit(f"export {day}: files over 50 MB: {big} (lower --levels or split)")
    return s


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "books", "meta", "export", "run", "status"])
    ap.add_argument("--start", default=ARCHIVE_FIRST)
    ap.add_argument("--end", default=ARCHIVE_LAST)
    ap.add_argument("--raw", type=Path, default=HERE / "raw_hf")
    ap.add_argument("--daily", type=Path, default=HERE / "daily")
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--dl", type=Path, default=HERE / "dl")
    ap.add_argument("--markets", type=Path, help="market list file (JSON / JSONL, see load_market_file) instead of the "
                    "gamma sports selection; use with its own --raw / --daily")
    ap.add_argument("--leagues", default="all")
    ap.add_argument("--market-types", default="moneyline")
    ap.add_argument("--min-volume", type=float, default=10000)
    ap.add_argument("--hours-before", type=float, default=48)
    ap.add_argument("--post-min", type=float, default=5)
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--export-min-volume", type=float, default=50000, help="daily/ export: gamma volume floor")
    ap.add_argument("--track-days", type=float, default=7, help="replay from listing, at most this long before ws")
    ap.add_argument("--recover-matches", type=int, default=rp.RECOVER_MATCHES)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--procs", type=int, default=3, help="row-group filter processes")
    ap.add_argument("--meta-workers", type=int, default=3)
    ap.add_argument("--min-free-gb", type=float, default=12)
    ap.add_argument("--reuse-dl", action="store_true", help="dev: reuse a complete local sparse file, keep it")
    ap.add_argument("--keep-raw", action="store_true", help="run: keep raw_hf books after export")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime
    a.raw.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(a.raw / "convert.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    if a.stage == "select":
        for kd in days(a.start, a.end):
            log.info("select %s: %d markets", kd, len(selection(a, kd)))
    elif a.stage == "books":
        for d in days(a.start, a.end):
            run_books_day(a, d)
    elif a.stage == "meta":
        log.info("meta: %s", run_meta_days(a, a.start, a.end))
    elif a.stage == "export":
        for d in days(a.start, a.end):
            log.info("export %s: %s", d, export_day(a, d))
    elif a.stage == "run":
        for d in days(a.start, a.end):
            free = shutil.disk_usage(a.raw).free / 1e9
            if free < a.min_free_gb:
                raise SystemExit(f"only {free:.1f} GB free (< --min-free-gb {a.min_free_gb}); stopping before {d}")
            st = run_books_day(a, d)
            if st.get("status") != "ok":
                continue
            run_meta_days(a, d, d)
            export_day(a, d)
    elif a.stage == "status":
        for f in sorted((a.raw / "_state" / "books").glob("*.json")):
            s = read_json(f)
            print(f.stem, s.get("status"), s.get("targets"), s.get("rows"), s.get("pc_match_frac"),
                  s.get("hours_absent"))


if __name__ == "__main__":
    main()
