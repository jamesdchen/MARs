#!/usr/bin/env python3
"""Cross-check the Joseph3222 (pmxt) replay against an independent capture: Houroux/polymarket-l2-history
`market_ws` (its own WS connection, local receive times, Jul 22 - Oct 2026). For markets present in both on
one day, both books are replayed and compared every second of a window (by exchange time).

    python -I validate_houroux.py --day 2026-08-06 --hours 12-14 --dl <scratch dir> --out qc/houroux_<day>.json

Reports per market and overall: touch agreement (best bid and best ask equal), top-5 level agreement (price
and size), and the timestamp offset Joseph event_ts_ms - Houroux exchange `timestamp` for price changes
matched on (asset, side, price, size). Downloads are untrusted data (parsed with json / pyarrow only).
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import io
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hfpq  # noqa: E402
import replay as rp  # noqa: E402

CKPT = defaultdict(list)
HOU = "https://huggingface.co/datasets/Houroux/polymarket-l2-history/resolve/main/"
TREE = "https://huggingface.co/api/datasets/Houroux/polymarket-l2-history/tree/main/market_ws/"


def hou_files(day: str) -> list[str]:
    y, m, d = day.split("-")
    out, url = [], f"{TREE}{y}/{m}/{d}"
    while url:
        r = urllib.request.urlopen(url, timeout=60)
        out += [x["path"] for x in json.load(r) if x["type"] == "file"]
        link = r.headers.get("Link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
    return sorted(out)


def hou_events(paths, dl: Path):
    """-> {asset: list of (ts_ms, kind, payload)} in file order; kind book/pc."""
    ev = defaultdict(list)
    cid = {}
    for p in paths:
        loc = dl / Path(p).name
        if not loc.exists():
            loc.write_bytes(urllib.request.urlopen(HOU + p, timeout=300).read())
        for line in gzip.open(loc):
            r = json.loads(line)
            pl = r.get("payload")
            if r.get("record_type") == "rest_book_checkpoint":
                q = r.get("payload") or {}
                if q.get("asset_id"):
                    CKPT[q["asset_id"]].append((int(q["timestamp"]), lv(q.get("bids")), lv(q.get("asks"))))
                continue
            if pl is None:
                continue
            for q in (pl if isinstance(pl, list) else [pl]):
                et = q.get("event_type")
                if et == "book" and q.get("asset_id"):
                    ev[q["asset_id"]].append((int(q.get("timestamp") or r["received_at_ns"] // 10**6), "book", q))
                    cid[q["asset_id"]] = q.get("market")
                elif et == "price_change":
                    for c in q.get("price_changes", []):
                        ev[c["asset_id"]].append((int(q["timestamp"]), "pc", c))
                        cid[c["asset_id"]] = q.get("market")
    return ev, cid


def lv(levels):
    return {rp._q(x["price"]): float(x["size"]) for x in levels or [] if float(x["size"]) > 0}


def hou_states(events, secs):
    """Book state of one asset at each second boundary in secs (state after events with ts < s*1000)."""
    bids, asks, has = {}, {}, False
    out, j = [], 0
    events = sorted(events, key=lambda e: e[0])           # stable: file order inside a ms
    for s in secs:
        while j < len(events) and events[j][0] < s * 1000:
            t, k, q = events[j]
            if k == "book":
                bids, asks, has = lv(q.get("bids")), lv(q.get("asks")), True
            else:
                side = bids if q["side"] == "BUY" else asks
                p, z = rp._q(q["price"]), float(q["size"])
                if z > 0:
                    side[p] = z
                else:
                    side.pop(p, None)
            j += 1
        out.append((dict(bids), dict(asks)) if has else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--hours", default="12-14", help="UTC hour range [a, b) of Houroux files to compare")
    ap.add_argument("--dl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-markets", type=int, default=40)
    a = ap.parse_args()
    h0, h1 = (int(x) for x in a.hours.split("-"))
    a.dl.mkdir(parents=True, exist_ok=True)
    day0 = int(np.datetime64(a.day, "s").astype(np.int64))
    files = [p for p in hou_files(a.day) if h0 <= int(Path(p).name.split("T")[1][:2]) < h1]
    hev, hcid = hou_events(files, a.dl)
    lo, hi = day0 + h0 * 3600, day0 + h1 * 3600
    # assets with a Houroux book snapshot before the compare window starts are anchored
    cand = []
    for asset, ev in hev.items():
        books = [t for t, k, _ in ev if k == "book"]
        if books and len(ev) > 50:
            cand.append((min(books), asset))
    cand.sort()
    assets = [x for _, x in cand[: a.max_markets]]
    print(f"houroux files {len(files)}, assets anchored {len(cand)}, comparing {len(assets)}")
    # Joseph: replay the day's rows of these assets up to hi (from 00:00, carried state unknown -> touch-cleaned)
    sp = hfpq.SparseParquet(f"orderbook/date={a.day}/data_0.parquet", str(a.dl / f"jos_{a.day}.parquet"))
    md = sp.meta
    conds = sorted({hcid[x] for x in assets if hcid.get(x)})
    want = []
    for i in range(md.num_row_groups):
        s = sp.rg_stats(i)
        t, cc = s["event_ts"], s["condition_id"]
        if t[0] >= hi:
            continue
        k = bisect.bisect_left(conds, cc[0])
        if k < len(conds) and conds[k] <= cc[1]:
            want.append(i)
    sp.fetch(want, workers=8)
    pf = pq.ParquetFile(sp.local)
    toks = pa.array(sorted(assets))
    parts = []
    for i in want:
        t = pf.read_row_group(i, columns=["event_ts_ms", "asset_id", "event_type", "bids_json", "asks_json", "price",
                                          "size", "side", "best_bid", "best_ask", "transaction_hash", "new_tick_size"])
        t = t.filter(pc.is_in(t["asset_id"], value_set=toks))
        if t.num_rows:
            parts.append(t)
    sp.close()
    jt = pa.concat_tables(parts)
    jt = jt.filter(pc.less(jt["event_ts_ms"].cast(pa.int64()), hi * 1000))
    jt = jt.take(pc.sort_indices(jt, [("asset_id", "ascending"), ("event_ts_ms", "ascending")]))
    secs = list(range(lo + 60, hi))
    res, offs = {}, []
    tot = Counter()
    for asset in assets:
        sub = jt.filter(pc.equal(jt["asset_id"], asset))
        if sub.num_rows == 0:
            res[asset] = {"joseph_events": 0}
            continue
        ev = rp.np_cols(sub.drop_columns(["asset_id"]))
        tb, rows = rp.TokenBook(), []
        rp.replay_token(ev, tb, 0, 1 << 40, rows, [])
        caps = [r[0] for r in rows]
        hs = hou_states(hev[asset], secs)
        c = Counter()
        for s, h in zip(secs, hs):
            k = bisect.bisect_right(caps, s) - 1
            if h is None or k < 0:
                c["n_skip"] += 1
                continue
            jb, ja = dict(rows[k][1]), dict(rows[k][2])
            hb, ha = h
            c["n"] += 1
            tb_ = (max(jb) if jb else None) == (max(hb) if hb else None)
            ta_ = (min(ja) if ja else None) == (min(ha) if ha else None)
            c["touch"] += tb_ and ta_
            top = lambda d, rev: sorted(d.items(), reverse=rev)[:5]
            c["top5"] += top(jb, True) == top(hb, True) and top(ja, False) == top(ha, False)
            c["touch_size"] += tb_ and ta_ and jb.get(max(jb, default=None)) == hb.get(max(hb, default=None)) and \
                ja.get(min(ja, default=None)) == ha.get(min(ha, default=None))
        # REST checkpoints (exchange snapshots) vs the Joseph replay and vs the Houroux WS replay, 1 s before
        for t_ck, cb, ca in CKPT.get(asset, []):
            s = t_ck // 1000
            if not (lo <= s < hi):
                continue
            k = bisect.bisect_right(caps, s) - 1
            if k < 0:
                continue
            jb, ja = dict(rows[k][1]), dict(rows[k][2])
            top = lambda d, rev, n: sorted(d.items(), reverse=rev)[:n]
            c["ck_n"] += 1
            c["ck_touch"] += top(jb, True, 1) == top(cb, True, 1) and top(ja, False, 1) == top(ca, False, 1)
            c["ck_touch_px"] += (max(jb, default=0) == max(cb, default=0)) and (min(ja, default=0) == min(ca, default=0))
            c["ck_top5"] += top(jb, True, 5) == top(cb, True, 5) and top(ja, False, 5) == top(ca, False, 5)
            c["ck_top5_px"] += [p for p, _ in top(jb, True, 5)] == [p for p, _ in top(cb, True, 5)] and \
                [p for p, _ in top(ja, False, 5)] == [p for p, _ in top(ca, False, 5)]
            c["ck_full"] += jb == cb and ja == ca
            h = hou_states(hev[asset], [s])[0]
            if h is not None:
                c["ckh_n"] += 1
                c["ckh_top5"] += top(h[0], True, 5) == top(cb, True, 5) and top(h[1], False, 5) == top(ca, False, 5)
        # timestamp offsets on matched price changes
        jpc = [(int(ev["event_ts_ms"][i]), ev["side"][i], rp._q(ev["price"][i]), round(float(ev["size"][i]), 4))
               for i in range(len(ev["event_ts_ms"])) if ev["event_type"][i] == "price_change"
               and int(ev["event_ts_ms"][i]) >= lo * 1000]
        hpc = defaultdict(list)
        for t, k, q in hev[asset]:
            if k == "pc":
                hpc[(q["side"], rp._q(q["price"]), round(float(q["size"]), 4))].append(t)
        for t, sd, p, z in jpc[:5000]:
            cand_ = hpc.get((sd, p, z))
            if cand_:
                d = min(cand_, key=lambda x: abs(x - t))
                if abs(d - t) < 60000:
                    offs.append(t - d)
        res[asset] = {"market": hcid.get(asset), "joseph_events": sub.num_rows, "n_book": tb.n_book,
                      "pc_match_frac": round(tb.pc_match / tb.pc_checked, 4) if tb.pc_checked else None,
                      **{k: v for k, v in c.items()},
                      "touch_frac": round(c["touch"] / c["n"], 4) if c["n"] else None,
                      "top5_frac": round(c["top5"] / c["n"], 4) if c["n"] else None}
        tot.update(c)
    o = np.asarray(offs)
    summ = {"day": a.day, "hours": a.hours, "houroux_files": len(files), "assets": len(assets),
            "seconds_compared": tot["n"], "touch_frac": round(tot["touch"] / tot["n"], 4) if tot["n"] else None,
            "touch_and_size_frac": round(tot["touch_size"] / tot["n"], 4) if tot["n"] else None,
            "top5_frac": round(tot["top5"] / tot["n"], 4) if tot["n"] else None,
            "checkpoints": tot["ck_n"],
            **{f"ckpt_{k}_frac": round(tot[f"ck_{k}"] / tot["ck_n"], 4) if tot["ck_n"] else None
               for k in ("touch", "touch_px", "top5", "top5_px", "full")},
            "ckpt_houroux_ws_top5_frac": round(tot["ckh_top5"] / tot["ckh_n"], 4) if tot["ckh_n"] else None,
            "ts_offset_ms_joseph_minus_houroux": {"n": int(len(o)), **({q: float(np.percentile(o, p)) for q, p in
                                                   (("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95))}
                                                  if len(o) else {})}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summ, "per_asset": res}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
