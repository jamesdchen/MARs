"""Sinks, recorder tick/reconcile flow, trades multiset, splits, coverage."""
import json
import sys
import time
from collections import deque
from pathlib import Path

import orjson
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from polyrec import coverage as cov  # noqa: E402
from polyrec import splits as sp  # noqa: E402
from polyrec.config import Config, Source  # noqa: E402
from polyrec.discovery import MarketInfo, market_from_gamma, resolution  # noqa: E402
from polyrec.recorder import Market, Recorder  # noqa: E402
from polyrec.schema import BOOK_SCHEMA, BOOK_SCHEMA_RAW, MARKETS_SCHEMA, TRADES_SCHEMA  # noqa: E402
from polyrec.sinks import RawLog, dataset_files, iter_raw, read_dataset  # noqa: E402
from polyrec.trades import (LegSet, build_trade_rows, rows_consistency, trade_rows_to_table,  # noqa: E402
                            trade_sanity)

DATA_RAW = ROOT.parent / "data" / "raw"
UP, DN = "111", "222"


class FakeShard:
    id = 1
    last_server_ts = 0
    connected = True
    closing = False


def make_recorder(tmp_path, end_in=10_000):
    cfg = Config()
    cfg.out_dir = str(tmp_path)
    cfg.output.raw_enabled = True
    rec = Recorder(cfg, run_id="test")
    now = time.time()
    info = MarketInfo(slug="btc-updown-5m-1", condition_id="0xc", tokens=[UP, DN], outcomes=["Up", "Down"],
                      end=int(now) + end_in, open=int(now) - 10, coin="btc", contract_length="5m", source="t")
    m = Market(info, Source(), now)
    m.state = "subscribed"
    for t in m.tokens:
        t.shard = FakeShard()
        t.book.rec_buf = deque(maxlen=4096)
        rec.tokens[t.id] = t
    rec.markets[info.slug] = m
    return rec, m


def book_ev(asset, bids, asks, ts):
    return {"event_type": "book", "asset_id": asset, "market": "0xc", "timestamp": str(ts), "hash": "h",
            "bids": [{"price": p, "size": s} for p, s in bids], "asks": [{"price": p, "size": s} for p, s in asks]}


def pc_ev(ts, *items):
    return {"event_type": "price_change", "market": "0xc", "timestamp": str(ts),
            "price_changes": [dict(asset_id=a, side=sd, price=p, size=s, hash="x", best_bid=bb, best_ask=ba)
                              for a, sd, p, s, bb, ba in items]}


def feed(rec, t, *events):
    raw = orjson.dumps(list(events) if len(events) > 1 else events[0])
    rec.on_frame(FakeShard(), raw, t)


def rows(rec):
    out = []
    for cols in rec.books.take_all():
        out.append(rec.books.to_table(cols)[1].to_pandas())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------------------- recorder flow
def test_ticks_are_causal_and_on_change(tmp_path):
    rec, m = make_recorder(tmp_path)
    T = int(time.time()) + 100
    rec.next_tick = T
    feed(rec, T - 0.5, book_ev(UP, [("0.48", "10")], [("0.52", "10")], 1), book_ev(DN, [("0.48", "10")], [("0.52", "10")], 1))
    assert m.live
    # an event received at T + 0.2 must not be in the row stamped T
    feed(rec, T + 0.2, pc_ev(2, (UP, "BUY", "0.49", "5", "0.49", "0.52"), (DN, "SELL", "0.51", "5", "0.48", "0.51")))
    rec._flush_tick(T + 1.0)          # closes tick T+1
    rec._flush_tick(T + 2.0)          # nothing changed: no rows
    df = rows(rec)
    assert list(df.captured_at.astype("int64") // 10**6) == [T, T, T + 1, T + 1]
    first_up = df[(df.outcome == "Up")].iloc[0]
    assert first_up.best_bid == 0.48              # the T+0.2 change is not in row T
    second_up = df[(df.outcome == "Up")].iloc[1]
    assert second_up.best_bid == 0.49 and list(second_up.bid_prices) == [0.49, 0.48]
    assert (df.seconds_to_close == (df.market_end_at - df.captured_at).dt.total_seconds()).all()
    assert not df.crossed.any() and not df.reconciled.any()


def test_crossed_row_and_stale_prune(tmp_path):
    rec, m = make_recorder(tmp_path)
    T = int(time.time()) + 100
    rec.next_tick = T
    feed(rec, T - 0.9, book_ev(UP, [("0.48", "10")], [("0.50", "10"), ("0.52", "4")], 1))
    feed(rec, T - 0.8, book_ev(DN, [("0.50", "10")], [("0.52", "10")], 1))
    # a trade consumed the 0.50 ask; the exchange says best ask is now 0.52 while posting a bid at 0.50
    feed(rec, T - 0.5, pc_ev(5, (UP, "BUY", "0.50", "3", "0.5", "0.52")))
    up = rec.tokens[UP].book
    assert up.best_ask == 0.52 and 0.50 not in up.asks           # pruned, not crossed
    assert rec.st["bba_pruned_levels"] == 1 and rec.st["bba_missing"] == 0
    rec.cfg.reconcile.prune_on_bba = False
    feed(rec, T - 0.4, book_ev(UP, [("0.48", "10")], [("0.50", "10"), ("0.52", "4")], 6))
    feed(rec, T - 0.3, pc_ev(7, (UP, "BUY", "0.50", "3", "0.5", "0.52")))
    rec._flush_tick(T)
    df = rows(rec)
    r = df[df.outcome == "Up"].iloc[-1]
    assert r.crossed and r.best_bid == 0.5 and r.best_ask == 0.5


def test_reconcile_mismatch_replaces_book_and_flags_row(tmp_path):
    rec, m = make_recorder(tmp_path)
    T = int(time.time()) + 100
    rec.next_tick = T
    feed(rec, T - 0.9, book_ev(UP, [("0.48", "10")], [("0.52", "10")], 100))
    feed(rec, T - 0.9, book_ev(DN, [("0.48", "10")], [("0.52", "10")], 100))
    tok = rec.tokens[UP]
    # stream event after the REST snapshot (ts 300 > rest ts 200): replayed, so no false mismatch
    feed(rec, T - 0.5, pc_ev(300, (UP, "SELL", "0.55", "2", "0.48", "0.52")))
    rest = {"asset_id": UP, "timestamp": "200", "bids": [{"price": "0.48", "size": "10"}],
            "asks": [{"price": "0.52", "size": "10"}]}
    rec._compare(tok, rest, 5.0)
    assert rec.st["reconcile"] == 1 and rec.st["reconcile_mismatch"] == 0
    # REST shows a level the stream never delivered (dropped message) -> mismatch, corrected, flagged
    rest2 = dict(rest, timestamp="350", bids=[{"price": "0.48", "size": "10"}, {"price": "0.47", "size": "8"}],
                 asks=[{"price": "0.52", "size": "10"}, {"price": "0.55", "size": "2"}])
    rec._compare(tok, rest2, 5.0)
    assert rec.st["reconcile_mismatch"] == 1
    assert tok.book.bids == {0.48: 10.0, 0.47: 8.0} and tok.book.asks == {0.52: 10.0, 0.55: 2.0}
    rec._flush_tick(T)
    df = rows(rec)
    up = df[df.outcome == "Up"].iloc[-1]
    assert up.reconciled and list(up.bid_prices) == [0.48, 0.47]
    rec.flush_sync = None
    items = rec.recon_sink.take()
    st = [r["status"] for _, rs in items for r in rs]
    assert st == ["ok", "mismatch"]


def test_disconnect_marks_down_and_reinit(tmp_path):
    rec, m = make_recorder(tmp_path)
    T = int(time.time()) + 100
    rec.next_tick = T
    feed(rec, T - 0.9, book_ev(UP, [("0.48", "10")], [("0.52", "10")], 1), book_ev(DN, [("0.48", "10")], [("0.52", "10")], 1))
    assert m.live
    rec._flush_tick(T)

    class S(FakeShard):
        assets = {UP, DN}
    rec.on_disconnect(S(), "closed 1013 slow consumer", True)
    assert not m.live and not rec.tokens[UP].book.initialized
    feed(rec, T + 0.5, pc_ev(5, (UP, "BUY", "0.49", "5", "0.49", "0.52")))
    rec._flush_tick(T + 1)                                         # not live: no rows
    feed(rec, T + 1.5, book_ev(UP, [("0.48", "10")], [("0.52", "10")], 9), book_ev(DN, [("0.48", "10")], [("0.52", "10")], 9))
    assert m.live
    rec._flush_tick(T + 2)                                         # identical re-dump: still no new rows
    df = rows(rec)
    assert list(df.captured_at.astype("int64") // 10**6) == [T, T] and (df.best_bid == 0.48).all()
    feed(rec, T + 2.5, book_ev(UP, [("0.47", "10")], [("0.52", "10")], 10))
    rec._flush_tick(T + 3)
    assert list(rows(rec).best_bid) == [0.47]
    rec.cov.close()
    ev = [json.loads(x)["ev"] for f in (tmp_path / "coverage").glob("*.jsonl") for x in f.read_text().splitlines()]
    assert ev == ["live", "down", "live"]


def test_rows_stop_after_market_end(tmp_path):
    rec, m = make_recorder(tmp_path, end_in=0)
    T = m.info.end + 1
    rec.next_tick = T
    feed(rec, T - 0.5, book_ev(UP, [("0.48", "10")], [("0.52", "10")], 1))
    rec._flush_tick(T)
    assert rows(rec).empty


# ----------------------------------------------------------------------------- sinks
def test_book_parts_schema_compaction_and_reader(tmp_path):
    rec, m = make_recorder(tmp_path)
    T = int(time.time()) // 3600 * 3600 - 7200 + 3595    # crosses an hour boundary, in the past
    rec.next_tick = T
    for k in range(10):
        feed(rec, T + k - 0.5, book_ev(UP, [("0.4", str(10 + k))], [("0.6", "1")], k),
             book_ev(DN, [("0.4", "1")], [("0.6", str(10 + k))], k))
        rec._flush_tick(T + k)
    for cols in rec.books.take_all():
        rec.books.write_cols(cols)
    parts = dataset_files(tmp_path, "books")
    assert len(parts) == 2                                 # one part per hour
    for p in parts:
        assert pq.read_schema(p).remove_metadata().equals(BOOK_SCHEMA)
    for h in rec.books.part_hours():
        rec.books.compact(h)
        rec.books.compact(h)                              # idempotent
    files = dataset_files(tmp_path, "books")
    assert len(files) == 2 and all(f.parent.name != "_parts" for f in files)
    t = read_dataset(tmp_path, "books")
    assert t.num_rows == 20 and t.schema.equals(BOOK_SCHEMA)
    df = pd.read_parquet(files[0])
    assert list(df.columns)[:17] == BOOK_SCHEMA_RAW.names
    if (DATA_RAW / "updown_5m.parquet").exists():
        ref = pq.read_schema(DATA_RAW / "updown_5m.parquet")
        got = pq.read_schema(files[0])
        assert [(f.name, str(f.type)) for f in got][:17] == [(f.name, str(f.type)) for f in ref]


def test_compaction_never_duplicates_after_crash(tmp_path):
    rec, m = make_recorder(tmp_path)
    T = int(time.time()) // 3600 * 3600 - 3600
    rec.next_tick = T
    feed(rec, T - 0.5, book_ev(UP, [("0.4", "1")], [("0.6", "1")], 1), book_ev(DN, [("0.4", "1")], [("0.6", "1")], 1))
    rec._flush_tick(T)
    for cols in rec.books.take_all():
        rec.books.write_cols(cols)
    part = dataset_files(tmp_path, "books")[0]
    saved = part.read_bytes()
    h = rec.books.part_hours()[0]
    rec.books.compact(h)
    part.parent.mkdir(exist_ok=True)
    part.write_bytes(saved)                              # crash between final rename and part delete
    assert read_dataset(tmp_path, "books").num_rows == 2  # reader ignores the absorbed part
    rec.books.compact(h)
    assert not part.exists() and read_dataset(tmp_path, "books").num_rows == 2


def test_raw_log_roundtrip_and_truncated_tail(tmp_path):
    rl = RawLog(tmp_path, "r1", level=3)
    t = time.time()
    for k in range(1000):
        rl.write(orjson.dumps({"k": k, "event_type": "price_change"}), t)
    rl.flush()                                            # block flushed: decodable prefix
    path = rl.path
    with open(path, "rb") as f:
        data = f.read()
    for k in range(1000, 3000):
        rl.write(orjson.dumps({"k": k, "event_type": "book", "pad": "x" * (k % 50)}), t)
    rl.flush()
    with open(path, "rb") as f:
        data2 = f.read()
    rl.close()
    assert [e["k"] for e in iter_raw(path)] == list(range(3000))
    torn = tmp_path / "torn.jsonl.zst"          # hard crash: second block only partly on disk
    torn.write_bytes(data2[: len(data) + (len(data2) - len(data)) // 2])
    assert [e["k"] for e in iter_raw(torn)] == list(range(1000))


# ----------------------------------------------------------------------------- trades
def leg(tx, wallet, side, oi, px, sz, ts=100, asset="a"):
    return {"transactionHash": tx, "proxyWallet": wallet, "side": side, "outcomeIndex": oi,
            "price": px, "size": sz, "timestamp": ts, "asset": asset}


def test_legset_max_merge_and_trade_rows():
    tx = "0x" + "ab" * 32
    taker = [leg(tx, "T", "BUY", 1, 0.40, 10)]                    # taker buys Down @0.40
    allf = taker + [leg(tx, "M1", "BUY", 0, 0.60, 6), leg(tx, "M2", "BUY", 0, 0.60, 4)]   # makers buy Up (mint)
    a, b = LegSet(), LegSet()
    a.merge(allf)
    a.merge(allf)                                                 # overlapping poll: no double count
    a.merge(allf[:2])
    assert len(a) == 3
    b.merge(taker)
    dup = LegSet()                                                # identical legs inside one tx are kept
    dup.merge([leg(tx, "M", "SELL", 0, 0.5, 1)] * 2)
    dup.merge([leg(tx, "M", "SELL", 0, 0.5, 1)])
    assert len(dup) == 2
    rows = build_trade_rows("s", b, a)
    t = trade_rows_to_table(rows)
    assert t.schema.equals(TRADES_SCHEMA)
    df = t.to_pandas()
    assert df.is_taker.sum() == 1 and len(df) == 3
    assert (df.aggressor_up == -1).all()                          # buying Down = selling Up
    assert df.iloc[0].is_taker                                    # taker leg first within the tx
    s = trade_sanity(df)["s"]
    assert s["ok"] and s["max_abs_maker_sum_minus_taker"] < 1e-6
    if (DATA_RAW / "trades.parquet").exists():
        ref = pq.read_schema(DATA_RAW / "trades.parquet")
        assert [(f.name, str(f.type)) for f in ref] == [(f.name, str(f.type)) for f in t.schema]


def test_trade_sanity_flags_missing_maker_legs():
    tx = "0x" + "cd" * 32
    a, b = LegSet(), LegSet()
    b.merge([leg(tx, "T", "SELL", 0, 0.5, 10)])
    a.merge([leg(tx, "T", "SELL", 0, 0.5, 10), leg(tx, "M", "BUY", 0, 0.5, 4)])
    rows = build_trade_rows("s", b, a)
    df = trade_rows_to_table(rows).to_pandas()
    assert not trade_sanity(df)["s"]["ok"]
    assert not rows_consistency(rows)["ok"]                         # maker sum 4 != taker 10
    b2 = LegSet()
    b2.merge([leg(tx, "T", "SELL", 0, 0.5, 10), leg("0x" + "ee" * 32, "T", "BUY", 0, 0.5, 3)])
    rows2 = build_trade_rows("s", b2, a)
    c = rows_consistency(rows2)
    assert not c["ok"] and c["txs_without_maker_legs"] == 1        # taker indexed, maker legs not yet
    a.merge([leg(tx, "M2", "BUY", 0, 0.5, 6)])
    assert rows_consistency(build_trade_rows("s", b, a))["ok"]


# ----------------------------------------------------------------------------- discovery / metadata
def test_market_from_gamma_and_resolution():
    import re
    g = {"slug": "eth-updown-15m-1792048500", "conditionId": "0xc", "clobTokenIds": '["1", "2"]',
         "outcomes": '["Up", "Down"]', "endDate": "2026-10-15T07:30:00Z", "eventStartTime": "2026-10-15T07:15:00Z"}
    src = Source()
    mi = market_from_gamma(g, {"eventMetadata": {"priceToBeat": 1.0}}, src, re.compile(src.slug_regex))
    assert mi.coin == "eth" and mi.contract_length == "15m" and mi.end - mi.open == 900 and mi.tokens == ["1", "2"]
    assert market_from_gamma(dict(g, slug="will-x-happen"), None, src, re.compile(src.slug_regex)) is None
    assert resolution({"outcomePrices": '["1", "0"]', "closed": True}) == (0, 1.0)
    assert resolution({"outcomePrices": '["0", "1"]', "closed": True}) == (1, 0.0)
    assert resolution({"outcomePrices": '["0.505", "0.495"]', "closed": False})[0] is None


def test_markets_schema_matches_data_raw():
    if not (DATA_RAW / "markets.parquet").exists():
        pytest.skip("data/raw not present")
    ref = pq.read_schema(DATA_RAW / "markets.parquet")
    assert [(f.name, str(f.type)) for f in ref] == [(f.name, str(f.type)) for f in MARKETS_SCHEMA]


# ----------------------------------------------------------------------------- splits / coverage
BUILD_DAYS = {"2026-08-20": 0, "2026-08-21": 0, "2026-08-22": 0, "2026-08-23": 1, "2026-08-24": 2}
FROZEN = {"2026-08-20": "train", "2026-08-21": "train", "2026-08-22": "train", "2026-08-23": "val", "2026-08-24": "test"}


def test_splits_deterministic_and_never_reshuffle():
    cfg = {"frozen": FROZEN, "default": "auto",
           "auto": {"anchor": "2026-10-15", "cycle": ["train"] * 6 + ["val", "test"]},
           "days": {"2026-10-30": "exclude", "2026-11-01..2026-11-02": "val"}}
    days = [f"2026-10-{d}" for d in range(15, 31)]
    a = sp.resolve(days, cfg, BUILD_DAYS, {})
    assert a["2026-08-24"] == "test" and a["2026-10-15"] == "train" and a["2026-10-21"] == "val"
    assert a["2026-10-22"] == "test" and a["2026-10-30"] == "exclude"
    b = sp.resolve(days + ["2026-11-01", "2026-11-05"], cfg, BUILD_DAYS, a)  # more days: old ones unchanged
    assert all(b[d] == a[d] for d in a) and b["2026-11-01"] == "val"
    with pytest.raises(sp.SplitError):                          # frozen must match data/build.py
        sp.resolve(days, dict(cfg, frozen=dict(FROZEN, **{"2026-08-24": "train"})), BUILD_DAYS, {})
    with pytest.raises(sp.SplitError):                          # a locked test day can never move
        sp.resolve(days, dict(cfg, days={"2026-10-22": "train"}), BUILD_DAYS, a, allow_reassign=True)
    with pytest.raises(sp.SplitError):                          # other locked days need --reassign
        sp.resolve(days, dict(cfg, days={"2026-10-16": "val"}), BUILD_DAYS, a)
    c = sp.resolve(days, dict(cfg, days={"2026-10-16": "val"}), BUILD_DAYS, a, allow_reassign=True)
    assert c["2026-10-16"] == "val"
    assert sp.to_build_split_days(a)["2026-08-24"] == 2 and "2026-10-30" not in sp.to_build_split_days(a)


def test_split_config_file_matches_build():
    cfg = json.loads((ROOT / "split_config.json").read_text())
    sys.path.insert(0, str(ROOT.parent / "data"))
    import build
    sp.resolve(["2026-10-15"], cfg, build.SPLIT_DAYS, {})


def test_coverage_intervals():
    ev = [{"t": 0, "run": "a", "ev": "start"}, {"t": 10, "run": "a", "ev": "live", "slug": "m"},
          {"t": 50, "run": "a", "ev": "down", "slug": "m"}, {"t": 52, "run": "a", "ev": "live", "slug": "m"},
          {"t": 100, "run": "a", "ev": "hb"},                       # run a crashes after t=100
          {"t": 105, "run": "b", "ev": "live", "slug": "m"}, {"t": 400, "run": "b", "ev": "stop"}]
    iv = cov.live_intervals(ev)["m"]
    assert iv == [(10, 50), (52, 100), (105, 400)]
    ok, tot, mx = cov.market_coverage(iv, 20, 300, max_gap_s=5, max_total_gap_s=15)
    assert ok and tot == 7 and mx == 5
    ok, *_ = cov.market_coverage(iv, 0, 300)                       # not live at open (10 s gap)
    assert not ok


# ----------------------------------------------------------------------------- backfill
def test_backfill_fills_winner_and_keeps_manifest(tmp_path):
    sys.path.insert(0, str(ROOT))
    from backfill_winners import fill_book_file
    rec, m = make_recorder(tmp_path)
    T = int(time.time()) // 3600 * 3600 - 3600
    rec.next_tick = T
    feed(rec, T - 0.5, book_ev(UP, [("0.4", "1")], [("0.6", "1")], 1), book_ev(DN, [("0.4", "1")], [("0.6", "1")], 1))
    rec._flush_tick(T)
    for cols in rec.books.take_all():
        rec.books.write_cols(cols)
    f = rec.books.compact(rec.books.part_hours()[0])
    md = pq.read_schema(f).metadata
    n = fill_book_file(f, {"btc-updown-5m-1": {"winner_idx": 1, "winning_token": DN},
                           "other": {"winner_idx": 0, "winning_token": "x"}})
    assert n == 2
    df = pd.read_parquet(f)
    assert (df.winner == "no").all()
    assert df.set_index("outcome").is_winning_outcome.to_dict() == {"Up": False, "Down": True}
    assert pq.read_schema(f).metadata == md and pq.read_schema(f).remove_metadata().equals(BOOK_SCHEMA)
    assert fill_book_file(f, {"btc-updown-5m-1": {"winner_idx": 0, "winning_token": UP}}) == 0  # never overwrites
