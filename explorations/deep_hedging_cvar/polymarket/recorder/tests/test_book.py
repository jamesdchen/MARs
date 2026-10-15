"""Book maintenance: snapshots, deltas, best-price cache, crossed detection, Up/Down mirror, reconcile replay."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polyrec.book import (EV_LEVEL, EV_SNAP, OrderBook, diff_books, mirror, parse_best,  # noqa: E402
                          parse_levels, replay)


def lv(*pairs):
    return [{"price": p, "size": s} for p, s in pairs]


def test_snapshot_ordering_and_parse():
    # the WS sends bids ascending (worst first) and asks descending; ladders must be best first
    b = OrderBook()
    assert b.set_snapshot(parse_levels(lv(("0.01", "100"), ("0.45", "10"), ("0.47", "5.5"))),
                          parse_levels(lv(("0.99", "7"), ("0.52", "3"), ("0.50", "1"))), ts=10)
    bp, bs, ap, asz = b.ladders()
    assert bp == [0.47, 0.45, 0.01] and bs == [5.5, 10.0, 100.0]
    assert ap == [0.5, 0.52, 0.99] and asz == [1.0, 3.0, 7.0]
    assert b.best_bid == 0.47 and b.best_ask == 0.5 and not b.crossed
    assert b.initialized and b.dirty and b.last_ts == 10


def test_zero_size_levels_dropped_and_string_forms_equal():
    d = parse_levels(lv(("0.50", "0"), ("0.5", "3"), ("0.4", "0.00")))
    assert d == {0.5: 3.0}
    assert float("0.50") == float("0.5")


def test_deltas_add_update_remove():
    b = OrderBook()
    b.set_snapshot({0.4: 10.0}, {0.6: 10.0})
    b.dirty = False
    assert b.apply_level(True, 0.45, 5.0)            # new best bid
    assert b.best_bid == 0.45 and b.dirty
    assert not b.apply_level(True, 0.45, 5.0)        # same size: no change
    assert b.apply_level(True, 0.45, 7.0)            # size update
    assert b.bids[0.45] == 7.0
    assert b.apply_level(True, 0.45, 0.0)            # remove best -> falls back
    assert b.best_bid == 0.4
    assert not b.apply_level(True, 0.33, 0.0)        # removing a missing level is a no-op
    assert b.apply_level(False, 0.55, 2.0)
    assert b.best_ask == 0.55
    b.apply_level(False, 0.55, 0)
    b.apply_level(False, 0.6, 0)
    assert b.best_ask is None and b.asks == {}


def test_best_cache_matches_bruteforce_randomized():
    rng = random.Random(7)
    b = OrderBook()
    b.set_snapshot({}, {})
    for _ in range(20000):
        is_bid = rng.random() < 0.5
        p = round(rng.randint(1, 99) / 100, 2)
        s = 0.0 if rng.random() < 0.4 else round(rng.uniform(1, 500), 2)
        b.apply_level(is_bid, p, s)
        assert b.best_bid == (max(b.bids) if b.bids else None)
        assert b.best_ask == (min(b.asks) if b.asks else None)
        assert all(v > 0 for v in b.bids.values()) and all(v > 0 for v in b.asks.values())


def test_crossed_detection():
    b = OrderBook()
    b.set_snapshot({0.5: 1.0}, {0.51: 1.0})
    assert not b.crossed
    b.apply_level(True, 0.51, 1.0)                    # bid == ask -> crossed (>=, as in the dataset)
    assert b.crossed
    b.apply_level(True, 0.51, 0)
    b.apply_level(True, 0.6, 1.0)                     # bid > ask
    assert b.crossed
    b.set_snapshot({0.6: 1.0}, {})                    # one-sided books are never crossed
    assert not b.crossed and b.best_ask is None
    b.set_snapshot({}, {})
    assert not b.crossed and b.best_bid is None


def test_snapshot_change_detection():
    b = OrderBook()
    assert b.set_snapshot({0.4: 1.0}, {0.6: 1.0})     # first snapshot always counts as a change
    b.dirty = False
    assert not b.set_snapshot({0.4: 1.0}, {0.6: 1.0})  # identical re-dump (e.g. after reconnect)
    assert not b.dirty
    b.initialized = False                              # disconnect
    assert not b.set_snapshot({0.4: 1.0}, {0.6: 1.0})
    assert b.initialized and not b.dirty
    assert b.set_snapshot({0.4: 2.0}, {0.6: 1.0})
    assert b.dirty


def test_mirror_up_down():
    up_b, up_a = {0.94: 1573.6, 0.93: 10.0}, {0.95: 20.0, 0.97: 5.0}
    dn_b, dn_a = mirror(up_b, up_a)
    # Down bid = 1 - Up ask, Down ask = 1 - Up bid, sizes carried (no float noise in keys)
    assert dn_b == {0.05: 20.0, 0.03: 5.0}
    assert dn_a == {0.06: 1573.6, 0.07: 10.0}
    assert mirror(dn_b, dn_a) == (up_b, up_a)          # involution
    assert diff_books(up_b, up_a, *mirror(dn_b, dn_a)) is None
    # the exchange's own message: Up BUY 0.94 and Down SELL 0.06 with the same size
    up, dn = OrderBook(), OrderBook()
    up.set_snapshot(dict(up_b), dict(up_a))
    dn.set_snapshot(*mirror(up_b, up_a))
    up.apply_level(True, float("0.94"), float("1600"))
    dn.apply_level(False, float("0.06"), float("1600"))
    assert diff_books(up.bids, up.asks, *mirror(dn.bids, dn.asks)) is None
    assert up.best_bid == 1 - dn.best_ask or abs(up.best_bid - (1 - dn.best_ask)) < 1e-12


def test_mirror_detects_stale_side():
    up = ({0.5: 1.0}, {0.52: 2.0})
    dn = mirror(*up)
    dn[1][0.5] = 9.0                                   # Down ask 0.5 <-> Up bid 0.5 size differs
    d = diff_books(*up, *mirror(*dn))
    assert d is not None and d.n_bid_diff == 1 and d.max_size_diff == pytest.approx(8.0)


def test_diff_books_summary():
    a = ({0.5: 1.0, 0.4: 2.0}, {0.6: 1.0})
    b = ({0.5: 1.0}, {0.6: 1.0, 0.7: 3.0})
    d = diff_books(*a, *b)
    assert d.n_bid_diff == 1 and d.n_ask_diff == 1 and d.max_size_diff == 3.0
    assert not d.top_differs
    assert diff_books(*a, {0.5: 1.0 + 1e-9, 0.4: 2.0}, {0.6: 1.0}) is None   # tolerance


def test_reconcile_replay_handles_race_and_detects_drop():
    """REST snapshot taken at ts=100 (its timestamp = last change); stream continues."""
    events = []
    live = OrderBook()
    live.set_snapshot({0.4: 10.0}, {0.6: 10.0}, ts=50)
    rest = ({0.4: 10.0, 0.45: 5.0}, {0.6: 10.0})        # REST already has the ts=100 change
    for ev in [(EV_LEVEL, 100, True, 0.45, 5.0),        # the change REST includes (ts == rest_ts)
               (EV_LEVEL, 120, False, 0.58, 3.0),       # after the REST snapshot
               (EV_LEVEL, 130, True, 0.4, 0.0)]:
        live.apply_level(ev[2], ev[3], ev[4], ev[1])
        events.append(ev)
    cand = replay(*rest, events, since_ts=100)
    assert diff_books(live.bids, live.asks, *cand) is None   # race, not a mismatch (idempotent replay)
    # now the stream dropped the ts=100 delta: the live book lacks level 0.45 -> genuine mismatch
    live2 = OrderBook()
    live2.set_snapshot({0.4: 10.0}, {0.6: 10.0}, ts=50)
    for ev in events[1:]:
        live2.apply_level(ev[2], ev[3], ev[4], ev[1])
    cand2 = replay(*rest, events[1:], since_ts=100)
    d = diff_books(live2.bids, live2.asks, *cand2)
    assert d is not None and d.n_bid_diff == 1 and d.top_differs
    # correction: replacing the live book with the candidate fixes it
    assert live2.set_snapshot(*cand2)
    assert diff_books(live.bids, live.asks, live2.bids, live2.asks) is None


def test_replay_snapshot_event_resets():
    events = [(EV_LEVEL, 10, True, 0.3, 1.0), (EV_SNAP, 20, {0.5: 2.0}, {0.7: 2.0}, None),
              (EV_LEVEL, 30, False, 0.65, 4.0)]
    b, a = replay({0.1: 1.0}, {0.9: 1.0}, events, since_ts=15)
    assert b == {0.5: 2.0} and a == {0.7: 2.0, 0.65: 4.0}


def test_parse_best_sentinels():
    assert parse_best("0", True) is None and parse_best("0.45", True) == 0.45
    assert parse_best("1", False) is None and parse_best("0.55", False) == 0.55
    assert parse_best("", True) is None and parse_best(None, False) is None
