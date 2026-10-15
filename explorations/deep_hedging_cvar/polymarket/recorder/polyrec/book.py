"""In-memory L2 order book for one token, plus pure helpers (mirror, diff, replay).

Prices and sizes arrive as decimal strings. A price string always parses to the same float, and
"0.5" / "0.50" parse to the same float, so floats are safe dict keys (no rounding needed). The
Up/Down mirror (p -> 1 - p) is the only arithmetic on prices and is rounded to 6 decimals.

`price_change` carries the NEW AGGREGATE size of a level (0 = level removed), so applying a delta
is idempotent. That is what makes the reconcile replay valid: REST snapshot + every buffered delta
with server ts >= the REST book's timestamp reproduces the current book even if some of those
deltas were already in the snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass

PX_DECIMALS = 6


def parse_levels(levels) -> dict:
    """[{'price': '0.5', 'size': '10'}, ...] -> {0.5: 10.0}; zero sizes dropped."""
    out = {}
    for lv in levels or ():
        s = float(lv["size"])
        if s > 0:
            out[float(lv["price"])] = s
    return out


class OrderBook:
    __slots__ = ("bids", "asks", "_bb", "_ba", "initialized", "ever_initialized", "last_ts",
                 "dirty", "reconciled", "rec_buf")

    def __init__(self):
        self.bids: dict = {}
        self.asks: dict = {}
        self._bb = None              # cached best bid / ask (None = side empty)
        self._ba = None
        self.initialized = False     # have a snapshot on the CURRENT connection
        self.ever_initialized = False
        self.last_ts = 0             # server ms of the last applied event
        self.dirty = False           # changed since the last emitted row
        self.reconciled = False      # next row reflects a REST correction
        self.rec_buf = None          # list of buffered events while a reconcile is in flight

    # ------------------------------------------------------------------ queries
    @property
    def best_bid(self):
        return self._bb

    @property
    def best_ask(self):
        return self._ba

    @property
    def crossed(self) -> bool:
        return self._bb is not None and self._ba is not None and self._bb >= self._ba

    def ladders(self, max_levels: int = 0):
        """(bid_px desc, bid_sz, ask_px asc, ask_sz) as lists; best level first."""
        b = sorted(self.bids.items(), reverse=True)
        a = sorted(self.asks.items())
        if max_levels:
            b, a = b[:max_levels], a[:max_levels]
        return [p for p, _ in b], [s for _, s in b], [p for p, _ in a], [s for _, s in a]

    # ------------------------------------------------------------------ updates
    def _recompute_best(self):
        self._bb = max(self.bids) if self.bids else None
        self._ba = min(self.asks) if self.asks else None

    def set_snapshot(self, bids: dict, asks: dict, ts: int = 0) -> bool:
        """Replace the book. Returns True if the content changed (or the book was not yet live)."""
        changed = (not self.ever_initialized) or bids != self.bids or asks != self.asks
        self.bids, self.asks = bids, asks
        self._recompute_best()
        self.initialized = self.ever_initialized = True
        if ts > self.last_ts:
            self.last_ts = ts
        if changed:
            self.dirty = True
        return changed

    def apply_level(self, is_bid: bool, price: float, size: float, ts: int = 0) -> bool:
        """Set the aggregate size at one level (size <= 0 removes it). Returns True if changed."""
        if ts > self.last_ts:
            self.last_ts = ts
        if is_bid:
            side = self.bids
            old = side.get(price)
            if size <= 0:
                if old is None:
                    return False
                del side[price]
                if price == self._bb:
                    self._bb = max(side) if side else None
            else:
                if old == size:
                    return False
                side[price] = size
                if old is None and (self._bb is None or price > self._bb):
                    self._bb = price
        else:
            side = self.asks
            old = side.get(price)
            if size <= 0:
                if old is None:
                    return False
                del side[price]
                if price == self._ba:
                    self._ba = min(side) if side else None
            else:
                if old == size:
                    return False
                side[price] = size
                if old is None and (self._ba is None or price < self._ba):
                    self._ba = price
        self.dirty = True
        return True

    def prune_better_than(self, is_bid: bool, limit) -> list:
        """Remove levels strictly better than `limit` (bids above / asks below; limit None = whole side).

        Used when a price_change reports the exchange's best bid/ask after the change and our book has
        levels beyond it: those were consumed by a trade whose follow-up `book` event has not arrived
        yet. Returns the removed prices."""
        if is_bid:
            gone = [p for p in self.bids if limit is None or p > limit]
            for p in gone:
                del self.bids[p]
        else:
            gone = [p for p in self.asks if limit is None or p < limit]
            for p in gone:
                del self.asks[p]
        if gone:
            self._recompute_best()
            self.dirty = True
        return gone


# ---------------------------------------------------------------------- helpers
def mirror(bids: dict, asks: dict) -> tuple[dict, dict]:
    """Book of the complementary token: its bid p' = 1 - (our ask p), sizes carried."""
    return ({round(1.0 - p, PX_DECIMALS): s for p, s in asks.items()},
            {round(1.0 - p, PX_DECIMALS): s for p, s in bids.items()})


@dataclass
class BookDiff:
    n_bid_diff: int
    n_ask_diff: int
    max_size_diff: float
    best_bid_a: float | None
    best_bid_b: float | None
    best_ask_a: float | None
    best_ask_b: float | None

    @property
    def top_differs(self) -> bool:
        return self.best_bid_a != self.best_bid_b or self.best_ask_a != self.best_ask_b


def _side_diff(a: dict, b: dict, tol: float):
    n, mx = 0, 0.0
    for p in a.keys() | b.keys():
        d = abs(a.get(p, 0.0) - b.get(p, 0.0))
        if d > tol:
            n += 1
            mx = max(mx, d)
    return n, mx


def diff_books(bids_a: dict, asks_a: dict, bids_b: dict, asks_b: dict, tol: float = 1e-6) -> BookDiff | None:
    """None if the books are equal within `tol` (size units), else a summary of the differences."""
    nb, mb = _side_diff(bids_a, bids_b, tol)
    na, ma = _side_diff(asks_a, asks_b, tol)
    if nb == 0 and na == 0:
        return None
    return BookDiff(nb, na, max(mb, ma),
                    max(bids_a) if bids_a else None, max(bids_b) if bids_b else None,
                    min(asks_a) if asks_a else None, min(asks_b) if asks_b else None)


# buffered event kinds for reconcile replay
EV_LEVEL, EV_SNAP = 0, 1


def replay(bids: dict, asks: dict, events, since_ts: int) -> tuple[dict, dict]:
    """Apply buffered stream events with server ts >= since_ts to a copy of (bids, asks).

    events: iterable of (kind, ts, a, b, c): EV_LEVEL -> (is_bid, price, size), EV_SNAP -> (bids, asks, None).
    """
    bids, asks = dict(bids), dict(asks)
    for kind, ts, a, b, c in events:
        if ts < since_ts:
            continue
        if kind == EV_SNAP:
            bids, asks = dict(a), dict(b)
        else:
            side = bids if a else asks
            if c <= 0:
                side.pop(b, None)
            else:
                side[b] = c
    return bids, asks


def parse_best(v, is_bid: bool):
    """best_bid/best_ask strings from price_change items; '0' (bid) / '1' (ask) or '' mean empty side."""
    if v is None or v == "":
        return None
    x = float(v)
    if is_bid:
        return x if x > 0 else None
    return x if 0 < x < 1 else None
