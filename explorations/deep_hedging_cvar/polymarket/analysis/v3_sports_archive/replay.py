"""Book replay of the Joseph3222/polymarket-orderbook raw WS stream (pmxt archive) for a set of tokens.

Semantics (dataset card, = poly-websockets OrderBookCache): a `book` event replaces the token's whole book;
`price_change` sets the level's ABSOLUTE size (0 removes it); `last_trade_price` does not touch the book.
Events are applied in (event_ts_ms, file order).

Touch cleaning: every v2 `price_change` row carries the exchange-emitted best_bid / best_ask of its token after
the change. At the last row of each same-millisecond group, replayed bid levels above the source best bid and
ask levels below the source best ask are ghosts (removals lost in the archive or mis-ordered inside a ms) and
are dropped. Then the replayed touch is compared with the source touch (pc_checked / pc_match). A mismatch
left after cleaning means a level is missing from the replay (its size is unknown).

Anchoring (`verified`): the replayed book is trusted when it descends from a full `book` snapshot, or from the
market's listing (an empty book at creation plus every change since), with no archive hour missing in between.
A missing archive hour clears it; a `book` snapshot, or --recover-matches consecutive touch matches after
cleaning, sets it again. Transitions are logged in `tb.trans` for the coverage log.

Output rows follow the recorder / backfill convention (data/raw updown schema): one row per token and second
in which the book changed; `captured_at` = the END of that second (s + 1), i.e. the row holds the book after
every event with event_ts_ms < (s + 1) * 1000 -- backfill_pob's causal convention (bucket t + resolution).
"""
from __future__ import annotations

import json

PX = 10000                      # prices as int 1e-4
RECOVER_MATCHES = 20


def _q(p) -> int:
    return int(round(float(p) * PX))


class TokenBook:
    __slots__ = ("bids", "asks", "has_book", "verified", "n_book", "pc_checked", "pc_match", "ghosts",
                 "last_emit_sig", "last_event_ms", "tick", "run_match", "trans", "recovered")

    def __init__(self):
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}
        self.has_book = False
        self.verified = False
        self.n_book = 0
        self.pc_checked = 0
        self.pc_match = 0
        self.ghosts = 0
        self.last_emit_sig = None
        self.last_event_ms = 0
        self.tick = None
        self.run_match = 0
        self.trans: list[tuple[float, bool]] = []
        self.recovered = 0

    def set_verified(self, v: bool, t: float):
        if v != self.verified:
            self.verified = v
            self.trans.append((t, v))

    def best(self):
        bb = max(self.bids) if self.bids else None
        ba = min(self.asks) if self.asks else None
        return bb, ba

    def state(self) -> dict:
        return {"b": [[k, v] for k, v in self.bids.items()], "a": [[k, v] for k, v in self.asks.items()],
                "h": self.has_book, "v": self.verified, "t": self.last_event_ms, "k": self.tick}

    @classmethod
    def from_state(cls, d: dict, t0: float) -> "TokenBook":
        tb = cls()
        tb.bids = {int(k): float(v) for k, v in d["b"]}
        tb.asks = {int(k): float(v) for k, v in d["a"]}
        tb.has_book = bool(d["h"])
        tb.last_event_ms = int(d.get("t") or 0)
        tb.tick = d.get("k")
        if d.get("v"):
            tb.set_verified(True, t0)
        return tb


def _levels(js) -> dict[int, float]:
    out = {}
    if not js:
        return out
    for p, s in json.loads(js):
        s = float(s)
        if s > 0:
            out[_q(p)] = s
    return out


def _ok(x) -> bool:
    return x is not None and x == x


def replay_token(ev, tb: TokenBook, emit_lo: int, emit_hi: int, rows: list, trades: list,
                 recover: int = RECOVER_MATCHES):
    """Apply one token's events (dict of numpy columns, time-sorted) to `tb`; emit changed books for
    captured_at = s + 1 in [emit_lo, emit_hi)."""
    ts = ev["event_ts_ms"]
    et = ev["event_type"]
    n = len(ts)
    cur_sec = None

    def flush(sec):
        cap = sec + 1
        if not (emit_lo <= cap < emit_hi) or not tb.has_book:
            return
        b = sorted(tb.bids.items(), key=lambda x: -x[0])
        a = sorted(tb.asks.items())
        sig = (tuple(b), tuple(a))
        if sig == tb.last_emit_sig:
            return
        tb.last_emit_sig = sig
        rows.append((cap, b, a))

    for i in range(n):
        t_ms = int(ts[i])
        sec = t_ms // 1000
        if cur_sec is not None and sec != cur_sec:
            flush(cur_sec)
        cur_sec = sec
        e = et[i]
        tb.last_event_ms = t_ms
        if e == "book":
            tb.bids = _levels(ev["bids_json"][i])
            tb.asks = _levels(ev["asks_json"][i])
            tb.has_book = True
            tb.n_book += 1
            tb.run_match = 0
            tb.set_verified(True, t_ms / 1000)
        elif e == "price_change":
            q = _q(ev["price"][i])
            s = float(ev["size"][i])
            side = tb.bids if ev["side"][i] == "BUY" else tb.asks
            if s > 0:
                side[q] = s
            else:
                side.pop(q, None)
            if i + 1 < n and ts[i + 1] == ts[i] and et[i + 1] == "price_change":
                continue                                    # check at the last row of the same-ms group
            sb, sa = ev["best_bid"][i], ev["best_ask"][i]
            if not (_ok(sb) and _ok(sa)):
                continue
            qb, qa = _q(sb), _q(sa)
            if qb > 0:
                for k in [k for k in tb.bids if k > qb]:
                    del tb.bids[k]
                    tb.ghosts += 1
            if 0 < qa < PX:
                for k in [k for k in tb.asks if k < qa]:
                    del tb.asks[k]
                    tb.ghosts += 1
            if not tb.has_book:
                continue
            bb, ba = tb.best()
            ok_b = (bb == qb) if qb > 0 else (bb is None)
            ok_a = (ba == qa) if 0 < qa < PX else (ba is None)
            tb.pc_checked += 1
            if ok_b and ok_a:
                tb.pc_match += 1
                tb.run_match += 1
                if not tb.verified and tb.run_match >= recover:
                    tb.recovered += 1
                    tb.set_verified(True, t_ms / 1000)
            else:
                tb.run_match = 0
        elif e == "last_trade_price":
            trades.append((t_ms, float(ev["price"][i]), float(ev["size"][i]), ev["side"][i],
                           ev["transaction_hash"][i]))
        elif e == "tick_size_change":
            tb.tick = float(ev["new_tick_size"][i])
    if cur_sec is not None:
        flush(cur_sec)


def to_lists(levels):
    return [q / PX for q, _ in levels], [s for _, s in levels]


def np_cols(tab):
    """pyarrow table -> dict of numpy/object arrays."""
    return {c: tab.column(c).to_numpy(zero_copy_only=False) for c in tab.column_names}
