"""Order-book replay of the Marketlens `polymarket-btc-5m-l2-depth` dataset (used by build_jun15.py).

The dataset has one book per market in the YES (= Up) frame: Down-token events are already folded
in as 1 - price with the side flipped, and `side` describes the resting order (BUY = bid). A
snapshot is a full book; each delta (`snapshot_id` = the chain it hangs off) REPLACES the size at
one level (0 removes it). Times: snapshots carry `collected_at_ms` (receive time), deltas and
trades `platform_timestamp_ms` (exchange time); both are epoch ms.

State at an evaluation time T (ms):
    k     = the latest snapshot with collected_at_ms <= T   (re-anchor on every snapshot, as the
            dataset README prescribes, so replay errors never accumulate past one snapshot)
    book  = snapshot k, then every delta of chain k with platform_timestamp_ms <= T, in time
            order (ties in file order)
Optional ghost cleaning (`trades` given): not every level removal arrives as a delta, so a level
emptied by a trade can survive in the replay. A trade with taker side BUY at price p at time tau
proves that no ask below p rested at tau (price priority), and a SELL at p that no bid above p
did; such a level is hidden from tau on, until a later delta for it arrives. A trade's timestamp
lags the deltas of its own fills by a few ms (median 3 ms, 95% within 27 ms on this file), so only
levels last updated more than GUARD_MS before tau are hidden: an order posted in that window is never
hidden by mistake. Only trades with timestamp <= T are used, so the cleaned state is still causal.

Prices are handled as integers in 1e-4 (the dataset's exact precision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NLEV = 5
PX = 10000            # price units per 1.0
GUARD_MS = 50         # trade-implied removals apply to levels last updated before tau - GUARD_MS


def snapshot_levels(levels) -> tuple[np.ndarray, np.ndarray]:
    """list[struct{price,size}] -> (q int64 ascending, size float64), zero sizes dropped."""
    if levels is None or len(levels) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    q = np.rint(np.array([x["price"] for x in levels], np.float64) * PX).astype(np.int64)
    s = np.array([x["size"] for x in levels], np.float64)
    keep = s > 0
    q, s = q[keep], s[keep]
    o = np.argsort(q, kind="stable")
    q, s = q[o], s[o]
    assert (np.diff(q) > 0).all(), "duplicate price level in a snapshot"
    return q, s


def _last_per_key(row, col, ncol):
    """Indices of the last occurrence of each (row, col) pair (inputs in application order)."""
    key = row * ncol + col
    _, first_rev = np.unique(key[::-1], return_index=True)
    return len(key) - 1 - first_rev


def _ffill_rows(M, init, init_mask=None):
    """Forward fill a [rows, cols] matrix with NaN holes along rows; row 0 holes take `init`."""
    M = M.copy()
    h = np.isnan(M[0])
    M[0, h] = init[h]
    idx = np.where(~np.isnan(M), np.arange(M.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return M[idx, np.arange(M.shape[1])[None, :]]


def side_matrix(q0, s0, t_snap, dq, ds, dts, B, trades_tau=None, trades_px=None, is_bid=True):
    """Sizes and last-update times of one side at evaluation times B (ms, ascending) for one chain.

    q0, s0: snapshot levels; t_snap: snapshot collected_at_ms; dq, ds, dts: this side's deltas of
    the chain in application order. Returns (levels q ascending, sizes [len(B), nlev], hidden mask).
    """
    L = np.union1d(q0, dq).astype(np.int64)
    n, m = len(B), len(L)
    sizes = np.zeros((n, m))
    upd = np.full((n, m), -np.inf)
    if m == 0:
        return L, sizes, np.zeros((n, 0), bool)
    init_s = np.zeros(m)
    init_s[np.searchsorted(L, q0)] = s0
    init_t = np.full(m, -np.inf)
    init_t[np.searchsorted(L, q0)] = float(t_snap)
    rd = np.searchsorted(B, dts, side="left")          # first evaluation time >= delta time
    ok = rd < n
    rd, col = rd[ok], np.searchsorted(L, dq[ok])
    V = np.full((n, m), np.nan)
    U = np.full((n, m), np.nan)
    if len(rd):
        last = _last_per_key(rd, col, m)
        V[rd[last], col[last]] = ds[ok][last]
        U[rd[last], col[last]] = dts[ok][last].astype(np.float64)
    sizes = _ffill_rows(V, init_s)
    upd = _ffill_rows(U, init_t)
    hidden = np.zeros((n, m), bool)
    if trades_tau is not None and len(trades_tau):
        R = np.full((n, m), -np.inf)
        rt = np.searchsorted(B, trades_tau, side="left")
        for j in np.flatnonzero(rt < n):
            if is_bid:      # SELL at p: no bid above p rested at tau
                c0 = np.searchsorted(L, trades_px[j], side="right")
                R[rt[j], c0:] = trades_tau[j] - GUARD_MS
            else:           # BUY at p: no ask below p rested at tau
                c1 = np.searchsorted(L, trades_px[j], side="left")
                R[rt[j], :c1] = trades_tau[j] - GUARD_MS
        np.maximum.accumulate(R, axis=0, out=R)
        hidden = R > upd
    return L, sizes, hidden


def topk(L, sizes, hidden, is_bid):
    """Top-NLEV levels (q, size) best first, total depth and level count per row."""
    n = sizes.shape[0]
    live = (sizes > 0) & ~hidden
    if is_bid:
        L, live, sizes = L[::-1], live[:, ::-1], sizes[:, ::-1]
    q = np.full((n, NLEV), -1, np.int64)
    s = np.zeros((n, NLEV))
    if L.size:
        cnt = np.cumsum(live, axis=1)
        rows = np.arange(n)
        for k in range(NLEV):
            sel = live & (cnt == k + 1)
            has = sel.any(1)
            j = sel.argmax(1)
            q[:, k] = np.where(has, L[j], -1)
            s[:, k] = np.where(has, sizes[rows, j], 0.0)
    depth = np.where(live, sizes, 0.0).sum(1)
    return q, s, depth, live.sum(1)


def replay(snaps: pd.DataFrame, deltas: pd.DataFrame, B: np.ndarray, trades: pd.DataFrame | None = None):
    """Book of one market at evaluation times B (int64 ms, ascending).

    snaps: one market's snapshots (collected_at_ms, bids, asks, snapshot_id), any order.
    deltas: one market's deltas (snapshot_id, platform_timestamp_ms, price, size, side) in file order.
    trades: optional (platform_timestamp_ms, price, side) taker trades for ghost cleaning.
    Returns dict of per-time arrays: k (snapshot index, -1 none), bq/bs/aq/as [n, 5] (q = -1 empty),
    bdep, adep, nb, na, last_msg (ms of the latest snapshot/delta used), n_hidden.
    """
    snaps = snaps.sort_values(["collected_at_ms", "snapshot_id"], kind="stable").reset_index(drop=True)
    sc = snaps.collected_at_ms.values.astype(np.int64)
    sid = {s: i for i, s in enumerate(snaps.snapshot_id.values)}
    ck = deltas.snapshot_id.map(sid)
    n_orphan = int(ck.isna().sum())
    d = deltas[ck.notna()]
    ck = ck[ck.notna()].values.astype(np.int64)
    dts = d.platform_timestamp_ms.values.astype(np.int64)
    dq = np.rint(d.price.values.astype(np.float64) * PX).astype(np.int64)
    ds = d["size"].values.astype(np.float64)
    dbid = (d.side.values == "BUY")
    assert np.isin(d.side.values, ["BUY", "SELL"]).all()
    order = np.lexsort((np.arange(len(dts)), dts, ck))    # by chain, then time, then file order
    ck, dts, dq, ds, dbid = ck[order], dts[order], dq[order], ds[order], dbid[order]
    if trades is not None:
        tt = trades.sort_values("platform_timestamp_ms", kind="stable")
        t_tau = tt.platform_timestamp_ms.values.astype(np.float64)
        t_px = np.rint(tt.price.values.astype(np.float64) * PX).astype(np.int64)
        t_buy = tt.side.values == "BUY"
    n = len(B)
    out = {"k": np.searchsorted(sc, B, side="right") - 1,
           "bq": np.full((n, NLEV), -1, np.int64), "bs": np.zeros((n, NLEV)),
           "aq": np.full((n, NLEV), -1, np.int64), "as": np.zeros((n, NLEV)),
           "bdep": np.zeros(n), "adep": np.zeros(n), "nb": np.zeros(n, np.int64), "na": np.zeros(n, np.int64),
           "last_msg": np.full(n, -1, np.int64), "n_hidden": np.zeros(n, np.int64), "n_orphan": n_orphan}
    kk = out["k"]
    cstart = np.searchsorted(ck, np.arange(len(sc)), side="left")
    cend = np.searchsorted(ck, np.arange(len(sc)), side="right")
    for k in np.unique(kk[kk >= 0]):
        rows = np.flatnonzero(kk == k)
        Bk = B[rows]
        a, b = cstart[k], cend[k]
        cts, cq, cs, cb = dts[a:b], dq[a:b], ds[a:b], dbid[a:b]
        nmsg = np.searchsorted(cts, Bk, side="right")      # cts is sorted within the chain
        last_d = np.where(nmsg > 0, cts[np.maximum(nmsg - 1, 0)], -1)
        out["last_msg"][rows] = np.maximum(last_d, sc[k])
        for is_bid, key in ((True, "b"), (False, "a")):
            q0, s0 = snapshot_levels(snaps.bids.iloc[k] if is_bid else snaps.asks.iloc[k])
            msk = cb == is_bid
            ttau = tpx = None
            if trades is not None:
                sel = (t_buy != is_bid) & (t_tau <= Bk[-1]) & (t_tau >= sc[k] - 60_000)
                ttau, tpx = t_tau[sel], t_px[sel]
            L, sizes, hidden = side_matrix(q0, s0, sc[k], cq[msk], cs[msk], cts[msk], Bk, ttau, tpx, is_bid)
            q, s, dep, nl = topk(L, sizes, hidden, is_bid)
            out[key + "q"][rows], out[key + "s"][rows] = q, s
            out[key + "dep"][rows], out["n" + key][rows] = dep, nl
            out["n_hidden"][rows] += ((sizes > 0) & hidden).sum(1)
    return out


def final_state(snap_bids, snap_asks, dq, ds, dbid):
    """Snapshot plus ALL deltas of its chain (application order) -> ({q: size} bids, asks)."""
    b = dict(zip(*snapshot_levels(snap_bids)))
    a = dict(zip(*snapshot_levels(snap_asks)))
    for q, s, isb in zip(dq, ds, dbid):
        book = b if isb else a
        if s > 0:
            book[int(q)] = float(s)
        else:
            book.pop(int(q), None)
    return b, a
