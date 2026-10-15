"""Offline finalization: markets that have book rows but no metadata row (recorder stopped before
their settle delay, or a crash) get the same full trade sweep + gamma metadata the live recorder does."""
from __future__ import annotations

import asyncio
import logging
import time

import pyarrow.compute as pc

from .config import Config
from .discovery import fetch_market_final, resolution
from .http import Http
from .metadata import market_row
from .schema import BOOK_SCHEMA, MARKET_QC_SCHEMA, MARKETS_SCHEMA, TRADES_SCHEMA
from .sinks import TableSink, read_dataset
from .trades import LegSet, build_trade_rows, fetch_all, trade_rows_to_table

log = logging.getLogger("polyrec.offline")


def recorded_markets(out) -> dict:
    """slug -> dict(coin, contract_length, end_s, n_rows) from the books dataset."""
    t = read_dataset(out, "books", columns=["market_slug", "coin", "contract_length", "market_end_at"])
    if t is None:
        return {}
    df = t.to_pandas()
    g = df.groupby("market_slug", observed=True).agg(coin=("coin", "first"), contract_length=("contract_length", "first"),
                                                     end=("market_end_at", "first"), n_rows=("coin", "size"))
    return {s: dict(coin=r.coin, contract_length=r.contract_length, end_s=r.end.timestamp(), n_rows=int(r.n_rows))
            for s, r in g.iterrows()}


def finalized_slugs(out) -> set:
    t = read_dataset(out, "markets", columns=["slug"])
    return set() if t is None else set(t.column(0).to_pylist())


async def finalize_offline(cfg: Config, slugs: list | None = None, min_age_s: float = 120,
                           require_resolved: bool = False) -> dict:
    out = cfg.out
    rec = recorded_markets(out)
    done = finalized_slugs(out)
    now = time.time()
    todo = [s for s in (slugs or sorted(rec)) if s in rec and s not in done and now - rec[s]["end_s"] >= min_age_s]
    run = "offline-" + time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    lvl = cfg.output.compression_level
    trades_sink = TableSink(out, "trades", TRADES_SCHEMA, run, lvl, dictionary_cols=["slug"])
    markets_sink = TableSink(out, "markets", MARKETS_SCHEMA, run, lvl)
    qc_sink = TableSink(out, "market_qc", MARKET_QC_SCHEMA, run, lvl)
    res = {"finalized": [], "unresolved": [], "errors": {}}
    async with Http(cfg.http_timeout_s) as http:
        sem = asyncio.Semaphore(cfg.trades.concurrency)

        async def one(slug):
            async with sem:
                try:
                    g = await fetch_market_final(http, slug)
                    if g is None:
                        raise RuntimeError("gamma: no market")
                    win, _ = resolution(g)
                    if win is None and require_resolved:
                        res["unresolved"].append(slug)
                        return
                    taker, allf = LegSet(), LegSet()
                    trunc = False
                    if cfg.trades.enabled:
                        tk, t1 = await fetch_all(http, g["conditionId"], True)
                        al, t2 = await fetch_all(http, g["conditionId"], False)
                        taker.merge(tk)
                        allf.merge(al)
                        trunc = t1 or t2
                        trades_sink.append_table(trade_rows_to_table(build_trade_rows(slug, taker, allf)), now)
                    r = rec[slug]
                    row = market_row(slug, r["coin"], r["contract_length"], g, len(taker), len(allf), trunc)
                    markets_sink.append(row, now)
                    qc_sink.append(dict(slug=slug, run_id=run, finalized_at=int(time.time() * 1e6), first_live=None,
                                        n_rows=r["n_rows"], n_reconcile=None, n_mismatch=None, n_bba_mismatch=None,
                                        n_disconnects=None, n_trades_incremental=0, trades_truncated=trunc,
                                        resolved=row["winner_up"] is not None), now)
                    res["finalized"].append(slug)
                    if win is None:
                        res["unresolved"].append(slug)
                except Exception as e:  # noqa: BLE001
                    res["errors"][slug] = str(e)
        await asyncio.gather(*(one(s) for s in todo))
    for s in (trades_sink, markets_sink, qc_sink):
        s.flush()
        for h in s.part_hours():
            s.compact(h)
    return res
