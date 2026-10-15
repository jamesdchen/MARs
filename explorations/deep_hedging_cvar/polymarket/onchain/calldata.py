#!/usr/bin/env python3
"""Decode matchOrders calldata for the txs in an extract: each matched order's original size, limit price,
creation timestamp (signed into the order) and signature type. Optional: `extract.py --calldata`, or standalone

  python calldata.py --out data/val_bnb [--date 2026-08-21] [--slugs bnb-updown-5m-1787277900]
  python calldata.py --out data/crypto-updown --date 2026-08-21 --sample-frac 0.05

Fetch modes (--mode auto picks blocks when the txs average > 3 per block)
  txs     eth_getTransactionByHash per tx (JSON-RPC batches; every element counts against the endpoint's rate
          limit, ~15/s quiknode). Right for a few markets.
  blocks  eth_getBlockByNumber(full=true) per block that holds a wanted tx, keeping only wanted txs. For the
          whole crypto up/down set (~1.37M txs over ~57k blocks per day) this is ~24x fewer calls: ~30 min/day
          at the keyless rate limits vs ~11 h/day per tx.
--sample-frac keeps a deterministic hash-based sample of txs (enough to calibrate order-age / partial-fill
distributions without the full cost). Work is chunked into parts under orders/date=D/_parts (resumable) and
merged into orders/date=D/orders.parquet.

matchOrders(bytes32 conditionId, Order takerOrder, Order[] makerOrders, uint256 takerFillAmount,
            uint256[] makerFillAmounts, uint256 takerFeeAmount, uint256[] makerFeeAmounts)    selector 0x3c2b4399
Order = (salt, maker, signer, tokenId, makerAmount, takerAmount, side, signatureType, timestamp, metadata,
         builder, signature)

Output columns (one row per order of each decoded tx)
  tx_hash, block, block_time, exchange, condition_id, role (taker/maker), order_index (0 = taker, 1.. makers in
  calldata order = OrderFilled log order), maker, signer, token_id, side, orig_maker_amount, orig_taker_amount,
  orig_size (shares), limit_price, sig_type, order_ts (raw signed timestamp, normally ms), created_s (unix s;
  NaN when the order carries timestamp 0; seconds-valued timestamps are recognised), age_s (block_time -
  created_s), fill_amount, fee_amount, partial (fill < original making amount), tx_from
Txs whose input is not matchOrders are counted by selector and skipped.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import abi
import common as C
import rpc

ORDER = "(uint256,address,address,uint256,uint256,uint256,uint8,uint8,uint256,bytes32,bytes32,bytes)"
SIG = f"matchOrders(bytes32,{ORDER},{ORDER}[],uint256,uint256[],uint256,uint256[])"
SEL = abi.selector(SIG)
ARGS = ["bytes32", ORDER, f"{ORDER}[]", "uint256", "uint256[]", "uint256", "uint256[]"]
I64 = 2**63 - 1


def _created_s(ts: int) -> float:
    """Order timestamp -> unix seconds. Signed as ms; a few orders carry seconds; 0 = not set (-> NaN)."""
    if ts >= 10**11:
        return ts / 1000.0
    if ts >= 10**9:
        return float(ts)
    return float("nan")


def decode_tx(tx: dict, block_time: int) -> list[dict] | None:
    inp = tx["input"]
    if inp[:10] != SEL:
        return None
    cid, tko, mkos, tfa, mfa, tfee, mfees = abi.decode(ARGS, bytes.fromhex(inp[10:]))
    rows = []
    blk = int(tx["blockNumber"], 16)
    exch = C.EXCHANGES.get(tx["to"].lower(), tx["to"].lower())
    for i, (o, fill, fee) in enumerate([(tko, tfa, tfee)] + list(zip(mkos, mfa, mfees))):
        side = "BUY" if o[6] == 0 else "SELL"
        mamt, tamt = o[4], o[5]
        px = (mamt / tamt if tamt else np.nan) if side == "BUY" else (tamt / mamt if mamt else np.nan)
        cs = _created_s(o[8])
        rows.append(dict(tx_hash=tx["hash"], block=blk, block_time=block_time, exchange=exch, condition_id=cid,
                         role="taker" if i == 0 else "maker", order_index=i, maker=o[1], signer=o[2],
                         token_id=str(o[3]), side=side, orig_maker_amount=min(mamt, I64),
                         orig_taker_amount=min(tamt, I64), orig_size=(tamt if side == "BUY" else mamt) / 1e6,
                         limit_price=px, sig_type=o[7], order_ts=min(o[8], I64), created_s=cs,
                         age_s=block_time - cs, fill_amount=min(fill, I64), fee_amount=min(fee, I64),
                         partial=bool(fill < mamt), tx_from=tx["from"]))
    return rows


def _fetch(pool: rpc.Pool, calls: list[tuple[str, list]], per_req: int, workers: int = 3) -> list:
    chunks = [calls[i:i + per_req] for i in range(0, len(calls), per_req)]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda ch: pool.request(lambda e: e.batch(ch, timeout=120)), chunks))
    return [r for ch in res for r in ch]


def decode_chunk(pool, mode: str, items: list, txb: pd.DataFrame, sels: collections.Counter) -> pd.DataFrame:
    """items: tx hashes (mode txs) or block numbers (mode blocks)."""
    rows = []
    if mode == "txs":
        txs = _fetch(pool, [("eth_getTransactionByHash", [h]) for h in items], 15)
    else:
        want = set(txb.index)
        blocks = _fetch(pool, [("eth_getBlockByNumber", [hex(b), True]) for b in items], 4)
        txs = [t for b in blocks for t in b["transactions"] if t["hash"] in want]
    for tx in txs:
        sels[tx["input"][:10]] += 1
        r = decode_tx(tx, int(txb.block_time.get(tx["hash"], -1)))
        if r:
            rows += r
    return pd.DataFrame(rows)


def run_day(out: Path, date: str, pool: rpc.Pool, clock=None, token_ids: set[str] | None = None,
            mode: str = "auto", sample_frac: float = 1.0, force: bool = False, chunk_blocks: int = 1000,
            chunk_txs: int = 3000) -> dict:
    import extract
    ddir = out / "orders" / f"date={date}"
    dst = ddir / "orders.parquet"
    if dst.exists() and not force:
        print(f"{date}: orders exist, skip")
        return {}
    f = extract.load(out, "fills", date)
    if token_ids is not None:
        f = f[f.token_id.isin(token_ids)]
    txb = f.drop_duplicates("tx_hash").set_index("tx_hash")[["block", "block_time"]]
    if sample_frac < 1:
        txb = txb[[int(h[-6:], 16) < sample_frac * 16**6 for h in txb.index]]
    blocks = np.sort(txb.block.unique())
    if mode == "auto":
        mode = "blocks" if len(txb) > 3 * max(1, len(blocks)) else "txs"
    if mode == "txs":
        hs = list(txb.sort_values("block").index)
        units = [hs[i:i + chunk_txs] for i in range(0, len(hs), chunk_txs)]
    else:
        units = [list(map(int, blocks[i:i + chunk_blocks])) for i in range(0, len(blocks), chunk_blocks)]
    pdir = ddir / "_parts"
    if force and pdir.exists():
        for p in pdir.glob("*.parquet"):
            p.unlink()
    t0, sels = time.time(), collections.Counter()
    for k, items in enumerate(units):
        part = pdir / f"part-{mode}-{k:05d}.parquet"
        if part.exists():
            continue
        df = decode_chunk(pool, mode, items, txb, sels)
        C.write_parquet_atomic(pa.Table.from_pandas(df, preserve_index=False), part)
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {date} orders chunk {k+1}/{len(units)}  {el:.0f}s", flush=True)
    parts = sorted(pdir.glob(f"part-{mode}-*.parquet"))
    tbls = [t for t in (pq.read_table(p) for p in parts) if t.num_rows]
    n = 0
    if tbls:
        tbl = pa.concat_tables(tbls, promote_options="default")
        n = tbl.num_rows
        C.write_parquet_atomic(tbl, dst)
    for p in parts:
        p.unlink()
    el = time.time() - t0
    st = dict(date=date, mode=mode, txs=len(txb), blocks=int(len(blocks)),
              calls=int(len(blocks)) if mode == "blocks" else len(txb), decoded_orders=n, selectors=dict(sels),
              secs=round(el, 1), txs_per_s=round(len(txb) / max(el, 1e-9), 1))
    print(json.dumps(st), flush=True)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="extract.py output dir")
    ap.add_argument("--date", help="one UTC date (default: every date with fills)")
    ap.add_argument("--slugs", help="restrict to these markets' tokens")
    ap.add_argument("--mode", choices=["auto", "txs", "blocks"], default="auto")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rpc", action="append", help="url[,conc[,span[,span_max[,rps[,max_batch]]]]]")
    args = ap.parse_args()
    out = Path(args.out)
    pool = rpc.Pool.from_specs(args.rpc)
    toks = None
    if args.slugs:
        import markets as M
        toks = set(M.Gamma().markets(args.slugs.split(",")).token_id)
    dates = [args.date] if args.date else sorted(p.name[5:] for p in (out / "fills").glob("date=*"))
    for d in dates:
        run_day(out, d, pool, token_ids=toks, mode=args.mode, sample_frac=args.sample_frac, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
