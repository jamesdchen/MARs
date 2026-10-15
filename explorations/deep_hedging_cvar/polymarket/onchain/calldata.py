#!/usr/bin/env python3
"""Decode matchOrders calldata for the txs in an extract: each matched order's original size, limit price,
creation timestamp (ms, signed into the order) and signature type. Optional (extract.py --calldata, or run
standalone); costs one eth_getTransactionByHash per tx, sent as JSON-RPC batches of --batch.

  python calldata.py --out data/val_bnb [--date 2026-08-21] [--slugs bnb-updown-5m-1787277900]

matchOrders(bytes32 conditionId, Order takerOrder, Order[] makerOrders, uint256 takerFillAmount,
            uint256[] makerFillAmounts, uint256 takerFeeAmount, uint256[] makerFeeAmounts)    selector 0x3c2b4399
Order = (salt, maker, signer, tokenId, makerAmount, takerAmount, side, signatureType, timestamp_ms, metadata,
         builder, signature)

Output: <out>/orders/date=D/orders.parquet, one row per order in each decoded tx
  tx_hash, block, block_time, exchange, condition_id, role (taker/maker), order_index (0 = taker, 1.. makers in
  calldata order = OrderFilled log order), maker, signer, token_id, side, orig_maker_amount, orig_taker_amount,
  orig_size (shares), limit_price, sig_type, order_ts (raw signed timestamp, normally ms), created_s (unix s;
  NaN when the order carries timestamp 0, seconds-valued timestamps are recognised), age_s (block_time -
  created_s), fill_amount, fee_amount, partial (fill < original making amount), tx_from
Txs whose input is not matchOrders (other selectors) are counted and skipped.
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

import abi
import common as C
import rpc

ORDER = "(uint256,address,address,uint256,uint256,uint256,uint8,uint8,uint256,bytes32,bytes32,bytes)"
SIG = f"matchOrders(bytes32,{ORDER},{ORDER}[],uint256,uint256[],uint256,uint256[])"
SEL = abi.selector(SIG)
ARGS = ["bytes32", ORDER, f"{ORDER}[]", "uint256", "uint256[]", "uint256", "uint256[]"]


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
    for i, (o, fill, fee) in enumerate([(tko, tfa, tfee)] + list(zip(mkos, mfa, mfees))):
        side = "BUY" if o[6] == 0 else "SELL"
        mamt, tamt = o[4], o[5]
        px = (mamt / tamt if tamt else np.nan) if side == "BUY" else (tamt / mamt if mamt else np.nan)
        rows.append(dict(tx_hash=tx["hash"], block=int(tx["blockNumber"], 16), block_time=block_time,
                         exchange=C.EXCHANGES.get(tx["to"].lower(), tx["to"].lower()), condition_id=cid,
                         role="taker" if i == 0 else "maker", order_index=i, maker=o[1], signer=o[2],
                         token_id=str(o[3]), side=side, orig_maker_amount=min(mamt, 2**63 - 1),
                         orig_taker_amount=min(tamt, 2**63 - 1),
                         orig_size=(tamt if side == "BUY" else mamt) / 1e6, limit_price=px, sig_type=o[7],
                         order_ts=o[8], created_s=_created_s(o[8]), age_s=block_time - _created_s(o[8]),
                         fill_amount=min(fill, 2**63 - 1),
                         fee_amount=min(fee, 2**63 - 1), partial=bool(fill < mamt), tx_from=tx["from"]))
    return rows


def fetch_txs(pool: rpc.Pool, hashes: list[str], batch: int = 15, workers: int = 3) -> dict[str, dict]:
    chunks = [hashes[i:i + batch] for i in range(0, len(hashes), batch)]

    def one(ch):
        res = pool.request(lambda e: e.batch([("eth_getTransactionByHash", [h]) for h in ch]))
        return {h: {k: r[k] for k in ("hash", "blockNumber", "to", "from", "input")} for h, r in zip(ch, res)}
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for d in ex.map(one, chunks):
            out.update(d)
    return out


def run_day(out: Path, date: str, pool: rpc.Pool, clock=None, token_ids: set[str] | None = None,
            batch: int = 15, force: bool = False) -> dict:
    import extract
    dst = out / "orders" / f"date={date}" / "orders.parquet"
    if dst.exists() and not force:
        print(f"{date}: orders exist, skip")
        return {}
    f = extract.load(out, "fills", date)
    if token_ids is not None:
        f = f[f.token_id.isin(token_ids)]
    bt = f.drop_duplicates("tx_hash").set_index("tx_hash").block_time
    hashes = sorted(bt.index)
    t0 = time.time()
    txs = fetch_txs(pool, hashes, batch)
    rows, sels = [], collections.Counter()
    for h in hashes:
        tx = txs[h]
        r = decode_tx(tx, int(bt[h]))
        sels[tx["input"][:10]] += 1
        if r:
            rows += r
    df = pd.DataFrame(rows)
    if len(df):
        C.write_parquet_atomic(pa.Table.from_pandas(df, preserve_index=False), dst)
    st = dict(date=date, txs=len(hashes), decoded_orders=len(df), selectors=dict(sels), secs=round(time.time() - t0, 1),
              txs_per_s=round(len(hashes) / max(1e-9, time.time() - t0), 1))
    print(json.dumps(st))
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="extract.py output dir")
    ap.add_argument("--date", help="one UTC date (default: every date with fills)")
    ap.add_argument("--slugs", help="restrict to these markets' tokens")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rpc", action="append")
    args = ap.parse_args()
    out = Path(args.out)
    pool = rpc.Pool.from_specs(args.rpc)
    toks = None
    if args.slugs:
        import markets as M
        toks = set(M.Gamma().markets(args.slugs.split(",")).token_id)
    dates = [args.date] if args.date else sorted(p.name[5:] for p in (out / "fills").glob("date=*"))
    for d in dates:
        run_day(out, d, pool, token_ids=toks, batch=args.batch, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
