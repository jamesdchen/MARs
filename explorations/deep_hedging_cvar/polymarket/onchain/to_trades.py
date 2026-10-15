#!/usr/bin/env python3
"""Convert extract.py fills into the exact schema of data/raw/trades.parquet (and gamma metadata into the schema
of data/raw/markets.parquet), so data/build.py-style code and sim/fit.py's flow fits can consume months of trades.

  python to_trades.py --src data/crypto-updown --start 2026-08-21 --end 2026-08-21
      -> data/crypto-updown/trades_2026-08-21_2026-08-21.parquet, markets_2026-08-21_2026-08-21.parquet

trades.parquet schema (identical dtypes, writer copied from data/fetch_trades.py)
  slug (dictionary), timestamp (int64 unix s = block time, same as the data-api), outcome_idx (int8, 0 Up / 1 Down
  from the token's position in gamma `outcomes`), side (int8 +1 BUY / -1 SELL from the leg's own perspective),
  price (float32: taker leg = average fill price usdc/shares, maker leg = its resting limit price, which on-chain
  equals usdc/shares of that maker fill), size (float32 shares), is_taker (bool: OrderFilled whose taker is the
  exchange contract), aggressor_up (int8, per tx from the taker leg in the Up frame: Down BUY = Up SELL),
  tx (fixed_size_binary(32)).
One OrderFilled = one data-api row (takerOnly=false): one taker leg + one leg per maker order.

markets.parquet: same 36 columns as data/raw/markets.parquet. n_taker / n_fills count the legs in this output,
and `truncated` is True when the converted date range does not cover the market's whole trading lifetime
(created .. end + 10 min), i.e. its counts are partial.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import common as C
import extract
import markets as M


def convert(fills: pd.DataFrame, tokmap: pd.DataFrame) -> pd.DataFrame:
    tm = tokmap.drop_duplicates("token_id").set_index("token_id")
    f = fills[fills.token_id.isin(tm.index)]
    oi = tm.outcome_idx.loc[f.token_id].values.astype(np.int8)
    t = pd.DataFrame(dict(
        slug=tm.slug.loc[f.token_id].values, tx=f.tx_hash.str.lower().values, timestamp=f.block_time.values.astype(np.int64),
        outcome_idx=oi, side=np.where(f.side.values == "BUY", 1, -1).astype(np.int8),
        price=f.price.values, size=f["size"].values, is_taker=(f.role.values == "taker"),
        log_index=f.log_index.values))
    tk = t[t.is_taker]
    up_side = np.where(tk.outcome_idx == 0, tk.side, -tk.side)
    agg = pd.Series(up_side, index=tk.tx.values)
    agg = agg[~agg.index.duplicated()]
    t["aggressor_up"] = t.tx.map(agg).fillna(0).astype(np.int8)
    t = t.sort_values(["slug", "timestamp", "tx", "is_taker", "log_index"],
                      ascending=[True, True, True, False, True], kind="stable")
    return t.drop(columns="log_index").reset_index(drop=True)


def write_trades(t: pd.DataFrame, path: Path):
    """Byte-for-byte the writer of data/fetch_trades.py (schema, dictionary slug, zstd-19)."""
    slugs = pd.Categorical(t.slug)
    txbuf = b"".join(bytes.fromhex(x[2:]) for x in t.tx)
    tbl = pa.table({
        "slug": pa.DictionaryArray.from_arrays(pa.array(slugs.codes.astype(np.int32)), pa.array(list(slugs.categories))),
        "timestamp": pa.array(t.timestamp.values.astype(np.int64)),
        "outcome_idx": pa.array(t.outcome_idx.values.astype(np.int8)),
        "side": pa.array(t.side.values.astype(np.int8)),
        "price": pa.array(t.price.values.astype(np.float32)),
        "size": pa.array(t["size"].values.astype(np.float32)),
        "is_taker": pa.array(t.is_taker.values),
        "aggressor_up": pa.array(t.aggressor_up.values.astype(np.int8)),
        "tx": pa.FixedSizeBinaryArray.from_buffers(pa.binary(32), len(t), [None, pa.py_buffer(txbuf)]),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(tbl, tmp, compression="zstd", compression_level=19, use_dictionary=["slug"], row_group_size=1 << 20)
    tmp.replace(path)


def _jl(x):
    return json.loads(x) if isinstance(x, str) else x


def markets_table(slugs: list[str], trades: pd.DataFrame, t0: int, t1: int, gamma: M.Gamma) -> pd.DataFrame:
    """Rows in data/raw/markets.parquet's schema (built like data/fetch_trades.build_outputs)."""
    raw = gamma.raw(slugs)
    cnt = trades.groupby("slug").agg(n_fills=("tx", "size"), n_taker=("is_taker", "sum"))
    rows = []
    for slug in slugs:
        g = raw.get(slug)
        if g is None:
            continue
        outcomes = _jl(g["outcomes"])
        if outcomes != ["Up", "Down"]:
            continue
        tok = dict(zip(outcomes, _jl(g["clobTokenIds"])))
        fs = g.get("feeSchedule") or {}
        em = g.get("eventMetadata") or {}
        cmc = g.get("cryptoMarketConfig") or {}
        prices = [float(p) for p in _jl(g.get("outcomePrices") or "[0.5, 0.5]")]
        m = pd.Series(slug).str.extract(M.CRYPTO_RE).iloc[0]
        created = pd.Timestamp(g["startDate"])
        end = pd.Timestamp(g["endDate"])
        cr_ts = min(x for x in (M._ts(g.get("createdAt")), M._ts(g.get("startDate"))) if x)
        covered = cr_ts >= t0 and end.timestamp() + M.POST_S <= t1
        rows.append(dict(
            slug=slug, coin=m[0], contract_length=m[1],
            gamma_id=int(g["id"]), condition_id=g["conditionId"], question_id=g["questionID"], question=g["question"],
            token_up=tok.get("Up"), token_down=tok.get("Down"), created_start=created,
            event_start=pd.Timestamp(g.get("eventStartTime") or g.get("eventStartTimeEv")), end=end,
            closed_time=pd.Timestamp(g["closedTime"]) if g.get("closedTime") else pd.NaT,
            tick_size=float(g["orderPriceMinTickSize"]), min_size=float(g["orderMinSize"]),
            maker_base_fee=float(g.get("makerBaseFee") or 0), taker_base_fee=float(g.get("takerBaseFee") or 0),
            fees_enabled=bool(g.get("feesEnabled")), fee_type=g.get("feeType"),
            fee_rate=float(fs.get("rate", np.nan)), fee_exponent=float(fs.get("exponent", np.nan)),
            fee_taker_only=bool(fs.get("takerOnly", False)), fee_rebate_rate=float(fs.get("rebateRate", np.nan)),
            maker_rebates_fee_share_bps=float(g.get("makerRebatesFeeShareBps") or 0),
            rewards_min_size=float(g.get("rewardsMinSize") or 0), rewards_max_spread=float(g.get("rewardsMaxSpread") or 0),
            resolution_source=g.get("resolutionSource"),
            twap_lookback_s=float(cmc.get("twapLookbackSeconds", np.nan)),
            outcome_price_up=prices[0], winner_up=int(prices[0] > 0.5),
            price_to_beat=float(em.get("priceToBeat", np.nan)), final_price=float(em.get("finalPrice", np.nan)),
            uma_status=g.get("umaResolutionStatus"),
            n_taker=int(cnt.n_taker.get(slug, 0)), n_fills=int(cnt.n_fills.get(slug, 0)), truncated=not covered))
    df = pd.DataFrame(rows)
    for c in ("created_start", "event_start", "end", "closed_time"):
        df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[us, UTC]")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="extract.py output dir")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--slug-regex", default=M.CRYPTO_RE, help="keep markets whose slug matches")
    ap.add_argument("--out")
    ap.add_argument("--markets-out")
    args = ap.parse_args()
    src = Path(args.src)
    fills = extract.load(src, "fills", args.start, args.end)
    tm = [pd.read_parquet(p) for d in C.date_range(args.start, args.end)
          if (p := src / "markets" / f"date={d}.parquet").exists()]
    g = M.Gamma()
    tokmap = pd.concat(tm).drop_duplicates("token_id") if tm else g.lookup_tokens(fills.token_id.unique())
    missing = set(fills.token_id) - set(tokmap.token_id)
    if missing:                               # e.g. complement-token legs of kept txs outside the day's map
        tokmap = pd.concat([tokmap, g.lookup_tokens(missing)]).drop_duplicates("token_id")
    tokmap = M.filter_slug(tokmap, args.slug_regex)
    t = convert(fills, tokmap)
    tag = f"{args.start}_{args.end}"
    out = Path(args.out) if args.out else src / f"trades_{tag}.parquet"
    write_trades(t, out)
    t0, t1 = C.utc_ts(args.start), C.utc_ts(args.end) + 86400
    mk = markets_table(sorted(t.slug.unique()), t, t0, t1, g)
    mout = Path(args.markets_out) if args.markets_out else src / f"markets_{tag}.parquet"
    mk.to_parquet(mout, index=False, compression="zstd")
    # schema check against the existing pipeline's files
    ref = pq.read_schema(C.REPO_DATA / "trades.parquet")
    got = pq.read_schema(out)
    ok_t = [(f.name, str(f.type)) for f in ref] == [(f.name, str(f.type)) for f in got]
    ref_m = pd.read_parquet(C.REPO_DATA / "markets.parquet").dtypes
    ok_m = list(ref_m.index) == list(mk.columns) and all(str(ref_m[c]) == str(mk[c].dtype) for c in mk.columns)
    print(f"{len(t)} legs ({int(t.is_taker.sum())} taker) in {t.slug.nunique()} markets -> {out}\n"
          f"{len(mk)} markets ({int(mk.truncated.sum())} with partial coverage) -> {mout}\n"
          f"schema identical to data/raw: trades {ok_t}, markets {ok_m}")
    if not ok_m:
        print({c: (str(ref_m.get(c)), str(mk[c].dtype)) for c in mk.columns if str(ref_m.get(c)) != str(mk[c].dtype)})
    return 0 if ok_t else 1


if __name__ == "__main__":
    sys.exit(main())
