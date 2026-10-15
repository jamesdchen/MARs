#!/usr/bin/env python3
"""Inputs of the v2 TEST build (DECISIONS D7): the 288 BTC 5-minute up/down markets of 2026-06-15.

Market list: Marketlens `polymarket-btc-5m-l2-depth` (data/raw/marketlens/markets.parquet; download
and license notes in data/build_jun15/TEST_LOCK.md). Slug = btc-updown-5m-<open unix s>.

For consistency with the training data, metadata and fills come from the same Polymarket APIs and
the SAME code as data/fetch_trades.py and data/fetch_underlying.py (imported, not copied):
  * gamma /markets?slug=<slug>&closed=true            -> metadata (strike, fees, rebate, tick, winner)
  * data-api /trades?market=<conditionId>&takerOnly=true|false -> taker + maker fill legs
  * data.binance.vision BTCUSDT 1 s klines 2026-06-14..15 (checksums verified)

Outputs (git-ignored, inside the test build):
  data/build_jun15/raw/markets.parquet     same schema as data/raw/markets.parquet
  data/build_jun15/raw/trades.parquet      same schema as data/raw/trades.parquet
  data/build_jun15/raw/underlying.parquet  same schema as data/raw/underlying.parquet (btc only)
  data/build_jun15/raw/fetch_report.json   token-id / winner / timing cross-checks
Per-market API responses are cached in data/raw/cache/jun15/ (git-ignored).

    python data/fetch_jun15.py          (~5 min)
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_trades as ft  # noqa: E402
import fetch_underlying as fu  # noqa: E402

ML = HERE / "raw" / "marketlens"
OUT = HERE / "build_jun15" / "raw"
DAYS = ["2026-06-14", "2026-06-15"]


def marketlens_markets() -> pd.DataFrame:
    m = pd.read_parquet(ML / "markets.parquet")
    m["open_ts"] = (m.open_time_ms // 1000).astype(np.int64)
    m["end_ts"] = (m.close_time_ms // 1000).astype(np.int64)
    assert (m.open_time_ms % 1000 == 0).all() and (m.close_time_ms % 1000 == 0).all()
    m["slug"] = "btc-updown-5m-" + m.open_ts.astype(str)
    return m


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ft.CACHE = HERE / "raw" / "cache" / "jun15"
    ft.CACHE.mkdir(parents=True, exist_ok=True)
    ml = marketlens_markets()
    slugs = pd.DataFrame({"coin": "btc", "contract_length": "5m"}, index=pd.Index(sorted(ml.slug), name="slug"))
    results, errors = {}, {}
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(ft.fetch_market, s): s for s in slugs.index}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                results[s] = f.result()
            except Exception as e:  # noqa: BLE001
                errors[s] = str(e)
            if i % 50 == 0 or i == len(futs):
                print(f"[{i}/{len(futs)}] {time.time() - t0:.0f}s errors={len(errors)}", flush=True)
    for s, e in errors.items():
        print("ERROR", s, e, file=sys.stderr)
    slugs = slugs.drop(list(errors))
    markets, trades = ft.build_outputs(slugs, results)

    # ---- cross-checks against the Marketlens market list
    mm = markets.set_index("slug")
    mlx = ml.set_index("slug").loc[mm.index]
    tok_bad = int(sum((list(t) != [u, d]) for t, u, d in zip(mlx.token_ids, mm.token_up, mm.token_down)))
    cond_bad = int((mlx.market_id != mm.condition_id).sum())
    names_bad = int(sum(list(n) != ["Up", "Down"] for n in mlx.outcome_names))
    win_bad = int(((mlx.winning_outcome == "Up").astype(int) != mm.winner_up).sum())
    end_bad = int((mm.end.map(lambda x: int(pd.Timestamp(x).timestamp())) != mlx.end_ts).sum())
    start_bad = int((mm.event_start.map(lambda x: int(pd.Timestamp(x).timestamp())) != mlx.open_ts).sum())
    report = {"n_marketlens": int(len(ml)), "n_fetched": int(len(markets)), "errors": errors,
              "token_id_mismatch_vs_marketlens": tok_bad, "condition_id_mismatch": cond_bad,
              "outcome_names_not_up_down": names_bad, "winner_mismatch_gamma_vs_marketlens": win_bad,
              "end_mismatch": end_bad, "event_start_mismatch": start_bad,
              "truncated_queries": int(markets.truncated.sum()),
              "missing_price_to_beat": int(markets.price_to_beat.isna().sum()),
              "fees": markets[["fees_enabled", "fee_type", "fee_rate", "fee_exponent", "fee_rebate_rate"]]
              .astype(str).value_counts().reset_index().to_dict("records"),
              "tick_size_gamma": markets.tick_size.value_counts().to_dict(),
              "min_size": markets.min_size.value_counts().to_dict(),
              "rewards": markets[["rewards_min_size", "rewards_max_spread"]].astype(str).value_counts()
              .reset_index().to_dict("records"),
              "n_fill_legs": int(len(trades)), "n_taker_legs": int(trades.is_taker.sum())}
    print(json.dumps(report, indent=1, default=str))
    markets.to_parquet(OUT / "markets.parquet", index=False, compression="zstd")
    ft.write_trades(trades, OUT / "trades.parquet")

    # ---- Binance BTCUSDT 1 s klines (same downloader / parser as data/fetch_underlying.py)
    fu.CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in DAYS:
        p = fu.get_zip("btc", d)
        assert p is not None, f"missing Binance day {d}"
        frames.append(fu.parse("btc", d, p))
    df = pd.concat(frames, ignore_index=True).sort_values(["coin", "open_time"])
    df["close"] = df.groupby("coin").close.ffill()
    n_nan = int(df.close.isna().sum())
    df = df.dropna(subset=["close"])
    report["underlying"] = {"days": DAYS, "rows": int(len(df)), "leading_nan_dropped": n_nan,
                            "open_time_min": int(df.open_time.min()), "open_time_max": int(df.open_time.max()),
                            "source": "data.binance.vision spot BTCUSDT 1s klines (sha256 CHECKSUM verified)",
                            "zips": [fu.url_for("btc", d) for d in DAYS]}
    tbl = pa.table({
        "coin": pa.array(df.coin.values).dictionary_encode(),
        "open_time": pa.array(df.open_time.values.astype(np.int64)),
        "close": pa.array(df.close.values.astype(np.float64)),
        "source": pa.array(df.source.values).dictionary_encode(),
    })
    pq.write_table(tbl, OUT / "underlying.parquet", compression="zstd", compression_level=19,
                   use_byte_stream_split=["close"], row_group_size=1 << 20)
    report["fetched_at_utc"] = pd.Timestamp.now("UTC").isoformat()
    (OUT / "fetch_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"markets {len(markets)}  fill legs {len(trades)}  taker {int(trades.is_taker.sum())}  "
          f"underlying {len(df)}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
