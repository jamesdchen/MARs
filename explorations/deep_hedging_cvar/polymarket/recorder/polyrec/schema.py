"""Arrow schemas of every recorder output.

BOOK_SCHEMA is data/raw/updown_*.parquet (PolyOrderBooks, see data/raw/README.md) column for column,
plus one trailing column `reconciled`. TRADES_SCHEMA is data/raw/trades.parquet and MARKETS_SCHEMA
is data/raw/markets.parquet (both written by data/fetch_trades.py). tests/test_schema.py checks them
against the files in data/raw when those are present.
"""
from __future__ import annotations

import pyarrow as pa

UTC_US = pa.timestamp("us", tz="UTC")
LIST_F64 = pa.list_(pa.field("element", pa.float64()))

BOOK_FIELDS = [
    ("market_slug", pa.string()),
    ("coin", pa.string()),
    ("contract_length", pa.string()),
    ("token_id", pa.string()),
    ("outcome", pa.string()),
    ("winner", pa.string()),               # 'yes' if outcome index 0 (Up) won, 'no' otherwise; null until backfill
    ("is_winning_outcome", pa.bool_()),    # null until backfill
    ("market_end_at", UTC_US),
    ("captured_at", UTC_US),
    ("seconds_to_close", pa.float64()),
    ("bid_prices", LIST_F64),
    ("bid_sizes", LIST_F64),
    ("ask_prices", LIST_F64),
    ("ask_sizes", LIST_F64),
    ("best_bid", pa.float64()),
    ("best_ask", pa.float64()),
    ("crossed", pa.bool_()),
]
BOOK_SCHEMA_RAW = pa.schema(BOOK_FIELDS)                       # exact data/raw/updown_*.parquet schema
BOOK_SCHEMA = pa.schema(BOOK_FIELDS + [("reconciled", pa.bool_())])
# reconciled: True when this row's book was replaced by a REST /book re-read that disagreed with the
# stream-maintained book (the mismatch itself is logged in quality/reconcile_*.parquet).

TRADES_SCHEMA = pa.schema([
    ("slug", pa.dictionary(pa.int32(), pa.string())),
    ("timestamp", pa.int64()),
    ("outcome_idx", pa.int8()),
    ("side", pa.int8()),
    ("price", pa.float32()),
    ("size", pa.float32()),
    ("is_taker", pa.bool_()),
    ("aggressor_up", pa.int8()),
    ("tx", pa.binary(32)),
])

_S = pa.large_string()
MARKETS_SCHEMA = pa.schema([
    ("slug", _S), ("coin", _S), ("contract_length", _S), ("gamma_id", pa.int64()),
    ("condition_id", _S), ("question_id", _S), ("question", _S), ("token_up", _S), ("token_down", _S),
    ("created_start", UTC_US), ("event_start", UTC_US), ("end", UTC_US), ("closed_time", UTC_US),
    ("tick_size", pa.float64()), ("min_size", pa.float64()), ("maker_base_fee", pa.float64()),
    ("taker_base_fee", pa.float64()), ("fees_enabled", pa.bool_()), ("fee_type", _S),
    ("fee_rate", pa.float64()), ("fee_exponent", pa.float64()), ("fee_taker_only", pa.bool_()),
    ("fee_rebate_rate", pa.float64()), ("maker_rebates_fee_share_bps", pa.float64()),
    ("rewards_min_size", pa.float64()), ("rewards_max_spread", pa.float64()), ("resolution_source", _S),
    ("twap_lookback_s", pa.float64()), ("outcome_price_up", pa.float64()), ("winner_up", pa.int64()),
    ("price_to_beat", pa.float64()), ("final_price", pa.float64()), ("uma_status", _S),
    ("n_taker", pa.int64()), ("n_fills", pa.int64()), ("truncated", pa.bool_()),
])

# Quality log: one row per REST reconcile check.
RECONCILE_SCHEMA = pa.schema([
    ("checked_at", UTC_US),
    ("market_slug", pa.string()),
    ("token_id", pa.string()),
    ("outcome", pa.string()),
    ("trigger", pa.string()),         # periodic | bba (stream best bid/ask disagreed) | seed (book not live)
    ("status", pa.string()),          # ok | mismatch | seeded | error
    ("rest_ts", pa.int64()),          # REST book timestamp (ms, time of the book's last change)
    ("stream_ts", pa.int64()),        # server ms of the newest stream event applied to this token
    ("rtt_ms", pa.float64()),
    ("n_replayed", pa.int32()),       # stream events with ts >= rest_ts re-applied on top of REST
    ("n_bid_diff", pa.int32()),
    ("n_ask_diff", pa.int32()),
    ("max_size_diff", pa.float64()),
    ("top_differs", pa.bool_()),
    ("stream_best_bid", pa.float64()),
    ("stream_best_ask", pa.float64()),
    ("rest_best_bid", pa.float64()),
    ("rest_best_ask", pa.float64()),
    ("error", pa.string()),
])

# One row per finalized market: recorder-side quality counters (coverage intervals are in the
# coverage JSONL log; see merge.py).
MARKET_QC_SCHEMA = pa.schema([
    ("slug", pa.string()),
    ("run_id", pa.string()),
    ("finalized_at", UTC_US),
    ("first_live", UTC_US),
    ("n_rows", pa.int64()),
    ("n_reconcile", pa.int64()),
    ("n_mismatch", pa.int64()),
    ("n_bba_mismatch", pa.int64()),
    ("n_disconnects", pa.int64()),
    ("n_trades_incremental", pa.int64()),
    ("trades_truncated", pa.bool_()),
    ("trades_consistent", pa.bool_()),   # passes the leg checks data/features.py asserts
    ("resolved", pa.bool_()),
])
