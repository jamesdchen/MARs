"""Recorder configuration: dataclass defaults, optionally overridden by a TOML file and CLI flags.

Every field has a production default, so `python -m polyrec record --out DIR` works without a file.
See config.example.toml for an annotated copy.
"""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36 polyrec/1.0")
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
WS_MARKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Up/down family: <coin>-updown-<len>-<open unix ts>. `coin` and `len` named groups fill the
# coin / contract_length columns; other sources may omit them (columns are then null).
UPDOWN_REGEX = r"^(?P<coin>[a-z0-9]+)-updown-(?P<len>5m|15m|4h)-(?P<ts>\d+)$"


@dataclass
class Source:
    """One gamma discovery query (GET /events/keyset) plus client-side filters."""
    name: str = "crypto-updown"
    # Any /events/keyset query parameters (tag_slug, tag_id, series_id, title_search, ...).
    params: dict = field(default_factory=lambda: {"tag_slug": "up-or-down"})
    slug_regex: str = UPDOWN_REGEX
    exclude_regex: str = ""
    # "start_time": window on event start time [now - max_duration_s, now + lookahead_s]
    #   (recurring markets created ~24 h ahead; we only want the ones about to open / running).
    # "end_date": every open market ending in [now, now + horizon_s] (generic categories).
    window: str = "start_time"
    max_duration_s: int = 4 * 3600 + 900
    lookahead_s: int = 900
    horizon_s: int = 7 * 86400
    pre_open_s: int = 120          # subscribe this long before eventStartTime
    post_close_s: int = 0          # keep writing book rows this long after end (0 = schema parity: stc >= 0)
    linger_s: int = 60             # keep the subscription after end (late events, market_resolved)
    max_pages: int = 20


@dataclass
class WSConfig:
    url: str = WS_MARKET
    markets_per_conn: int = 2      # small shards: one busy connection gets 1013 "slow consumer" closes
    ping_interval_s: float = 10.0  # docs: send text PING every 10 s, server answers PONG
    stale_timeout_s: float = 30.0  # dead-man per connection: no frame (incl. PONG) for this long -> reconnect
    custom_features: bool = True   # best_bid_ask, new_market, market_resolved events
    max_queue: int = 4096          # frames buffered by the client before TCP backpressure
    max_frame_bytes: int = 32 * 1024 * 1024
    backoff_min_s: float = 1.0
    backoff_max_s: float = 60.0
    open_timeout_s: float = 15.0


@dataclass
class ReconcileConfig:
    interval_s: float = 60.0       # REST re-read period per token
    grace_s: float = 1.5           # wait after the REST reply so in-flight WS deltas land before comparing
    mode: str = "book"             # "book": GET /book per token; "books": POST /books batches
    batch: int = 40                # tokens per POST /books
    concurrency: int = 4
    on_bba_mismatch: bool = True   # early reconcile when price_change best_bid/ask disagree with our book
    min_gap_s: float = 10.0        # minimum spacing of early reconciles per token
    size_tol: float = 1e-6


@dataclass
class TradesConfig:
    enabled: bool = True
    poll_interval_s: float = 30.0  # incremental data-api polls per live market (both takerOnly modes)
    page: int = 500
    overlap_s: int = 180           # re-read this far behind the newest seen trade (late indexing)
    settle_delay_s: float = 300.0  # finalize (full sweep + gamma metadata) this long after end
    concurrency: int = 3
    finalize_lookback_s: int = 6 * 3600  # on startup, finalize markets that ended this recently


@dataclass
class OutputConfig:
    flush_interval_s: float = 60.0  # book rows -> parquet part files (crash loses at most this much)
    max_buffer_rows: int = 50_000
    compression_level: int = 9      # parquet zstd level of hourly files
    raw_enabled: bool = True
    raw_zstd_level: int = 6
    raw_keep_order_hashes: bool = False  # per-order hashes are ~70% of the compressed raw log
    raw_flush_interval_s: float = 10.0


@dataclass
class HealthConfig:
    status_interval_s: float = 10.0
    summary_interval_s: float = 60.0
    loop_stall_exit_s: float = 120.0     # watchdog thread: event loop stuck this long -> exit (restart)
    no_data_exit_s: float = 600.0        # subscribed but no market data at all this long -> exit
    max_rss_mb: int = 1500               # exit (and get restarted) above this RSS
    min_free_disk_mb: int = 2000         # stop writing raw log below this; warn


@dataclass
class Config:
    out_dir: str = "./out"
    log_level: str = "INFO"
    log_file: str = ""              # optional rotating log file (journald gets stderr anyway)
    discovery_interval_s: float = 60.0
    sources: list = field(default_factory=lambda: [Source()])
    ws: WSConfig = field(default_factory=WSConfig)
    reconcile: ReconcileConfig = field(default_factory=ReconcileConfig)
    trades: TradesConfig = field(default_factory=TradesConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    http_timeout_s: float = 20.0

    @property
    def out(self) -> Path:
        return Path(self.out_dir)


def _merge(dc, d: dict):
    names = {f.name: f for f in dataclasses.fields(dc)}
    for k, v in d.items():
        if k not in names:
            raise ValueError(f"unknown config key {type(dc).__name__}.{k}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    d: dict = {}
    if path:
        with open(path, "rb") as f:
            d = tomllib.load(f)
    srcs = d.pop("source", None)
    _merge(cfg, d)
    if srcs is not None:
        cfg.sources = []
        for s in srcs:
            src = Source()
            _merge(src, s)
            cfg.sources.append(src)
    for k, v in (overrides or {}).items():
        obj = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], v)
    return cfg
