"""Fair value for Polymarket crypto Up/Down markets as short-dated digital options.

Resolution rule (gamma `description`, checked on train markets, see REPORT.md §1):
    Up  <=>  TWAP60(end) >= TWAP60(start)
where TWAP60(T) is the value of the Chainlink "<coin>-usd-twap-60s-streams" data stream at T,
i.e. the time-weighted average of the Chainlink price over [T-60s, T]. `priceToBeat` of a market
equals `finalPrice` of the market that ends at its start time, so the strike K is fully known at
market open (the opening TWAP covers the 60 s *before* open). Ties resolve Up.

Model: driftless lognormal, p = Phi(ln(F/K)/sqrt(V) - sqrt(V)/2) with
    tau = end - now (s), a = 60 s TWAP window
    tau >= a : F = S,                     V = sigma^2 * (tau - a + a/3)
    tau <  a : F = w*M + (1-w)*S          V = sigma^2 * tau^3 / (3 a^2) * (S/F)^2
               (M = running average of the already-observed part of the closing window, w = (a-tau)/a)
sigma is a per-sqrt(second) vol (EWMA of Binance 1 s log returns, see `ewma_sigma`).
Optional fat tails: Student-t with `nu` dof scaled to unit variance.

Train-only loaders: everything is filtered at read time to markets whose whole [open, end] window
lies inside 2026-08-21 00:00 .. 2026-08-23 00:00 UTC, and Binance data to < 2026-08-23 00:00 UTC
(2026-08-20 is used only as EWMA warm-up). Val (Aug 23) / test (Aug 24) rows are never materialised.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import special, stats
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "data" / "raw"
CACHE = HERE / "cache"
FIG = HERE / "figures"

TRAIN_START = pd.Timestamp("2026-08-21", tz="UTC")
TRAIN_END = pd.Timestamp("2026-08-23", tz="UTC")          # exclusive for data, inclusive for market end
UNDERLYING_START = pd.Timestamp("2026-08-20", tz="UTC")   # EWMA warm-up only
T0 = int(UNDERLYING_START.timestamp())
T_TRAIN_START = int(TRAIN_START.timestamp())
T_TRAIN_END = int(TRAIN_END.timestamp())
LEN_S = {"5m": 300, "15m": 900, "4h": 14400}
TWAP_S = 60.0
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]


# --------------------------------------------------------------------------------------------
# Pricer
# --------------------------------------------------------------------------------------------
def _forward_and_var(S, sigma, tau, known_avg=None, twap=TWAP_S):
    S = np.asarray(S, float)
    sigma = np.asarray(sigma, float)
    tau = np.maximum(np.asarray(tau, float), 1e-9)
    inwin = tau < twap
    M = S if known_avg is None else np.where(np.isnan(np.asarray(known_avg, float)), S, known_avg)
    w = np.where(inwin, (twap - tau) / twap, 0.0)
    F = w * M + (1.0 - w) * S
    V = np.where(inwin, sigma**2 * tau**3 / (3.0 * twap**2) * (S / F) ** 2,
                 sigma**2 * (tau - twap + twap / 3.0))
    return F, np.maximum(V, 1e-18), w


def digital_up_prob(S, K, sigma, tau, known_avg=None, nu=None, twap=TWAP_S):
    """P(TWAP_end >= K) for an Up share.

    S: spot now; K: opening reference (TWAP60 at open); sigma: vol per sqrt(second);
    tau: seconds to market end; known_avg: running mean of S over the elapsed part of the closing
    TWAP window (only used when tau < twap); nu: Student-t dof (None = Gaussian).
    All array-like, broadcast.
    """
    F, V, _ = _forward_and_var(S, sigma, tau, known_avg, twap)
    sd = np.sqrt(V)
    d = np.log(F / np.asarray(K, float)) / sd - sd / 2.0
    if nu is None:
        return special.ndtr(d)
    return stats.t.cdf(d * np.sqrt(nu / (nu - 2.0)), nu)


def digital_up_delta(S, K, sigma, tau, known_avg=None, nu=None, twap=TWAP_S):
    """dp/dS: coins of underlying per Up share (multiply by S for USD notional)."""
    F, V, w = _forward_and_var(S, sigma, tau, known_avg, twap)
    sd = np.sqrt(V)
    d = np.log(F / np.asarray(K, float)) / sd - sd / 2.0
    if nu is None:
        dens = np.exp(-0.5 * d * d) / np.sqrt(2 * np.pi)
    else:
        c = np.sqrt(nu / (nu - 2.0))
        dens = stats.t.pdf(d * c, nu) * c
    return dens * (1.0 - w) / (F * sd)


# --------------------------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------------------------
def ewma_sigma(close: np.ndarray, halflife_s: float, sample_s: int = 1, seed_s: int = 3600) -> np.ndarray:
    """EWMA per-sqrt-second vol at each second from a contiguous 1 s close array.

    Uses overlapping `sample_s`-second log returns r_k (variance-per-second r_k^2/k), so sample_s>1
    damps microstructure noise. Output[i] uses returns ending at or before i (causal).
    """
    lc = np.log(close)
    k = int(sample_s)
    x = np.zeros_like(lc)
    x[k:] = (lc[k:] - lc[:-k]) ** 2 / k
    lam = 0.5 ** (1.0 / halflife_s)
    v0 = x[k:k + seed_s].mean()
    v, _ = lfilter([1 - lam], [1, -lam], x, zi=[lam * v0])
    v[:k] = v0
    return np.sqrt(v)


# --------------------------------------------------------------------------------------------
# Train-only loaders
# --------------------------------------------------------------------------------------------
def _assert_train(df: pd.DataFrame):
    o = df["market_open"]
    e = df["market_end_at"]
    assert (o >= TRAIN_START).all() and (e <= TRAIN_END).all(), "LEAKAGE: non-train market"
    if "captured_at" in df:
        assert (df["captured_at"] < TRAIN_END).all(), "LEAKAGE: snapshot on/after 2026-08-23"


def load_books(cls=("5m", "15m", "4h"), columns=None, up_only=True) -> pd.DataFrame:
    """Train-split order-book snapshots (Up token by default)."""
    cols = columns or ["market_slug", "coin", "contract_length", "outcome", "is_winning_outcome",
                       "market_end_at", "captured_at", "seconds_to_close", "best_bid", "best_ask",
                       "crossed"]
    need = set(cols) | {"market_end_at", "captured_at", "contract_length", "outcome"}
    out = []
    for cl in cls:
        filt = [("market_end_at", "<=", TRAIN_END), ("captured_at", "<", TRAIN_END),
                ("market_end_at", ">=", TRAIN_START + pd.Timedelta(seconds=LEN_S[cl]))]
        if up_only:
            filt.append(("outcome", "==", "Up"))
        t = pq.read_table(RAW / f"updown_{cl}.parquet", columns=sorted(need), filters=filt).to_pandas()
        t["market_open"] = t["market_end_at"] - pd.Timedelta(seconds=LEN_S[cl])
        t = t[t["market_open"] >= TRAIN_START]
        out.append(t)
    df = pd.concat(out, ignore_index=True)
    _assert_train(df)
    return df


def train_markets() -> pd.DataFrame:
    b = load_books(columns=["market_slug", "coin", "contract_length", "outcome", "is_winning_outcome",
                            "market_end_at", "captured_at"])
    m = (b.groupby("market_slug")
         .agg(coin=("coin", "first"), cl=("contract_length", "first"), open=("market_open", "first"),
              end=("market_end_at", "first"), winner_up=("is_winning_outcome", "first"),
              n_snap=("captured_at", "size"))
         .reset_index().rename(columns={"market_slug": "slug"}))
    m["winner_up"] = m["winner_up"].astype(int)
    m["t_open"] = (m["open"].astype("int64") // 10**6).astype(np.int64)
    m["t_end"] = (m["end"].astype("int64") // 10**6).astype(np.int64)
    assert (m.t_open >= T_TRAIN_START).all() and (m.t_end <= T_TRAIN_END).all()
    return m


def load_underlying() -> dict[str, np.ndarray]:
    """{coin: close array}; index i <-> kline open_time T0+i (close knowable at T0+i+1).
    Covers 2026-08-20 00:00 .. 2026-08-22 23:59:59 only."""
    t = pq.read_table(RAW / "underlying.parquet", columns=["coin", "open_time", "close"],
                      filters=[("open_time", ">=", T0), ("open_time", "<", T_TRAIN_END)]).to_pandas()
    assert t.open_time.max() < T_TRAIN_END, "LEAKAGE: underlying beyond train"
    out = {}
    n = T_TRAIN_END - T0
    for c, g in t.groupby("coin", observed=True):
        g = g.sort_values("open_time")
        arr = np.full(n, np.nan)
        arr[g.open_time.values - T0] = g.close.values
        arr = pd.Series(arr).ffill().bfill().values
        out[str(c)] = arr
    return out


def spot_at(close: np.ndarray, t) -> np.ndarray:
    """Last price knowable at unix time t (float ok): close of kline opened at floor(t)-1."""
    i = np.floor(np.asarray(t, float)).astype(np.int64) - 1 - T0
    return close[i]


def twap_binance(close: np.ndarray, T: int, a: int = 60) -> float:
    """Binance proxy of TWAP60(T): mean of prices at instants T-a+1..T = closes of klines T-a..T-1."""
    return close[T - a - T0:T - T0].mean()


# ---------------- gamma metadata + taker trades (train slugs only) ----------------
HEADERS = {"User-Agent": "Mozilla/5.0 (research)", "Accept": "application/json"}


def _get(url, tries=6):
    d = 1.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(d)
            d *= 2


def load_market_meta(slug: str) -> dict:
    """{'gamma': {...}, 'taker': [[tx, ts, wallet, side, outcomeIdx, price, size, asset], ...]}.
    Reads the data agent's per-slug cache (read-only) or fetches into ./cache/trades/."""
    assert int(slug.rsplit("-", 1)[1]) >= T_TRAIN_START, slug
    p = RAW / "cache" / f"{slug}.json.gz"
    if p.exists():
        try:
            with gzip.open(p, "rt") as f:
                d = json.load(f)
            return {"gamma": d["gamma"], "taker": d["taker"]}
        except Exception:  # noqa: BLE001  (file being written)
            pass
    q = CACHE / "trades" / f"{slug}.json.gz"
    if q.exists():
        with gzip.open(q, "rt") as f:
            return json.load(f)
    g = _get(f"https://gamma-api.polymarket.com/markets?slug={slug}&closed=true")[0]
    ev = (g.get("events") or [{}])[0]
    g["eventMetadata"] = ev.get("eventMetadata")
    rows, off = [], 0
    while True:
        page = _get(f"https://data-api.polymarket.com/trades?market={g['conditionId']}"
                    f"&limit=1000&offset={off}&takerOnly=true")
        rows += [[r["transactionHash"], int(r["timestamp"]), r["proxyWallet"], r["side"],
                  int(r["outcomeIndex"]), float(r["price"]), float(r["size"]), r["asset"]] for r in page]
        if len(page) < 1000 or off >= 9000:
            break
        off += 1000
    d = {"gamma": {k: g.get(k) for k in ("conditionId", "description", "resolutionSource", "eventMetadata",
                                          "feeSchedule", "outcomes", "eventStartTime", "endDate")},
         "taker": rows}
    q.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(q, "wt") as f:
        json.dump(d, f)
    return d


# --------------------------------------------------------------------------------------------
# Convenience: model p / delta at arbitrary (integer-second) decision times for one coin
# --------------------------------------------------------------------------------------------
def model_at(close: np.ndarray, sigma: np.ndarray, t_now, t_end, K, shift: int = 0, nu=None,
             twap: int = 60, with_delta: bool = False):
    """Model p (and delta) at unix times t_now for markets ending t_end with strike K.

    Uses only information knowable at t_now (kline closes up to floor(t_now)-1, sigma at the same
    index). `shift` moves the effective TWAP window by `shift` seconds (Chainlink alignment:
    best fit shift=-4, see REPORT §1).
    """
    t_now = np.floor(np.asarray(t_now, float)).astype(np.int64)
    t_end_eff = np.asarray(t_end, np.int64) + shift
    i = t_now - 1 - T0
    S = close[i]
    sg = sigma[i]
    tau = (t_end_eff - t_now).astype(float)
    # running mean of prices at instants (t_end_eff-twap, t_now]  = closes of klines [t_end_eff-twap, t_now-1]
    cs = np.concatenate([[0.0], np.cumsum(close)])
    lo = t_end_eff - twap - T0
    hi = i + 1
    n = hi - lo
    known = np.where(n > 0, (cs[np.clip(hi, 0, len(close))] - cs[np.clip(lo, 0, len(close))]) / np.maximum(n, 1), np.nan)
    tau_c = np.maximum(tau, 0.5)
    p = digital_up_prob(S, K, sg, tau_c, known_avg=known, nu=nu, twap=twap)
    # after the effective window closed the outcome (on Binance) is known
    done = tau <= 0
    if done.any():
        fin = (cs[np.clip(t_end_eff - T0, 0, len(close))] - cs[np.clip(t_end_eff - twap - T0, 0, len(close))]) / twap
        p = np.where(done, (fin >= K).astype(float), p)
    if not with_delta:
        return p
    dlt = np.where(done, 0.0, digital_up_delta(S, K, sg, tau_c, known_avg=known, nu=nu, twap=twap))
    return p, dlt, S
