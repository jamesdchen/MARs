"""Validate the touch-based mid proxy of maker_econ.py against recorded book mids (tonight's finished markets).

Trades come from data-api /trades (both takerOnly modes): legs of a tx that are not the taker row are maker legs.
The proxy is rebuilt exactly as in maker_econ.mid_series (YES frame = outcome 0) and compared with the polyrec book
mid of outcome 0 at the trade (block) time - 3 s (block time lags the CLOB match by ~2-4 s, onchain/README).
Usage: proxy_check.py <rec_out_dir> slug [slug ...]
"""
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maker_econ import mid_series  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
D = "https://data-api.polymarket.com/trades"


def trades(cond, taker_only):
    out, off = [], 0
    while off <= 3000:
        r = requests.get(D, params={"market": cond, "limit": 500, "offset": off, "takerOnly": str(taker_only).lower()}, timeout=30)
        j = r.json() if r.ok else []
        out += j
        if len(j) < 500:
            break
        off += 500
    return pd.DataFrame(out)


def main():
    rec = Path(sys.argv[1]); slugs = sys.argv[2:]
    fs = glob.glob(str(rec / "books" / "*" / "*.parquet")) + glob.glob(str(rec / "books" / "*" / "_parts" / "*.parquet"))
    B = pd.concat([pd.read_parquet(f, columns=["market_slug", "outcome", "token_id", "captured_at", "best_bid", "best_ask"]) for f in fs])
    rows = []
    for slug in slugs:
        g = requests.get("https://gamma-api.polymarket.com/markets", params={"slug": slug}, timeout=30).json()[0]
        toks = json.loads(g["clobTokenIds"])
        A = trades(g["conditionId"], False); T = trades(g["conditionId"], True)
        if A.empty or T.empty:
            continue
        T["k"] = T.transactionHash + T.proxyWallet + T.side + T["size"].astype(str)
        A["k"] = A.transactionHash + A.proxyWallet + A.side + A["size"].astype(str)
        A["role"] = np.where(A.k.isin(set(T.k)), "taker", "maker")
        A["outcome_idx"] = A.outcomeIndex.astype(int)
        A["p_yes"] = np.where(A.outcome_idx == 0, A.price, 1 - A.price)
        A["dir"] = np.where((A.side == "BUY") == (A.outcome_idx == 0), 1, -1)
        A["t"] = A.timestamp.astype(int); A = A.rename(columns={"transactionHash": "tx_hash"}).sort_values("t")
        t, m = mid_series(A)
        b = B[(B.market_slug == slug) & (B.token_id == toks[0])].sort_values("captured_at")
        bt = (b.captured_at.astype("int64") // 10**6).to_numpy(); bm = ((b.best_bid + b.best_ask) / 2).to_numpy()
        ok = (t >= bt.min() + 5) & (t <= bt.max())
        i = np.searchsorted(bt, t[ok] - 3, "right") - 1
        err = (m[ok] - bm[i]) * 100
        sp = ((b.best_ask - b.best_bid).to_numpy()[i]) * 100
        rows.append(dict(slug=slug, n=int(ok.sum()), mean_err_c=np.mean(err), mae_c=np.mean(np.abs(err)), p90_abs_c=np.percentile(np.abs(err), 90),
                         within_half_spread=np.mean(np.abs(err) <= sp / 2 + 1e-9), book_spread_c=np.median(sp)))
        time.sleep(0.5)
    R = pd.DataFrame(rows).round(3)
    txt = __doc__ + "\n" + R.to_string(index=False) + "\n"
    (HERE / "out" / "proxy_check.txt").write_text(txt); print(txt)


if __name__ == "__main__":
    main()
