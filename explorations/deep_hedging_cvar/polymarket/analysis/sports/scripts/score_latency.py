"""Does Polymarket's public score feed arrive before or after the book reacts?

For every score change seen on wss://sports-api.polymarket.com/ws (sports_ws_log.py) for a game we also recorded with
polyrec, take the outcome-0 book mid (1 Hz rows) and find the reaction: the largest 60 s mid move in a window of
[-90 s, +90 s] around the feed time, and the first second at which half of that move is done.
lag = feed receive time - book half-move time (positive: the book moved before the public feed).
Only score changes followed by a book move >= 2 cents are counted (others did not move the price).
Usage: score_latency.py <rec_out_dir> <sports_ws.jsonl>
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent
# polyrec slug -> (substring of homeTeam or awayTeam on the feed, league abbreviation)
GAMES = {"nfl-sea-den-2026-10-16": ("Broncos", "nfl"), "nhl-utah-edm-2026-10-15": ("Oilers", "nhl"),
         "nhl-nyr-nj-2026-10-15": ("Devils", "nhl"), "nhl-nyi-ott-2026-10-15": ("Senators", "nhl"),
         "nhl-buf-tor-2026-10-15": ("Maple Leafs", "nhl"), "nba-okc-hou-2026-10-15": ("Rockets", "nba"),
         "nba-tor-nyk-2026-10-15": ("Knicks", "nba"), "lol-c9-kc-2026-10-15": ("Cloud9", "lol"),
         "cs2-lgc-furia-2026-10-15": ("FURIA", "cs2"), "cfb-gas-old-2026-10-15": ("Old Dominion", "cfb")}


def feed(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if "m" not in d:
            continue
        try:
            m = json.loads(d["m"])
        except ValueError:
            continue
        rows.append(dict(r=d["r"], home=m.get("homeTeam") or "", away=m.get("awayTeam") or "", lg=m.get("leagueAbbreviation"),
                         score=m.get("score"), period=m.get("period"), status=m.get("status"), gid=m.get("gameId")))
    return pd.DataFrame(rows)


def main():
    rec, fp = Path(sys.argv[1]), sys.argv[2]
    F = feed(fp)
    fs = glob.glob(str(rec / "books" / "*" / "*.parquet")) + glob.glob(str(rec / "books" / "*" / "_parts" / "*.parquet"))
    B = pd.concat([pd.read_parquet(f, columns=["market_slug", "token_id", "captured_at", "best_bid", "best_ask"]) for f in fs])
    out = []
    for slug, (team, lg) in GAMES.items():
        f = F[(F.home.str.contains(team, regex=False) | F.away.str.contains(team, regex=False))].sort_values("r")
        if f.empty:
            continue
        ch = f[f.score.ne(f.score.shift())].iloc[1:]
        b = B[B.market_slug == slug]
        if b.empty:
            continue
        tok = sorted(b.token_id.unique())[0]
        b = b[b.token_id == tok].sort_values("captured_at")
        t = (b.captured_at.astype("int64") // 10**6).to_numpy()
        mid = ((b.best_bid + b.best_ask) / 2).to_numpy()
        grid = np.arange(t.min(), t.max() + 1)
        m = pd.Series(mid, index=t).groupby(level=0).last().reindex(grid).ffill().to_numpy()
        for _, e in ch.iterrows():
            r = e.r; i0 = int(r - 90 - grid[0]); i1 = int(r + 90 - grid[0])
            if i0 < 60 or i1 >= len(grid) - 60:
                continue
            w = m[i0 - 60:i1 + 60]
            mv = np.array([w[k + 60] - w[k] for k in range(len(w) - 60)])
            k = int(np.nanargmax(np.abs(mv)))
            if abs(mv[k]) < 0.02:
                out.append(dict(slug=slug, feed_t=r, score=e.score, move_c=mv[k] * 100, moved=False, lag_s=np.nan)); continue
            base, sgn = w[k], np.sign(mv[k])
            half = next(j for j in range(k, k + 61) if sgn * (w[j] - base) >= abs(mv[k]) / 2)
            t_half = grid[i0 - 60] + half
            out.append(dict(slug=slug, feed_t=r, score=e.score, period=e.period, move_c=round(mv[k] * 100, 1), moved=True, lag_s=round(r - t_half, 1)))
    O = pd.DataFrame(out)
    txt = [__doc__]
    if len(O):
        mv = O[O.moved]
        txt += [O.to_string(index=False), f"\nscore changes matched: {len(O)}, with a >=2c book move: {len(mv)}",
                f"lag (feed - book half-move), s: median {mv.lag_s.median():.1f}, mean {mv.lag_s.mean():.1f}, "
                f"share with book first {np.mean(mv.lag_s > 0):.2f}" if len(mv) else ""]
    (HERE / "out" / "score_latency.txt").write_text("\n".join(txt) + "\n"); print("\n".join(txt))


if __name__ == "__main__":
    main()
