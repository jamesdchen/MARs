"""Compact per-kickoff-day export of the PolyOrderBooks backfill (raw_pob/, git-ignored) for git.

Same format as w1's ../v3_sports/export_daily.py (rec_daily/), grouped by the market's KICKOFF day
(gamma gameStartTime, UTC) instead of by captured_at day:
  books_<day>.parquet     outcome-0 token only (outcome 1 is its exact mirror: bid p <-> ask 1-p,
                          checked 200/200 on the probe), top --levels levels per side, zstd 19,
                          data/raw book schema (build_sports.py --rec reads it)
  trades_<day>.parquet    every fill leg in the market's window (data/raw trades schema)
  markets_<day>.parquet   one row per market (data/raw markets schema + sports columns)
  coverage_<day>.jsonl    one line per market: book span, rows, 1s/1m chunk counts, start detection
  summary.json
Each run rewrites the days it touches (later runs can add markets to a day still inside the plan's
history window). Refuses to write a file over 50 MB.

    python export_pob_daily.py --raw raw_pob --out pob_daily [--day 2026-10-13]
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_MB = 50


def read_all(raw, kind):
    files = sorted(glob.glob(os.path.join(raw, kind, '*', '*.parquet')) +
                   glob.glob(os.path.join(raw, kind, '*', '_parts', '*.parquet')))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def merge_old(df, path, keys):
    """Union with the already-committed file, so a fresh raw_pob never shrinks a day."""
    if os.path.exists(path):
        old = pd.read_parquet(path)
        old = old[~old.set_index(keys).index.isin(df.set_index(keys).index)] if len(df) else old
        df = pd.concat([old, df], ignore_index=True)
    return df


def write(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression='zstd',
                   compression_level=19)
    mb = os.path.getsize(path) / 1e6
    if mb > MAX_MB:
        os.remove(path)
        sys.exit(f'{path}: {mb:.1f} MB > {MAX_MB} MB; split it before committing')
    return round(mb, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default=os.path.join(HERE, 'raw_pob'))
    p.add_argument('--out', default=os.path.join(HERE, 'pob_daily'))
    p.add_argument('--day', action='append')
    p.add_argument('--levels', type=int, default=10)
    a = p.parse_args()
    mk = read_all(a.raw, 'markets')
    if mk.empty:
        sys.exit('no markets in raw')
    mk = mk.drop_duplicates('slug', keep='last')
    mk['kday'] = mk['kickoff'].dt.strftime('%Y-%m-%d')
    books = read_all(a.raw, 'books')
    trades = read_all(a.raw, 'trades')
    have_books = set(books['market_slug'].unique()) if len(books) else set()
    summary = {}
    for day in sorted(a.day or mk['kday'].unique()):
        m = mk[(mk['kday'] == day) & mk['slug'].isin(have_books)]
        if m.empty:
            continue
        od = os.path.join(a.out, day)
        os.makedirs(od, exist_ok=True)
        o0 = dict(zip(m['slug'], m['outcome_0']))
        b = books[books['market_slug'].isin(o0)]
        b = b[b['outcome'] == b['market_slug'].map(o0)].copy()
        for c in ('bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes'):
            b[c] = b[c].map(lambda x: list(x[:a.levels]) if x is not None else x)
        b = b.drop_duplicates(['market_slug', 'token_id', 'captured_at'], keep='last')
        bp_, tp_, mp_ = (os.path.join(od, f'{k}_{day}.parquet') for k in ('books', 'trades', 'markets'))
        b = merge_old(b, bp_, ['market_slug', 'token_id', 'captured_at'])
        b = b.sort_values(['market_slug', 'captured_at']).reset_index(drop=True)
        t = trades[trades['slug'].isin(o0)].drop_duplicates() if len(trades) else trades
        t = merge_old(t, tp_, ['slug'])                  # a market's legs come whole from one fetch
        m = merge_old(m.drop(columns=['kday']), mp_, ['slug'])
        s = {'markets': int(len(m)), 'books_markets': int(b['market_slug'].nunique()), 'books_rows': int(len(b)),
             'books_mb': write(b, bp_), 'trades_rows': int(len(t)), 'trades_mb': write(t, tp_),
             'markets_mb': write(m, mp_)}
        cp_ = os.path.join(od, f'coverage_{day}.jsonl')
        cov_old = {}
        if os.path.exists(cp_):
            cov_old = {json.loads(x)['slug']: x for x in open(cp_) if x.strip()}
        new_slugs = set(b['market_slug'].unique()) & set(o0)
        with open(cp_, 'w') as f:
            for slug, line in cov_old.items():
                if slug not in new_slugs:
                    f.write(line if line.endswith('\n') else line + '\n')
            for slug, g in b[b['market_slug'].isin(new_slugs)].groupby('market_slug'):
                ko = m.loc[m['slug'] == slug, 'kickoff'].iloc[0]
                sd = os.path.join(a.raw, '_state', 'books', slug)
                ch = [json.load(open(x)) for x in glob.glob(os.path.join(sd, '*_*.json'))]
                stp = os.path.join(a.raw, '_state', 'start', f'{slug}.json')
                st = json.load(open(stp)) if os.path.exists(stp) else None
                rec = {'slug': slug, 'league': slug.split('-')[0], 'kickoff': ko.isoformat(),
                       'book_first': g['captured_at'].min().isoformat(), 'book_last': g['captured_at'].max().isoformat(),
                       'rows': int(len(g)), 'hours_before_kickoff': round((ko - g['captured_at'].min()).total_seconds() / 3600, 2),
                       'chunks': {r: sum(1 for c in ch if c.get('res') == r and c.get('status') in ('ok', 'empty'))
                                  for r in ('1s', '1m')},
                       'chunks_forbidden': sum(1 for c in ch if c.get('status') == 'forbidden'),
                       'start_detect': st}
                f.write(json.dumps(rec) + '\n')
        summary[day] = s
        with open(os.path.join(od, 'summary.json'), 'w') as f:
            json.dump(s, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
