"""Compact daily export of the pre-game sports recorder (track S, DECISIONS.md D11) for git.

The recorder output (rec_out/, git-ignored) is ~100+ MB/day of book parquet. This writes, per UTC
day, a compact copy to rec_daily/<day>/ that is small enough to commit:
  books_<day>.parquet   outcome-0 token only (outcome 1 is its mirror: bid p <-> ask 1-p),
                        top --levels levels per side, rows before kickoff + post_close only,
                        zstd 19; same columns as the recorder books (data/raw schema).
  trades_<day>.parquet  every finalized leg (recorder trades/, data/raw trades schema)
  markets_<day>.parquet finalized market rows (data/raw markets schema)
  coverage_<day>.jsonl  liveness log (merge/build use it to drop gaps)
build_sports.py reads rec_daily/ as a --rec input only for books; use rec_out/ when available.

    python export_daily.py --rec rec_out --out rec_daily [--day 2026-10-15] [--levels 10]
"""

import argparse
import glob
import json
import os
import shutil
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'recorder'))
from polyrec.sinks import read_dataset  # noqa: E402


def days_of(rec, kind):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(rec, kind, '*')) if os.path.isdir(d))


def compact_books(df, levels):
    first = df.groupby('market_slug')['outcome'].transform('first')
    df = df[df['outcome'] == first].copy()
    for c in ('bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes'):
        df[c] = df[c].map(lambda x: list(x[:levels]) if x is not None else x)
    return df.sort_values(['market_slug', 'captured_at']).reset_index(drop=True)


def write(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression='zstd',
                   compression_level=19)
    return os.path.getsize(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rec', default=os.path.join(HERE, 'rec_out'))
    p.add_argument('--out', default=os.path.join(HERE, 'rec_daily'))
    p.add_argument('--day', action='append', help='UTC day(s); default every day in the recorder output')
    p.add_argument('--levels', type=int, default=10)
    a = p.parse_args()
    books_all = None
    days = a.day or sorted(set(days_of(a.rec, 'books')) | set(days_of(a.rec, 'trades')))
    summary = {}
    for day in days:
        od = os.path.join(a.out, day)
        os.makedirs(od, exist_ok=True)
        s = {}
        if books_all is None:
            books_all = read_dataset(a.rec, 'books')
            books_all = books_all.to_pandas() if hasattr(books_all, 'to_pandas') else books_all
        b = books_all[books_all['captured_at'].dt.strftime('%Y-%m-%d') == day]
        if len(b):
            b = compact_books(b, a.levels)
            s['books_rows'] = len(b)
            s['books_markets'] = int(b['market_slug'].nunique())
            s['books_mb'] = round(write(b, os.path.join(od, f'books_{day}.parquet')) / 1e6, 2)
        for kind in ('trades', 'markets'):
            files = sorted(glob.glob(os.path.join(a.rec, kind, day, '*.parquet')) +
                           glob.glob(os.path.join(a.rec, kind, day, '_parts', '*.parquet')))
            if files:
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates()
                s[f'{kind}_rows'] = len(df)
                s[f'{kind}_mb'] = round(write(df, os.path.join(od, f'{kind}_{day}.parquet')) / 1e6, 2)
        cov = os.path.join(a.rec, 'coverage', f'coverage_{day}.jsonl')
        if os.path.exists(cov):
            shutil.copy(cov, os.path.join(od, f'coverage_{day}.jsonl'))
        summary[day] = s
        with open(os.path.join(od, 'summary.json'), 'w') as f:
            json.dump(s, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
