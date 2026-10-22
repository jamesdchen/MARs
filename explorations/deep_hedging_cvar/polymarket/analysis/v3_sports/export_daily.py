"""Compact daily export of the pre-game sports recorder (track S, DECISIONS.md D11) for git.

The recorder output (rec_out/, git-ignored) is ~100+ MB/day of book parquet. This writes, per UTC
day, a compact copy to rec_daily/<day>/ that is small enough to commit:
  books_<day>.parquet   outcome-0 token only (outcome 1 is its mirror: bid p <-> ask 1-p),
                        top --levels levels per side, rows before kickoff + post_close only,
                        zstd 19; same columns as the recorder books (data/raw schema); rows in
                        capture-time order per source file (not sorted by market).
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
import pyarrow.compute as pc  # noqa: E402
from polyrec.sinks import dataset_files  # noqa: E402


def days_of(rec, kind):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(rec, kind, '*')) if os.path.isdir(d))


def export_books_day(rec, day, levels, path):
    """Outcome-0 rows (the first outcome seen per market, in file/time order), top `levels` levels,
    streamed file by file into one parquet (rows in capture-time order per source file). pandas
    list columns of full ladders took > 10 GB and an in-memory concat + sort > 4 GB (OOM kills of
    the container at the hourly export, 2026-10-22). Returns (rows, markets) or None."""
    chosen, writer, rows, slugs = {}, None, 0, set()
    tmp = path + '.tmp'
    try:
        for f in dataset_files(rec, 'books', [day]):
            try:
                t = pq.read_table(f)
            except Exception:  # noqa: BLE001  torn part of a live hour
                continue
            t = t.replace_schema_metadata(None)
            so = t.select(['market_slug', 'outcome']).to_pandas()
            for slug, oc in so.drop_duplicates('market_slug').itertuples(index=False):
                chosen.setdefault(slug, oc)
            keep = so["outcome"].values == so["market_slug"].map(chosen).values
            t = t.filter(pa.array(keep))
            if not t.num_rows:
                continue
            for c in ('bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes'):
                t = t.set_column(t.schema.get_field_index(c), c, pc.list_slice(t[c], 0, levels))
            if writer is None:
                writer = pq.ParquetWriter(tmp, t.schema, compression='zstd', compression_level=19)
            writer.write_table(t.cast(writer.schema) if t.schema != writer.schema else t)
            rows += t.num_rows
            slugs |= set(so['market_slug'].unique())
            del t, so
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        return None
    os.replace(tmp, path)
    return rows, len(slugs)


def write(df, path):
    t = df if isinstance(df, pa.Table) else pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(t, path, compression='zstd', compression_level=19)
    return os.path.getsize(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rec', default=os.path.join(HERE, 'rec_out'))
    p.add_argument('--out', default=os.path.join(HERE, 'rec_daily'))
    p.add_argument('--day', action='append', help='UTC day(s); default the last --recent days present')
    p.add_argument('--recent', type=int, default=2, help='default: re-export only the newest N days')
    p.add_argument('--levels', type=int, default=10)
    p.add_argument('--no-books', action='store_true',
                   help='hourly heartbeat: skip the books parquet (keep its counts from the last summary)')
    a = p.parse_args()
    days = a.day or sorted(set(days_of(a.rec, 'books')) | set(days_of(a.rec, 'trades')))[-a.recent:]
    summary = {}
    for day in days:
        od = os.path.join(a.out, day)
        os.makedirs(od, exist_ok=True)
        s = {}
        bp = os.path.join(od, f'books_{day}.parquet')
        sp = os.path.join(od, 'summary.json')
        if a.no_books:
            old = json.load(open(sp)) if os.path.exists(sp) else {}
            s.update({k: v for k, v in old.items() if k.startswith('books_')})
            r = None
        else:
            r = export_books_day(a.rec, day, a.levels, bp)
            s['books_exported_at'] = pd.Timestamp.now('UTC').strftime('%Y-%m-%dT%H:%MZ')
        if r:
            s['books_rows'], s['books_markets'] = r
            s['books_mb'] = round(os.path.getsize(bp) / 1e6, 2)
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
