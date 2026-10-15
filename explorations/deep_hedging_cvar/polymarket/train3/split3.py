"""Pre-register a protocol-v3 split by date (DECISIONS.md D11) from a build directory.

    python split3.py BUILD --dry-run                         # list dates, print the split, write nothing
    python split3.py BUILD --dry-run --out /scratch/s.json   # smoke only: a file marked _dry_run
    python split3.py BUILD --out train3/splits/sports_v3.json   # THE pre-registration (once per track)

It lists the dates available in a build (any builder output: build_sports.py / build_elections.py,
made loadable or not by adapt3.py), per track, and assigns them by D11's rule:

- track S (sports, by game day = slug date, else kickoff UTC date; build_sports.game_day): the
  oldest ~60% of game days train, the next ~20% val, the newest ~20% test (the builder's own
  `assign_splits` arithmetic: n_train = max(1, round(0.6 n)), n_val = round(0.2 n) for n >= 3,
  1 for n = 2). Days recorded after the split is fixed form an extra forward test: they are
  not in the file, so they load as split 3 (purged) until a later entry names them.
- track E (elections, by UTC day of the window open): days at or after --cutoff (default
  2026-11-03, election night, out of scope) are excluded; of the rest, the LAST recorded days are
  test with the same 60 / 20 / 20 arithmetic (oldest train, then val, newest test).
`--exclude DAY ...` drops days (e.g. a partial first recording day) before the rule is applied;
`--min-markets` drops days with fewer markets.

Output: a JSON file the builders read directly (sports: `build_sports.py --splits FILE`, layout
{"days": {...}, "default": "exclude"}; elections: `build_elections.py --split-days FILE`, a flat
{day: name}) and train3/envs3.py V3Adapter reads as `split_map`. Keys starting with '_' are
provenance (track, rule, build, its markets.bin sha256, per-day market counts, time). The file's
sha256 is written next to it (`<file>.sha256`) and appended to train3/split_registry.jsonl.
A non-dry run refuses to overwrite an existing split file, and refuses a build or output path
containing 'jun15'.
"""

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import envs3  # noqa: E402

REGISTRY = os.path.join(HERE, 'split_registry.jsonl')


def d11_rule(days):
    """build_sports.assign_splits's auto rule (D11): oldest ~60% train, next ~20% val, newest ~20% test."""
    days = sorted(days)
    n = len(days)
    n_tr = max(1, int(round(0.6 * n))) if n else 0
    n_va = int(round(0.2 * n)) if n >= 3 else (1 if n == 2 else 0)
    return {d: ('train' if i < n_tr else ('val' if i < n_tr + n_va else 'test')) for i, d in enumerate(days)}


def available(build):
    """{track: {day: n_markets}} over the markets of a build."""
    out = {}
    for m in envs3.market_table(build):
        track = {1: 'sports', 2: 'elections'}.get(m['category'])
        if track is None:
            track = envs3.detect_kind(build)
        out.setdefault(track, {}).setdefault(m['date'], 0)
        out[track][m['date']] += 1
    return out


def make_split(build, track, exclude=(), cutoff='2026-11-03', min_markets=1):
    avail = available(build)
    if track not in avail:
        raise SystemExit(f'{build}: no {track} markets (tracks present: {sorted(avail)})')
    per_day = avail[track]
    dropped = {}
    days = []
    for d, n in sorted(per_day.items()):
        if d in exclude:
            dropped[d] = 'excluded (--exclude)'
        elif track == 'elections' and d >= cutoff:
            dropped[d] = f'at or after --cutoff {cutoff} (election night out of scope, D11)'
        elif n < min_markets:
            dropped[d] = f'fewer than {min_markets} markets'
        else:
            days.append(d)
    assign = d11_rule(days)
    rule = ('D11 track S: by game day (slug date, else kickoff UTC date); oldest ~60% train, next ~20% val, '
            'newest ~20% test' if track == 'sports' else
            'D11 track E: by UTC day of the window open, days before the cutoff; oldest ~60% train, next ~20% val, '
            'the last recorded ~20% test')
    meta = {'_split3': 1, '_track': track, '_rule': rule, '_build': os.path.realpath(build),
            '_build_markets_sha256': envs3.sha256_file(os.path.join(build, 'markets.bin')),
            '_markets_per_day': per_day, '_dropped': dropped, '_created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            '_counts': {s: sorted(d for d, v in assign.items() if v == s) for s in ('train', 'val', 'test')}}
    full = {**assign, **{d: 'exclude' for d in dropped}}
    if track == 'sports':
        doc = {**meta, 'days': full, 'default': 'exclude'}       # build_sports.py --splits layout
    else:
        doc = {**meta, **full}                                    # build_elections.py --split-days layout
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('build')
    ap.add_argument('--track', choices=('sports', 'elections'), help='default: the build\'s kind')
    ap.add_argument('--out', help='split file to write (required unless --dry-run)')
    ap.add_argument('--dry-run', action='store_true', help='print only; with --out, write a file marked _dry_run '
                    '(V3Adapter refuses it outside --smoke) and register nothing')
    ap.add_argument('--exclude', nargs='*', default=[], help='days to drop before the rule')
    ap.add_argument('--cutoff', default='2026-11-03', help='elections: days at or after this are dropped')
    ap.add_argument('--min-markets', type=int, default=1)
    a = ap.parse_args(argv)
    envs3.no_jun15(a.build)
    envs3.no_jun15(a.out)
    track = a.track or envs3.detect_kind(a.build)
    print(json.dumps({'build': os.path.realpath(a.build), 'available': available(a.build)}, indent=1))
    doc = make_split(a.build, track, set(a.exclude), a.cutoff, a.min_markets)
    if a.dry_run:
        doc['_dry_run'] = True
    raw = (json.dumps(doc, indent=1, sort_keys=False) + '\n').encode()
    sha = hashlib.sha256(raw).hexdigest()
    print(json.dumps({'track': track, 'counts': doc['_counts'], 'dropped': doc['_dropped'], 'sha256': sha,
                      'dry_run': a.dry_run}, indent=1))
    if a.dry_run and not a.out:
        print(raw.decode())
        return doc, sha
    if not a.out:
        raise SystemExit('--out is required (or --dry-run)')
    if not a.dry_run and os.path.exists(a.out):
        raise SystemExit(f'{a.out} exists: a pre-registered split is never overwritten')
    if a.dry_run and os.path.realpath(a.out).startswith(os.path.realpath(HERE) + os.sep):
        raise SystemExit('--dry-run files are not written inside train3/ (use a scratch path)')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'wb') as f:
        f.write(raw)
    with open(a.out + '.sha256', 'w') as f:
        f.write(f'{sha}  {os.path.basename(a.out)}\n')
    if not a.dry_run:
        import ledger                                    # train/ledger.py: locked append
        ledger.append({'kind': 'split_registration', 'track': track, 'file': os.path.abspath(a.out), 'sha256': sha,
                       'build': doc['_build'], 'build_markets_sha256': doc['_build_markets_sha256'],
                       'counts': doc['_counts'], 'dropped': doc['_dropped']}, REGISTRY)
    print(f'wrote {a.out} (sha256 {sha})')
    return doc, sha


if __name__ == '__main__':
    main()
