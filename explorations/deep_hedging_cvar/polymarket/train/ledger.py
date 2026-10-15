"""The trial ledger (EVALUATION.md T4) and the run index.

Two append-only JSONL files, written under an exclusive lock so parallel runs can
share them:

- `trials.jsonl` (LEDGER): ONE ROW PER VALIDATION EVALUATION, kind 'val_eval',
  split 1: every periodic and final evaluation, every fill/latency variant.
  final/common.py count_trials counts one val evaluation per record (records
  without a split field count as val; kind 'annotation' rows are skipped), and
  globs every file named trials.jsonl under polymarket/, so nothing else here is
  named trials.jsonl.
- `runs.jsonl` (RUNS): one row per training run (kind 'train_run', or
  'train_run_failed'), with the full config, the final checkpoint and its sha256 and
  the final validation metrics per variant. select.py, summary.py and seeds.py read it.

`config_hash(cfg)` is the sha256 of the canonical JSON (sorted keys, no
whitespace; stats.sha256_json) of the training-relevant config, i.e. the config
without `seed` and without the run-only keys in RUN_ONLY. Two seeds of one config
share the hash; select.py groups by it.
"""

import copy
import fcntl
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'eval'))
from stats import sha256_json  # noqa: E402

LEDGER = os.path.join(HERE, 'trials.jsonl')
RUNS = os.path.join(HERE, 'runs.jsonl')
STANDIN_LEDGER = os.path.join(HERE, 'runs', 'standin_trials.jsonl')
STANDIN_RUNS = os.path.join(HERE, 'runs', 'standin_runs.jsonl')
RUN_ONLY = ('seed', 'run')


def runs_path(ledger_path):
    """The run index that goes with a ledger file (same directory)."""
    d, b = os.path.split(ledger_path)
    return os.path.join(d, b.replace('trials', 'runs') if 'trials' in b else 'runs_' + b)


def semantic(cfg):
    c = copy.deepcopy(cfg)
    for k in RUN_ONLY:
        c.pop(k, None)
    return c


def config_hash(cfg):
    return sha256_json(semantic(cfg))


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=HERE,
                             capture_output=True, text=True, timeout=10).stdout
        return bool(out.strip())
    except Exception:
        return None


_GIT = {}


def _git():
    if not _GIT:
        _GIT.update(commit=git_commit(), train_dirty=git_dirty())
    return dict(_GIT)


def append(row, path=LEDGER):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    row = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': socket.gethostname(), **_git(), **row}
    line = json.dumps(row, sort_keys=True, default=float) + '\n'
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return row


def read(path=LEDGER):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def n_val_evals(path=LEDGER):
    """N as final/common.py counts it: every record except annotations."""
    return sum(1 for r in read(path) if str(r.get('kind', '')).lower() not in
               ('note', 'comment', 'meta', 'header', 'annotation'))
