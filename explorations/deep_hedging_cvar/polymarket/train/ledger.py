"""The trial ledger (EVALUATION.md T4): one JSON row per training run that was
scored on validation, appended under an exclusive lock so parallel runs can
share the file.

`config_hash(cfg)` is the sha256 of the canonical JSON (sorted keys, no
whitespace; stats.sha256_json) of the training-relevant config, i.e. the config
without `seed` and without the run-only keys in RUN_ONLY (paths, threads,
evaluation schedule). Two seeds of one config share the hash; select.py groups
by it.
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
STANDIN_LEDGER = os.path.join(HERE, 'runs', 'standin_trials.jsonl')
RUN_ONLY = ('seed', 'run')


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


def append(row, path=LEDGER):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    row = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': socket.gethostname(),
           'commit': git_commit(), 'train_dirty': git_dirty(), **row}
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
