#!/usr/bin/env python3
"""Protocol v2 build (DECISIONS D7): the v1 build with the Aug 24 markets relabelled as validation.

data/build/ is not touched. data/build_v2/ gets
  ticks.bin feat.bin book.bin fills.bin fill_index.bin   hard links to data/build/ (no copies;
                                                         symlinks if hard links are impossible)
  markets.bin     the v1 records with split 2 (test, Aug 24) -> 1 (val); order, offsets unchanged
  meta.json       v1 meta with the split counts/rule updated, test_rows emptied (the v1 test row
                  hash is kept as v1_test_rows), v2 note and sha256 of every other file
  feat_meta.json  v1 feat_meta plus the v2 note (column layout identical)
  splits.json     {slug: split} with the new labels, plus a "_v2_note" string entry

Markets stay sorted by (split, open_ts, slug): the Aug 24 markets already followed the Aug 23 ones.

    python data/build_v2.py && python -c "import sys; sys.path.insert(0, 'data'); \
        from load import verify; verify('data/build_v2')"
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "build"
OUT = HERE / "build_v2"
NOTE = "v2: Aug 24 relabelled val (DECISIONS D7)"
LINKED = ["ticks.bin", "feat.bin", "book.bin", "fills.bin", "fill_index.bin"]
SPLIT_NAMES = {0: "train", 1: "val", 2: "test", 3: "purged"}
CLEN = {"5m": 300, "15m": 900, "4h": 14400}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(path: Path, data):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    if isinstance(data, np.ndarray):
        data.tofile(tmp)
    else:
        tmp.write_text(data)
    os.replace(tmp, path)


def link(src: Path, dst: Path) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.exists() and os.path.samefile(src, dst):
            return "hardlink" if not dst.is_symlink() else "symlink"
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return "symlink"


def main():
    import sys
    sys.path.insert(0, str(HERE))
    from load import load, verify
    verify(SRC)                                   # the source must be the intact v1 build
    OUT.mkdir(exist_ok=True)
    kinds = {f: link(SRC / f, OUT / f) for f in LINKED}
    _, markets, meta = load(SRC)
    src_sha = {f: sha256_file(SRC / f) for f in ["markets.bin", "meta.json", "feat_meta.json", "splits.json"]}

    mk = markets.copy()
    was_test = mk["split"] == 2
    mk["split"][was_test] = 1
    assert (mk["split"] != 2).all()
    write_atomic(OUT / "markets.bin", mk)

    slugs = meta["slugs"]
    splits = {s: int(v) for s, v in zip(slugs, mk["split"])}
    splits["_v2_note"] = NOTE + "; split 2 (test, 2026-08-24) relabelled 1 (val)"
    write_atomic(OUT / "splits.json", json.dumps(splits, indent=0, sort_keys=True))

    counts = {SPLIT_NAMES[k]: {"markets": int((mk["split"] == k).sum()),
                               "rows": int(mk["n_rows"][mk["split"] == k].sum()),
                               **{cl: int(((mk["split"] == k) & (mk["contract_len_s"] == L)).sum())
                                  for cl, L in CLEN.items()}}
              for k in SPLIT_NAMES}
    m2 = dict(meta)
    m2["protocol_version"] = 2
    m2["v2_note"] = NOTE
    m2["v2_built_at_utc"] = pd.Timestamp.now("UTC").isoformat()
    m2["derived_from"] = {"dir": "data/build", "v1_built_at_utc": meta["built_at_utc"],
                          "v1_sha256": src_sha, "links": kinds,
                          "relabelled": {"from_split": 2, "to_split": 1, "markets": int(was_test.sum()),
                                         "slugs_open_day_utc": "2026-08-24"}}
    m2["splits"] = {"ids": meta["splits"]["ids"],
                    "rule": "v2 (DECISIONS D7): by market open date UTC: train 2026-08-21..22 (also 08-20 opens), "
                            "val 08-23 and 08-24 (08-24 was the v1 test, used once; relabelled val); no test "
                            "split in this build (the v2 test is data/build_jun15); purged (3) as in v1",
                    "counts": counts, "v1_counts": meta["splits"]["counts"]}
    m2["v1_test_rows"] = meta["test_rows"]
    m2["test_rows"] = {"row_lo": 0, "row_hi": 0, "sha256": None}
    lo, hi = meta["test_rows"]["row_lo"], meta["test_rows"]["row_hi"]
    va = mk[mk["split"] == 1]
    m2["val_rows"] = {"row_lo": int(va["row_offset"].min()), "row_hi": int((va["row_offset"] + va["n_rows"]).max())}
    assert m2["val_rows"]["row_hi"] - m2["val_rows"]["row_lo"] == int(va["n_rows"].sum()), "val rows not contiguous"
    assert m2["val_rows"]["row_hi"] == hi and lo > m2["val_rows"]["row_lo"]

    fm = json.loads((SRC / "feat_meta.json").read_text())
    fm["v2_note"] = NOTE
    fm["derived_from"] = {"dir": "data/build", "v1_built_at_utc": fm["built_at_utc"],
                          "feat.bin/book.bin/fills.bin/fill_index.bin": "hard links to data/build (identical bytes)"}
    write_atomic(OUT / "feat_meta.json", json.dumps(fm, indent=1))

    files = LINKED + ["markets.bin", "splits.json", "feat_meta.json"]
    m2["file_sha256"] = {f: sha256_file(OUT / f) for f in files}
    m2["file_sha256_note"] = "sha256 of every other file in data/build_v2 (meta.json cannot hash itself)"
    write_atomic(OUT / "meta.json", json.dumps(m2, indent=1))
    verify(OUT)
    print("links:", kinds)
    print("counts:", json.dumps(counts))


if __name__ == "__main__":
    main()
