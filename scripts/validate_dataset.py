#!/usr/bin/env python3
"""Dataset integrity & spec-conformance validator.

Checks:
- image index completeness (dataset_id numbering is contiguous 0001..N)
- each non-corrupt image uploaded to object storage
- dataset_index.csv exists and row count matches DB
- for every APPROVED image: per-plane masks + merged mask + meta JSON exist in storage,
  plane mask filenames follow plane-NN[-parapet].png, parapet suffix matches meta.parapets
- mask files are valid single-channel 255/0 PNGs

Usage:
    python scripts/validate_dataset.py
"""
import io
import json
import os
import sys

sys.path.insert(0, "/app/backend")
import config  # noqa: E402
import storage  # noqa: E402
from pymongo import MongoClient  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402


def main():
    db = MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
    errors, warnings, checks = [], [], []

    imgs = list(db.images.find({}, {"_id": 0}).sort("dataset_id", 1))
    n = len(imgs)
    checks.append(f"images in DB: {n}")

    # contiguous numbering
    ids = [im["dataset_id"] for im in imgs]
    expected = [f"{i:04d}" for i in range(1, n + 1)]
    if ids != expected:
        errors.append("dataset_id numbering not contiguous 0001..N")
    else:
        checks.append("dataset_id numbering contiguous 0001..%04d" % n)

    # storage upload for non-corrupt
    for im in imgs:
        if im.get("corrupted"):
            warnings.append(f"{im['dataset_id']} corrupted: {im.get('corrupt_reason')}")
            continue
        if not im.get("storage_uploaded"):
            errors.append(f"{im['dataset_id']} not uploaded to object storage")

    # csv
    if not config.DATASET_INDEX_CSV.exists():
        errors.append("dataset_index.csv missing")
    else:
        with open(config.DATASET_INDEX_CSV) as f:
            rows = sum(1 for _ in f) - 1
        checks.append(f"dataset_index.csv rows: {rows}")
        if rows != n:
            errors.append(f"csv rows ({rows}) != DB images ({n})")

    # approved annotations
    anns = list(db.annotations.find({"status": "approved"}, {"_id": 0}))
    checks.append(f"approved annotations: {len(anns)}")
    for a in anns:
        did = a["dataset_id"]
        planes = a.get("planes", [])
        if not planes:
            errors.append(f"{did} approved but has no planes")
        # meta
        try:
            meta_bytes, _ = storage.get_object(config.meta_path(did))
            meta = json.loads(meta_bytes)
        except Exception as e:
            errors.append(f"{did} meta JSON missing/invalid: {e}")
            meta = None
        # merged
        try:
            mb, _ = storage.get_object(config.merged_path(did))
            _check_binary_png(mb, f"{did} merged", errors)
        except Exception as e:
            errors.append(f"{did} merged mask missing: {e}")
        # per-plane
        for idx, plane in enumerate(planes, start=1):
            suffix = "-parapet" if plane.get("is_parapet") else ""
            fname = f"plane-{idx:02d}{suffix}.png"
            try:
                pb, _ = storage.get_object(config.plane_path(did, fname))
                _check_binary_png(pb, f"{did} {fname}", errors)
            except Exception as e:
                errors.append(f"{did} plane mask {fname} missing: {e}")
            if meta:
                mp = [p for p in meta.get("planes", []) if p["index"] == idx]
                if mp and mp[0].get("is_parapet") != bool(plane.get("is_parapet")):
                    errors.append(f"{did} plane {idx} parapet flag mismatch vs meta")
                if plane.get("is_parapet") and fname not in meta.get("parapets", []):
                    errors.append(f"{did} parapet plane {fname} not listed in meta.parapets")

    print("=== VALIDATE DATASET ===")
    for c in checks:
        print("  [check]", c)
    for w in warnings:
        print("  [warn ]", w)
    for e in errors:
        print("  [ERROR]", e)
    print(f"\nRESULT: {'PASS' if not errors else 'FAIL'} "
          f"({len(errors)} errors, {len(warnings)} warnings)")
    sys.exit(0 if not errors else 1)


def _check_binary_png(data: bytes, label: str, errors: list):
    try:
        im = Image.open(io.BytesIO(data))
        arr = np.array(im.convert("L"))
        vals = set(np.unique(arr).tolist())
        if not vals.issubset({0, 255}):
            errors.append(f"{label}: mask not strictly 255/0 (values={sorted(vals)[:6]}...)")
    except Exception as e:
        errors.append(f"{label}: invalid PNG ({e})")


if __name__ == "__main__":
    main()
