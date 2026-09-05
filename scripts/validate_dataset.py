#!/usr/bin/env python3
"""Dataset integrity & spec-conformance validator (PASS / WARN / FAIL).

Warnings are never silently treated as passes. Reads real state from MongoDB + object storage.
"""
import io
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "/app/backend")
import config  # noqa: E402
import storage  # noqa: E402
import spec_checklist  # noqa: E402
from pymongo import MongoClient  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402

FAIL, WARN = [], []


def fail(m): FAIL.append(m)
def warn(m): WARN.append(m)


def _binary_png(data, label):
    try:
        arr = np.array(Image.open(io.BytesIO(data)).convert("L"))
        vals = set(np.unique(arr).tolist())
        if not vals.issubset({0, 255}):
            fail(f"{label}: mask not strictly 255/0 (values={sorted(vals)[:6]})")
        return arr
    except Exception as e:
        fail(f"{label}: invalid PNG ({e})")
        return None


def main():
    db = MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
    checks = []
    imgs = list(db.images.find({}, {"_id": 0}).sort("dataset_id", 1))
    n = len(imgs)
    checks.append(f"images in DB: {n}")

    # --- numbering + duplicate ids ---
    ids = [im["dataset_id"] for im in imgs]
    if ids != [f"{i:04d}" for i in range(1, n + 1)]:
        fail("dataset_id numbering not contiguous 0001..N")
    if len(set(ids)) != len(ids):
        fail("duplicate internal dataset_id found")

    # --- duplicate original filenames (forbidden) ---
    fn_counter = Counter(im["original_filename"] for im in imgs)
    dups = [f for f, c in fn_counter.items() if c > 1]
    if dups:
        fail(f"duplicate original filenames: {dups[:5]}")

    # --- storage upload + original hash preservation ---
    for im in imgs:
        if im.get("corrupted"):
            warn(f"{im['dataset_id']} corrupted: {im.get('corrupt_reason')}")
            continue
        if not im.get("storage_uploaded"):
            fail(f"{im['dataset_id']} not uploaded to object storage")
        if not im.get("sha1"):
            warn(f"{im['dataset_id']} missing original sha1 hash")

    # --- orientation metadata consistency (EXIF vs display) ---
    orient_issues = 0
    for im in imgs:
        w, h = im.get("width"), im.get("height")
        if w and h:
            expect = "portrait" if h > w else ("landscape" if w > h else "square")
            if im.get("orientation") != expect:
                orient_issues += 1
        if im.get("exif_transposed") and (im.get("raw_width"), im.get("raw_height")) == (w, h):
            warn(f"{im['dataset_id']} flagged exif_transposed but dims unchanged")
    if orient_issues:
        fail(f"{orient_issues} images: effective orientation inconsistent with display dims")
    else:
        checks.append("orientation metadata consistent with EXIF-applied display dims")
    checks.append("orientation dist: " + str(dict(Counter(im["orientation"] for im in imgs))))

    # --- csv ---
    if not config.DATASET_INDEX_CSV.exists():
        fail("dataset_index.csv missing")
    else:
        with open(config.DATASET_INDEX_CSV) as f:
            rows = sum(1 for _ in f) - 1
        if rows != n:
            fail(f"csv rows ({rows}) != DB images ({n})")
        checks.append(f"dataset_index.csv rows: {rows}")

    # --- approved annotations spec conformance ---
    anns = list(db.annotations.find({"status": "approved"}, {"_id": 0}))
    checks.append(f"approved annotations: {len(anns)}")
    for a in anns:
        did = a["dataset_id"]
        planes = a.get("planes", [])
        if not planes:
            fail(f"{did} approved but has no planes")
        try:
            meta = json.loads(storage.get_object(config.meta_path(did))[0])
        except Exception as e:
            fail(f"{did} meta JSON missing/invalid: {e}")
            meta = None
        # merged == union of plane masks
        plane_arrays = []
        for i2, plane in enumerate(planes, start=1):
            suffix = "-parapet" if plane.get("is_parapet") else ""
            fname = f"plane-{i2:02d}{suffix}.png"
            try:
                arr = _binary_png(storage.get_object(config.plane_path(did, fname))[0], f"{did} {fname}")
                if arr is not None:
                    plane_arrays.append(arr)
            except Exception as e:
                fail(f"{did} plane mask {fname} missing: {e}")
            if meta:
                mp = [p for p in meta.get("planes", []) if p["index"] == i2]
                if mp:
                    if mp[0].get("is_parapet") != bool(plane.get("is_parapet")):
                        fail(f"{did} plane {i2} parapet flag mismatch vs meta")
                    if not mp[0].get("generation_method"):
                        warn(f"{did} plane {i2} missing generation_method provenance")
                if plane.get("is_parapet") and f"plane-{i2:02d}" not in meta.get("parapets", []):
                    fail(f"{did} parapet plane not in meta.parapets as 'plane-{i2:02d}'")
        try:
            merged = _binary_png(storage.get_object(config.merged_path(did))[0], f"{did} merged")
            if merged is not None and plane_arrays:
                union = np.zeros_like(plane_arrays[0])
                for pa in plane_arrays:
                    union = np.maximum(union, pa)
                if not np.array_equal(union, merged):
                    fail(f"{did} merged mask != exact union of plane masks")
        except Exception as e:
            fail(f"{did} merged mask missing: {e}")
        # occlusions normalized + valid hides
        plane_ids = {f"plane-{i2:02d}" for i2 in range(1, len(planes) + 1)}
        for occ in a.get("occlusions", []):
            bb = occ.get("bbox", [])
            if len(bb) != 4 or any((not isinstance(v, (int, float)) or v < 0 or v > 1) for v in bb):
                fail(f"{did} occlusion bbox not normalized 0-1: {bb}")
            if occ.get("type") not in spec_checklist.OCCLUSION_TYPES:
                warn(f"{did} occlusion type '{occ.get('type')}' not in spec enum")
            for h in occ.get("hides", []):
                if h not in plane_ids:
                    fail(f"{did} occlusion hides unknown plane '{h}'")

    print("=== VALIDATE DATASET (PASS / WARN / FAIL) ===")
    for c in checks:
        print("  [check]", c)
    for w in WARN:
        print("  [WARN ]", w)
    for e in FAIL:
        print("  [FAIL ]", e)
    result = "FAIL" if FAIL else ("PASS_WITH_WARNINGS" if WARN else "PASS")
    print(f"\nRESULT: {result} ({len(FAIL)} failures, {len(WARN)} warnings)")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
