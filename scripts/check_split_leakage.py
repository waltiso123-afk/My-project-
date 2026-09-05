#!/usr/bin/env python3
"""Anti-leakage split validator.

Fails (exit 1) if ANY house_group appears in more than one split. Reads the real
state from MongoDB (source of truth). Prints a machine-readable + human summary.

Usage:
    python scripts/check_split_leakage.py
"""
import json
import os
import sys

sys.path.insert(0, "/app/backend")
import config  # noqa: E402
from pymongo import MongoClient  # noqa: E402


def main():
    db = MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
    imgs = list(db.images.find({}, {"_id": 0, "dataset_id": 1, "house_group_id": 1, "split": 1}))

    group_splits = {}
    unassigned = []
    for im in imgs:
        g, s = im.get("house_group_id"), im.get("split")
        if g is None or s is None:
            unassigned.append(im["dataset_id"])
            continue
        group_splits.setdefault(g, set()).add(s)

    leaks = {g: sorted(list(ss)) for g, ss in group_splits.items() if len(ss) > 1}
    passed = len(leaks) == 0 and len(unassigned) == 0

    result = {
        "passed": passed,
        "images_checked": len(imgs),
        "groups_checked": len(group_splits),
        "unassigned_images": unassigned,
        "leaks": leaks,
    }
    print(json.dumps(result, indent=2))
    print("\n=== LEAKAGE CHECK ===")
    if passed:
        print("PASS: no house_group spans multiple splits; all images assigned. Zero leakage.")
    else:
        if leaks:
            print(f"FAIL: {len(leaks)} group(s) leak across splits.")
        if unassigned:
            print(f"FAIL: {len(unassigned)} image(s) have no group/split assignment.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
