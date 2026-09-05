#!/usr/bin/env python3
"""Auditable Batch-2 orientation report (raw vs EXIF vs effective display orientation).

Explains the previous '0 portrait' bug and lists exact portrait / EXIF-rotated filenames.
Reads real values from MongoDB (populated by the EXIF-aware ingestion).
"""
import sys
sys.path.insert(0, "/app/backend")
import os
import json
import config  # noqa: F401 loads .env
from pymongo import MongoClient


def main():
    db = MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
    b2 = list(db.images.find({"batch": "batch2"}, {"_id": 0}).sort("dataset_id", 1))
    portrait = [d for d in b2 if d["orientation"] == "portrait"]
    landscape = [d for d in b2 if d["orientation"] == "landscape"]
    square = [d for d in b2 if d["orientation"] == "square"]
    exif_rot = [d for d in b2 if d.get("exif_transposed")]

    report = {
        "total_b2": len(b2),
        "portrait": len(portrait),
        "landscape": len(landscape),
        "square": len(square),
        "exif_rotation_required": len(exif_rot),
        "portrait_filenames": [d["original_filename"] for d in portrait],
        "exif_rotated_filenames": [d["original_filename"] for d in exif_rot],
        "previous_bug_explanation": (
            "The first ingestion computed orientation from RAW pixel dimensions only. "
            "18 Batch-2 phone photos carry EXIF orientation=6 (rotate 90°): their raw pixels "
            "are landscape (e.g. 4032x3024) but the effective/display image is portrait "
            "(3024x4032). Without applying EXIF, all were mislabeled landscape -> '0 portrait'. "
            "Ingestion now records raw dims, exif_orientation, effective display dims, and "
            "effective orientation via PIL ImageOps.exif_transpose (matching what the browser renders)."
        ),
        "rows": [{
            "dataset_id": d["dataset_id"], "file": d["original_filename"],
            "raw": [d.get("raw_width"), d.get("raw_height")],
            "exif_orientation": d.get("exif_orientation"),
            "display": [d["width"], d["height"]],
            "effective_orientation": d["orientation"],
            "exif_rotation_required": bool(d.get("exif_transposed")),
        } for d in b2],
    }
    print(json.dumps(report, indent=2))
    print(f"\n=== B2 ORIENTATION AUDIT ===")
    print(f"portrait={report['portrait']} landscape={report['landscape']} "
          f"square={report['square']} exif_rotated={report['exif_rotation_required']}")
    return report


if __name__ == "__main__":
    main()
