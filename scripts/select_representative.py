#!/usr/bin/env python3
"""Tag ~5 representative images for the Phase-1 human validation checkpoint.

Selection is based on measured metadata + a documented rationale (NOT fabricated
annotations). The images span the real variety observed in this dataset.
"""
import sys
sys.path.insert(0, "/app/backend")
import os
import config  # noqa: F401  (loads .env)
from pymongo import MongoClient

# (dataset_id, rank, reason) — chosen after real visual inspection of the dataset.
REPRESENTATIVE = [
    ("0001", 1, "Multi-plane hip roof + attached garage + partial neighbour roof; palm occlusions (batch1)."),
    ("0050", 2, "Front-facing gable end with awnings + hip sections; palm/tree occlusion — gable case (batch1)."),
    ("0138", 3, "Wide framing (aspect 1.85), tree occlusion top-right — difficult cadrage (batch1)."),
    ("0169", 4, "High-res phone photo (4032x3024), dormer + hip, heavy tree occlusion — batch2 difficult."),
    ("0173", 5, "Modern flat/stepped roof (parapet case) + EXIF-rotated capture — batch2 edge case."),
]


def main():
    db = MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
    db.images.update_many({}, {"$set": {"representative": False, "representative_rank": None,
                                        "representative_reason": None}})
    for did, rank, reason in REPRESENTATIVE:
        r = db.images.update_one({"dataset_id": did},
                                 {"$set": {"representative": True, "representative_rank": rank,
                                           "representative_reason": reason}})
        print(f"{did}: {'tagged' if r.matched_count else 'NOT FOUND'} — {reason}")
    print(f"\n{db.images.count_documents({'representative': True})} representative images tagged.")


if __name__ == "__main__":
    main()
