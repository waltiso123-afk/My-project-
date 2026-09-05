"""Shared configuration for the Roofline Phase-1 foundation pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

APP_NAME = os.environ.get("APP_NAME", "roofline")

# Local staging where the two extracted zip batches live (originals, untouched)
DATA_DIR = Path("/app/data")
STAGING_BATCH1 = DATA_DIR / "staging" / "batch1"
STAGING_BATCH2 = DATA_DIR / "staging" / "batch2"

# Local dataset artifacts (index + reports). Images/masks live in object storage.
DATASET_DIR = Path("/app/dataset")
DATASET_INDEX_CSV = DATASET_DIR / "dataset_index.csv"
REPORTS_DIR = DATASET_DIR / "reports"

# Object storage logical prefixes (canonical layout required by the spec)
def img_path(dataset_id: str, ext: str) -> str:
    return f"{APP_NAME}/images/{dataset_id}.{ext}"

def plane_path(dataset_id: str, plane_file: str) -> str:
    return f"{APP_NAME}/planes/{dataset_id}/{plane_file}"

def merged_path(dataset_id: str) -> str:
    return f"{APP_NAME}/merged/{dataset_id}.png"

def meta_path(dataset_id: str) -> str:
    return f"{APP_NAME}/meta/{dataset_id}.json"

# Split configuration (holdout is FIXED once assigned)
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "holdout": 0.15}

# House grouping: phash Hamming distance threshold for "possible same property"
PHASH_THRESHOLD = 10

MIME_TYPES = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "gif": "image/gif", "webp": "image/webp", "json": "application/json",
    "csv": "text/csv", "txt": "text/plain",
}

for d in (DATASET_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
