#!/usr/bin/env python3
"""Run the full Phase-1 ingestion pipeline (idempotent / resumable).

Steps: validate + index 217 images -> upload originals to object storage ->
build house_groups (perceptual hash) -> anti-leakage split -> dataset_index.csv.

Usage:
    python scripts/ingest_dataset.py
"""
import sys
sys.path.insert(0, "/app/backend")

import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.run_all(log=print)
