# Roofline — Milestone 1, Phase 1 (Foundation)

Dataset ingestion + anti-leakage split + human-in-the-loop roof-plane annotation studio.
React + FastAPI + MongoDB + Object Storage. ML: PyTorch/SAM2 **CUDA-ready** (heavy inference
runs only on GPU); CPU environments use a clearly-labeled classical assist for point prompts.

## What Phase 1 delivers (verifiable)
- Real ingestion of 217 images (Batch1=168, Batch2=49 `b2-001`..`b2-049`), validated & indexed.
- `dataset_index.csv` + MongoDB index (dims, EXIF orientation, sha1, phash, batch, split, status).
- Originals uploaded **unchanged** to object storage under `roofline/images/<id>.<ext>`.
- **Real** same-property (house_group) analysis: ORB local features + RANSAC homography
  geometric verification (not perceptual hashing). Strong matches auto-group; borderline pairs
  are surfaced as human-review candidates. No silent decisions.
- Anti-leakage train/val/holdout split **at house_group level** (holdout fixed, content-stable seed).
- Honest GPU/CPU/CUDA/RAM/disk/PyTorch/SAM2 environment report.
- Human-in-the-loop annotation studio: point-assist proposal → polygon/plane editing → overlay →
  Approve → per-plane masks (255/0, `-parapet` suffix) + merged mask + metadata JSON → `approved`.
- QA scripts: `validate_dataset.py`, `check_split_leakage.py`.

## Environment
- Backend `/api` on :8001 (supervisor), Frontend on :3000. Mongo via `MONGO_URL`/`DB_NAME`.
- Object storage uses `EMERGENT_LLM_KEY` (Emergent managed).

## Reproduce
```bash
# 1) Download the two Google Drive zips into /app/data/downloads and extract to
#    /app/data/staging/batch1 and /app/data/staging/batch2 (flat image files).

# 2) Full pipeline (idempotent / resumable): validate+index -> upload -> ORB house groups
#    -> anti-leakage split -> dataset_index.csv
python scripts/ingest_dataset.py

# 3) Tag 5 representative test images
python scripts/select_representative.py

# 4) QA
python scripts/env_report.py            # real hardware/software report
python scripts/check_split_leakage.py   # PASS => no house_group across splits
python scripts/validate_dataset.py      # integrity, numbering, 255/0 masks, -parapet suffix
```

## GPU note (honest)
This build environment is **CPU-only** (no CUDA GPU). SAM2 heavy inference is therefore NOT run
here; point prompts use a labeled CPU region-growing assist. The SAM2 wrapper auto-detects CUDA
and uses the real model when a GPU is present (`SAM2_CHECKPOINT` / `SAM2_CONFIG` env vars). Mass
annotation of the 217 images is intentionally deferred to a GPU run.

## Dataset layout (object storage, prefix `roofline/`)
```
images/<id>.<ext>
planes/<id>/plane-NN[-parapet].png   (255/0)
merged/<id>.png                      (255/0 union)
meta/<id>.json                       (file, occlusions[type,bbox,hides], parapets[], notes, planes[])
```
`<id>` = zero-padded sequential 0001..0217 (batch1: 0001–0168, batch2: 0169–0217). Original
filenames are preserved in `dataset_index.csv` / MongoDB `original_filename`.
