# FOUNDATION V2 REPORT — Roofline Milestone 1, Phase 1

Generated after the Foundation Audit. Every value below is measured from the real pipeline
(MongoDB + object storage), verified by scripts and 8/8 regression tests + backend/frontend tests.
No fabrication. Client written spec (`/app/PROJECT_SPEC.md`) is the source of truth.

## Dataset
- total = 217 · Batch1 = 168 · Batch2 = 49 · corrupted = 0 · exact duplicates (sha1) = 0
- Object storage `roofline/images/<id>.<ext>` (originals unchanged) · MongoDB + `dataset_index.csv` (217 rows)
- Batch2 original filenames preserved exactly `b2-001`..`b2-049`; internal ids 0169–0217 map to them.

## Orientation (CORRECTED — previous "0 portrait" was a bug)
- B2 portrait = 18 · B2 landscape = 31 · square = 0 · EXIF-rotation-required = 18
- Portrait filenames: b2-001, b2-002, b2-003, b2-004, b2-006, b2-007, b2-008, b2-009, b2-010,
  b2-011, b2-012, b2-014, b2-017, b2-018, b2-044, b2-046, b2-048, b2-049
- **Why the previous run showed 0 portrait:** orientation was computed from RAW pixel dims only.
  18 B2 photos carry EXIF orientation=6 (rotate 90°): raw pixels are landscape (e.g. 4032×3024) but
  the effective/display image is portrait (3024×4032). Ingestion now records raw dims, `exif_orientation`,
  effective display dims, and effective orientation via `ImageOps.exif_transpose` (matching the browser).
- Regression tests PASS: EXIF-6 landscape→portrait, raw portrait, ordinary landscape, propose() in
  display space, UI aspect ratio (canvas fits any orientation). Original files never modified.

## Coordinate / EXIF policy (documented, one consistent convention)
- Canonical annotation/mask space = **EXIF-applied display orientation** (what the annotator and
  browser see). Masks are generated at those pixel dims → image and mask never misalign.
- Original file preserved unchanged in storage; mapping to it retained (`raw_width/height`, `sha1`).
- `meta.coordinate_space = "exif_applied_display"`, `meta.image_size = {width,height}`.

## House grouping (candidate evidence, NOT ground truth)
- Method: ORB local features + RANSAC homography geometric verification. 217 house_groups (all
  singletons currently). 7 candidate pairs surfaced with full metrics (inliers, good matches, inlier
  ratio, keypoints, batch, orientation) in the House Group Manager.
- Human ruling per pair: SAME / DIFFERENT / UNSURE (SAME merges + recomputes split; others recorded).
- Top candidates (12–19 inliers) visually inspected = look-alike Florida homes (shared white stucco /
  dark window frames / tile-metal roofs), none reaching the strong auto-group threshold (22).
- **Unresolved / awaiting human ruling:** all 7 pairs (0085↔0213, 0042↔0176, 0051↔0177, 0005↔0051,
  0005↔0195, 0012↔0023, 0037↔0153). Complementary embedding search deferred (no GPU/model on this env);
  documented as a follow-up candidate-generation step for the GPU run.

## Split (group-aware, holdout FIXED, content-stable seed)
| split    | images | groups | B1 | B2 | portrait | landscape |
|----------|--------|--------|----|----|----------|-----------|
| train    | 152    | 152    |120 | 32 | 10       | 142       |
| val      | 31     | 31     | 21 | 10 | 7        | 24        |
| holdout  | 34     | 34     | 27 | 7  | 1        | 33        |
- Seed on `min(image_id)` per group (content-stable) → same holdout across 50/100/150/200 checkpoints.
- `scripts/check_split_leakage.py` → **PASS — no house_group spans train/validation/holdout.**

## Environment (really verified — no fabrication)
- CPU 8 logical cores · RAM 31.3 GB · disk free ~7 GB
- GPU present: **false** · CUDA available: **false** · PyTorch: **not installed** · SAM2: **not installed**
- Current inference backend: `cpu_assist_fallback` (classical region-growing; **NOT SAM2**)
- SAM2 wrapper is CUDA-ready (`SAM2_CHECKPOINT`/`SAM2_CONFIG`), fails honestly without CUDA/weights,
  and never labels a CPU proposal as SAM2. Disk too tight to materialize SAM2 weights here; deferred to GPU.

## Annotation
- Client spec loaded: **YES** (`/app/PROJECT_SPEC.md`, spec_version roofline-labeling-spec-v1)
- Operational checklist surfaced in the studio via `/api/spec/checklist` (16 derived spec rules).
- SAM2 real available: **NO** (CPU env). CPU fallback clearly labeled everywhere: **YES**.
- Annotation UI status: functional (point-assist proposal → polygon/plane edit → overlay/opacity →
  parapet toggle → spec-enum occlusions with **normalized bbox + hides** → view_type → approve/needs_review →
  per-plane 255/0 masks with `-parapet` + merged union + spec meta JSON). Autosave, nav, shortcuts, zoom/pan.
- Mask provenance support: **YES** (`generation_method`, `human_corrected`, `human_approved` per plane).
- Pilot images selected: 0001, 0050, 0138, 0169 (portrait B2), 0173 (oblique B2). Pilot approved: **NO** (gate).
- Unresolved annotation questions: exact eave line on heavily-occluded B2 obliques; any true parapet
  instances to be confirmed during the pilot; house-group pair rulings.

## QA
- backend tests: PASS (prior 13/13) · frontend tests: PASS · orientation regression: 4 PASS
- production/test isolation: PASS (tests use `roofline_regression` DB + prefix; production 217 intact)
- dataset validator: **PASS** (0 failures, 0 warnings) · leakage validator: **PASS**

## FINAL GATE
**READY FOR 5-IMAGE PILOT.**
Reasons: client spec loaded; orientation bug fixed and regression-tested; EXIF/mask coordinate policy
defined; leakage PASS; provenance + spec checklist wired; CPU fallback never mislabeled as SAM2; test
data isolated from production. Remaining human step before scaling to 50: (1) rule the 7 house-group
candidate pairs, (2) annotate + approve the 5 pilot images. No mass annotation, no SegFormer training,
no fabricated metrics until the pilot is approved.
