# Roofline — Milestone 1, Phase 1 (Foundation) — PRD

## Original problem statement
Web app for roof-plane segmentation. Phase 1 = foundation only (NOT mass annotation):
environment verification, ingestion/indexing of ~217 images (2 batches), house grouping +
anti-leakage split at house_group level, human-in-the-loop SAM2-assisted annotation studio,
QA scripts, and annotation of 5 representative images pending human validation before scaling.
Client rules: no fabrication, labeling spec is source of truth, resumable pipeline, Batch2 never
excluded, geometric quality prioritized (perpendicular error ≤1% img width, ≥90% coverage).

## User choices
- Data: 2 public Google Drive zips (provided). Batch1=168, Batch2=49 (b2-001..b2-049).
- GPU/SAM2: env is CPU-only → prepare SAM2 CUDA-ready, keep functional CPU assist, never claim GPU ran.
- Labeling spec: user SKIPPED providing it ("assuming defaults for now") — spec-based rules deferred.
- Object storage: Emergent managed. Auth: none.

## Architecture
- Backend FastAPI (`/app/backend`): server.py (routes), pipeline.py (ingest/ORB grouping/split/merge),
  ml.py (SAM2 wrapper + CPU floodfill assist + mask generation), envreport.py, storage.py, config.py.
- Frontend React (`/app/frontend/src`): App.js + components (HeaderNav, DashboardCheckpoint,
  DatasetGallery, AnnotationWorkspace canvas, HouseGroupManager, LeakageReport), lib/api.js.
- MongoDB collections: images, house_groups, group_pairs, splits, annotations, environment_reports.
- Object storage prefix `roofline/`: images/ planes/ merged/ meta/.
- Scripts (`/app/scripts`): ingest_dataset.py, select_representative.py, validate_dataset.py,
  check_split_leakage.py, env_report.py, README.md.

## Implemented (2026-06, verified by testing agent 13/13 backend + 100% frontend)
- Ingestion: 217 images validated (0 corrupted, 0 duplicates), EXIF orientation detected/handled,
  uploaded to object storage, dataset_index.csv written. Resumable/idempotent.
- House grouping: REAL ORB + RANSAC-homography same-property analysis (replaced weak phash).
  Result: 7 evidence candidate pairs (12-19 inliers), none reach strong threshold (22) → 217
  singletons (evidence-based, look-alike homes not same property). Candidates surfaced for human
  review; manual merge endpoint recomputes split.
- Anti-leakage split at house_group level, holdout fixed via content-stable seed (min image id):
  train 152 / val 31 / holdout 34. Leakage check PASS.
- Environment: real detection — NO_GPU_CPU_ONLY, 8 cores, 31GB RAM, ~7GB free, torch not installed,
  SAM2 active_backend=cpu_assist_fallback.
- Annotation studio: point-assist proposal (cpu_assist_fallback), polygon/plane editing, overlay,
  parapet toggle, occlusions, checklist, autosave, prev/next, shortcuts, Approve → per-plane 255/0
  masks + `-parapet` suffix + merged mask + meta JSON → approved. needs_review flow.
- 5 representative images tagged: 0001 (multi-plane), 0050 (gable), 0138 (wide framing),
  0169 (batch2 high-res occlusion), 0173 (flat/parapet + EXIF rotation). Currently pending.

## Honest findings / blockers for human review
- Images are GROUND-LEVEL real-estate photos (Zillow/MLS), NOT aerial/drone/satellite.
- Zero portrait orientation exists (all landscape); 18 batch2 images are EXIF-rotated (handled).
- No confident multi-view duplicates auto-detected; 7 look-alike candidate pairs need human ruling.
- Labeling spec NOT provided → eave/gutter/gable/parapet/15% rules use problem-statement defaults;
  spec-compliant annotation of the 5 is deferred to human + spec.
- SAM2 GPU inference not run (CPU-only env), by design.

## Backlog (post human validation of the 5)
- P0: Human runs the 5-image pilot in the studio (SAM2-CPU proposal → manual correction per PROJECT_SPEC.md) and approves.
- DONE 2026-06: Real SAM2 on CPU (sam2 1.1.0 + hiera_tiny), embedding cache, manual tools (polygon/brush/eraser/vertex/undo-redo), flood-fill demoted to "Colour Region Helper" (never SAM2). Verified iteration_3 100%.
- P1: Install SAM2 + checkpoints on a GPU env; enable real SAM2 proposals; brush/eraser tools.
- P1: Scale annotation 50 → 100 → 150 → 200 on the fixed split.
- P2: SegFormer-B0 baseline (BCE+Dice), geometric eval on holdout, holdout_error_curve.csv/.png.
- P2: Split server.py into per-resource routers.
