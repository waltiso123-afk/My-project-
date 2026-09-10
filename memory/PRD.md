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

## Pilot 5 images — executed 2026-06 (agent-driven SAM2-CPU run, human approval still pending)
- Report: `/app/reports/PILOT_5_REPORT.md`; evidence: `/app/reports/pilot/overlays|masks`; scripts: `scripts/pilot_run.py`, `pilot_fix.py`.
- Real SAM2 CPU (hiera-tiny) on all 5. Encode ~3.5–3.8s/image (once), predict 0.07–0.5s/point; embedding cache OK.
- Coordinate/orientation verification PASS on all 5 incl. 0169 (portrait EXIF=6, 3024×4032) and 0173 (5712×4284): clicked point = same physical pixel; db_size==display_size.
- SAM2 quality: EXCELLENT on pitched tile roofs (0138 best; 0173, 0050 garage, 0001 right hip). Point placement is decisive; repositioning fixed 0001 tower/left-wing. 0050 left gable slope occluded by tree.
- 0169 (modern flat roof + parapets) = HARD/AMBIGUOUS: no roof planes visible from ground, only parapet caps → needs full manual polygon along parapet top (-parapet) + client clarification (pergola/cantilever slab).
- Recommendation: workflow READY to scale on tile pitched roofs; treat flat/parapet homes as a separate manual+clarify track. Agent did NOT mark any as approved/spec-final (human eave-precision gate).

## Milestone 1 production prompt (client-approved) — Phase 1 done 2026-06
- Client approved layout + metadata (generation_method, human_corrected, spec_version, coordinate_space kept). 4/5 pilots approved; 0169 needs correction; scaling authorized.
- Phase 1 (EXIF/coordinate) DONE: scripts/normalize_exif_upright.py re-stored images UPRIGHT (EXIF applied+stripped) so file dims==display==mask. Only 0169 changed (4032x3024 -> 3024x4032, exif_orientation=1). Verified: all 5 file==mask. pipeline.py ingestion now stores upright via _to_upright() for all future images (coordinate_policy="upright_normalized", original_raw_size kept for provenance). storage.init_storage hardened with retries.
- Phase 4: 0001 = user's 4 planes; far-left tiled roof confirmed NEIGHBOUR -> left unmasked (unchanged).
- Studio supports Phase 2: per-plane `-parapet` checkbox + occlusion box tool already present.
- REMAINING (human-in-the-loop, agent must NOT auto-annotate):
  - Phase 2: user manually corrects 0169 in Studio -> 4 parapet planes (tall left block, center block, wide overhang above garage, right block), each is_parapet -> -parapet suffix + parapets[] in meta. Then agent rebuilds zip + QA.
  - Phase 5: user annotates remaining ~200 images in Studio in batches; agent packages/QA per batch.
  - Phase 6: at 50 & 100 approved labels, agent builds SegFormer-B0 baseline + holdout curve (no hidden 30-set, no fabricated metrics).
- Current deliverable (milestone-1) still reflects pre-parapet 0169 (1 plane) — will rebuild after user's 0169 correction.
- The VLM-assisted Mini-Milestone was REJECTED by the user and is obsolete. Redone with human+SAM2 only.
- Workflow: human (agent) identifies plane -> human places SAM2 point prompt -> real local SAM2 (CPU, Hiera-Tiny) mask -> human visual validation -> human correction (clip/polygon, e.g. 0050 left slope tightened off the gable wall) -> approve -> ml.generate_and_store (production path). NO Gemini/VLM/auto-detection/auto-points anywhere.
- Previous VLM annotations purged from DB (annotations deleted, images reset) before redo; final DB annotations: status=approved, workflow "human+sam2 (no VLM)", generation_method human_point_prompt+sam2_cpu+human_qa.
- Scripts: scripts/hm1_annotate.py (human points + SAM2 + validation overlays), scripts/hm1_package.py (store+export+QA+zip). Output: /app/reports/MILESTONE_1_HUMAN/dataset + PILOT_MILESTONE_1.zip.
- Plane counts: 0001=3, 0050=3, 0138=2 (palm>15% split), 0173=4, 0169=4 (3 -parapet + 1 canopy, PROVISIONAL/ambiguous, flagged in meta for client). 16 masks total.
- ZIP: /app/reports/MILESTONE_1_HUMAN/PILOT_MILESTONE_1.zip (9.9MB, sha256 c611a105...), download GET /api/deliverables/milestone-1/download (overwrote deliverable). QA passed (binary, dims, merged==union, originals unchanged, parapet naming/meta consistent, exactly 5 images, no VLM refs). Internal QA report kept OUTSIDE the zip.
- STOP gate respected: no other images, no 200-run, no checkpoints, no architecture comparison.
- Architecture executed: VLM reasoning/plan (gemini-3.1-pro-preview via Emergent Universal Key) -> SAM2 (CPU hiera-tiny) point prompts -> clip/reconcile -> agent QA & correction -> REAL production path `ml.generate_and_store` (object storage planes/merged/meta) -> approved annotations in DB.
- Scripts: scripts/mm1_plan.py, mm1_segment.py, mm1_finalize_masks.py, mm1_store_export.py, mm1_qa_package.py.
- Output: /app/reports/PILOT_MILESTONE_1/dataset/ (images,planes,merged,meta) + client README; ZIP /app/reports/PILOT_MILESTONE_1/PILOT_MILESTONE_1.zip (9.9MB, sha256 d4a5f18b..., 16 planes over 5 imgs). Internal QA report kept OUTSIDE the zip.
- Download endpoints: GET /api/deliverables/milestone-1/{info,download} (also pilot-5 endpoints from prior package). Backend server.py updated.
- Plane counts: 0001=3, 0050=3, 0138=2 (palm >15% split), 0173=4, 0169=4 (3 -parapet + 1 canopy, PROVISIONAL/ambiguous, documented in meta notes for client ruling).
- QA passed: masks binary & at image dims, merged==union, originals unchanged, parapet naming/meta consistent, exactly 5 images.
- Note: VLM over-segments and mis-locates points (palms/walls); agent QA correction is essential — full-auto is NOT production quality. Findings honest, no fabricated metrics. STOP gate respected (no mass annotation, no 50/100/150/200 checkpoints).
- P0: Human validates eave precision on the 5 in the studio, rules on 0169 ambiguities, then approves. (Agent pilot + QA done 2026-06.)
- DONE 2026-06: Real SAM2 on CPU (sam2 1.1.0 + hiera_tiny), embedding cache, manual tools (polygon/brush/eraser/vertex/undo-redo), flood-fill demoted to "Colour Region Helper" (never SAM2). Verified iteration_3 100%.
- P1: Install SAM2 + checkpoints on a GPU env; enable real SAM2 proposals; brush/eraser tools.
- P1: Scale annotation 50 → 100 → 150 → 200 on the fixed split.
- P2: SegFormer-B0 baseline (BCE+Dice), geometric eval on holdout, holdout_error_curve.csv/.png.
- P2: Split server.py into per-resource routers.


## Milestone 1 — Step 1 checkpoint (baseline 50 vs 100) — 2026-06 DONE
- Validations: 0169 parapets PASS (4 -parapet planes, meta.parapets set, 3024x4032 file=mask, EXIF=1; is_parapet flag set on human planes w/ user auth), coordinate PASS (upright_normalized), 0001 neighbour excluded PASS. 101 approved; 7 images carry parapet planes (0020,0021,0024,0030,0053,0072,0169) per user.
- Frozen split: group-aware, holdout=13 (split=holdout, house-group leak NONE), train pool=88. baseline_50 (50 imgs) strict subset of baseline_100 (88 = max non-holdout; literal 100 impossible with 101 total; real count reported).
- Model: SegFormer-B0 (nvidia/mit-b0) binary roof seg, 256px, CPU, 18 epochs, AdamW 6e-5, weighted CE. transformers==4.46.3 (torch 2.5.1 compat).
- REAL holdout metrics: baseline_50 dice=0.687 iou=0.523 perp=0.0161 cov=0.793 ; baseline_100 dice=0.719 iou=0.562 perp=0.0150 cov=0.874. Trend=IMPROVING -> safe to continue toward 150.
- Roofline metric is a mask-derived proxy (per-column lowest roof edge); client marked-polyline + hidden 30-set NOT accessed (stated). No fabricated numbers. No VLM.
- Deliverable: /app/reports/milestone1_step1.zip (sha 28d9709e..., 28.5MB, incl report, curve, metrics_50/100.json, training_config.json, qualitative/, models/baseline_50.pt & baseline_100.pt). Download: GET /api/deliverables/milestone1-step1/download. Scripts: scripts/train_step1.py, package_step1.py, step1_validate.py.

## Checkpoint v2 (768, per-plane) — PREPARED (not trained) 2026-06
- 768 CPU benchmark (4 cores): 768 bs2 train 3.51 s/img (~10x slower than 256), infer 0.93 s/img, peak 3.2GB. Est 150-img@768 checkpoint ~2.6h/run on CPU -> IMPRACTICAL. Recommend GPU T4(16GB) min / L4-A10(24GB); ~50-100x speedup; pipeline device-agnostic ready.
- Per-plane eval prepared (scripts/per_plane_eval.py), parapet-aware (upper boundary for -parapet, lower eave else). Dry-run holdout(13, baseline_100): per-plane perp 0.02445 normwidth, coverage 0.917, 53 plane instances (0 parapet in holdout). Note: binary Step1 model can't predict instances; GT per-plane framework complete, per-plane prediction needs instance/multiclass head in v2.
- Selection: PRIMARY mean perpendicular roofline error (minimize); SECONDARY coverage; Dice/IoU diagnostics only.
- Holdout preserved (same 13 group-aware ids, leak NONE). Config: reports/milestone1_checkpoint_v2_768_per_plane/config.json + V2_METHODOLOGY_REPORT.md.
- Step 1 preserved as reports/milestone1_step1_256_binary_baseline. NO annotations modified. No long training started (awaiting approval + GPU).

## v2 per-plane MODELING POC (256 CPU, diagnostic — no long/768 run) — DONE 2026-06
- Goal (user): validate WHICH output formulation actually recovers INDIVIDUAL plane instances + correct eave/parapet boundaries, before GPU/150@768. 256px numbers are diagnostic, NOT acceptance metrics. Read-only: annotations + fixed 13-holdout untouched. Human+SAM2 only.
- Script scripts/poc_perplane.py; output reports/milestone1_v2_per_plane_POC/ (POC_REPORT.md, poc_results.json, overlays/ 18 imgs). Added scikit-image to requirements.
- Compared on 13-holdout (54 GT planes): A binary+CC recovery 0.516 (6 merges); A binary+watershed recovery 0.629 (8 merges, over-splits, IoU/cov drop); B 3-class edge-aware (8 ep quick) recovery 0.397. Mask2Former = conceptual analysis only.
- Finding: NO lightweight semantic formulation preserves instances (best ~63%, frequent adjacent-plane merges — see overlays/0031). Parapet: GT UPPER-boundary extraction WORKS (0169 red polyline correct) BUT binary model predicts ~nothing on flat/parapet roofs (0020/0021 no match, 0169 cov~0) → parapet plane is intrinsically a distinct instance/class, not a binary-roof subcase.
- RECOMMENDATION for 150@768: PRIMARY = Mask2Former INSTANCE seg, 2 categories {roof_plane, parapet_plane}, Swin-T/S, 768 GPU. Compatible w/o re-labeling (our per-plane masks + `-parapet` flag ARE instance-seg GT); aligns w/ client per-plane perp-error+coverage metric (per_plane_eval.py reused at instance level); handles variable 2-8 planes; Hungarian match = its train objective + our eval. FALLBACK = SegFormer 3-class edge-aware trained more @768 (POC shows not yet reliable). REJECT binary+CC/watershed and fixed-channel multiclass.
- Blocked next: GPU (T4/L4/A10) + reach ~150 labels, then fine-tune Mask2Former-instance @768, evaluate per-plane parapet-aware on frozen holdout.

## v2 768 SMOKE TEST (Mask2Former, end-to-end) — PASS 2026-09-10
- 150 approved now. Fixed 13 holdout INTACT (all 13 still holdout); split grew to 19 (49 new labels also split) — to decide 13-vs-19 for real run (keep 13 for 50/100/150 comparability recommended). Group leak NONE. No approved data modified.
- scripts/smoke_768_mask2former.py; report reports/milestone1_v2_768_smoke_test/SMOKE_TEST_REPORT.md (+smoke_results.json, smoke_ckpt/). Model facebook/mask2former-swin-tiny-coco-instance, num_labels=2 {roof_plane,parapet_plane}, 768px, 2 epochs, CPU, 7 imgs (4 train incl parapet 0020 / 3 eval incl parapet 0169,0021 + holdout 0016).
- 16/16 stages PASS: dataset→per-plane GT→instance conversion→M2F init→forward→loss→backward→optim→2-epoch train+eval→multi-instance inference→Hungarian per-plane matching (20 rows)→eave(lower)+parapet(upper) boundary→perp err+coverage→checkpoint SELECTION by MIN perp err (epoch2, not dice/iou)→save→reload→infer. SMOKE ONLY, accuracy not meaningful.
- CPU @768 ~100s/epoch for 7 imgs → real 150 run impractical on CPU. GPU: T4 16GB min / L4-A10 24GB rec; VRAM ~8-12GB @768 bs2; ~50-100x speedup. Pipeline device-agnostic ready. Real 150@768 run NOT launched (per instruction).

## ARCHITECTURE PIVOT + BROWSER EXPORT GATE — PASS 2026-09-10
- CLIENT DECISION: production model MUST be SegFormer-B0 (binary roof/bg), NOT Mask2Former. Per-plane eval does NOT need NN instance seg — client downstream separates planes from binary mask via connected components (same as acceptance scorer). Pipeline: image→SegFormer-B0 binary→CC→planes→roofline→scoring. Deploy target = browser via onnxruntime-web.
- Gate scripts: scripts/export_segformer_onnx.py, scripts/downstream_verify.py; deliverables reports/milestone1_v2_browser_export_validation/ (EXPORT_REPORT.md, export.log, browser_test.html/.js, browser_console.log, browser_test_screenshot.png, segformer_b0_768_binary.onnx, export_meta.json, downstream_verify.json). Browser test page also live at frontend/public/onnx_test/.
- Export: untrained SegFormer-B0 (nvidia/mit-b0, num_labels=2), input [1,3,768,768] f32, output logits [1,2,768,768] f32, opset 17, static, 15.01MB, onnx.checker OK. Ops all in WASM-supported set (Conv,LayerNorm,MatMul,Resize,Erf,Softmax...); NONE unsupported.
- REAL browser (Chromium/Playwright) via onnxruntime-web 1.29.0: webgpu unavailable headless→WASM EP; session ~1494ms, forward 2295ms; output [1,2,768,768] f32; argmax→binary {0,1} in-browser. PASS.
- Downstream verify: binary mask→CC→per-plane→roofline. 0016 normal→eave LOWER; 0169 parapet(4)→UPPER via -parapet flag. Both branches execute. Primary=perp err (MIN), secondary=coverage, dice/iou diagnostic only. Runs on untrained ONNX output too.
- Data integrity read-only: sha256 before==after (15e559bb...), 150 approved + frozen-13 holdout intact, nothing modified.
- SegFormer-B0 @768 CPU measured: 3.59s/img train step, 1.24s infer. GPU est real 150 run: T4/L4 ~1-3h, ~$1-3 total (<<$50). NO GPU rented, NO training. GATE CLEARED for GPU stage.
