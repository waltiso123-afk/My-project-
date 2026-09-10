SMOKE TEST: PASS

# SegFormer-B0 @768 — Training-pipeline smoke test

> Proves the **already-agreed SegFormer-B0 @768 binary** training pipeline executes end-to-end on a
> tiny subset **before any GPU spending**. Reuses `scripts/train_step1.py` logic verbatim (model,
> loss, eval, config) — only RES→768, tiny subset, 1–2 epochs, + val-loss + reload/inference checks.
> **NOT an accuracy benchmark.** No architecture/resolution/split/annotation changes. No GPU.

| # | Item | Result |
|---|---|---|
| 1 | **PASS / FAIL** | **PASS** (13/13 stages) |
| 2 | Exact model | **SegFormer-B0 · `nvidia/mit-b0` · num_labels=2** (0=background, 1=roof), 3.71M params |
| 3 | Input resolution | **768 × 768** (input `[b,3,768,768]` → output logits `[b,2,768,768]`) |
| 4 | Dataset subset | train (10, pool): `0001–0010` · val (3, from frozen-13 holdout): `0016,0023,0031` · merged binary masks via existing pipeline |
| 5 | Epochs | **2** (batch_size=2 for 768 CPU memory safety) |
| 6 | **Train loss / epoch** | ep1 = **0.6555** · ep2 = **0.5883** |
| 7 | **Val loss / epoch** | ep1 = **0.6751** · ep2 = **0.6481** |
| 8 | Dice / IoU (diagnostic) | ep1 Dice 0.1298 / IoU 0.0694 · ep2 Dice 0.1933 / IoU 0.1070 |
| 9 | Runtime / epoch | ep1 ≈ **49 s** · ep2 ≈ **50 s** |
| 10 | Total runtime | **110.7 s** (peak memory **3321 MB**) |
| 11 | Checkpoint path | `/app/reports/milestone1_v2_768_segformer_smoke/models/smoke_best.pt` (14.5 MB, best epoch=2 by Dice — diagnostic only) |
| 12 | Checkpoint reload test | **PASS** — state_dict loaded into a fresh SegFormer-B0 |
| 13 | Inference test | **PASS** — from saved checkpoint: logits `(1,2,768,768)`, argmax→binary mask {0,1}, roof_px=60418 |
| 14 | Holdout integrity | **PASS** — fixed 13-image holdout all still `holdout`, untouched (before & after) |
| 15 | Annotation integrity | **PASS** — SHA-256 over approved (split/group/paths/plane-refs) **before == after** |
| 16 | Warnings / errors | Expected HF notice: decode-head weights newly initialized (untrained) — benign. No errors, no OOM, no device issues. |

## Roofline (diagnostic only)
Reused existing eave (lower-boundary) roofline eval from `train_step1.py`: mean perpendicular error (norm. by width) ep1=0.27789 / ep2=0.10322; horizontal coverage ep1=1.0 / ep2=0.9974. **PRIMARY = mean perpendicular roofline error (MINIMIZE)**, coverage secondary, Dice/IoU diagnostics — unchanged. **These tiny-subset numbers are SMOKE-TEST DIAGNOSTICS ONLY; no production checkpoint is selected/advertised from them.**

## Integrity checks
**Before**: model=SegFormer-B0 ✓ · num_labels=2 ✓ · res=768 ✓ · frozen-13 holdout intact ✓ · no train↔holdout group leakage (group_leak=NONE) ✓ · annotations read-only ✓.
**After**: 2 epochs completed no errors ✓ · validation completed ✓ · checkpoint written ✓ · checkpoint reloaded ✓ · inference from saved checkpoint ✓ · annotations unchanged (hash equal) ✓.

## Config used (unchanged from established SegFormer-B0 spec)
AdamW, lr=6e-5, weight_decay=1e-4, weighted CE class weights [1,3], seed=0, CPU, binary roof/bg. Data = existing approved merged binary masks (no regeneration).

## Conclusion
**SMOKE TEST: PASS** — the SegFormer-B0 @768 binary training pipeline executes end-to-end (init → load → forward → loss → backprop → validation → checkpoint save → reload → inference) on the small subset, with holdout and annotations fully intact. This does **NOT** indicate the production model has met the client's acceptance accuracy — that is established during the real 150-image GPU run.

> No GPU rented. No full-dataset training. No production annotations modified. Human+SAM2 workflow only.
Artifacts: `smoke_results.json`, `models/smoke_best.pt`. Script: `scripts/smoke_768_segformer.py`.
