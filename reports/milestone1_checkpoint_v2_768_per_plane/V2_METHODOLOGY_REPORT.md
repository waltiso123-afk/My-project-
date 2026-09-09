# Milestone 1 — Checkpoint v2 (768, per-plane) — PREPARATION & 768 BENCHMARK

> Preparation only. No full 150-image training was started. No annotations were modified.
> Step 1 (256 binary) results preserved as `milestone1_step1_256_binary_baseline`. Human + SAM2 only (no VLM).

## 1. Is 768 feasible on the current CPU?
**No — impractical for iterative 150-image checkpoints.** Real short benchmark (4-core CPU, SegFormer-B0):
| config | train s/img | infer s/img | peak RSS |
|--------|-------------|-------------|----------|
| 256 bs2 | 0.34 | 0.05 | 0.78 GB |
| 768 bs1 | 3.74 | 1.21 | 1.9 GB |
| 768 bs2 | 3.51 | 0.93 | 3.2 GB |

768 is ~10× slower per image than 256 (≈9× pixels). Memory is fine (~3.2 GB), so the limit is **compute time**, not RAM.

## 2. Benchmark runtime (estimated full checkpoint)
150 imgs × 3.5 s/img ≈ **525 s/epoch (~8.75 min)**; 18 epochs ≈ **~2.6 hours per single 150-image checkpoint** on this CPU. Iterating (tuning, re-runs) becomes multi-hour each — impractical.

## 3. Recommended GPU (CPU impractical)
- **Minimum:** NVIDIA **T4 (16 GB)**. **Recommended:** **L4 / A10 (24 GB)** for batch headroom.
- Estimated VRAM at 768, bs8: **~6–8 GB**.
- Expected speedup vs CPU: **~50–100×** (150-img epoch drops from ~9 min to a few seconds; full checkpoint = minutes, not hours).
- **Pipeline is GPU-ready** (device-agnostic Torch/Transformers): set `device='cuda'`, `.to(device)`, raise batch size. No code rewrite needed.

## 4. Per-plane evaluation methodology (prepared)
`scripts/per_plane_eval.py`. Each plane mask `plane-0N[-parapet].png` is evaluated **individually** (merged mask is only a secondary diagnostic). Per plane we extract a roofline polyline (per-column boundary), then report:
- **mean perpendicular roofline error, normalized by image width** (PRIMARY)
- **horizontal roofline coverage** (SECONDARY)
- **number of valid plane instances**

**Dry-run on the current 13-image holdout (baseline_100, interim binary prediction restricted to each GT plane's column span):**
- mean per-plane perpendicular error (norm width): **0.02445**
- mean horizontal coverage: **0.9171**
- valid plane instances: **53** (parapet planes in holdout: **0**)

> Interim limitation: the Step-1 model is BINARY (roof vs bg), so it cannot output separate plane instances. The **ground-truth per-plane extraction + metric framework is complete**; true per-plane *prediction* requires an instance/multi-class head — planned for the v2 checkpoint. Numbers above are real, computed via the prepared pipeline.

## 5. Parapet plane handling
Determined from the `-parapet` filename suffix **and** `meta.parapets`. 
- **Parapet planes → UPPER parapet boundary** (min-y per column).
- **Normal planes → LOWER eave boundary** (max-y per column).
Parapets are never silently ignored; the branch is implemented and unit-exercised (activates whenever a parapet plane appears in the evaluated set; the current holdout happens to contain none — 0169 & the 7 parapet homes are in train).

## 6. Checkpoint-selection metric
- **PRIMARY: mean perpendicular roofline error (normalized by width) → MINIMIZE.**
- **SECONDARY: horizontal coverage → MAXIMIZE** (tie-breaker).
- **Dice / IoU: reported as segmentation diagnostics only — NOT used to select the best checkpoint.**

## 7. 150-image training configuration
See `config.json`. Summary: SegFormer-B0, input **768**, AdamW 6e-5 / wd 1e-4, weighted CE, 18 epochs, seed 0; **same fixed group-aware holdout** (13 ids listed, leakage NONE); target ~150 approved labels (currently 101 available — will use ~150 once labeled); per-plane parapet-aware evaluation; best epoch chosen by per-plane roofline error. Batch: GPU bs4–8 (CPU bs1, impractical).

## 8. No annotations modified
Read-only throughout. No `annotations`/`images`/masks were changed by this preparation.

## 9. Step 1 preserved
Original Step 1 archived intact at `reports/milestone1_step1_256_binary_baseline/` (and `reports/milestone1_step1.zip`). New methodology lives in `reports/milestone1_checkpoint_v2_768_per_plane/`.

## Recommendation
Do **not** run the 150 @ 768 checkpoint on this CPU. Provision a **T4/L4/A10 GPU**; the pipeline is ready to move over. Await explicit approval before any long training run.
