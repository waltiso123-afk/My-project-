# Milestone 1 — Step 1 Checkpoint

> Internal holdout evaluation — client's hidden 30-photo acceptance set NOT accessed. All numbers are real, measured, reproducible. Workflow: Human + SAM2 (no VLM).

## Dataset
- Current labeled (approved) count: 101 images
- 50-image checkpoint: `baseline_50` (train ids = strict subset of the 100-image set)
- 100-image checkpoint: `baseline_100` — trains on all 88 non-holdout labels (a literal 100-image train set is impossible with only 101 total labels while keeping a group-aware holdout; real count reported, not fabricated)
- Fixed internal holdout: 13 images = ['0016', '0023', '0031', '0032', '0037', '0044', '0051', '0054', '0058', '0064', '0085', '0120', '0138']
- Group-aware split: house-group leakage between train and holdout = NONE

## Validation of Previous Issues
### 0169 parapets — PASS
4 planes in upright space; files plane-01-parapet.png..plane-04-parapet.png; meta.parapets=[plane-01..04]; stored image, all masks & merged = 3024x4032 (match); EXIF normalized (=1). Represents the 4 client light-run areas (tall left block, center block, wide overhang above garage, right block). Downstream convention = UPPER parapet boundary. Human labels not overwritten (only the is_parapet flag was set on the existing human planes with user authorization).
### Coordinate space — PASS
coordinate_policy=upright_normalized. 0169 stored=3024x4032, mask=3024x4032, merged=3024x4032, meta=3024x4032, EXIF=1 → alignment PASS. Ingestion pipeline stores future images upright (_to_upright).
### 0001 neighbor roof — PASS
Far-left tiled roof beyond the left wing is the neighbour's; excluded (not masked). Annotation notes confirm neighbours excluded.

## 50-image checkpoint (best holdout)
{
  "iou": 0.5231,
  "dice": 0.6869,
  "mean_perp_roofline_err_normwidth": 0.01605,
  "horizontal_coverage": 0.7931,
  "n_holdout": 13
}

## 100-image checkpoint (best holdout)
{
  "iou": 0.5615,
  "dice": 0.7192,
  "mean_perp_roofline_err_normwidth": 0.01503,
  "horizontal_coverage": 0.8736,
  "n_holdout": 13
}

## Comparison (50 -> 88)
- Dice: 0.6869 -> 0.7192
- IoU: 0.5231 -> 0.5615
- Mean perpendicular roofline error (norm by width): 0.01605 -> 0.01503
- Horizontal coverage: 0.7931 -> 0.8736

## Holdout curve
See `holdout_curve.png`.

## Qualitative results
See `qualitative/` (source, ground-truth, baseline_50, baseline_100 for holdout examples: ['0016', '0023', '0031', '0032', '0037', '0044']).

## Evaluation target (honest note)
The client's marked-polyline metric cannot be reproduced (their marked polylines & hidden 30-set are unavailable). The reported roofline error/coverage are a **mask-derived proxy**: per-column lowest roof-mask edge (eave proxy) compared between the human ground-truth merged mask and the model merged mask, on the internal holdout at 256px, normalized by image width. Parapet UPPER-boundary convention is NOT applied in this proxy (documented limitation). Segmentation IoU/Dice are the primary robust metrics.

## Conclusion
Technical status: **IMPROVING**.
50->100 shows higher Dice/IoU and lower roofline error: the model benefits from more labels.

Safe to continue labeling toward 150? YES — the curve is still improving, more labels are expected to help; continue to 150 then re-checkpoint.

_Baseline is intentionally non-optimized and reproducible (SegFormer-B0, CPU). Not a client-facing acceptance result._
