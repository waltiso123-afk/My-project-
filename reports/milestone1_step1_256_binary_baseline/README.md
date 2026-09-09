# Milestone 1 — Step 1 Checkpoint package

Internal reproducible baseline (SegFormer-B0, CPU) at 50 and 100(max non-holdout) labels on the frozen 101-image approved set.
- `milestone1_step1_report.md` — full report (dataset, validations, metrics, comparison, conclusion)
- `metrics_50.json`, `metrics_100.json` — per-epoch + best holdout metrics (real)
- `training_config.json` — model/train config + split ids (reproducibility)
- `holdout_curve.png` — 50 vs 100 curves
- `qualitative/` — source/GT/baseline_50/baseline_100 on holdout examples
- `models/baseline_50.pt`, `models/baseline_100.pt` — best checkpoints

Internal holdout only — client's hidden 30-photo acceptance set NOT accessed. No fabricated numbers. No VLM.
