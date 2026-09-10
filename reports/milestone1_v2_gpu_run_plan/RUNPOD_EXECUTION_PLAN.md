# RunPod execution plan — REAL training run (SegFormer-B0 + B1 @768)

> **PLAN ONLY.** Nothing has been provisioned, rented, or trained. No production data/holdout modified.
> Turnkey package already prepared in `/app/gpu_run/` and validated locally (preflight PASS, integrity
> checksummed). Client approvals: RTX 4090 24GB · RunPod Secure Cloud · ~3–4 h · ~$2–3 · hard cap $50.

## Methodology lock (unchanged) + one client-directed change
- SegFormer-B0 (`nvidia/mit-b0`) **and** SegFormer-B1 (`nvidia/mit-b1`), `num_labels=2` (0=bg,1=roof), **768×768**, binary.
- **Identical** dataset, fixed split, schedule and eval for both: AdamW **lr 6e-5**, **wd 1e-4**, **weighted CE [1,3]**, **seed 0**, **18 epochs**, **bs 4** (same schedule as validated `train_step1.py`).
- Eval: single-pass **and** horizontal-flip averaging; save raw logits + probability maps for all holdout images.
- **⚠ FLAGGED (client req #9, not silent):** checkpoint selection uses **mean perpendicular roofline error (MINIMIZE)** as PRIMARY (coverage secondary; Dice/IoU diagnostic). `train_step1.py` originally selected by Dice — we switch to perp-error per the client's explicit acceptance metric. No other methodology change.
- ONNX export: **static `[1,3,768,768]`→`[1,2,768,768]`, opset 17, in-graph bilinear upsample** (the already-validated config).
- **Fixed 13-image holdout unchanged** (`0016,0023,0031,0032,0037,0044,0051,0054,0058,0064,0085,0120,0138`). Training pool = 137 approved (0 group-overlap exclusions — verified). No Mask2Former/U-Net/instance seg.

---

## A. Recommended RunPod Pod configuration
| Item | Recommendation |
|---|---|
| GPU | 1× **RTX 4090 24GB** (approved). Trivially fits SegFormer-B0/B1 @768. |
| Cloud tier | **Secure Cloud** (approved). Community Cloud is cheaper if ever needed. |
| Container image | **`runpod/pytorch:2.5.1-py3.11-cuda12.1.0-devel-ubuntu22.04`** (or the closest official RunPod PyTorch 2.5.1 / CUDA 12.1 template) — matches our validated torch 2.5.1. |
| CUDA | **12.1** (matches torch cu121 build). |
| CPU / RAM | ~**8 vCPU / 32 GB RAM** (default 4090 pod is fine; our peak host RAM was ~3.3 GB). |
| Disk | **30 GB container disk** (bundle 573 MB + all outputs < 1 GB; headroom for pip/caches). |
| Persistent vs ephemeral | Optional **Network Volume (~20 GB)** mounted at `/workspace` for safety; otherwise ephemeral is fine **provided artifacts are downloaded before termination** (§I). |
| Ports / network | **None** (no server). Access via **SSH** or the pod **web terminal / Jupyter**; outbound HTTPS for pip + HF model download. |

## B. Exact environment (validated versions)
- Python **3.11**, PyTorch **2.5.1** (cu121, from the container), Transformers **4.46.3**, ONNX **1.22.0**, ONNX Runtime **1.29.0**, numpy 1.26.4, pillow 10.4.0, safetensors.
- Pinned in `gpu_run/requirements-gpu.txt`. If the container lacks torch: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` **first**.

## C. Code + data transfer (safest options)
**No GitHub remote is currently configured on this project** (`git remote -v` is empty). Do not assume a repo URL.
- **Code — preferred:** use Emergent chat **“Save to GitHub”** to push this project, then on the pod `git clone <your-new-repo> && cd <repo>/gpu_run`.
- **Code — alternative (no GitHub):** `runpodctl send gpu_run/` from a machine with `runpodctl`, or upload a `gpu_run.tar.gz` to the pod via the web file browser / `scp`.
- **Data:** the frozen bundle `gpu_run/dataset_bundle/` (**573 MB**, `bundle_sha256=ca19cd07…65eb1`) — transfer via `runpodctl send dataset_bundle/` **or** a cloud link (S3/GDrive) + `wget`. Verify with `preflight.py` (checksums) after transfer. **Do NOT put MongoDB/object-storage credentials on the pod** — the bundle is fully self-contained, so the pod never touches our backend.

## D. Preflight checks (exact) — `python preflight.py dataset_bundle`
Verifies: package versions; `torch.cuda.is_available()` + device name + capability; **VRAM total** (`torch.cuda.get_device_properties(0)`); a live CUDA matmul; **disk free**; **dataset counts**; **holdout == fixed 13**; **train↔holdout group leakage = NONE**; **per-file SHA-256** vs `manifest.json`; `git rev-parse HEAD`. Exits non-zero on any hard failure. (Locally, minus GPU, this already returns **PREFLIGHT: PASS**.)
Quick manual equivalents: `nvidia-smi` · `python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"` · `df -h .`

## E. Training execution sequence (`train_gpu.py`, both models, same code path)
1. Load frozen bundle (images 768 BILINEAR, merged binary masks 768 NEAREST). 2. **B0**: init `mit-b0` → 18 epochs (AdamW/CE[1,3]) → **per-epoch single-pass holdout eval** → track **best by MIN perp error** → save `b0_best.pt` + `b0_last.pt`. 3. Reload B0 best → **single-pass eval + hflip-avg eval** → **dump raw logits + probs** for all 13 holdout (`holdout_outputs/b0/*.npy`) → **export ONNX** `onnx/segformer_b0_768_binary.onnx`. 4. **B1**: repeat identically with `mit-b1`. 5. Write `metrics_b0.json`, `metrics_b1.json`, `run_summary.json` (config, per-epoch history, selected epoch, single & hflip metrics, peak VRAM, total runtime).
One command runs the whole thing (see §J). Selection metric = perpendicular roofline error (primary), coverage (secondary), Dice/IoU (diagnostic).

## F. Runtime estimate (recalculated from our CPU smoke — NOT the old 3–4 h guess)
Basis: CPU (4 threads) smoke = 768, bs2, ~50 s/epoch for 10 imgs ⇒ ~5 s/img (fwd+bwd). RTX 4090 ≈ **40–80× faster** for this workload (conservative 40×).
- **B0** (137 imgs, bs4, 18 ep): ~18 s/epoch train + ~2 s eval ⇒ **~6–10 min** total.
- **B1** (~2.5–3× heavier than B0): **~18–28 min** total.
- Final best-eval + hflip + logits dump + 2× ONNX export: **~3–5 min**.
- **Pure compute ≈ 30–45 min.** Pod overhead (container ready + pip + 573 MB data upload + preflight + artifact tar/download): **~15–30 min**.
- **Realistic total Pod time ≈ ~1 h** (budget 1.5 h with buffer). The client's 3–4 h approval is a comfortable ceiling, not the expectation.

## G. Cost estimate (RTX 4090, verify live rate in console)
Current rates (2026-06, verify before launch): **Secure Cloud $0.74/hr**, Community $0.34/hr.
- **Expected** (~1 h Secure): **~$0.75–$1.10**.
- **Conservative budget** (2 h Secure): **~$1.50** · (3–4 h ceiling): **$2.2–$3.0** (matches client's $2–3).
- **Hard cap before re-approval: $50** ⇒ ~67 h Secure / ~147 h Community — far beyond need. **If a run trends past ~4 h / $3, stop and reassess before approaching $50.**

## H. Failure / recovery plan (no silent methodology changes)
- **OOM** (unlikely on 24GB): lower `--bs` (4→2→1). Batch size is a memory knob, not the schedule; note it in the report. Do **not** change resolution/architecture/loss.
- **Package incompatibility:** reinstall exact pins from `requirements-gpu.txt`; if torch mismatches the container, install `torch==2.5.1+cu121` explicitly. If HF model download fails, retry / set `HF_HOME` on the volume.
- **Training crash mid-run:** `*_last.pt` is saved after each model; resume by re-running the remaining model via `--models b1`. Per-epoch history is in `metrics_*.json`.
- **Checkpoint save failure:** check disk (`df -h`); write to the network volume; re-run the affected model.
- **Disk fills:** logits are saved float16 (~60 MB total); clear pip cache (`pip cache purge`); mount a larger volume.
- **Runtime exceeds estimate:** if wall time approaches ~4 h, stop training, still run §I to salvage artifacts, and report before spending toward $50.
- **Any genuinely required deviation** (e.g., bs change) → report: what failed, why, proposed change, whether it deviates from spec — **before** proceeding.

## I. Shutdown procedure (copy-off BEFORE terminate) — `collect_and_checksum.sh`
1. Verify **checkpoints** `models/{b0,b1}_{best,last}.pt`. 2. Verify **logits/probability maps** `holdout_outputs/{b0,b1}/*_logits.npy` + `*_probs.npy` (≥13 each). 3. Verify **ONNX** `onnx/segformer_{b0,b1}_768_binary.onnx`. 4. Verify **reports/logs** `run_summary.json`, `metrics_b0.json`, `metrics_b1.json`, `run_outputs_train.log`. 5. **Package** → `run_outputs.tar.gz`. 6. **Checksums** → `SHA256SUMS.txt` + `run_outputs.tar.gz.sha256`. 7. **Download** the tarball (`runpodctl receive` / web file browser / `scp`) and **verify sha256 locally**. 8. **Only then TERMINATE the Pod.** The script prints `ARTIFACT CHECK: FAIL — do NOT terminate` if anything is missing.

---

## J. FINAL OUTPUT (copy-paste)

**1. Recommended RunPod configuration**
`RTX 4090 24GB · Secure Cloud · image runpod/pytorch:2.5.1-py3.11-cuda12.1.0-devel · 8 vCPU / 32 GB RAM · 30 GB disk · no open ports · (optional) 20 GB network volume at /workspace`

**2. Installation commands**
```bash
cd gpu_run
# torch 2.5.1+cu121 ships in the container; if not:
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements-gpu.txt
```

**3. Preflight commands**
```bash
nvidia-smi
python preflight.py dataset_bundle     # GPU + VRAM + disk + dataset integrity + holdout==13 + no-leak + checksums
```

**4. Training command(s)**
```bash
python train_gpu.py --data dataset_bundle --out run_outputs \
  --models b0,b1 --res 768 --epochs 18 --bs 4 --lr 6e-5 --device auto 2>&1 | tee run_outputs_train.log
# (or simply: bash run.sh  — does install + preflight + train + collect)
```

**5. Artifact collection commands**
```bash
bash collect_and_checksum.sh          # verifies everything exists, checksums, builds run_outputs.tar.gz
```

**6. Shutdown checklist**
`checkpoints ✓ → logits/probs ✓ → ONNX ✓ → reports/logs ✓ → tar+sha256 ✓ → download + verify sha256 locally ✓ → TERMINATE POD`

**7. Estimated runtime** — B0 ~6–10 min · B1 ~18–28 min · eval+export ~3–5 min · **total Pod ≈ ~1 h** (budget 1.5 h).

**8. Estimated cost** — **~$0.75–$1.10 expected** (Secure $0.74/hr, ~1 h) · conservative ~$1.5–$3 · **hard cap $50 (needs re-approval above this)**.

---
*Prepared files:* `gpu_run/{requirements-gpu.txt, export_dataset_bundle.py, preflight.py, train_gpu.py, export_onnx.py, run.sh, collect_and_checksum.sh, dataset_bundle/}`. **No GPU rented. No training started. Holdout & annotations untouched.**
