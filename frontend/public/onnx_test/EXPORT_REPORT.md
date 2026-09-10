BROWSER EXPORT: PASS

**SegFormer-B0 at 768px successfully exported to ONNX and loaded/executed one forward pass in onnxruntime-web in a browser.**

# Milestone 1 v2 — Pre-GPU deployment validation (SegFormer-B0 → ONNX → onnxruntime-web)

> Gate to prove the **production architecture (SegFormer-B0, binary roof/bg)** exports and runs
> in a **real browser** before any GPU rental/training. Model is **UNTRAINED** — accuracy is
> irrelevant here. **Read-only** w.r.t. production annotations. **No GPU rented. No training.**

- **Date/heure** : 2026-09-10 ~12:13 UTC
- **Architecture** : **SegFormer-B0** (`nvidia/mit-b0`), `num_labels=2` (roof/bg), 3.71M params — comme spécifié par le client. **Pas** de Mask2Former, pas d'instance-seg dans le réseau.
- **Statut** : **PASS**

## 1. Model
- SegFormer-B0 untrained, entrée **768×768**, sortie **binaire roof/background**.
- Wrapper de production : logits (H/4) → **upsample bilinéaire → 768×768** → `logits[1,2,768,768]` ; le downstream fait `argmax(dim=1)` → masque binaire roof/bg.

## 2. ONNX export
| Champ | Valeur |
|---|---|
| PyTorch | 2.5.1 |
| Transformers | 4.46.3 |
| ONNX | 1.22.0 |
| onnxruntime (python, checker/inspection) | 1.29.0 |
| **opset** | **17** |
| input name | `pixel_values` |
| input shape | `[1, 3, 768, 768]` |
| input dtype | `float32` (FLOAT) |
| output name | `logits` |
| output shape | `[1, 2, 768, 768]` |
| output dtype | `float32` (FLOAT) |
| dynamic/static | **static** (1×3×768×768 in / 1×2×768×768 out) |
| ONNX file size | **15.01 MB** |
| onnx.checker | **OK (valid graph)** |

**Export command** (`scripts/export_segformer_onnx.py`) :
```
torch.onnx.export(model, dummy[1,3,768,768], "segformer_b0_768_binary.onnx",
                  opset_version=17, input_names=['pixel_values'],
                  output_names=['logits'], dynamic_axes=None, do_constant_folding=True)
```
Complete export log → `export.log`. Metadata → `export_meta.json`.

## 3. Real browser test (authoritative)
Executed in a **real Chromium browser** (Playwright), loading the ACTUAL `.onnx` via **onnxruntime-web** from the app URL `/onnx_test/browser_test.html`. Not Python-only.

- **onnxruntime-web version** : common **1.29.0**, web **1.29.0** (CDN jsdelivr).
- **Execution provider requested** : `webgpu` → **unavailable in headless** (`Failed to get GPU adapter`) → **fell back to `wasm`**.
- **EP used : WASM** (single-thread; no cross-origin isolation on dev server).
- Session create : ~1494 ms · **one forward pass : 2295 ms**.
- **Output** : name=`logits`, dtype=`float32`, shape=`[1,2,768,768]`.
- **logits → argmax → binary mask 768×768, values {0,1}** (roof_pixels reported) — output→binary conversion done **in the browser**.
- Screenshot: `browser_test_screenshot.png` · console: `browser_console.log` · page/script: `browser_test.html`, `browser_test.js`.

## 4. Operator compatibility
Graph operators inventory (`export_meta.json`):
`Conv×20, LayerNormalization×30, MatMul×68, Add×76, Reshape×76, Transpose×76, Constant×176, Concat×22, Shape×21, Slice×21, Div×16, Mul×16, Softmax×8, Erf×8, Resize×5, Identity×4, Relu×1`.
- **Operators NOT in the known WASM-supported set : NONE.**
- The session was created and ran successfully on the **WASM EP** → operator support **confirmed at runtime**, not just theoretically. No unsupported op; **no custom/silent replacement**.

## 5. Output compatibility
`logits[1,2,768,768]` → `argmax(channel)` → binary roof/background mask ∈ {0,1}. Verified in **Python onnxruntime** (roof_px=149415 on random input) **and in the browser** (roof_px reported live). Compatible with the downstream binary-mask consumer.

## 6. Downstream per-plane pipeline (no NN instance seg)
`scripts/downstream_verify.py` → `downstream_verify.json`. Binary roof mask → **connected components** → individual planes → roofline → metrics. Ran on a **binary roof mask (GT union)** and on the **actual untrained ONNX SegFormer binary output**:
- **0016 (normal)** : 3 planes matched → **eave / LOWER boundary** → perp+coverage computed.
- **0169 (parapet, 4 parapet planes)** : 4 planes matched → **parapet / UPPER boundary** (via existing `-parapet` flag) → perp+coverage computed.
- Untrained ONNX binary output also flows through CC (17 / 24 components) and matches — **path executes on real model output**.
- **Boundary branches executed : normal eave (lower) = TRUE, parapet upper = TRUE.**
- **Checkpoint-selection rule (unchanged client logic)** : PRIMARY = **mean perpendicular roofline error (MINIMIZE)** ; SECONDARY = horizontal coverage (tie-break) ; **Dice/IoU diagnostic only, NOT used for selection**. Evaluation remains **downstream of the binary model**.

## 7. Data integrity (read-only — verified before AND after)
SHA-256 over (dataset_id, split, house_group_id, storage_path, plane paths) for all approved images:
- before = after = `15e559bb25f874954dcb585aac69287650ca834314661b068e1d79d490dcaea2` → **UNCHANGED**.
- 150 approved annotations unchanged · fixed **13-image holdout intact** · no masks overwritten · no metadata/split changed.

## 8. GPU / training estimate (NO GPU RENTED, NO TRAINING)
**A. Smoke-test runtime already measured (768px)**
- Pipeline smoke (Mask2Former, previous gate): 7 images, 2 epochs, CPU, ~100 s/epoch.
- **SegFormer-B0 @768 CPU (this task, minimal timing)** : **3.59 s/image** train step (fwd+bwd+opt, bs1) ; 1.24 s/image inference.

**B. Estimated runtime — real run (150 images, SegFormer-B0, 768px, single consumer GPU)**
- Work ≈ 150 imgs × ~60–120 epochs = ~9k–18k steps.
- CPU would be ~9–18 h (impractical). GPU speedup for this tiny model (3.71M params) ≈ 20–60×.
- **Estimated GPU wall time : ~0.5–2 h** training + eval/checkpointing/setup ⇒ **budget ~1–3 h** (buffer to 5 h for retries/hyperparam tries).

**C. Recommended GPU**
| GPU | VRAM | ~Hourly (RunPod/Vast) | Est. hours | Est. total |
|---|---|---|---|---|
| **NVIDIA T4** (cheapest, sufficient) | 16 GB | $0.35–0.50 | 1–3 | **$0.35–$1.50** |
| **NVIDIA L4** (recommended, best value) | 24 GB | $0.40–0.80 | 1–3 | **$0.40–$2.40** |
| RTX 4090 (consumer, Vast) | 24 GB | $0.35–0.70 | 1–2 | $0.35–$1.40 |
- **VRAM needed** : SegFormer-B0 @768 bs4–8 ≈ 4–8 GB → 16 GB is ample.
- **Estimated total cost ≈ $1–$3** (even with 5 h buffer, < $5). **Well within the approved $50 budget.**

## 9. Decision
The exact production architecture (SegFormer-B0 binary @768) **exports to ONNX and runs one forward pass in a real browser via onnxruntime-web (WASM EP)**, output converts to a binary mask, and the existing downstream connected-components per-plane pipeline consumes it with correct eave/parapet boundary handling and perpendicular-error-primary selection. **Estimated GPU cost ~$1–3 (< $50).**

**GATE: BROWSER EXPORT = PASS** → cleared to proceed to the GPU training stage in a later task.
**No GPU rented. No training performed. No production annotations modified.**

> Human + SAM2 workflow only (no VLM). Untrained model — SMOKE/DEPLOYMENT VALIDATION ONLY, NOT FINAL ACCURACY.
