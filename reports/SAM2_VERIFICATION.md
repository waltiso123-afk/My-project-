# SAM2 CPU VERIFICATION REPORT (option C)

Verified by testing_agent iteration_3 (backend 100% 9/9, frontend 100%, retest_needed=false) + manual curl.

| item | result |
|------|--------|
| **Active backend** | `sam2_cpu` (real SAM2, device=cpu) |
| **SAM2 CPU confirmation** | YES — `/api/ml/status` → sam2_installed=true, checkpoint_present=true, cuda_available=false |
| **Exact SAM2 / checkpoint** | sam2 1.1.0 · `sam2.1_hiera_tiny.pt` (149 MB) · cfg `configs/sam2.1/sam2.1_hiera_t.yaml` |
| **PyTorch version** | 2.5.1 (CPU wheel) |
| **CPU inference time** | first click ~3.8 s (image encode) · subsequent clicks ~35–170 ms (predict) |
| **Embedding caching** | WORKS — 2nd click same image: encode_ms≈0, embedding_cached=true |
| **Coordinate test** | PASS — screen→image transform verified; EXIF-applied on both browser & backend; click maps to correct pixel (portrait 0169 + landscape 0001) |
| **Roof-point test (0001)** | PASS — point [645,391] → backend sam2_cpu, score 0.844, **area 12.3%** (plausible roof plane), 15 polygons. NOT "No region found", NOT 96–99%, NOT road/grass, NOT colour-similarity |
| **2nd roof plane** | PASS — [952,414] → real region returned |
| **Colour Region helper** | Separate `method=colour_region` → backend `cpu_colour_region`, note "NOT SAM2". Never returned for method=sam2. Never a silent fallback |
| **SAM2 failure handling** | `backend:"sam2_error"` + error message + empty polygons; UI tells annotator to switch to manual. No mislabeled flood-fill |
| **Manual annotation test** | polygon + brush + eraser + vertex-edit + undo/redo + clear all work; function independent of ML |
| **Approve artifacts** | polygon plane AND brush-raster plane → per-plane 255/0 + merged union + spec meta (coordinate_space=exif_applied_display, provenance per plane) |
| **Regression tests** | prior foundation (8) + audit_v2 (9) + sam2 (9) all green; dataset untouched (217 pending) |
| **testing-agent result** | iteration_3 PASS (100% / 100%), retest_needed=false |

## Honest environment note
No GPU / no CUDA in this environment. SAM2 runs on **CPU** (slow encode, fast clicks) — fine for the 5-image
pilot, marginal for the full 200 (GPU recommended for scale). The badge and API always report the true backend
(`SAM2 (CPU)`); the OpenCV flood-fill is only the optional "Colour Region Helper (CPU)" and is never called SAM2.

## Stop-gate status
Assist pipeline is validated and reliable. **Ready for the human 5-image pilot** in the Annotation Studio
(SAM2 initial mask → manual correction per `/app/PROJECT_SPEC.md`). Mass annotation (200) and 50/100/150
training remain gated until the 5 pilot masks are reviewed and approved by you.
