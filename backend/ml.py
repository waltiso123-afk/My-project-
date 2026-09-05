"""SAM2 (real) + manual mask generation. Backend truthfully reports which engine ran.

HONESTY CONTRACT:
- 'sam2_cuda'  = real SAM2 on a CUDA GPU.
- 'sam2_cpu'   = real SAM2 on CPU (slower, embedding cached per image).
- 'cpu_colour_region' = OpenCV flood-fill COLOUR helper. It is NEVER called SAM2 and is
  NEVER silently used as a SAM2 fallback. If SAM2 fails we return an explicit error.
"""
import io
import os
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2

import config
import storage

SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/app/models/sam2.1_hiera_tiny.pt")
SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")

_sam2_predictor = None
_sam2_device = None
_sam2_last_image_id = None  # embedding cache key


def torch_cuda_ready() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def sam2_installed() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("sam2") is not None
    except Exception:
        return False


def _checkpoint_present() -> bool:
    return os.path.exists(SAM2_CHECKPOINT)


def sam2_status() -> dict:
    cuda = torch_cuda_ready()
    installed = sam2_installed()
    ckpt = _checkpoint_present()
    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        torch_ver = None

    if installed and ckpt and cuda:
        backend, device = "sam2_cuda", "cuda"
        msg = "SAM2 ready on CUDA GPU (real point-prompt segmentation)."
    elif installed and ckpt:
        backend, device = "sam2_cpu", "cpu"
        msg = "SAM2 (CPU) ready — real point-prompt segmentation, ~4s encode/image then fast clicks."
    else:
        backend, device = "unavailable", None
        missing = []
        if not installed:
            missing.append("sam2 package")
        if not ckpt:
            missing.append(f"checkpoint {SAM2_CHECKPOINT}")
        msg = "SAM2 unavailable (" + ", ".join(missing) + "). Use manual tools or the Colour Region helper."
    return {
        "sam2_installed": installed,
        "checkpoint_present": ckpt,
        "checkpoint": SAM2_CHECKPOINT,
        "config": SAM2_CONFIG,
        "cuda_available": cuda,
        "device": device,
        "torch_version": torch_ver,
        "active_backend": backend,
        "message": msg,
    }


def _load_sam2(device: str):
    global _sam2_predictor, _sam2_device
    if _sam2_predictor is not None and _sam2_device == device:
        return _sam2_predictor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    _sam2_predictor = SAM2ImagePredictor(model)
    _sam2_device = device
    return _sam2_predictor


def _mask_to_polygons(mask: np.ndarray, max_points: int = 80):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 50:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.004 * peri, True)
        pts = [[int(p[0][0]), int(p[0][1])] for p in approx]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def sam2_propose(image_bytes: bytes, image_id: str, positive_points, negative_points=None) -> dict:
    """Real SAM2 point-prompt segmentation. Embedding cached per image_id.
    Raises on failure — the caller must surface the error, never silently fall back."""
    global _sam2_last_image_id
    status = sam2_status()
    if status["active_backend"] not in ("sam2_cpu", "sam2_cuda"):
        raise RuntimeError(status["message"])
    device = status["device"]
    import torch
    img = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB"))
    h, w = img.shape[:2]
    predictor = _load_sam2(device)

    t0 = time.time()
    cached = (_sam2_last_image_id == image_id)
    if not cached:
        with torch.inference_mode():
            predictor.set_image(img)
        _sam2_last_image_id = image_id
    encode_ms = int((time.time() - t0) * 1000)

    pts = np.array(positive_points + (negative_points or []), dtype=np.float32)
    labels = np.array([1] * len(positive_points) + [0] * len(negative_points or []), dtype=np.int32)
    t1 = time.time()
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=labels, multimask_output=True)
    predict_ms = int((time.time() - t1) * 1000)
    best = int(np.argmax(scores))
    mask = (masks[best] > 0).astype(np.uint8) * 255
    polygons = _mask_to_polygons(mask)
    return {
        "backend": status["active_backend"], "device": device,
        "score": round(float(scores[best]), 3),
        "area_frac": round(float(mask.sum() / 255 / (h * w)), 4),
        "encode_ms": encode_ms, "embedding_cached": cached, "predict_ms": predict_ms,
        "polygons": polygons,
        "note": f"Real SAM2 on {device.upper()} point-prompt segmentation.",
    }


def colour_region(image_bytes: bytes, positive_points, tolerance: int = 22) -> dict:
    """Optional COLOUR REGION helper (OpenCV flood-fill). NOT SAM2, NOT semantic.
    Only runs when the user explicitly selects the Colour Region tool."""
    img = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB"))
    h, w = img.shape[:2]
    seed = positive_points[0]
    seed = (int(np.clip(seed[0], 0, w - 1)), int(np.clip(seed[1], 0, h - 1)))
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(img.copy(), flood_mask, seed, 0, (tolerance,) * 3, (tolerance,) * 3, flags)
    m = flood_mask[1:-1, 1:-1]
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return {"backend": "cpu_colour_region", "polygons": _mask_to_polygons(m),
            "note": "Colour Region Helper (CPU flood-fill). NOT SAM2 — low-level colour similarity only."}



# ---------------- Mask generation on approve ----------------

def polygons_to_mask(polygons, size):
    im = Image.new("L", size, 0)
    d = ImageDraw.Draw(im)
    for poly in polygons or []:
        if len(poly) >= 3:
            d.polygon([tuple(p) for p in poly], fill=255)
    return im


import base64


def _raster_to_mask(raster_png_b64: str, size):
    """Decode a base64 PNG brush raster into a strict 255/0 mask at `size`."""
    raw = base64.b64decode(raster_png_b64.split(",")[-1])
    im = Image.open(io.BytesIO(raw)).convert("L").resize(size)
    arr = (np.array(im) >= 128).astype(np.uint8) * 255
    return Image.fromarray(arr)


def generate_and_store(dataset_id: str, ext: str, size, planes: list,
                       occlusions: list, notes: str, view_type: str = None,
                       coordinate_space: str = "exif_applied_display") -> dict:
    """Generate per-plane masks (255/0), merged mask, and spec-compliant meta JSON.
    Masks are generated in the EXIF-applied display coordinate space (same pixels the
    annotator saw and the browser rendered), so image and mask never misalign."""
    w, h = size
    merged = np.zeros((h, w), np.uint8)
    plane_records = []
    parapets = []

    for idx, plane in enumerate(planes, start=1):
        is_parapet = bool(plane.get("is_parapet"))
        suffix = "-parapet" if is_parapet else ""
        plane_id = f"plane-{idx:02d}"
        fname = f"{plane_id}{suffix}.png"
        if plane.get("raster_png"):
            mask_img = _raster_to_mask(plane["raster_png"], (w, h))
        else:
            mask_img = polygons_to_mask(plane.get("polygons", []), (w, h))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        storage.put_object(config.plane_path(dataset_id, fname), buf.getvalue(), "image/png")
        merged = np.maximum(merged, np.array(mask_img))
        rec = {
            "index": idx, "plane_id": plane_id, "name": plane.get("name", plane_id),
            "file": fname, "is_parapet": is_parapet,
            "generation_method": plane.get("generation_method", "manual_polygon"),
            "human_corrected": bool(plane.get("human_corrected", True)),
            "human_approved": True,
        }
        plane_records.append(rec)
        if is_parapet:
            parapets.append(plane_id)  # spec: "parapets": ["plane-02"]

    merged_buf = io.BytesIO()
    Image.fromarray(merged).save(merged_buf, format="PNG")
    storage.put_object(config.merged_path(dataset_id), merged_buf.getvalue(), "image/png")

    meta = {
        "file": f"{dataset_id}.{ext}",
        "occlusions": occlusions or [],   # spec: [{type, bbox:[x0,y0,x1,y1] normalized, hides:[plane-NN]}]
        "parapets": parapets,             # spec: ["plane-02"]
        "notes": notes or "",
        "view_type": view_type or "uncertain",
        "planes": plane_records,
        "spec_version": "roofline-labeling-spec-v1",
        "coordinate_space": coordinate_space,
        "image_size": {"width": w, "height": h},
    }
    storage.put_object(config.meta_path(dataset_id),
                       json.dumps(meta, indent=2).encode(), "application/json")
    return {"merged": config.merged_path(dataset_id),
            "planes": [config.plane_path(dataset_id, p["file"]) for p in plane_records],
            "meta": meta}
