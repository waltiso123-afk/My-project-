#!/usr/bin/env python3
"""Validate REAL SAM2 CPU inference on image 0001 (roof points). Read-only."""
import sys, time
sys.path.insert(0, "/app/backend")
import numpy as np
from PIL import Image, ImageOps
import torch
import config
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CKPT = "/app/models/sam2.1_hiera_tiny.pt"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

img = np.array(ImageOps.exif_transpose(Image.open(config.DATA_DIR / "staging/batch1/0001.webp")).convert("RGB"))
H, W = img.shape[:2]
print(f"device=cpu  torch={torch.__version__}  image={W}x{H}")

t0 = time.time()
model = build_sam2(CFG, CKPT, device="cpu")
predictor = SAM2ImagePredictor(model)
print(f"model build: {time.time()-t0:.1f}s")

t0 = time.time()
with torch.inference_mode():
    predictor.set_image(img)
print(f"set_image (encode) : {time.time()-t0:.1f}s")

points = {"red_roof_a": (811, 440), "roof_center": (int(W*0.62), int(H*0.36)), "roof_left": (int(W*0.42), int(H*0.34))}
for name, pt in points.items():
    t0 = time.time()
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=np.array([pt], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True)
    best = int(np.argmax(scores))
    area = masks[best].sum()
    dt = time.time() - t0
    print(f"{name:12} pt={pt} predict={dt*1000:.0f}ms  best_score={scores[best]:.3f}  area={100*area/(H*W):.1f}%")
print("PASS: real SAM2 CPU inference works")
