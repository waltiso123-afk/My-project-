"""5-image pilot: REAL SAM2 (CPU) point-prompt run with QA evidence.
Runs in-process (imports ml/storage) so we can measure encode/predict timing and
demonstrate embedding caching (one encode per image, multiple cached predicts).
Saves per-plane SAM2 masks, per-plane overlays and a combined overlay + metrics JSON.
NO fabrication: every number is measured, every mask is what SAM2 actually returned.
"""
import os, io, sys, json, time
sys.path.insert(0, "/app/backend")
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage, ml

OUT = Path("/app/reports/pilot"); (OUT / "masks").mkdir(parents=True, exist_ok=True)
(OUT / "overlays").mkdir(parents=True, exist_ok=True)
c = pymongo.MongoClient(os.environ["MONGO_URL"]); db = c[os.environ["DB_NAME"]]

# One positive point per roof plane, in EXIF-APPLIED DISPLAY pixel coordinates.
POINTS = {
    "0001": [("tower_hip", (889, 406)), ("right_hip_garage", (1097, 461)), ("left_wing", (609, 470))],
    "0169": [("left_tower_parapet", (576, 1319)), ("mid_block_parapet", (1152, 1253)),
             ("roof_deck_pergola", (1613, 1109)), ("right_canopy_slab", (2016, 1728))],
    "0050": [("left_gable_slope", (580, 258)), ("right_gable_slope", (825, 270)), ("garage_hip", (600, 352))],
    "0138": [("left_hip", (263, 439)), ("center_front_hip", (614, 437)), ("right_garage_hip", (1097, 470))],
    "0173": [("upper_tower_hip", (2530, 979)), ("left_hip", (1142, 1530)),
             ("main_lower_hip", (1632, 2081)), ("right_entry_roof", (4692, 898))],
}
COLORS = [(255,60,60),(60,160,255),(60,220,120),(255,200,40),(200,80,255),(255,120,0)]

predictor = ml._load_sam2("cpu")
report = {}

for did, planes in POINTS.items():
    img_doc = db.images.find_one({"dataset_id": did})
    data, _ = storage.get_object(img_doc["storage_path"])
    img = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
    h, w = img.shape[:2]
    base = Image.fromarray(img).convert("RGBA")

    t0 = time.time()
    with torch.inference_mode():
        predictor.set_image(img)   # ENCODE once (embedding)
    encode_ms = int((time.time() - t0) * 1000)

    combined = base.copy()
    plane_results = []
    for i, (name, (px, py)) in enumerate(planes):
        pts = np.array([[px, py]], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        t1 = time.time()
        with torch.inference_mode():   # PREDICT reuses cached embedding
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
        predict_ms = int((time.time() - t1) * 1000)
        best = int(np.argmax(scores))
        m = (masks[best] > 0).astype(np.uint8) * 255
        area_frac = round(float(m.sum() / 255 / (h * w)), 4)

        # save raw per-plane binary mask
        Image.fromarray(m).save(OUT / "masks" / f"{did}_{i+1:02d}_{name}.png")

        # per-plane overlay
        col = COLORS[i % len(COLORS)]
        tint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        tarr = np.zeros((h, w, 4), np.uint8); tarr[m > 0] = (*col, 110); tint = Image.fromarray(tarr)
        one = Image.alpha_composite(base, tint)
        d = ImageDraw.Draw(one); r = max(6, w // 200)
        d.ellipse([px-r, py-r, px+r, py+r], fill=(255,255,0,255), outline=(0,0,0,255), width=3)
        one.convert("RGB").save(OUT / "overlays" / f"{did}_{i+1:02d}_{name}.jpg", quality=70)
        combined = Image.alpha_composite(combined, tint)
        cd = ImageDraw.Draw(combined); cd.ellipse([px-r, py-r, px+r, py+r], fill=(255,255,0,255), outline=(0,0,0,255), width=3)

        plane_results.append({"plane": name, "point_disp": [px, py], "score": round(float(scores[best]),3),
                              "area_frac": area_frac, "predict_ms": predict_ms,
                              "sam2_multimask_scores": [round(float(s),3) for s in scores]})

    combined.convert("RGB").save(OUT / "overlays" / f"{did}_00_COMBINED.jpg", quality=72)
    report[did] = {"batch": img_doc["batch"], "exif_orientation": img_doc.get("exif_orientation"),
                   "display_size": [w, h], "encode_ms": encode_ms,
                   "embedding_cached_after_first_predict": True, "n_points": len(planes),
                   "planes": plane_results}
    print(f"{did}: encode={encode_ms}ms, planes={len(planes)}, "
          f"scores={[p['score'] for p in plane_results]}, areas={[p['area_frac'] for p in plane_results]}")

(OUT / "sam2_metrics.json").write_text(json.dumps(report, indent=2))
print("METRICS ->", OUT / "sam2_metrics.json")
