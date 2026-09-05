"""Mini-Milestone 1 — STEP 2: SAM2 segmentation + reconciliation from the VLM plan.
For each VLM-planned plane: SAM2 point-prompt -> best mask, clipped to the (dilated)
VLM polygon bbox to kill gross overreach; fall back to the VLM polygon when SAM2 misfires.
Saves reconciled per-plane masks (display size) + a combined overlay for human QA.
"""
import os, io, sys, json
sys.path.insert(0, "/app/backend")
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2, torch
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage, ml

ROOT = Path("/app/reports/PILOT_MILESTONE_1")
PLANS = ROOT / "plans"
WORK = ROOT / "work"; WORK.mkdir(parents=True, exist_ok=True)
IDS = ["0001", "0050", "0138", "0169", "0173"]
COLORS = [(255,64,64),(64,160,255),(64,220,120),(255,205,40),(200,90,255),(255,130,0),
          (0,200,200),(255,0,150),(150,255,0),(120,120,255),(255,150,150)]
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
predictor = ml._load_sam2("cpu")

def poly_px(poly, W, H):
    return np.array([[int(x*W), int(y*H)] for x, y in poly], np.int32)

for did in IDS:
    plan = json.loads((PLANS / f"{did}.json").read_text())
    doc = db.images.find_one({"dataset_id": did})
    raw, _ = storage.get_object(doc["storage_path"])
    disp = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    W, H = disp.size
    arr = np.array(disp)
    with torch.inference_mode():
        predictor.set_image(arr)
    wd = WORK / did; wd.mkdir(exist_ok=True)
    overlay = disp.convert("RGBA")
    tint = np.zeros((H, W, 4), np.uint8)
    recs = []
    for i, pl in enumerate(plan.get("planes", []), start=1):
        px, py = int(pl["point"][0]*W), int(pl["point"][1]*H)
        poly = poly_px(pl.get("polygon", []), W, H)
        # polygon bbox dilated by 8% of image size
        polymask = np.zeros((H, W), np.uint8)
        if len(poly) >= 3:
            cv2.fillPoly(polymask, [poly], 255)
            x0, y0 = poly[:,0].min(), poly[:,1].min(); x1, y1 = poly[:,0].max(), poly[:,1].max()
        else:
            x0, y0, x1, y1 = px-1, py-1, px+1, py+1
        dx, dy = int(0.08*W), int(0.08*H)
        bx0, by0 = max(0, x0-dx), max(0, y0-dy); bx1, by1 = min(W, x1+dx), min(H, y1+dy)
        clipbox = np.zeros((H, W), np.uint8); clipbox[by0:by1, bx0:bx1] = 255
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(point_coords=np.array([[px,py]],np.float32),
                                                 point_labels=np.array([1],np.int32), multimask_output=True)
        best = int(np.argmax(scores)); sam = (masks[best]>0).astype(np.uint8)*255
        sam_clip = cv2.bitwise_and(sam, clipbox)
        af_sam = sam_clip.sum()/255/(H*W)
        # reconcile: accept clipped SAM2 if plausible size, else polygon fallback
        if 0.0004 <= af_sam <= 0.30 and sam_clip[max(0,min(H-1,py)), max(0,min(W-1,px))] > 0:
            final = sam_clip; source = "sam2_clipped"
        elif len(poly) >= 3:
            final = polymask; source = "vlm_polygon_fallback"
        else:
            final = sam_clip; source = "sam2_raw"
        Image.fromarray(final).save(wd / f"plane_{i:02d}.png")
        col = COLORS[(i-1) % len(COLORS)]
        tint[final > 0] = (*col, 95)
        recs.append({"i": i, "name": pl.get("name"), "is_parapet": bool(pl.get("is_parapet")),
                     "point_px": [px, py], "source": source,
                     "area_frac": round(float(final.sum()/255/(H*W)), 4),
                     "sam_score": round(float(scores[best]),3), "eave_note": pl.get("eave_note","")})
    overlay = Image.alpha_composite(overlay, Image.fromarray(tint)); ov = np.array(overlay)
    for i, pl in enumerate(plan.get("planes", []), start=1):
        m = np.array(Image.open(wd / f"plane_{i:02d}.png").convert("L"))
        col = COLORS[(i-1) % len(COLORS)]
        cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (*col,255), max(2, W//600))
        px, py = recs[i-1]["point_px"]; r = max(5, W//220)
        cv2.circle(ov, (px,py), r, (255,255,0), -1); cv2.circle(ov, (px,py), r, (0,0,0), 2)
    ovimg = Image.fromarray(ov).convert("RGB")
    d = ImageDraw.Draw(ovimg)
    ovimg.thumbnail((1600,1600))
    ovimg.save(WORK / f"{did}_overlay.jpg", quality=72)
    (wd / "recs.json").write_text(json.dumps({"display_size":[W,H],"planes":recs}, indent=2))
    print(f"{did}: {len(recs)} planes | sources={[r['source'][:4] for r in recs]} | areas={[r['area_frac'] for r in recs]}")
print("WORK ->", WORK)
