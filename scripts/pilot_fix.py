"""Corrective re-run: reposition misfired points onto actual roof tile (human-in-loop step)."""
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

OUT = Path("/app/reports/pilot")
c = pymongo.MongoClient(os.environ["MONGO_URL"]); db = c[os.environ["DB_NAME"]]

# Repositioned points (display coords) after inspecting the first-pass overlays.
FIX = {
    "0001": [("tower_hip", (955, 378)), ("left_wing", (665, 452))],
    "0050": [("left_gable_slope", (500, 240))],
    "0169": [("mid_block_parapet", (1140, 1180))],
}
predictor = ml._load_sam2("cpu")
res = {}
for did, planes in FIX.items():
    img_doc = db.images.find_one({"dataset_id": did})
    data, _ = storage.get_object(img_doc["storage_path"])
    img = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
    h, w = img.shape[:2]
    base = Image.fromarray(img).convert("RGBA")
    with torch.inference_mode():
        predictor.set_image(img)
    for i, (name, (px, py)) in enumerate(planes):
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(point_coords=np.array([[px,py]],dtype=np.float32),
                                                 point_labels=np.array([1],dtype=np.int32), multimask_output=True)
        best = int(np.argmax(scores)); m = (masks[best]>0).astype(np.uint8)*255
        af = round(float(m.sum()/255/(h*w)),4)
        tarr = np.zeros((h,w,4),np.uint8); tarr[m>0]=(255,60,60,120)
        one = Image.alpha_composite(base, Image.fromarray(tarr))
        d = ImageDraw.Draw(one); r=max(6,w//200)
        d.ellipse([px-r,py-r,px+r,py+r], fill=(255,255,0,255), outline=(0,0,0,255), width=3)
        one.convert("RGB").save(OUT/"overlays"/f"{did}_FIX_{name}.jpg", quality=72)
        res.setdefault(did,[]).append({"plane":name,"point":[px,py],"score":round(float(scores[best]),3),"area_frac":af})
        print(f"{did} {name}: score={round(float(scores[best]),3)} area_frac={af}")
(OUT/"sam2_fix_metrics.json").write_text(json.dumps(res,indent=2))
