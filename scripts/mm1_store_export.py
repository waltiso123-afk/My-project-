"""Mini-Milestone 1 — STEP 4: push FINAL masks through the REAL production code path
(ml.generate_and_store -> object storage planes/merged/meta), persist approved annotations,
then export the canonical dataset/ layout (images/planes/merged/meta) for the client package.
"""
import os, io, sys, json, base64
sys.path.insert(0, "/app/backend")
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage, ml, config

ROOT = Path("/app/reports/PILOT_MILESTONE_1")
FINAL = ROOT / "final"
DATASET = ROOT / "dataset"
for s in ["images", "planes", "merged", "meta"]:
    (DATASET / s).mkdir(parents=True, exist_ok=True)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

# Per-image production metadata (names / parapet flags / occlusions / notes / view) — matches STEP 3.
META = {
 "0001": {"view":"oblique","planes":[("left_wing_front",False),("central_upper_hip",False),("garage_front_hip",False)],
   "occlusions":[{"type":"palm","bbox":[0.32,0.15,0.52,0.55],"hides":["plane-02","plane-03"]},
                 {"type":"palm","bbox":[0.79,0.30,0.86,0.45],"hides":["plane-03"]}],
   "notes":"Subject house only; left and right neighbour houses excluded. Palms partially cross planes (<15% image width each) so planes continue underneath per the 15% rule."},
 "0050": {"view":"frontal","planes":[("gable_left_slope",False),("gable_right_slope",False),("garage_front_hip",False)],
   "occlusions":[{"type":"tree","bbox":[0.00,0.30,0.28,0.60],"hides":["plane-01"]}],
   "notes":"Central front gable (both visible slopes) + front garage hip of the subject villa. Attached villa to the right is a neighbour and excluded. A tree partly occludes the left gable slope."},
 "0138": {"view":"oblique","planes":[("left_front_hip",False),("right_garage_hip",False)],
   "occlusions":[{"type":"palm","bbox":[0.40,0.35,0.62,0.75],"hides":["plane-01","plane-02"]}],
   "notes":"Long front hip roof. A large palm occludes the centre (>15% image width) so the readable front plane is delivered as two segments (left of palm and right/garage side). Neighbour house at far right excluded."},
 "0173": {"view":"oblique","planes":[("left_upper_hip",False),("main_lower_hip",False),("central_tower_front",False),("right_upper_hip",False)],
   "occlusions":[{"type":"palm","bbox":[0.54,0.29,0.86,0.85],"hides":["plane-04"]}],
   "notes":"Tile hip roof: left upper, main lower (over garage), central tower front, right upper. Palm crosses the right upper plane (~12% image width, <15%) so that plane continues underneath."},
 "0169": {"view":"frontal","planes":[("highest_block_parapet",True),("left_block_parapet",True),("main_upper_parapet",True),("lower_canopy_front",False)],
   "occlusions":[{"type":"palm","bbox":[0.00,0.05,0.13,0.40],"hides":["plane-02"]}],
   "notes":"PROVISIONAL / AMBIGUOUS — pending client confirmation. Modern flat-roof home behind parapet walls; from ground only parapet caps and a lower canopy slab are visible. Parapet planes are lit along the UPPER parapet edge (-parapet suffix). Open questions for the client: (a) does each flat roof behind a parapet count as its own lit plane; (b) is the louvered pergola/trellis a lit element (currently excluded); (c) the lower canopy is lit along the TOP of its dark fascia (top = drip edge), not the fascia bottom. Masks are provisional strips along the parapet caps and require client sign-off before scaling."},
}

def png_data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()

result_index = {}
for did, m in META.items():
    doc = db.images.find_one({"dataset_id": did})
    W, H = doc["width"], doc["height"]
    ext = doc["ext"]
    planes = []
    for i, (name, is_par) in enumerate(m["planes"], start=1):
        planes.append({"name": name, "is_parapet": is_par,
                       "raster_png": png_data_url(FINAL / did / f"plane_{i:02d}.png"),
                       "generation_method": "vlm_plan+sam2_pointprompt+human_qa", "human_corrected": True})
    # REAL production call
    res = ml.generate_and_store(did, ext, (W, H), planes, m["occlusions"], m["notes"], m["view"])
    now = datetime.now(timezone.utc).isoformat()
    ann = {"dataset_id": did, "planes": planes, "occlusions": m["occlusions"], "notes": m["notes"],
           "checklist": {}, "status": "approved", "approved_at": now, "updated_at": now,
           "generated": {"merged": res["merged"], "planes": res["planes"], "meta": config.meta_path(did)}}
    db.annotations.update_one({"dataset_id": did}, {"$set": ann}, upsert=True)
    db.images.update_one({"dataset_id": did}, {"$set": {"status": "approved", "view_type": m["view"]}})

    # ---- Export canonical dataset/ by reading storage back (byte-identical to production) ----
    raw, _ = storage.get_object(doc["storage_path"])
    (DATASET / "images" / f"{did}.{ext}").write_bytes(raw)
    (DATASET / "planes" / did).mkdir(exist_ok=True)
    plane_files = []
    for p in res["planes"]:
        fname = p.split("/")[-1]
        data, _ = storage.get_object(p)
        (DATASET / "planes" / did / fname).write_bytes(data)
        plane_files.append(fname)
    md, _ = storage.get_object(config.merged_path(did))
    (DATASET / "merged" / f"{did}.png").write_bytes(md)
    meta_bytes, _ = storage.get_object(config.meta_path(did))
    (DATASET / "meta" / f"{did}.json").write_bytes(meta_bytes)
    result_index[did] = {"ext": ext, "size": [W, H], "n_planes": len(plane_files), "plane_files": plane_files,
                         "parapets": json.loads(meta_bytes).get("parapets", [])}
    print(f"{did}: stored+exported {len(plane_files)} planes, parapets={result_index[did]['parapets']}")

(ROOT / "dataset_index.json").write_text(json.dumps(result_index, indent=2))
print("DATASET ->", DATASET)
