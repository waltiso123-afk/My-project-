"""Milestone 1 Step 1 — READ-ONLY validation of the 3 open items + dataset readiness.
Reports exactly what is present. Does NOT modify any annotation."""
import os, io, sys, json
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
import storage, config
import pymongo
from PIL import Image
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

def dims(path):
    d, _ = storage.get_object(path); return Image.open(io.BytesIO(d)).size

approved = list(db.images.find({"status": "approved"}))
print("== DATASET ==")
print("approved images:", len(approved))
from collections import Counter
print("by split:", dict(Counter(a.get("split") for a in approved)))
print("by batch:", dict(Counter(a.get("batch") for a in approved)))

print("\n== A. 0169 PARAPETS ==")
img = db.images.find_one({"dataset_id": "0169"})
ann = db.annotations.find_one({"dataset_id": "0169"})
print("image coord_policy:", img.get("coordinate_policy"), "exif:", img.get("exif_orientation"),
      "stored dims:", dims(img["storage_path"]), "DB w/h:", img["width"], img["height"])
if ann:
    print("status:", ann.get("status"), "n planes:", len(ann.get("planes", [])))
    for p in ann.get("planes", []):
        print("   plane:", p.get("name"), "| is_parapet:", p.get("is_parapet"))
    gp = ann.get("generated", {}).get("planes", [])
    print("stored plane files:", [g.split("/")[-1] for g in gp])
    for g in gp[:8]:
        print("   ", g.split("/")[-1], "dims:", dims(g))
    print("merged dims:", dims(config.merged_path("0169")))
    meta = json.loads(storage.get_object(config.meta_path("0169"))[0])
    print("meta image_size:", meta.get("image_size"), "| meta parapets:", meta.get("parapets"))
else:
    print("NO ANNOTATION for 0169")

print("\n== B. COORDINATE SPACE (0169 + a few) ==")
for did in ["0169", "0001", "0173"]:
    d = db.images.find_one({"dataset_id": did}); a = db.annotations.find_one({"dataset_id": did})
    sd = dims(d["storage_path"]); mdim = dims(a["generated"]["planes"][0]) if a and a.get("generated") else None
    mgd = dims(config.merged_path(did)) if a else None
    meta = json.loads(storage.get_object(config.meta_path(did))[0]) if a else {}
    print(f"{did}: stored={sd} mask={mdim} merged={mgd} meta_size={meta.get('image_size')} exif={d.get('exif_orientation')} "
          f"align={'PASS' if (sd==mdim==mgd and meta.get('image_size')=={'width':sd[0],'height':sd[1]}) else 'FAIL'}")

print("\n== C. 0001 NEIGHBOUR ==")
a1 = db.annotations.find_one({"dataset_id": "0001"})
print("0001 planes:", [p.get("name") for p in a1.get("planes", [])], "| status:", a1.get("status"))
print("notes:", (a1.get("notes") or "")[:200])
