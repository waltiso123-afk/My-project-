"""Set is_parapet=True on 0169's 4 existing planes (pixels unchanged) -> re-run production
generate_and_store to add -parapet suffix + parapets[] in meta. Authorized by user (option b)."""
import os, io, sys, json, base64
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from datetime import datetime, timezone
import storage, ml, config, pymongo
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

img = db.images.find_one({"dataset_id": "0169"}); ann = db.annotations.find_one({"dataset_id": "0169"})
ext = img["ext"]; W, H = img["width"], img["height"]; view = img.get("view_type") or ann.get("view_type") or "frontal"
planes = []
for i, g in enumerate(ann["generated"]["planes"]):
    data, _ = storage.get_object(g)
    old = ann["planes"][i]
    planes.append({"name": old.get("name", f"Plane {i+1}"), "is_parapet": True,
                   "raster_png": "data:image/png;base64," + base64.b64encode(data).decode(),
                   "generation_method": old.get("generation_method", "human_point_prompt+sam2_cpu+human_qa"),
                   "human_corrected": True})
res = ml.generate_and_store("0169", ext, (W, H), planes, ann.get("occlusions", []), ann.get("notes", ""), view)
now = datetime.now(timezone.utc).isoformat()
db.annotations.update_one({"dataset_id": "0169"}, {"$set": {
    "planes": planes, "generated": {"merged": res["merged"], "planes": res["planes"], "meta": config.meta_path("0169")},
    "updated_at": now, "parapet_flag_fix": "is_parapet set True on 4 planes per client (pixels unchanged)"}})
# cleanup stale non-suffixed plane files so exports/storage stay clean
for i in range(1, 12):
    p = config.plane_path("0169", i, False)  # plane-0i.png (non-parapet name)
    try:
        if p not in res["planes"]:
            storage.delete_object(p)
    except Exception:
        pass
meta = json.loads(storage.get_object(config.meta_path("0169"))[0])
print("new plane files:", [x.split("/")[-1] for x in res["planes"]])
print("meta parapets:", meta.get("parapets"))
print("done")
