"""Phase 1 — EXIF/coordinate normalization (production-wide, idempotent).
Ensures every stored production image is UPRIGHT (EXIF applied + stripped) so that
file pixel dims == displayed dims == mask dims == one deterministic coordinate space.
Preserves source provenance (keeps sha1 + records original raw size). Masks (already in
display/upright space) stay valid. Only images that actually have rotation are re-encoded."""
import os, io, sys, json
sys.path.insert(0, "/app/backend")
from PIL import Image, ImageOps
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage

db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
PILOTS = set(["0001","0050","0138","0169","0173"])
only_pilots = "--all" not in sys.argv

fmt = {"jpg":"JPEG","jpeg":"JPEG","png":"PNG","webp":"WEBP"}
changed=[]; skipped=[]
q = {"dataset_id":{"$in":list(PILOTS)}} if only_pilots else {}
for doc in db.images.find(q):
    did=doc["dataset_id"]; ext=(doc.get("ext") or "jpg").lower()
    eo=doc.get("exif_orientation")
    raw_upright = (doc.get("raw_width")==doc.get("width") and doc.get("raw_height")==doc.get("height"))
    if eo in (None,1) and raw_upright:
        skipped.append(did); continue
    raw,_=storage.get_object(doc["storage_path"])
    im=Image.open(io.BytesIO(raw)); up=ImageOps.exif_transpose(im)
    buf=io.BytesIO()
    save_fmt=fmt.get(ext,"JPEG")
    params={"quality":95} if save_fmt in ("JPEG","WEBP") else {}
    up.convert("RGB" if save_fmt in ("JPEG","WEBP") else up.mode).save(buf, format=save_fmt, **params)
    data=buf.getvalue()
    storage.put_object(doc["storage_path"], data, doc.get("content_type") or "image/jpeg")
    db.images.update_one({"dataset_id":did},{"$set":{
        "raw_width":up.width,"raw_height":up.height,"width":up.width,"height":up.height,
        "exif_orientation":1,"exif_transposed":False,"coordinate_policy":"upright_normalized",
        "original_sha1":doc.get("sha1"),"original_raw_size":[doc.get("raw_width"),doc.get("raw_height")],
        "file_size_bytes":len(data)}})
    changed.append({"id":did,"from":[doc.get("raw_width"),doc.get("raw_height")],"to":[up.width,up.height]})
print("CHANGED:", json.dumps(changed))
print("SKIPPED (already upright):", skipped)
