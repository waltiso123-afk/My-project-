"""Rebuild MINI-MILESTONE zip from the USER's current manual annotations (object storage),
using each annotation's authoritative generated.planes list (no stale/globbed files).
Structure: dataset/{images,planes,merged,meta}. Light QA, then upload + download link."""
import os, io, sys, json, hashlib, zipfile, shutil
sys.path.insert(0, "/app/backend")
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage, config

ROOT = Path("/app/reports/MINI_MILESTONE_FINAL")
DS = ROOT / "dataset"
if DS.exists(): shutil.rmtree(DS)
for s in ["images","planes","merged","meta"]: (DS/s).mkdir(parents=True, exist_ok=True)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
IDS = ["0001","0050","0138","0169","0173"]

qa_issues=[]; index={}
for did in IDS:
    doc=db.images.find_one({"dataset_id":did}); ann=db.annotations.find_one({"dataset_id":did})
    if not ann or ann.get("status")!="approved": qa_issues.append(f"{did}: not approved"); continue
    W,H=doc["width"],doc["height"]; ext=doc["ext"]
    # original (unchanged)
    raw,_=storage.get_object(doc["storage_path"]); (DS/"images"/f"{did}.{ext}").write_bytes(raw)
    # planes — strictly from annotation.generated.planes (current, authoritative)
    (DS/"planes"/did).mkdir(exist_ok=True); union=np.zeros((H,W),np.uint8); pnames=[]
    for p in ann["generated"]["planes"]:
        fn=p.split("/")[-1]; data,_=storage.get_object(p); (DS/"planes"/did/fn).write_bytes(data); pnames.append(fn)
        a=np.array(Image.open(io.BytesIO(data)).convert("L"))
        if a.shape!=(H,W): qa_issues.append(f"{did}/{fn}: dims {a.shape[::-1]}!={W}x{H}")
        if not set(np.unique(a).tolist()).issubset({0,255}): qa_issues.append(f"{did}/{fn}: non-binary")
        union=np.maximum(union,a)
    # merged + meta
    md,_=storage.get_object(config.merged_path(did)); (DS/"merged"/f"{did}.png").write_bytes(md)
    if not np.array_equal(np.array(Image.open(io.BytesIO(md)).convert("L"))>0, union>0):
        qa_issues.append(f"{did}: merged != union of current planes")
    mj,_=storage.get_object(config.meta_path(did)); (DS/"meta"/f"{did}.json").write_bytes(mj)
    meta=json.loads(mj)
    index[did]={"ext":ext,"size":[W,H],"n_planes":len(pnames),"planes":pnames,"parapets":meta.get("parapets",[])}
    print(f"{did}: {len(pnames)} planes {pnames} parapets={meta.get('parapets',[])}")

# no extras / exactly 5
imgset=sorted(p.stem for p in (DS/"images").glob("*"))
if imgset!=sorted(IDS): qa_issues.append(f"image set {imgset}")
print("QA issues:", qa_issues if qa_issues else "none")

# ZIP with top folder 'dataset'
ZIP=ROOT/"MINI_MILESTONE_1.zip"
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
    for f in sorted(DS.rglob("*")):
        if f.is_file(): z.write(f, f.relative_to(ROOT))  # arcname: dataset/...
data=ZIP.read_bytes(); size=len(data); sha=hashlib.sha256(data).hexdigest()
storage.put_object("roofline/deliverables/PILOT_MILESTONE_1.zip", data, "application/zip")
burl=next((l.split("=",1)[1].strip() for l in Path("/app/frontend/.env").read_text().splitlines() if l.startswith("REACT_APP_BACKEND_URL=")),None)
manifest={"_id":"pilot-milestone-1","filename":"MINI_MILESTONE_1.zip","storage_path":"roofline/deliverables/PILOT_MILESTONE_1.zip",
 "size_bytes":size,"sha256":sha,"image_count":len(IDS),"planes_total":sum(index[d]["n_planes"] for d in index),
 "structure":"dataset/{images,planes,merged,meta}","source":"user manual annotations (Studio, approved)",
 "download_api":"GET /api/deliverables/milestone-1/download","info_api":"GET /api/deliverables/milestone-1/info",
 "download_url":(burl.rstrip("/")+"/api/deliverables/milestone-1/download") if burl else None,
 "qa_issues":qa_issues,"built_at":datetime.now(timezone.utc).isoformat()}
db.deliverables.replace_one({"_id":"pilot-milestone-1"},manifest,upsert=True)
with zipfile.ZipFile(ZIP) as z: print("zip files:",len(z.namelist()),"bad:",z.testzip()); print("sample:",z.namelist()[:6])
print(json.dumps({k:manifest[k] for k in ["filename","size_bytes","sha256","image_count","planes_total","download_url"]},indent=2))
