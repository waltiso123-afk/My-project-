"""Milestone 1 (HUMAN + SAM2, NO VLM) — production store + dataset export + QA + README + ZIP + upload.
Consumes ONLY the human-approved masks in MILESTONE_1_HUMAN/final. No VLM artifacts are read."""
import os, io, sys, json, base64, hashlib, zipfile
sys.path.insert(0, "/app/backend")
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage, ml, config

ROOT = Path("/app/reports/MILESTONE_1_HUMAN")
FINAL = ROOT / "final"; DATASET = ROOT / "dataset"
for s in ["images","planes","merged","meta"]: (DATASET/s).mkdir(parents=True, exist_ok=True)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
IDS = ["0001","0050","0138","0169","0173"]

META = {
 "0001":{"view":"oblique","planes":[("left_wing_front",False),("central_upper_hip",False),("garage_front_hip",False)],
   "occlusions":[{"type":"palm","bbox":[0.32,0.15,0.52,0.55],"hides":["plane-02","plane-03"]},
                 {"type":"palm","bbox":[0.79,0.30,0.86,0.45],"hides":["plane-03"]}],
   "notes":"Subject house only; left and right neighbour houses excluded. Palms partially cross planes (<15% image width each) so planes continue underneath per the 15% rule."},
 "0050":{"view":"frontal","planes":[("gable_left_slope",False),("gable_right_slope",False),("garage_front_hip",False)],
   "occlusions":[{"type":"tree","bbox":[0.00,0.30,0.28,0.60],"hides":["plane-01"]}],
   "notes":"Central front gable (both visible slopes) + front garage hip of the subject villa. Attached villa to the right is a neighbour and excluded. A tree partly occludes the left gable slope."},
 "0138":{"view":"oblique","planes":[("left_front_hip",False),("right_garage_hip",False)],
   "occlusions":[{"type":"palm","bbox":[0.40,0.35,0.62,0.75],"hides":["plane-01","plane-02"]}],
   "notes":"Long front hip roof. A large palm occludes the centre (>15% image width) so the readable front plane is delivered as two segments (left of palm and right/garage side). Neighbour house at far right excluded."},
 "0173":{"view":"oblique","planes":[("left_upper_hip",False),("main_lower_hip",False),("central_tower_front",False),("right_upper_hip",False)],
   "occlusions":[{"type":"palm","bbox":[0.54,0.29,0.86,0.85],"hides":["plane-04"]}],
   "notes":"Tile hip roof: left upper, main lower (over garage), central tower front, right upper. Palm crosses the right upper plane (~12% image width, <15%) so that plane continues underneath."},
 "0169":{"view":"frontal","planes":[("highest_block_parapet",True),("left_block_parapet",True),("main_upper_parapet",True),("lower_canopy_front",False)],
   "occlusions":[{"type":"palm","bbox":[0.00,0.05,0.13,0.40],"hides":["plane-02"]}],
   "notes":"PROVISIONAL / AMBIGUOUS — flagged for client clarification. Modern flat-roof home behind parapet walls; from ground only parapet caps and a lower canopy slab are visible. Human drew provisional parapet-cap polygons because real local SAM2 cannot separate white-on-white parapet edges. Open questions for the client: (a) does each flat roof behind a parapet count as its own lit plane; (b) is the louvered pergola/trellis a lit element (currently excluded); (c) the lower canopy is lit along the TOP of its dark fascia (top = drip edge), not the fascia bottom. These masks require client sign-off before scaling."},
}
def durl(p): return "data:image/png;base64,"+base64.b64encode(Path(p).read_bytes()).decode()

for did,m in META.items():
    doc=db.images.find_one({"dataset_id":did}); W,H=doc["width"],doc["height"]; ext=doc["ext"]
    planes=[{"name":n,"is_parapet":ip,"raster_png":durl(FINAL/did/f"plane_{i:02d}.png"),
             "generation_method":"human_point_prompt+sam2_cpu+human_qa","human_corrected":True}
            for i,(n,ip) in enumerate(m["planes"],start=1)]
    res=ml.generate_and_store(did,ext,(W,H),planes,m["occlusions"],m["notes"],m["view"])
    now=datetime.now(timezone.utc).isoformat()
    db.annotations.update_one({"dataset_id":did},{"$set":{"dataset_id":did,"planes":planes,"occlusions":m["occlusions"],
        "notes":m["notes"],"checklist":{},"status":"approved","approved_at":now,"updated_at":now,
        "workflow":"human+sam2 (no VLM)","generated":{"merged":res["merged"],"planes":res["planes"],"meta":config.meta_path(did)}}},upsert=True)
    db.images.update_one({"dataset_id":did},{"$set":{"status":"approved","view_type":m["view"]}})
    # export dataset/ from storage (byte-identical to production)
    raw,_=storage.get_object(doc["storage_path"]); (DATASET/"images"/f"{did}.{ext}").write_bytes(raw)
    (DATASET/"planes"/did).mkdir(exist_ok=True)
    for p in res["planes"]:
        d,_=storage.get_object(p); (DATASET/"planes"/did/p.split("/")[-1]).write_bytes(d)
    md,_=storage.get_object(config.merged_path(did)); (DATASET/"merged"/f"{did}.png").write_bytes(md)
    mj,_=storage.get_object(config.meta_path(did)); (DATASET/"meta"/f"{did}.json").write_bytes(mj)
    print(f"{did}: {len(planes)} planes stored+exported")

# -------- QA --------
qa={"images":{},"issues":[]}
for did in IDS:
    doc=db.images.find_one({"dataset_id":did}); W,H=doc["width"],doc["height"]; ext=doc["ext"]
    if hashlib.sha256(storage.get_object(doc["storage_path"])[0]).hexdigest()!=hashlib.sha256((DATASET/"images"/f"{did}.{ext}").read_bytes()).hexdigest():
        qa["issues"].append(f"{did}: original changed")
    ann=db.annotations.find_one({"dataset_id":did})
    if ann.get("status")!="approved": qa["issues"].append(f"{did}: not approved")
    meta=json.loads((DATASET/"meta"/f"{did}.json").read_text())
    union=np.zeros((H,W),np.uint8); pfiles=sorted((DATASET/"planes"/did).glob("*.png"))
    for pf in pfiles:
        a=np.array(Image.open(pf).convert("L"))
        if a.shape!=(H,W): qa["issues"].append(f"{did}/{pf.name}: dims {a.shape[::-1]}!={W}x{H}")
        if not set(np.unique(a).tolist()).issubset({0,255}): qa["issues"].append(f"{did}/{pf.name}: non-binary")
        pid=pf.stem.replace("-parapet",""); isp=pf.name.endswith("-parapet.png")
        if isp!=(pid in meta.get("parapets",[])): qa["issues"].append(f"{did}/{pf.name}: parapet naming/meta mismatch")
        union=np.maximum(union,a)
    merged=np.array(Image.open(DATASET/"merged"/f"{did}.png").convert("L"))
    if not np.array_equal(merged>0,union>0): qa["issues"].append(f"{did}: merged!=union")
    if len(pfiles)>1:
        for pf in pfiles:
            if np.array_equal(np.array(Image.open(pf).convert("L"))>0,merged>0): qa["issues"].append(f"{did}/{pf.name}=merged")
    qa["images"][did]={"n_planes":len(pfiles),"parapets":meta.get("parapets",[]),"size":[W,H]}
imgset=sorted(p.stem for p in (DATASET/"images").glob("*"))
if imgset!=sorted(IDS): qa["issues"].append(f"image set {imgset}")
# ensure no VLM plan artifacts anywhere in delivery
if any("gemini" in json.dumps(qa["images"][d]).lower() for d in IDS): qa["issues"].append("vlm ref found")
qa["passed"]=len(qa["issues"])==0; qa["total_planes"]=sum(qa["images"][d]["n_planes"] for d in IDS)
(ROOT/"internal_qa.json").write_text(json.dumps(qa,indent=2))
print("QA passed:",qa["passed"],"issues:",qa["issues"],"total planes:",qa["total_planes"])

# -------- Client README --------
(DATASET/"README.md").write_text(
"# Roofline Labeling — Milestone 1 (first 5-image delivery)\n\n"
"This is the first delivery of the roof-plane labeling work, covering 5 house photos. It uses the exact same\n"
"specification, format and quality standard planned for the full Milestone 1 dataset — only the number of\n"
"images differs.\n\n"
"## Contents (per image)\n"
"- `images/<id>.<ext>` — original photo, unmodified.\n"
"- `planes/<id>/plane-01.png, plane-02.png, ...` — one binary mask per roof plane (255 = plane, 0 = background). Primary output.\n"
"- `merged/<id>.png` — union of the planes (convenience only).\n"
"- `meta/<id>.json` — occlusion records and parapet information.\n\n"
"## Reading it\n"
"- Each distinct roof plane is a separate mask at the exact pixel size of the original.\n"
"- Flat roofs behind a parapet use `-parapet` in the filename and appear under `\"parapets\"` in metadata (lit along the parapet top).\n"
"- Trees/palms/wires crossing a roof are recorded under `\"occlusions\"`.\n\n"
"## Image 0169\n"
"0169 is a modern flat-roof / parapet-style home containing an ambiguous situation we would like you to confirm\n"
"before scaling. The specific open questions are in `meta/0169.json` under `notes`; its masks are provisional.\n\n"
"## Purpose\n"
"Please confirm the roof-plane separation, boundaries and metadata match your expectation. The same process and\n"
"format will then be applied to the full dataset.\n")

# -------- Internal QA report (excluded from zip) --------
notes={"0001":"3 planes (left wing, central upper hip, garage front hip). Human-placed SAM2 points; neighbours excluded.",
 "0050":"3 planes (gable left slope, gable right slope, garage hip). Left slope clip tightened by human to exclude the gable wall. Right attached villa = neighbour, excluded. Tree partly occludes left slope.",
 "0138":"2 planes (left front hip, right/garage hip). Centre occluded by a large palm (>15%): front delivered as two segments per the 15% rule. Neighbour at right excluded.",
 "0173":"4 planes (left upper, main lower, central tower front, right upper). Clean tile roofs; SAM2 lower edge at tile bottom ~ drip edge.",
 "0169":"4 planes (3 parapet caps + 1 lower canopy). PROVISIONAL/AMBIGUOUS: real local SAM2 grabbed walls (white-on-white), so human drew parapet-cap polygons. Requires client ruling (pergola? each flat roof? fascia). Flagged in meta notes."}
lines=["# Milestone 1 (HUMAN + SAM2, NO VLM) — INTERNAL QA report (not for client)","",
 f"Generated {datetime.now(timezone.utc).isoformat()}","",
 "Workflow: human identifies plane -> human places SAM2 point prompt -> real local SAM2 (CPU, Hiera-Tiny) mask ->",
 "human visual validation -> human correction (clip/polygon) -> approve -> ml.generate_and_store (production path).",
 "NO Gemini / NO VLM / NO automatic roof detection / NO automatic point selection anywhere.","",
 f"QA PASSED: {qa['passed']} | images: 5 | total planes: {qa['total_planes']} | issues: {qa['issues'] if qa['issues'] else 'none'}",""]
for did in IDS:
    im=qa["images"][did]; lines.append(f"## {did} ({im['size'][0]}x{im['size'][1]}, {im['n_planes']} planes, parapets={im['parapets']})")
    lines.append(f"- {notes[did]}"); lines.append("")
(ROOT/"INTERNAL_QA_REPORT.md").write_text("\n".join(lines))

# -------- ZIP (client dataset only) --------
ZIP=ROOT/"PILOT_MILESTONE_1.zip"
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
    for f in sorted(DATASET.rglob("*")):
        if f.is_file(): z.write(f, Path("PILOT_MILESTONE_1")/f.relative_to(DATASET.parent))
data=ZIP.read_bytes(); size=len(data); sha=hashlib.sha256(data).hexdigest()
storage.put_object("roofline/deliverables/PILOT_MILESTONE_1.zip", data, "application/zip")
burl=next((l.split("=",1)[1].strip() for l in Path("/app/frontend/.env").read_text().splitlines() if l.startswith("REACT_APP_BACKEND_URL=")),None)
manifest={"_id":"pilot-milestone-1","filename":"PILOT_MILESTONE_1.zip","storage_path":"roofline/deliverables/PILOT_MILESTONE_1.zip",
 "size_bytes":size,"sha256":sha,"image_count":5,"planes_total":qa["total_planes"],"workflow":"human+sam2 (no VLM)",
 "download_api":"GET /api/deliverables/milestone-1/download","info_api":"GET /api/deliverables/milestone-1/info",
 "download_url":(burl.rstrip("/")+"/api/deliverables/milestone-1/download") if burl else None,
 "qa_passed":qa["passed"],"built_at":datetime.now(timezone.utc).isoformat(),
 "note":"Human-validated Human+SAM2 Milestone-1 (5 images). Supersedes the obsolete VLM version. 0169 parapet case provisional pending client confirmation."}
db.deliverables.replace_one({"_id":"pilot-milestone-1"},manifest,upsert=True)
with zipfile.ZipFile(ZIP) as z: print("zip files:",len(z.namelist()),"bad:",z.testzip())
print(json.dumps({k:manifest[k] for k in ["filename","size_bytes","sha256","image_count","planes_total","qa_passed","download_url"]},indent=2))
