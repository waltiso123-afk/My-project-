"""Milestone 1 Step 1 — packaging: curve, qualitative comparisons, report, README, ZIP, upload.
Runs AFTER train_step1.py. Uses REAL saved metrics + model checkpoints."""
import os, io, sys, json, hashlib, zipfile
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
import torch, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import storage, config, pymongo
import warnings; warnings.filterwarnings("ignore")

OUT = Path("/app/reports/milestone1_step1"); (OUT/"qualitative").mkdir(parents=True, exist_ok=True)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
RES=256; MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
cfg=json.loads((OUT/"training_config.json").read_text())
m50=json.loads((OUT/"metrics_50.json").read_text()); m100=json.loads((OUT/"metrics_100.json").read_text())
hold_ids=cfg["holdout_ids"]

def load_model(tag):
    m=SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",num_labels=2,ignore_mismatched_sizes=True)
    m.load_state_dict(torch.load(OUT/"models"/f"{tag}.pt",map_location="cpu")); m.eval(); return m
net50=load_model("baseline_50"); net100=load_model("baseline_100")

def img_mask(did):
    raw,_=storage.get_object(db.images.find_one({"dataset_id":did})["storage_path"])
    im=np.array(Image.open(io.BytesIO(raw)).convert("RGB").resize((RES,RES)))
    md,_=storage.get_object(config.merged_path(did))
    gt=np.array(Image.open(io.BytesIO(md)).convert("L").resize((RES,RES),Image.NEAREST))
    return im,(gt>127).astype(np.uint8)
def pred(net,im):
    x=torch.from_numpy(((im/255.0-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32))
    with torch.no_grad():
        up=F.interpolate(net(pixel_values=x).logits,size=(RES,RES),mode="bilinear",align_corners=False)
    return up.argmax(1)[0].numpy().astype(np.uint8)
def overlay(im,mask,color):
    o=im.copy(); o[mask>0]=(0.45*np.array(color)+0.55*o[mask>0]).astype(np.uint8); return o

# ---- holdout error curve ----
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
xs=[m50["n_train"],m100["n_train"]]
ax[0].plot(xs,[m50["best_holdout"]["dice"],m100["best_holdout"]["dice"]],"o-",color="#22d3ee",label="Dice")
ax[0].plot(xs,[m50["best_holdout"]["iou"],m100["best_holdout"]["iou"]],"s-",color="#a78bfa",label="IoU")
ax[0].set_title("Holdout segmentation vs #labels"); ax[0].set_xlabel("# training images"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(xs,[m50["best_holdout"]["mean_perp_roofline_err_normwidth"],m100["best_holdout"]["mean_perp_roofline_err_normwidth"]],"o-",color="#f59e0b",label="Perp roofline err (norm width)")
ax[1].plot(xs,[m50["best_holdout"]["horizontal_coverage"],m100["best_holdout"]["horizontal_coverage"]],"s-",color="#34d399",label="Coverage")
ax[1].set_title("Holdout roofline proxy vs #labels"); ax[1].set_xlabel("# training images"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT/"holdout_curve.png",dpi=110); plt.close()

# ---- qualitative (representative incl parapet/occlusion if in holdout) ----
para_ids=set(cfg.get("parapet_ids",[]))
picks=hold_ids[:6]
for did in picks:
    im,gt=img_mask(did); p50=pred(net50,im); p100=pred(net100,im)
    panels=[("Original",im),("Ground truth",overlay(im,gt,(0,220,120))),
            ("baseline_50",overlay(im,p50,(64,160,255))),("baseline_100",overlay(im,p100,(255,120,0)))]
    W=RES*4+30; canv=Image.new("RGB",(W,RES+26),(20,24,32)); d=ImageDraw.Draw(canv)
    for j,(lbl,arr) in enumerate(panels):
        canv.paste(Image.fromarray(arr.astype(np.uint8)),(j*(RES+10),26)); d.text((j*(RES+10)+4,6),f"{did} {lbl}",fill=(150,230,255))
    canv.save(OUT/"qualitative"/f"{did}.png")
print("qualitative:",picks)

# ---- report ----
b50=m50["best_holdout"]; b100=m100["best_holdout"]
def trend():
    if b100["dice"]>b50["dice"]+0.02 and b100["mean_perp_roofline_err_normwidth"]<=b50["mean_perp_roofline_err_normwidth"]+1e-6: return "IMPROVING"
    if abs(b100["dice"]-b50["dice"])<=0.02: return "PLATEAUING"
    return "PROBLEMATIC"
tr=trend()
rep=f"""# Milestone 1 — Step 1 Checkpoint

> Internal holdout evaluation — client's hidden 30-photo acceptance set NOT accessed. All numbers are real, measured, reproducible. Workflow: Human + SAM2 (no VLM).

## Dataset
- Current labeled (approved) count: 101 images
- 50-image checkpoint: `baseline_50` (train ids = strict subset of the 100-image set)
- 100-image checkpoint: `baseline_100` — trains on all {cfg['pool_size']} non-holdout labels (a literal 100-image train set is impossible with only 101 total labels while keeping a group-aware holdout; real count reported, not fabricated)
- Fixed internal holdout: {len(hold_ids)} images = {hold_ids}
- Group-aware split: house-group leakage between train and holdout = {cfg['house_group_leak'] or 'NONE'}

## Validation of Previous Issues
### 0169 parapets — PASS
4 planes in upright space; files plane-01-parapet.png..plane-04-parapet.png; meta.parapets=[plane-01..04]; stored image, all masks & merged = 3024x4032 (match); EXIF normalized (=1). Represents the 4 client light-run areas (tall left block, center block, wide overhang above garage, right block). Downstream convention = UPPER parapet boundary. Human labels not overwritten (only the is_parapet flag was set on the existing human planes with user authorization).
### Coordinate space — PASS
coordinate_policy=upright_normalized. 0169 stored=3024x4032, mask=3024x4032, merged=3024x4032, meta=3024x4032, EXIF=1 → alignment PASS. Ingestion pipeline stores future images upright (_to_upright).
### 0001 neighbor roof — PASS
Far-left tiled roof beyond the left wing is the neighbour's; excluded (not masked). Annotation notes confirm neighbours excluded.

## 50-image checkpoint (best holdout)
{json.dumps(b50,indent=2)}

## 100-image checkpoint (best holdout)
{json.dumps(b100,indent=2)}

## Comparison (50 -> {m100['n_train']})
- Dice: {b50['dice']} -> {b100['dice']}
- IoU: {b50['iou']} -> {b100['iou']}
- Mean perpendicular roofline error (norm by width): {b50['mean_perp_roofline_err_normwidth']} -> {b100['mean_perp_roofline_err_normwidth']}
- Horizontal coverage: {b50['horizontal_coverage']} -> {b100['horizontal_coverage']}

## Holdout curve
See `holdout_curve.png`.

## Qualitative results
See `qualitative/` (source, ground-truth, baseline_50, baseline_100 for holdout examples: {picks}).

## Evaluation target (honest note)
The client's marked-polyline metric cannot be reproduced (their marked polylines & hidden 30-set are unavailable). The reported roofline error/coverage are a **mask-derived proxy**: per-column lowest roof-mask edge (eave proxy) compared between the human ground-truth merged mask and the model merged mask, on the internal holdout at {RES}px, normalized by image width. Parapet UPPER-boundary convention is NOT applied in this proxy (documented limitation). Segmentation IoU/Dice are the primary robust metrics.

## Conclusion
Technical status: **{tr}**.
{"50->100 shows higher Dice/IoU and lower roofline error: the model benefits from more labels." if tr=="IMPROVING" else ("Metrics are close between 50 and 100: gains are flattening at this scale." if tr=="PLATEAUING" else "Metrics did not improve (or degraded) from 50->100; investigate label consistency / capacity before scaling.")}

Safe to continue labeling toward 150? {"YES — the curve is still improving, more labels are expected to help; continue to 150 then re-checkpoint." if tr=="IMPROVING" else ("CAUTIOUS — improvement is flattening; 150 may give marginal gains. Recommend continuing but re-evaluate the metric/labeling before larger investment." if tr=="PLATEAUING" else "NOT YET — resolve the regression cause before scaling to 150.")}

_Baseline is intentionally non-optimized and reproducible (SegFormer-B0, CPU). Not a client-facing acceptance result._
"""
(OUT/"milestone1_step1_report.md").write_text(rep)
(OUT/"README.md").write_text("# Milestone 1 — Step 1 Checkpoint package\n\n"
 "Internal reproducible baseline (SegFormer-B0, CPU) at 50 and 100(max non-holdout) labels on the frozen 101-image approved set.\n"
 "- `milestone1_step1_report.md` — full report (dataset, validations, metrics, comparison, conclusion)\n"
 "- `metrics_50.json`, `metrics_100.json` — per-epoch + best holdout metrics (real)\n"
 "- `training_config.json` — model/train config + split ids (reproducibility)\n"
 "- `holdout_curve.png` — 50 vs 100 curves\n"
 "- `qualitative/` — source/GT/baseline_50/baseline_100 on holdout examples\n"
 "- `models/baseline_50.pt`, `models/baseline_100.pt` — best checkpoints\n\n"
 "Internal holdout only — client's hidden 30-photo acceptance set NOT accessed. No fabricated numbers. No VLM.\n")

# ---- ZIP ----
ZIP=Path("/app/reports/milestone1_step1.zip")
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file(): z.write(f, Path("milestone1_step1")/f.relative_to(OUT))
data=ZIP.read_bytes(); sha=hashlib.sha256(data).hexdigest()
storage.put_object("roofline/deliverables/milestone1_step1.zip", data, "application/zip")
burl=next((l.split("=",1)[1].strip() for l in Path("/app/frontend/.env").read_text().splitlines() if l.startswith("REACT_APP_BACKEND_URL=")),None)
db.deliverables.replace_one({"_id":"milestone1-step1"},{"_id":"milestone1-step1","filename":"milestone1_step1.zip",
  "storage_path":"roofline/deliverables/milestone1_step1.zip","size_bytes":len(data),"sha256":sha,
  "download_api":"GET /api/deliverables/milestone1-step1/download","download_url":(burl.rstrip('/')+"/api/deliverables/milestone1-step1/download") if burl else None,
  "trend":tr,"built_at":datetime.now(timezone.utc).isoformat()},upsert=True)
with zipfile.ZipFile(ZIP) as z: print("zip files:",len(z.namelist()),"bad:",z.testzip())
print("TREND:",tr,"| sha256:",sha,"| size:",len(data))
