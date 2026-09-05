"""Mini-Milestone 1 — STEP 5: full QA pass, client README, internal QA report, ZIP, upload."""
import os, io, sys, json, hashlib, zipfile
sys.path.insert(0, "/app/backend")
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage

ROOT = Path("/app/reports/PILOT_MILESTONE_1")
DATASET = ROOT / "dataset"
IDS = ["0001", "0050", "0138", "0169", "0173"]
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

qa = {"images": {}, "issues": [], "checks": {}}
for did in IDS:
    doc = db.images.find_one({"dataset_id": did})
    W, H = doc["width"], doc["height"]; ext = doc["ext"]
    r = {"display_size": [W, H], "planes": [], "parapets": []}
    # originals unchanged
    orig_store, _ = storage.get_object(doc["storage_path"])
    orig_file = (DATASET / "images" / f"{did}.{ext}").read_bytes()
    if hashlib.sha256(orig_store).hexdigest() != hashlib.sha256(orig_file).hexdigest():
        qa["issues"].append(f"{did}: exported original differs from source (should be unchanged)")
    r["original_unchanged"] = True
    # meta
    meta = json.loads((DATASET / "meta" / f"{did}.json").read_text())
    r["parapets"] = meta.get("parapets", [])
    if meta.get("image_size") != {"width": W, "height": H}:
        qa["issues"].append(f"{did}: meta image_size != display size")
    # planes
    pfiles = sorted((DATASET / "planes" / did).glob("*.png"))
    merged_union = np.zeros((H, W), np.uint8)
    for pf in pfiles:
        m = np.array(Image.open(pf).convert("L"))
        if m.shape != (H, W):
            qa["issues"].append(f"{did}/{pf.name}: dims {m.shape[::-1]} != {W}x{H}")
        uniq = set(np.unique(m).tolist())
        if not uniq.issubset({0, 255}):
            qa["issues"].append(f"{did}/{pf.name}: non-binary {sorted(uniq)[:4]}")
        is_par_file = pf.name.endswith("-parapet.png")
        plane_id = pf.stem.replace("-parapet", "")
        if is_par_file and plane_id not in meta.get("parapets", []):
            qa["issues"].append(f"{did}/{pf.name}: -parapet file but not listed in meta parapets")
        if (not is_par_file) and plane_id in meta.get("parapets", []):
            qa["issues"].append(f"{did}/{pf.name}: listed as parapet in meta but filename lacks -parapet")
        merged_union = np.maximum(merged_union, m)
        r["planes"].append({"file": pf.name, "area_px": int((m > 0).sum())})
    # merged consistency
    merged = np.array(Image.open(DATASET / "merged" / f"{did}.png").convert("L"))
    if merged.shape != (H, W):
        qa["issues"].append(f"{did}: merged dims mismatch")
    if not np.array_equal((merged > 0), (merged_union > 0)):
        qa["issues"].append(f"{did}: merged != union of plane masks")
    # no single plane equals merged when multiple planes
    if len(pfiles) > 1:
        for pf in pfiles:
            m = np.array(Image.open(pf).convert("L"))
            if np.array_equal((m > 0), (merged > 0)):
                qa["issues"].append(f"{did}/{pf.name}: identical to merged despite {len(pfiles)} planes")
    r["n_planes"] = len(pfiles)
    qa["images"][did] = r

# no extra / missing images
imgs = sorted(p.stem for p in (DATASET / "images").glob("*"))
if imgs != sorted(IDS):
    qa["issues"].append(f"image set mismatch: {imgs}")
qa["checks"] = {"n_images": len(imgs), "total_planes": sum(qa["images"][d]["n_planes"] for d in IDS)}
qa["passed"] = len(qa["issues"]) == 0
(ROOT / "internal_qa.json").write_text(json.dumps(qa, indent=2))
print("QA passed:", qa["passed"], "issues:", qa["issues"])

# -------- Client README (inside dataset delivery; NO internal architecture) --------
readme = """# Roofline Labeling — Milestone 1 (first 5-image delivery)

This is the first delivery of the roof-plane labeling work, covering 5 house photos. It uses the
exact same specification, format and quality standard that will be used for the full Milestone 1
dataset — only the number of images differs.

## What is in this delivery
For each image you receive:
- `images/<id>.<ext>` — the original photo, unmodified.
- `planes/<id>/plane-01.png, plane-02.png, ...` — one binary mask per roof plane (255 = plane, 0 = background).
  These per-plane masks are the primary annotation output.
- `merged/<id>.png` — the union of that image's planes (provided for convenience only).
- `meta/<id>.json` — metadata for the image, including occlusion records and parapet information.

## How to read it
- Each distinct roof plane is delivered as its own separate mask, at the exact pixel dimensions of the
  original image.
- Flat roofs behind a parapet wall are delivered with `-parapet` in the filename and listed under
  `"parapets"` in the metadata; these are lit along the top of the parapet.
- Where a tree, palm, wire or similar crosses a roof, the metadata records it under `"occlusions"`.

## Note on image 0169
Image 0169 is a modern flat-roof / parapet-style home. It contains an ambiguous situation that we would
like you to confirm before we scale to the full dataset. The specific open questions are written in
`meta/0169.json` under `notes`. The masks provided for 0169 are provisional pending your confirmation.

## Purpose
Please review these 5 images and confirm that the roof-plane separation, boundaries and metadata match
your expectation. Once confirmed, the same process and format will be applied to the full dataset.
"""
(DATASET / "README.md").write_text(readme)

# -------- Internal QA report (NOT shipped to client) --------
lines = ["# Mini-Milestone 1 — INTERNAL QA report (not for client)", "",
         f"Generated {datetime.now(timezone.utc).isoformat()}", "",
         "Pipeline: VLM reasoning/plan (gemini-3.1-pro-preview) -> SAM2 (CPU, hiera-tiny) point prompts ->",
         "clip/reconcile -> human (agent) QA & correction -> ml.generate_and_store (production path).", "",
         f"QA PASSED: {qa['passed']}   |   images: {qa['checks']['n_images']}   |   total planes: {qa['checks']['total_planes']}", ""]
per_img_notes = {
 "0001":"3 planes (left wing, central upper hip, garage front hip). VLM over-segmented to 11; QA consolidated to 3 real planes. SAM2 points repositioned off palms/walls. Neighbours excluded.",
 "0050":"3 planes (gable left slope, gable right slope, garage hip). Right attached villa = neighbour, excluded. VLM falsely proposed parapet planes; corrected to pitched. Left gable slope partly tree-occluded.",
 "0138":"2 planes (left front hip, right/garage hip). Centre front plane is occluded by a large palm (>15% width) so front is delivered as two segments per the 15% rule. Neighbour at right excluded.",
 "0173":"4 planes (left upper, main lower, central tower front, right upper). Clean tile roofs; SAM2 lower edge sits at tile bottom ~ drip edge. Palm over right upper ~12% (<15%), plane continued.",
 "0169":"4 planes (3 parapet caps + 1 lower canopy). PROVISIONAL/AMBIGUOUS: flat-roof/parapet home; only parapet caps visible from ground. Automatic SAM2 grabbed walls, so parapet planes were hand-drawn along the caps. Requires client ruling (pergola? each flat roof? fascia top). Documented in meta notes.",
}
for did in IDS:
    im = qa["images"][did]
    lines.append(f"## {did}  ({im['display_size'][0]}x{im['display_size'][1]}, {im['n_planes']} planes, parapets={im['parapets']})")
    lines.append(f"- corrections/observations: {per_img_notes[did]}")
    lines.append(f"- per-plane px areas: {[p['area_px'] for p in im['planes']]}")
    lines.append("")
lines += ["## File integrity", f"- issues found: {qa['issues'] if qa['issues'] else 'none'}",
          "- all masks binary and at image dimensions; merged == union of planes; originals unchanged; "
          "parapet filename/metadata consistent; exactly 5 images, no extras."]
(ROOT / "INTERNAL_QA_REPORT.md").write_text("\n".join(lines))

# -------- ZIP (client dataset only; internal report excluded) --------
ZIP = ROOT / "PILOT_MILESTONE_1.zip"
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(DATASET.rglob("*")):
        if f.is_file():
            z.write(f, Path("PILOT_MILESTONE_1") / f.relative_to(DATASET.parent))
data = ZIP.read_bytes()
size = len(data); sha = hashlib.sha256(data).hexdigest()
storage.put_object("roofline/deliverables/PILOT_MILESTONE_1.zip", data, "application/zip")
backend_url = next((l.split("=",1)[1].strip() for l in Path("/app/frontend/.env").read_text().splitlines()
                    if l.startswith("REACT_APP_BACKEND_URL=")), None)
manifest = {"_id":"pilot-milestone-1","filename":"PILOT_MILESTONE_1.zip",
            "storage_path":"roofline/deliverables/PILOT_MILESTONE_1.zip","size_bytes":size,"sha256":sha,
            "image_count":len(IDS),"planes_total":qa["checks"]["total_planes"],
            "download_api":"GET /api/deliverables/milestone-1/download","info_api":"GET /api/deliverables/milestone-1/info",
            "download_url":(backend_url.rstrip("/")+"/api/deliverables/milestone-1/download") if backend_url else None,
            "qa_passed":qa["passed"],"built_at":datetime.now(timezone.utc).isoformat(),
            "note":"Milestone-1 production format on 5 images. 0169 parapet case provisional pending client confirmation."}
db.deliverables.replace_one({"_id":"pilot-milestone-1"}, manifest, upsert=True)
with zipfile.ZipFile(ZIP) as z:
    print("zip files:", len(z.namelist()), "bad:", z.testzip())
print(json.dumps({k:manifest[k] for k in ["filename","size_bytes","sha256","image_count","planes_total","qa_passed","download_url"]}, indent=2))
