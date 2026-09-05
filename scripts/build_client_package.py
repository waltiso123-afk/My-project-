"""Build the 5-image CLIENT REVIEW package from EXISTING pilot artifacts.
STRICTLY packaging only: copies existing originals + existing SAM2 pilot masks, renders
overlays/before-after/merged FROM those masks, assembles metadata from DB + measured metrics.
NO annotation is re-run. Nothing in the DB or pilot masks is modified.
"""
import os, io, sys, json, shutil, hashlib, zipfile
sys.path.insert(0, "/app/backend")
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage

PILOT = Path("/app/reports/pilot")
MASKS_SRC = PILOT / "masks"
ROOT = Path("/app/reports/client_review/PILOT_5_CLIENT_REVIEW")
for sub in ["originals", "planes", "merged", "overlays", "before_after", "special_cases", "metadata"]:
    if (ROOT / sub).exists():
        shutil.rmtree(ROOT / sub)
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

c = pymongo.MongoClient(os.environ["MONGO_URL"]); db = c[os.environ["DB_NAME"]]
metrics = json.loads((PILOT / "sam2_metrics.json").read_text())
fix = json.loads((PILOT / "sam2_fix_metrics.json").read_text())

IDS = ["0001", "0169", "0050", "0138", "0173"]
COLORS = [(255,64,64),(64,160,255),(64,220,120),(255,205,40),(200,90,255),(255,130,0)]

# per-plane human-readable observations, backed ONLY by the pilot report / measured data
OBS = {
    "0001": {"planes": 3, "note": "Toits tuiles: pan droit (garage) bien capté par SAM2; tourelle et aile "
             "gauche ont nécessité un repositionnement du point (1re passe tombée sur mur/végétation). "
             "Maison voisine à gauche exclue.", "confirm": "Oui"},
    "0169": {"planes": 4, "note": "CAS SPÉCIAL: maison moderne à toits plats derrière parapets. Depuis le sol, "
             "seules les arases de parapet, une pergola à lames et une dalle en porte-à-faux sont visibles. "
             "SAM2 capte murs/fenêtre/pergola (peu pertinent). Éléments de type parapet = éclairage au bord "
             "SUPÉRIEUR (spec). AMBIGUÏTÉS non tranchées: la pergola à lames et la dalle comptent-elles comme "
             "plans à éclairer? À confirmer par le client.", "confirm": "Oui (obligatoire)"},
    "0050": {"planes": 3, "note": "Pan droit du pignon + croupe garage bien captés. Pan gauche du pignon "
             "OCCULTÉ par un arbre fleuri (le point tombe sur la végétation). Unités latérales = villas "
             "voisines, exclues.", "confirm": "Oui"},
    "0138": {"planes": 3, "note": "Croupe gauche (plane-01) et croupe garage droite (plane-03) captées "
             "proprement et bien séparées. plane-02 (centre) est un RATÉ SAM2: le point est tombé sur "
             "l'entrée/le palmier (occlusion) et non sur le pan central -> nécessite repositionnement/"
             "correction manuelle. Voisin à droite exclu.", "confirm": "Oui"},
    "0173": {"planes": 4, "note": "Tuiles: tourelle, croupe gauche, croupe basse (garage) et toit d'entrée "
             "droite captés proprement. Pleine résolution 5712×4284, EXIF géré.", "confirm": "Oui"},
}
FIX_PLANES = {"0001": {"tower_hip", "left_wing"}, "0050": {"left_gable_slope"}, "0169": {"mid_block_parapet"}}

def load_display(did):
    doc = db.images.find_one({"dataset_id": did})
    raw, _ = storage.get_object(doc["storage_path"])
    disp = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    return doc, raw, disp

def mask_files_for(did):
    fs = sorted(MASKS_SRC.glob(f"{did}_*.png"), key=lambda p: p.name)
    return fs  # already NN-ordered

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

summary_rows = []
qa = {"issues": []}
package_meta = {}

for did in IDS:
    doc, raw_bytes, disp = load_display(did)
    W, H = disp.size
    dispn = np.array(disp)

    # 1) ORIGINAL — untouched raw bytes, real filename
    (ROOT / "originals" / did).mkdir(parents=True, exist_ok=True)
    (ROOT / "originals" / did / doc["original_filename"]).write_bytes(raw_bytes)

    # 2) PER-PLANE MASKS — copy existing pilot masks exactly, renumbered plane-0N
    (ROOT / "planes" / did).mkdir(parents=True, exist_ok=True)
    mfiles = mask_files_for(did)
    plane_map = []
    merged = np.zeros((H, W), np.uint8)
    plane_arrays = []
    for i, mf in enumerate(mfiles, start=1):
        name = mf.stem.split("_", 2)[2]  # e.g. tower_hip
        m = np.array(Image.open(mf).convert("L"))
        # QA: dimensions must match display image
        if m.shape[:2] != (H, W):
            qa["issues"].append(f"{did} {mf.name}: mask {m.shape[1]}x{m.shape[0]} != display {W}x{H}")
        # QA: binary
        uniq = set(np.unique(m).tolist())
        if not uniq.issubset({0, 255}):
            qa["issues"].append(f"{did} {mf.name}: non-binary values {sorted(uniq)[:5]}")
        out = ROOT / "planes" / did / f"plane-{i:02d}.png"
        Image.fromarray(m).save(out)
        plane_arrays.append(m)
        merged = np.maximum(merged, m)
        # provenance from measured metrics
        pm = next((p for p in metrics[did]["planes"] if p["plane"] == name), None)
        repositioned = name in FIX_PLANES.get(did, set())
        fx = None
        if repositioned:
            fx = next((p for p in fix.get(did, []) if p["plane"] == name), None)
        plane_map.append({
            "plane_id": f"plane-{i:02d}", "internal_name": name, "file": f"plane-{i:02d}.png",
            "generation_method": "sam2_cpu_point_prompt",
            "sam2_point_prompt_display_xy": pm["point_disp"] if pm else None,
            "sam2_score": pm["score"] if pm else None,
            "sam2_area_fraction": pm["area_frac"] if pm else None,
            "sam2_predict_ms": pm["predict_ms"] if pm else None,
            "point_repositioned_during_pilot": repositioned,
            "repositioned_point_display_xy": fx["point"] if fx else None,
            "manual_human_correction_applied": False,
            "human_approved": False,
            "is_parapet_confirmed": False,
            "parapet_status": ("ambiguous_needs_client_ruling" if did == "0169" else "not_applicable"),
        })

    # 3) MERGED — union convenience output
    Image.fromarray(merged).save(ROOT / "merged" / f"{did}.png")
    # QA: with multiple planes, no single plane must equal merged
    if len(plane_arrays) > 1:
        for i, pa in enumerate(plane_arrays, start=1):
            if np.array_equal(pa, merged):
                qa["issues"].append(f"{did} plane-{i:02d} identical to merged despite {len(plane_arrays)} planes")

    # 4) OVERLAY — colored masks + boundaries, transparent enough to see roof
    ov = disp.convert("RGBA")
    tint = np.zeros((H, W, 4), np.uint8)
    for idx, m in enumerate(plane_arrays):
        col = COLORS[idx % len(COLORS)]
        tint[m > 0] = (*col, 90)  # ~35% fill
    ov = Image.alpha_composite(ov, Image.fromarray(tint))
    ovn = np.array(ov)
    for idx, m in enumerate(plane_arrays):
        col = COLORS[idx % len(COLORS)]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ovn, cnts, -1, (*col, 255), max(2, W // 500))
    ov = Image.fromarray(ovn)
    # legend
    d = ImageDraw.Draw(ov); f = font(max(16, W // 70)); y = 10
    for idx, pm in enumerate(plane_map):
        col = COLORS[idx % len(COLORS)]
        d.rectangle([10, y, 10 + f.size, y + f.size], fill=(*col, 255))
        d.text((14 + f.size, y), f"{pm['plane_id']} ({pm['internal_name']})", fill=(255,255,255), font=f,
               stroke_width=2, stroke_fill=(0,0,0))
        y += f.size + 8
    # downscale visualisation only (masks + originals stay full-res); primary outputs untouched
    VIZ_MAX = 2200
    ov_rgb = ov.convert("RGB")
    disp_v = disp.copy()
    if max(W, H) > VIZ_MAX:
        ov_rgb.thumbnail((VIZ_MAX, VIZ_MAX)); disp_v.thumbnail((VIZ_MAX, VIZ_MAX))
    ov_rgb.save(ROOT / "overlays" / f"{did}_overlay.png")

    # 5) BEFORE / AFTER (from downscaled viz)
    vw, vh = disp_v.size
    bar = max(40, vh // 20)
    canvas = Image.new("RGB", (vw * 2 + 30, vh + bar), (245, 245, 245))
    canvas.paste(disp_v, (0, bar)); canvas.paste(ov_rgb, (vw + 30, bar))
    d = ImageDraw.Draw(canvas); fb = font(max(20, bar // 2))
    d.text((10, bar // 4), "Original", fill=(20, 20, 20), font=fb)
    d.text((vw + 40, bar // 4), "Annoté / Overlay", fill=(20, 20, 20), font=fb)
    canvas.save(ROOT / "before_after" / f"{did}_before_after.png")

    # 7) METADATA — assembled from DB + measured metrics only (no fabrication)
    md = {
        "image_id": did,
        "original_filename": doc["original_filename"],
        "batch": doc["batch"], "split": doc["split"], "house_group_id": doc.get("house_group_id"),
        "original_dimensions": {"width": doc["raw_width"], "height": doc["raw_height"]},
        "display_dimensions": {"width": W, "height": H},
        "exif_orientation": doc.get("exif_orientation"),
        "orientation": doc.get("orientation"),
        "coordinate_space": "exif_applied_display (masks generated in the pixels the annotator saw)",
        "roof_plane_count": len(plane_map),
        "planes": plane_map,
        "occlusions_observed": ("Pan gauche du pignon occulté par un arbre fleuri (bbox non mesurée -> non "
                                "inventée)." if did == "0050" else
                                ("Occlusion partielle par arbre/palmier." if did == "0138" else "")),
        "parapets_confirmed": [],
        "parapet_note": ("Éléments de type parapet présents mais NON confirmés — à statuer par le client. "
                         "Aucun suffixe -parapet appliqué pour ne pas décider en silence." if did == "0169" else ""),
        "notes_and_ambiguities": OBS[did]["note"],
        "sam2": {"backend": "sam2_cpu", "checkpoint": "sam2.1_hiera_tiny.pt",
                 "encode_ms_once_per_image": metrics[did]["encode_ms"],
                 "embedding_cache_used": True, "n_point_prompts": metrics[did]["n_points"]},
        "review_status": "pilot_sam2_proposal_PENDING_human_review_and_client_confirmation",
        "human_approved": False,
    }
    (ROOT / "metadata" / f"{did}.json").write_text(json.dumps(md, indent=2, ensure_ascii=False))
    package_meta[did] = md
    summary_rows.append((did, doc["original_filename"], len(plane_map), OBS[did]["note"], OBS[did]["confirm"]))

# 6) SPECIAL CASES
def add_special(did, why):
    sd = ROOT / "special_cases" / did; sd.mkdir(parents=True, exist_ok=True)
    doc = db.images.find_one({"dataset_id": did})
    shutil.copy(ROOT / "originals" / did / doc["original_filename"], sd / doc["original_filename"])
    shutil.copy(ROOT / "overlays" / f"{did}_overlay.png", sd / f"{did}_overlay.png")
    for p in (ROOT / "planes" / did).glob("*.png"):
        shutil.copy(p, sd / p.name)
    # include the repositioning-evidence overlays that already exist (pilot artifacts)
    for fj in PILOT.glob(f"overlays/{did}_FIX_*.jpg"):
        shutil.copy(fj, sd / fj.name)
    (sd / "WHY_SPECIAL.txt").write_text(why, encoding="utf-8")

add_special("0169", "Toits plats modernes derrière parapets. Depuis le sol, quasi aucun pan de toit visible "
            "(arases de parapet, pergola à lames, dalle en porte-à-faux). SAM2 seul insuffisant -> polygone "
            "manuel le long du HAUT du parapet (spec). AMBIGUÏTÉS À CONFIRMER PAR LE CLIENT: la pergola à "
            "lames et la dalle comptent-elles comme plans à éclairer? Aucune règle nouvelle n'a été décidée.")
add_special("0050", "Occlusion par végétation: le pan gauche du pignon est masqué par un arbre fleuri. "
            "Illustre la limite d'occlusion (spec: seuil 15% largeur image) et l'exclusion des villas voisines.")
add_special("0001", "Placement du point déterminant: en 1re passe, tourelle et aile gauche captées sur "
            "mur/végétation; repositionnées ensuite sur la tuile (voir *_FIX_*). Illustre la boucle "
            "proposition SAM2 -> correction du point.")

# 10) QA — presence checks
for did in IDS:
    doc = db.images.find_one({"dataset_id": did})
    checks = [
        (ROOT / "originals" / did / doc["original_filename"]).exists(),
        (ROOT / "merged" / f"{did}.png").exists(),
        (ROOT / "overlays" / f"{did}_overlay.png").exists(),
        (ROOT / "before_after" / f"{did}_before_after.png").exists(),
        (ROOT / "metadata" / f"{did}.json").exists(),
        any((ROOT / "planes" / did).glob("*.png")),
    ]
    if not all(checks):
        qa["issues"].append(f"{did}: missing package files {checks}")

total_planes = sum(len(list((ROOT / 'planes' / d).glob('*.png'))) for d in IDS)
qa["total_planes"] = total_planes
qa["passed"] = len(qa["issues"]) == 0
(ROOT.parent / "PACKAGE_QA.json").write_text(json.dumps(qa, indent=2))
print("QA passed:", qa["passed"], "| issues:", qa["issues"], "| total planes:", total_planes)
json.dump({"summary_rows": summary_rows}, open("/tmp/summary_rows.json", "w"))
print("PACKAGE BUILT at", ROOT)
