"""Milestone 1 v2 — PER-PLANE modeling POC (technical validation, NOT the 150@768 run).
Goal: determine which OUTPUT FORMULATION can actually recover INDIVIDUAL roof-plane
instances (not one merged blob) from our existing per-plane GT masks, and produce correct
eave (lower) / parapet (upper) roofline boundaries.

Formulations compared (fast, 256px, CPU — diagnostic metrics only, NOT acceptance metrics):
  A) Binary semantic + connected-components / watershed instancing (reuses baseline_100.pt).
  B) 3-class edge-aware semantic (bg / plane-body / inter-plane-seam) -> body CC instancing.
     Trained briefly on the 76 train images. Seam class exists to split touching planes.
  (Mask2Former instance seg: analysed conceptually only in the report — no training.)

Read-only w.r.t. DB annotations and the FIXED 13-image holdout. Output ->
/app/reports/milestone1_v2_per_plane_POC/.
"""
import os, io, sys, json, time, random, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.measure import label as cc_label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import storage, config, pymongo

torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.set_num_threads(os.cpu_count() or 4)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

RES = 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
OUT = Path("/app/reports/milestone1_v2_per_plane_POC"); (OUT / "overlays").mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_data_cache.pkl"
BASELINE = "/app/reports/milestone1_step1/models/baseline_100.pt"
MIN_INST_AREA = int(0.002 * RES * RES)   # ignore predicted instances < 0.2% of image
MATCH_IOU = 0.25                          # IoU to count a GT plane as recovered

# ----------------------------------------------------------------------------- data
def _load_mask(path):
    md, _ = storage.get_object(path)
    m = np.array(Image.open(io.BytesIO(md)).convert("L").resize((RES, RES), Image.NEAREST))
    return (m > 127).astype(np.uint8)

def load_all():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    appr = sorted(db.images.find({"status": "approved"}), key=lambda d: d["dataset_id"])
    data = {}
    for i, d in enumerate(appr):
        did = d["dataset_id"]
        raw, _ = storage.get_object(d["storage_path"])
        im = np.array(Image.open(io.BytesIO(raw)).convert("RGB").resize((RES, RES), Image.BILINEAR)).astype(np.uint8)
        ann = db.annotations.find_one({"dataset_id": did})
        planes = []
        for p in ann["generated"]["planes"]:
            planes.append({"mask": _load_mask(p), "is_parapet": p.endswith("-parapet.png"),
                           "name": p.split("/")[-1]})
        merged = _load_mask(config.merged_path(did))
        data[did] = {"img": im, "planes": planes, "merged": merged,
                     "split": d.get("split"), "hg": d.get("house_group_id")}
        if (i + 1) % 20 == 0: print(f"  loaded {i+1}/{len(appr)}")
    pickle.dump(data, open(CACHE, "wb"))
    return data

# ------------------------------------------------------------------- geometry utils
def boundary(mask, upper):
    """per-column boundary y; upper=True -> min-y (parapet top), else max-y (eave bottom)."""
    ys = {}
    for x in range(mask.shape[1]):
        col = np.where(mask[:, x] > 0)[0]
        if len(col): ys[x] = int(col.min()) if upper else int(col.max())
    return ys

def roofline_metrics(gt_mask, pred_mask, upper, W):
    gb = boundary(gt_mask, upper)
    if not gb: return None
    pb = boundary(pred_mask, upper)
    errs = [abs(gb[x] - pb[x]) for x in gb if x in pb]
    perp = (float(np.mean(errs)) / W) if errs else None
    cov = len([x for x in gb if x in pb]) / len(gb)
    return {"perp_err_normwidth": round(perp, 5) if perp is not None else None,
            "coverage": round(float(cov), 4), "gt_cols": len(gb)}

def iou(a, b):
    inter = np.logical_and(a, b).sum(); uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 0.0

def match_instances(pred_insts, gt_planes):
    """Hungarian match predicted instance masks to GT plane masks by IoU."""
    n_g = len(gt_planes); n_p = len(pred_insts)
    if n_g == 0: return [], {"n_gt": 0, "n_pred": n_p, "n_matched": 0, "mean_iou": None,
                             "recovery_rate": None, "n_over_merge": 0}
    M = np.zeros((n_g, n_p))
    for i, g in enumerate(gt_planes):
        for j, p in enumerate(pred_insts):
            M[i, j] = iou(g["mask"], p)
    matches = []
    if n_p:
        ri, ci = linear_sum_assignment(-M)
        for i, j in zip(ri, ci):
            if M[i, j] >= MATCH_IOU:
                matches.append((i, j, float(M[i, j])))
    # over-merge: a single predicted instance overlapping >=2 GT planes at >=0.15
    n_over = 0
    for j, p in enumerate(pred_insts):
        hits = sum(1 for g in gt_planes if iou(g["mask"], p) >= 0.15)
        if hits >= 2: n_over += 1
    ious = [m[2] for m in matches]
    stats = {"n_gt": n_g, "n_pred": n_p, "n_matched": len(matches),
             "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
             "recovery_rate": round(len(matches) / n_g, 4), "n_over_merge": n_over}
    return matches, stats

# ------------------------------------------------------------------- instancing ops
def instances_from_mask(mask, use_watershed):
    lbl = cc_label(mask, connectivity=2)
    insts = []
    for k in range(1, lbl.max() + 1):
        comp = (lbl == k)
        if comp.sum() < MIN_INST_AREA: continue
        if use_watershed:
            dist = distance_transform_edt(comp)
            coords = peak_local_max(dist, min_distance=12, labels=comp, exclude_border=False)
            if len(coords) > 1:
                markers = np.zeros_like(lbl)
                for mi, (yy, xx) in enumerate(coords, 1): markers[yy, xx] = mi
                ws = watershed(-dist, markers, mask=comp)
                for w in range(1, ws.max() + 1):
                    sub = (ws == w)
                    if sub.sum() >= MIN_INST_AREA: insts.append(sub.astype(np.uint8))
                continue
        insts.append(comp.astype(np.uint8))
    return insts

# -------------------------------------------------------------- 3-class GT (form. B)
def build_3class_target(planes):
    """0=bg, 1=plane-body, 2=inter-plane seam (where two dilated planes meet)."""
    H = W = RES
    union = np.zeros((H, W), bool)
    dil_count = np.zeros((H, W), np.int16)
    for p in planes:
        m = p["mask"].astype(bool)
        union |= m
        dil_count += binary_dilation(m, iterations=3).astype(np.int16)
    seam = dil_count >= 2                # region where >=2 expanded planes overlap -> the seam
    cls = np.zeros((H, W), np.int64)
    cls[union] = 1
    cls[seam] = 2
    return cls

# --------------------------------------------------------------- forward / eval core
def predict_argmax(net, im, n_labels):
    x = torch.from_numpy(((im / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))
    with torch.no_grad():
        up = F.interpolate(net(pixel_values=x).logits, size=(RES, RES), mode="bilinear", align_corners=False)
    return up.argmax(1)[0].numpy().astype(np.uint8)

def evaluate_formulation(name, ids, data, inst_fn):
    """inst_fn(did)-> list of predicted instance masks. Runs per-plane match + roofline metrics."""
    per_img = {}; perp_all = []; cov_all = []; rec_all = []; iou_all = []
    n_over_tot = 0; n_gt_tot = 0
    for did in ids:
        d = data[did]; gt_planes = d["planes"]
        pred_insts = inst_fn(did)
        matches, stats = match_instances(pred_insts, gt_planes)
        n_gt_tot += stats["n_gt"]; n_over_tot += stats["n_over_merge"]
        if stats["recovery_rate"] is not None: rec_all.append(stats["recovery_rate"])
        if stats["mean_iou"] is not None: iou_all.append(stats["mean_iou"])
        rows = []
        for gi, pj, iouv in matches:
            g = gt_planes[gi]
            rm = roofline_metrics(g["mask"], pred_insts[pj], upper=g["is_parapet"], W=RES)
            if rm is None: continue
            if rm["perp_err_normwidth"] is not None: perp_all.append(rm["perp_err_normwidth"])
            cov_all.append(rm["coverage"])
            rows.append({"gt_plane": g["name"], "is_parapet": g["is_parapet"],
                         "match_iou": round(iouv, 4), **rm})
        per_img[did] = {"split": d["split"], **stats, "matched_planes": rows}
    summary = {
        "formulation": name,
        "n_images": len(ids),
        "gt_plane_instances_total": n_gt_tot,
        "mean_instance_recovery_rate": round(float(np.mean(rec_all)), 4) if rec_all else None,
        "mean_matched_iou": round(float(np.mean(iou_all)), 4) if iou_all else None,
        "predicted_instances_that_merge_multiple_planes": n_over_tot,
        "mean_perp_err_normwidth_matched": round(float(np.mean(perp_all)), 5) if perp_all else None,
        "mean_horizontal_coverage_matched": round(float(np.mean(cov_all)), 4) if cov_all else None,
    }
    return {"summary": summary, "per_image": per_img}

# -------------------------------------------------------------------------- overlays
PALETTE = [(255,80,80),(80,180,255),(120,230,120),(255,200,60),(210,120,255),(80,230,230),(255,140,190),(180,180,80)]
def save_overlay(did, im, insts, gt_planes, tag):
    base = im.copy()
    ov = base.copy()
    for k, inst in enumerate(insts):
        c = PALETTE[k % len(PALETTE)]
        ov[inst > 0] = (0.45 * np.array(c) + 0.55 * ov[inst > 0]).astype(np.uint8)
    # draw GT eave/parapet boundary polyline (green normal / red parapet)
    for g in gt_planes:
        bb = boundary(g["mask"], upper=g["is_parapet"])
        col = (255, 40, 40) if g["is_parapet"] else (30, 255, 30)
        for x, y in bb.items():
            ov[max(0,y-1):y+2, x] = col
    Image.fromarray(ov).save(OUT / "overlays" / f"{did}_{tag}.png")

# -------------------------------------------------------------------------- training B
def train_form_B(train_ids, data, epochs=8, bs=4, lr=6e-5):
    net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=3, ignore_mismatched_sizes=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor([1.0, 3.0, 8.0])   # bg / body / seam (seam is thin -> upweight)
    tids = list(train_ids)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train(); random.shuffle(tids); tl = 0.0; nb = 0
        for k in range(0, len(tids), bs):
            ids = tids[k:k + bs]
            x = torch.from_numpy(np.stack([((data[i]["img"]/255.0 - MEAN)/STD).transpose(2,0,1) for i in ids]).astype(np.float32))
            y = torch.from_numpy(np.stack([build_3class_target(data[i]["planes"]) for i in ids]))
            up = F.interpolate(net(pixel_values=x).logits, size=(RES, RES), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(up, y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        print(f"  [B] ep{ep}/{epochs} loss={tl/nb:.4f} ({int(time.time()-t0)}s)")
    net.eval()
    return net

# ------------------------------------------------------------------------------- main
def main():
    print("loading data (image + per-plane GT masks @256)...")
    data = load_all(); print("loaded", len(data), "images")
    holdout = sorted([k for k, v in data.items() if v["split"] == "holdout"])
    train_ids = sorted([k for k, v in data.items() if v["split"] == "train"])
    parapet_ids = sorted([k for k, v in data.items() if any(p["is_parapet"] for p in v["planes"])])
    print("holdout:", holdout)
    print("parapet-bearing images:", parapet_ids)

    results = {"config": {"res": RES, "match_iou": MATCH_IOU, "min_inst_area_px": MIN_INST_AREA,
                          "holdout_ids": holdout, "train_ids_count": len(train_ids),
                          "parapet_ids": parapet_ids, "note": "256px diagnostic POC, not acceptance metrics"}}

    # ---- Formulation A: binary baseline_100 + CC and CC+watershed instancing
    print("\n=== Formulation A: binary + connected-components / watershed ===")
    netA = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True)
    netA.load_state_dict(torch.load(BASELINE, map_location="cpu")); netA.eval()
    predA_cache = {did: predict_argmax(netA, data[did]["img"], 2) for did in data}
    A_cc = evaluate_formulation("A_binary_cc", holdout, data, lambda did: instances_from_mask(predA_cache[did], False))
    A_ws = evaluate_formulation("A_binary_watershed", holdout, data, lambda did: instances_from_mask(predA_cache[did], True))
    results["A_binary_cc"] = A_cc
    results["A_binary_watershed"] = A_ws
    print(" A/cc:", A_cc["summary"]); print(" A/ws:", A_ws["summary"])

    # ---- Formulation B: 3-class edge-aware, quick train, body-CC instancing
    print("\n=== Formulation B: 3-class edge-aware (train quick @256) ===")
    netB = train_form_B(train_ids, data, epochs=8)
    def instB(did):
        pm = predict_argmax(netB, data[did]["img"], 3)
        return instances_from_mask((pm == 1).astype(np.uint8), False)
    predB_cache = {did: predict_argmax(netB, data[did]["img"], 3) for did in data}
    B = evaluate_formulation("B_3class_edge", holdout, data, lambda did: instances_from_mask((predB_cache[did]==1).astype(np.uint8), False))
    results["B_3class_edge"] = B
    print(" B:", B["summary"])

    # ---- Parapet-branch demonstration (diagnostic; parapet imgs are in train/val)
    print("\n=== Parapet upper-boundary demonstration (train/val diagnostic) ===")
    par = {}
    for did in parapet_ids:
        d = data[did]
        instsA = instances_from_mask(predA_cache[did], True)
        instsB = instances_from_mask((predB_cache[did] == 1).astype(np.uint8), False)
        rowsA = []
        for g in d["planes"]:
            # match to best A instance
            best = max(instsA, key=lambda p: iou(g["mask"], p)) if instsA else None
            rm = roofline_metrics(g["mask"], best, upper=g["is_parapet"], W=RES) if best is not None else None
            rowsA.append({"gt_plane": g["name"], "is_parapet": g["is_parapet"], "boundary": "UPPER" if g["is_parapet"] else "LOWER", "metrics_A": rm})
        par[did] = {"split": d["split"], "n_planes": len(d["planes"]),
                    "n_parapet": sum(p["is_parapet"] for p in d["planes"]), "planes": rowsA}
        save_overlay(did, d["img"], instsA, d["planes"], "A")
        save_overlay(did, d["img"], instsB, d["planes"], "B")
    results["parapet_demo"] = par
    print(" parapet demo done for", parapet_ids)

    # ---- qualitative overlays for a few multi-plane holdout images
    for did in holdout[:6]:
        d = data[did]
        save_overlay(did, d["img"], instances_from_mask(predA_cache[did], True), d["planes"], "A")
        save_overlay(did, d["img"], instances_from_mask((predB_cache[did]==1).astype(np.uint8), False), d["planes"], "B")

    (OUT / "poc_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved", OUT / "poc_results.json")
    print("overlays ->", OUT / "overlays")

if __name__ == "__main__":
    main()
