"""Downstream verification: binary roof mask -> connected components -> per-plane ->
roofline (eave lower / parapet upper) -> mean perpendicular error + coverage.
Proves the client's downstream path consumes a BINARY SegFormer mask (no NN instance seg).
Runs on (a) a real GT merged binary mask and (b) the ACTUAL untrained ONNX SegFormer output.
Read-only w.r.t. production annotations."""
import os, io, sys, json
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as cc_label
import storage, config, pymongo

db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
RES = 768
OUT = Path("/app/reports/milestone1_v2_browser_export_validation")
ONNX_PATH = OUT / "segformer_b0_768_binary.onnx"
MIN_AREA = 50  # small: GT per-plane masks are sparse (~0.2% of frame); accuracy irrelevant here

def load(did):
    d = db.images.find_one({"dataset_id": did})
    raw, _ = storage.get_object(d["storage_path"])
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((RES, RES), Image.BILINEAR)
    ann = db.annotations.find_one({"dataset_id": did})
    planes = []
    for p in ann["generated"]["planes"]:
        md, _ = storage.get_object(p)
        m = np.array(Image.open(io.BytesIO(md)).convert("L").resize((RES, RES), Image.NEAREST))
        planes.append({"mask": (m > 127).astype(np.uint8), "is_parapet": p.endswith("-parapet.png"),
                       "name": p.split("/")[-1]})
    mm, _ = storage.get_object(config.merged_path(did))
    merged = (np.array(Image.open(io.BytesIO(mm)).convert("L").resize((RES, RES), Image.NEAREST)) > 127).astype(np.uint8)
    union = np.zeros((RES, RES), np.uint8)
    for p in planes:
        union |= p["mask"]
    return im, planes, union

def boundary(mask, upper):
    ys = {}
    for x in range(mask.shape[1]):
        col = np.where(mask[:, x] > 0)[0]
        if len(col): ys[x] = int(col.min()) if upper else int(col.max())
    return ys

def roofline(gt, pred, upper, W):
    gb = boundary(gt, upper)
    if not gb: return None
    pb = boundary(pred, upper)
    errs = [abs(gb[x] - pb[x]) for x in gb if x in pb]
    return {"perp_err_normwidth": round(float(np.mean(errs))/W, 5) if errs else None,
            "coverage": round(len([x for x in gb if x in pb])/len(gb), 4)}

def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i/u) if u else 0.0

def cc_instances(binmask):
    lbl = cc_label(binmask, connectivity=2)
    out = []
    for k in range(1, lbl.max()+1):
        c = (lbl == k)
        if c.sum() >= MIN_AREA: out.append(c.astype(np.uint8))
    return out

def per_plane_eval(binmask, planes, tag):
    insts = cc_instances(binmask)
    rows = []
    if insts and planes:
        M = np.zeros((len(planes), len(insts)))
        for i, g in enumerate(planes):
            for j, p in enumerate(insts):
                M[i, j] = iou(g["mask"], p)
        ri, ci = linear_sum_assignment(-M)
        for i, j in zip(ri, ci):
            g = planes[i]
            rm = roofline(g["mask"], insts[j], upper=g["is_parapet"], W=RES)
            rows.append({"gt_plane": g["name"], "boundary": "UPPER_parapet" if g["is_parapet"] else "LOWER_eave",
                         "matched_instance": int(j), "match_iou": round(float(M[i, j]), 4),
                         **(rm or {})})
    perp = [r["perp_err_normwidth"] for r in rows if r.get("perp_err_normwidth") is not None]
    cov = [r["coverage"] for r in rows if r.get("coverage") is not None]
    return {"source": tag, "n_cc_instances": len(insts), "n_gt_planes": len(planes),
            "matched": len(rows),
            "mean_perp_err_normwidth_PRIMARY": round(float(np.mean(perp)), 5) if perp else None,
            "mean_coverage_SECONDARY": round(float(np.mean(cov)), 4) if cov else None,
            "planes": rows}

def main():
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
    res = {"note": "Downstream consumes BINARY SegFormer mask via connected components (no NN instance seg). "
                   "Accuracy irrelevant (model untrained); goal = path executes + boundary branches + metric order.",
           "cases": []}

    for did in ["0016", "0169"]:   # 0016 normal (holdout), 0169 parapet
        im, planes, roof_bin = load(did)
        # (a) downstream on a BINARY roof mask (union of GT planes) -> CC -> per-plane roofline
        case_gt = per_plane_eval(roof_bin, planes, f"{did}:binary_roof_mask(GT_union)")
        # (b) downstream on ACTUAL untrained ONNX SegFormer binary output -> path executes
        x = (((np.asarray(im, np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)[None]).astype(np.float32)
        logits = sess.run(None, {"pixel_values": x})[0]
        pred_bin = (logits.argmax(1)[0]).astype(np.uint8)
        case_pred = per_plane_eval(pred_bin, planes, f"{did}:ONNX_SegFormer_binary_output(untrained)")
        res["cases"].append({"image": did, "n_parapet_planes": sum(p["is_parapet"] for p in planes),
                             "on_GT_binary": case_gt, "on_ONNX_binary": case_pred})
        print(f"{did}: parapet_planes={sum(p['is_parapet'] for p in planes)} | "
              f"GT_binary matched={case_gt['matched']} perp={case_gt['mean_perp_err_normwidth_PRIMARY']} cov={case_gt['mean_coverage_SECONDARY']} | "
              f"ONNX_binary CC={case_pred['n_cc_instances']} matched={case_pred['matched']}")

    res["checkpoint_selection_rule"] = {
        "primary": "mean_perpendicular_roofline_error_normalized_by_width (MINIMIZE)",
        "secondary": "horizontal_coverage (tie-break)",
        "dice_iou": "diagnostic only — NOT used for selection",
        "location": "downstream of the binary model (unchanged client scoring logic)"}
    # boundary-branch confirmation
    res["boundary_branches_executed"] = {
        "normal_eave_lower": any(any(r["boundary"] == "LOWER_eave" for r in c["on_GT_binary"]["planes"] + c["on_ONNX_binary"]["planes"]) for c in res["cases"]),
        "parapet_upper": any(any(r["boundary"] == "UPPER_parapet" for r in c["on_GT_binary"]["planes"] + c["on_ONNX_binary"]["planes"]) for c in res["cases"]),
    }
    (OUT / "downstream_verify.json").write_text(json.dumps(res, indent=2))
    print("boundary branches:", res["boundary_branches_executed"])
    print("wrote downstream_verify.json")

if __name__ == "__main__":
    main()
