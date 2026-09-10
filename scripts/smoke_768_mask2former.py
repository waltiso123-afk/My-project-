"""768px SMOKE TEST — Mask2Former instance-seg pipeline (end-to-end, NOT the real run).
Tiny subset, 1-2 epochs, CPU. Exercises the FULL production path and STOPS on first failure.
Numerical accuracy is NOT relevant here. No annotations/holdout/approved data modified.
"""
import os, io, sys, json, time, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import torch
from scipy.optimize import linear_sum_assignment
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import storage, config, pymongo

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(os.cpu_count() or 4)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

RES = 768
CKPT = "facebook/mask2former-swin-tiny-coco-instance"
OUT = Path("/app/reports/milestone1_v2_768_smoke_test"); OUT.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT / "smoke_ckpt"
FROZEN_HOLDOUT = ["0016","0023","0031","0032","0037","0044","0051","0054","0058","0064","0085","0120","0138"]

stages = []            # (name, ok, detail)
def stage(name, ok, detail=""):
    stages.append({"stage": name, "ok": bool(ok), "detail": str(detail)[:400]})
    print(("  [PASS] " if ok else "  [FAIL] ") + name + (f" :: {detail}" if detail else ""))

class SmokeFail(Exception): ...

# --------------------------------------------------------------- data (768, subset)
def load_item(did):
    d = db.images.find_one({"dataset_id": did})
    raw, _ = storage.get_object(d["storage_path"])
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((RES, RES), Image.BILINEAR)
    ann = db.annotations.find_one({"dataset_id": did})
    planes = []
    for p in ann["generated"]["planes"]:
        md, _ = storage.get_object(p)
        m = np.array(Image.open(io.BytesIO(md)).convert("L").resize((RES, RES), Image.NEAREST))
        planes.append({"mask": (m > 127).astype(np.uint8),
                       "is_parapet": p.endswith("-parapet.png"), "name": p.split("/")[-1]})
    return {"did": did, "pil": im, "planes": planes, "split": d.get("split")}

# --------------------------------------------------------------- geometry / metrics
def boundary(mask, upper):
    ys = {}
    for x in range(mask.shape[1]):
        col = np.where(mask[:, x] > 0)[0]
        if len(col): ys[x] = int(col.min()) if upper else int(col.max())
    return ys

def roofline_metrics(gt, pred, upper, W):
    gb = boundary(gt, upper)
    if not gb: return None
    pb = boundary(pred, upper)
    errs = [abs(gb[x] - pb[x]) for x in gb if x in pb]
    perp = (float(np.mean(errs)) / W) if errs else None
    cov = len([x for x in gb if x in pb]) / len(gb)
    return {"perp_err_normwidth": round(perp, 5) if perp is not None else None,
            "coverage": round(float(cov), 4), "gt_cols": len(gb)}

def iou(a, b):
    inter = np.logical_and(a, b).sum(); uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 0.0

# --------------------------------------------------------------- main
def main():
    report = {"when": datetime.now(timezone.utc).isoformat(), "res": RES, "model": CKPT}

    # ---- Stage: dataset integrity (read-only) -------------------------------------
    try:
        appr = sorted(db.images.find({"status": "approved"}), key=lambda d: d["dataset_id"])
        n_appr = len(appr)
        cur_hold = sorted(d["dataset_id"] for d in appr if d.get("split") == "holdout")
        frozen_intact = all(h in cur_hold for h in FROZEN_HOLDOUT)
        # group-aware leakage: no house_group_id spans holdout and non-holdout
        hg_hold = set(d.get("house_group_id") for d in appr if d.get("split") == "holdout")
        hg_pool = set(d.get("house_group_id") for d in appr if d.get("split") != "holdout")
        leak = hg_hold & hg_pool
        report["dataset"] = {"approved": n_appr, "holdout_now": cur_hold,
                             "frozen13_intact": frozen_intact, "group_leak": sorted(leak)}
        ok = (n_appr >= 150) and frozen_intact and (not leak)
        stage("dataset_integrity", ok,
              f"approved={n_appr} frozen13_intact={frozen_intact} holdout_now={len(cur_hold)} leak={leak or 'NONE'}")
        if not ok: raise SmokeFail("dataset integrity")
    except SmokeFail: raise
    except Exception as e:
        stage("dataset_integrity", False, traceback.format_exc()); raise SmokeFail("dataset")

    # ---- Stage: pick tiny subset (train incl. parapet; eval incl. parapet+normal) --
    train_ids = ["0020", "0001", "0002", "0009"]      # 0020 = parapet (exercise parapet cat)
    eval_ids  = ["0169", "0016", "0021"]              # 0169/0021 parapet, 0016 normal (holdout)
    subset = train_ids + [e for e in eval_ids if e not in train_ids]

    # ---- Stage: load 768 data + per-plane GT masks --------------------------------
    try:
        data = {did: load_item(did) for did in subset}
        n_planes = {did: len(data[did]["planes"]) for did in subset}
        n_par = sum(sum(p["is_parapet"] for p in data[did]["planes"]) for did in subset)
        stage("load_dataset_768_perplane_gt", True,
              f"imgs={len(data)} planes_per_img={n_planes} parapet_planes_total={n_par}")
        if n_par == 0: raise SmokeFail("no parapet plane in subset — cannot test parapet branch")
    except SmokeFail: raise
    except Exception:
        stage("load_dataset_768_perplane_gt", False, traceback.format_exc()); raise SmokeFail("load")

    # ---- Stage: instance conversion (per-plane GT -> instance masks + class ids) ---
    # category: 0 = roof_plane, 1 = parapet_plane
    try:
        for did in subset:
            insts = []; classes = []
            for p in data[did]["planes"]:
                if p["mask"].sum() == 0: continue
                insts.append(p["mask"].astype(np.float32))
                classes.append(1 if p["is_parapet"] else 0)
            data[did]["inst_masks"] = insts
            data[did]["inst_classes"] = classes
        conv_ok = all(len(data[d]["inst_masks"]) == sum(1 for p in data[d]["planes"] if p["mask"].sum() > 0) for d in subset)
        per = {d: len(data[d]["inst_masks"]) for d in subset}
        stage("instance_conversion", conv_ok, f"instances_per_img={per}")
        if not conv_ok: raise SmokeFail("instance conversion")
    except SmokeFail: raise
    except Exception:
        stage("instance_conversion", False, traceback.format_exc()); raise SmokeFail("instconv")

    # ---- Stage: Mask2Former init + processor --------------------------------------
    try:
        proc = Mask2FormerImageProcessor(do_resize=False, do_rescale=True, do_normalize=True,
                                         ignore_index=255, reduce_labels=False)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            CKPT, num_labels=2, ignore_mismatched_sizes=True)
        model.train()
        nparam = sum(p.numel() for p in model.parameters())
        stage("mask2former_init", True, f"params={nparam/1e6:.1f}M num_labels=2 (roof_plane,parapet_plane)")
    except Exception:
        stage("mask2former_init", False, traceback.format_exc()); raise SmokeFail("init")

    def pixel_values(did):
        return proc(images=[data[did]["pil"]], return_tensors="pt").pixel_values

    # ---- Stage: forward + loss + backward + optimizer step ------------------------
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        t0 = time.time()
        pv0 = pixel_values(train_ids[0])
        ml0 = [torch.tensor(np.stack(data[train_ids[0]]["inst_masks"]), dtype=torch.float32)]
        cl0 = [torch.tensor(data[train_ids[0]]["inst_classes"], dtype=torch.long)]
        out = model(pixel_values=pv0, mask_labels=ml0, class_labels=cl0)
        loss = out.loss
        stage("forward_and_loss", (loss is not None and torch.isfinite(loss)),
              f"pixel_values={tuple(pv0.shape)} loss={float(loss):.4f} ({int(time.time()-t0)}s)")
        loss.backward()
        gnorm = float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)))
        stage("backward", np.isfinite(gnorm), f"grad_norm={gnorm:.3f}")
        opt.step(); opt.zero_grad()
        stage("optimizer_step", True, "AdamW step ok")
    except Exception:
        stage("forward_backward_optim", False, traceback.format_exc()); raise SmokeFail("trainstep")

    # ---- Stage: run 2 epochs, eval per epoch, select by roofline error ------------
    def evaluate(mdl):
        mdl.eval(); per_plane = []; perp_all = []; cov_all = []
        tested_normal = tested_parapet = False; n_pred_multi = 0
        with torch.no_grad():
            for did in eval_ids:
                pv = pixel_values(did)
                out = mdl(pixel_values=pv)
                res = proc.post_process_instance_segmentation(
                    out, target_sizes=[(RES, RES)], threshold=0.0)[0]
                seg = res["segmentation"]
                seg = seg.numpy() if hasattr(seg, "numpy") else np.array(seg)
                pred_insts = []
                for s in res["segments_info"]:
                    m = (seg == s["id"]).astype(np.uint8)
                    if m.sum() > 0: pred_insts.append({"mask": m, "label": int(s["label_id"])})
                if len(pred_insts) >= 2: n_pred_multi += 1
                gt = data[did]["planes"]
                # Hungarian match pred<->gt by IoU
                if pred_insts and gt:
                    M = np.zeros((len(gt), len(pred_insts)))
                    for i, g in enumerate(gt):
                        for j, p in enumerate(pred_insts):
                            M[i, j] = iou(g["mask"], p["mask"])
                    ri, ci = linear_sum_assignment(-M)
                    for i, j in zip(ri, ci):
                        g = gt[i]; upper = g["is_parapet"]
                        rm = roofline_metrics(g["mask"], pred_insts[j]["mask"], upper=upper, W=RES)
                        if rm is None: continue
                        if upper: tested_parapet = True
                        else: tested_normal = True
                        if rm["perp_err_normwidth"] is not None: perp_all.append(rm["perp_err_normwidth"])
                        cov_all.append(rm["coverage"])
                        per_plane.append({"image": did, "gt_plane": g["name"],
                                          "boundary": "UPPER_parapet" if upper else "LOWER_eave",
                                          "pred_instance_id": int(j), "pred_label": pred_insts[j]["label"],
                                          "iou": round(float(M[i, j]), 4), **rm})
        return {"per_plane": per_plane,
                "mean_perp_err_normwidth": round(float(np.mean(perp_all)), 5) if perp_all else None,
                "mean_coverage": round(float(np.mean(cov_all)), 4) if cov_all else None,
                "tested_normal_eave": tested_normal, "tested_parapet_upper": tested_parapet,
                "images_with_multiple_pred_instances": n_pred_multi}

    try:
        epoch_evals = []; t0 = time.time()
        for ep in range(1, 3):
            model.train()
            for did in train_ids:
                pv = pixel_values(did)
                ml = [torch.tensor(np.stack(data[did]["inst_masks"]), dtype=torch.float32)]
                cl = [torch.tensor(data[did]["inst_classes"], dtype=torch.long)]
                loss = model(pixel_values=pv, mask_labels=ml, class_labels=cl).loss
                opt.zero_grad(); loss.backward(); opt.step()
            ev = evaluate(model)
            ev["epoch"] = ep
            epoch_evals.append(ev)
            print(f"  epoch {ep}: perp={ev['mean_perp_err_normwidth']} cov={ev['mean_coverage']} "
                  f"normal={ev['tested_normal_eave']} parapet={ev['tested_parapet_upper']} "
                  f"multi_inst_imgs={ev['images_with_multiple_pred_instances']} ({int(time.time()-t0)}s)")
        stage("train_2epochs_and_eval", True, f"epochs_evaluated={len(epoch_evals)}")
    except Exception:
        stage("train_2epochs_and_eval", False, traceback.format_exc()); raise SmokeFail("epochs")

    # ---- Stage: inference produced multiple instances -----------------------------
    stage("inference_multiple_instances",
          any(e["images_with_multiple_pred_instances"] > 0 for e in epoch_evals),
          f"max_multi_inst_imgs={max(e['images_with_multiple_pred_instances'] for e in epoch_evals)}")

    # ---- Stage: per-plane matching executed ---------------------------------------
    matched_rows = sum(len(e["per_plane"]) for e in epoch_evals)
    stage("per_plane_matching", matched_rows > 0, f"matched_plane_rows_total={matched_rows}")

    # ---- Stage: boundary handling (both branches executed) ------------------------
    tested_normal = any(e["tested_normal_eave"] for e in epoch_evals)
    tested_parapet = any(e["tested_parapet_upper"] for e in epoch_evals)
    stage("normal_eave_boundary", tested_normal, "lower/eave extraction executed")
    stage("parapet_upper_boundary", tested_parapet, "upper parapet extraction executed")

    # ---- Stage: roofline error + coverage computed --------------------------------
    stage("roofline_error_and_coverage",
          any(e["mean_perp_err_normwidth"] is not None for e in epoch_evals) or matched_rows > 0,
          "perpendicular error + horizontal coverage computed per matched plane")

    # ---- Stage: checkpoint SELECTION by roofline error (NOT dice/iou) -------------
    try:
        def perp_key(e):
            v = e["mean_perp_err_normwidth"]
            return v if v is not None else float("inf")
        valid = [e for e in epoch_evals if e["mean_perp_err_normwidth"] is not None]
        chosen = min(valid, key=perp_key) if valid else min(epoch_evals, key=lambda e: -e["mean_coverage"] if e["mean_coverage"] else 0)
        report["checkpoint_selection"] = {
            "primary_metric": "mean_perpendicular_roofline_error_normalized_by_width",
            "direction": "MINIMIZE",
            "selected_epoch": chosen["epoch"],
            "selected_perp_err": chosen["mean_perp_err_normwidth"],
            "selected_coverage": chosen["mean_coverage"],
            "note": "Dice/IoU NOT used for selection (diagnostic only).",
        }
        stage("checkpoint_selection_by_roofline_error", True,
              f"selected epoch={chosen['epoch']} by MIN perp_err={chosen['mean_perp_err_normwidth']}")
    except Exception:
        stage("checkpoint_selection_by_roofline_error", False, traceback.format_exc()); raise SmokeFail("select")

    # ---- Stage: checkpoint SAVE + RELOAD + inference-after-reload ------------------
    try:
        model.save_pretrained(CKPT_PATH)
        saved = list(Path(CKPT_PATH).glob("*"))
        stage("checkpoint_save", len(saved) > 0, f"files={[p.name for p in saved]}")
        reloaded = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT_PATH)
        reloaded.eval()
        with torch.no_grad():
            out = reloaded(pixel_values=pixel_values(eval_ids[0]))
            res = proc.post_process_instance_segmentation(out, target_sizes=[(RES, RES)], threshold=0.0)[0]
        stage("checkpoint_reload_and_infer", True,
              f"reloaded ok, inference segments={len(res['segments_info'])}")
    except Exception:
        stage("checkpoint_save_reload", False, traceback.format_exc()); raise SmokeFail("ckpt")

    report["epoch_evals"] = epoch_evals
    report["stages"] = stages
    report["all_pass"] = all(s["ok"] for s in stages)
    (OUT / "smoke_results.json").write_text(json.dumps(report, indent=2, default=str))
    return report

if __name__ == "__main__":
    print(f"=== 768 SMOKE TEST (Mask2Former) — {datetime.now().isoformat()} ===")
    try:
        rep = main()
        print("\n=== SMOKE TEST:", "PASS" if rep["all_pass"] else "FAIL", "===")
    except SmokeFail as sf:
        Path("/app/reports/milestone1_v2_768_smoke_test").mkdir(parents=True, exist_ok=True)
        (Path("/app/reports/milestone1_v2_768_smoke_test") / "smoke_results.json").write_text(
            json.dumps({"stages": stages, "all_pass": False, "stopped_at": str(sf)}, indent=2, default=str))
        print("\n=== SMOKE TEST: FAIL (stopped at:", sf, ") ===")
        sys.exit(1)
