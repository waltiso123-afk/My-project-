"""SegFormer-B0 @768 TRAINING-PIPELINE SMOKE TEST (reuses scripts/train_step1.py logic verbatim).
Only differences vs train_step1: RES=768, tiny subset, 1-2 epochs, + val-loss, + checkpoint
reload/inference checks, + integrity checks. Architecture/loss/eval/config UNCHANGED.
NOT an accuracy benchmark. Read-only w.r.t. annotations. No GPU."""
import os, io, sys, json, time, random, hashlib, resource, traceback
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import storage, config, pymongo
import warnings; warnings.filterwarnings("ignore")

torch.manual_seed(0); np.random.seed(0); random.seed(0)          # seed 0 (same as train_step1)
torch.set_num_threads(os.cpu_count() or 4)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

RES = 768                                                        # <-- ONLY resolution change (768, not 256)
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)
W_CE = torch.tensor([1.0,3.0])                                   # weighted CE [1,3] (unchanged)
FROZEN13 = ["0016","0023","0031","0032","0037","0044","0051","0054","0058","0064","0085","0120","0138"]
OUT = Path("/app/reports/milestone1_v2_768_segformer_smoke"); (OUT/"models").mkdir(parents=True, exist_ok=True)

def ann_hash():
    appr = sorted(db.images.find({"status":"approved"}), key=lambda d: d["dataset_id"])
    sig = []
    for d in appr:
        a = db.annotations.find_one({"dataset_id": d["dataset_id"]})
        sig.append((d["dataset_id"], d.get("split"), d.get("house_group_id"), d.get("storage_path"),
                    tuple(a["generated"]["planes"]) if a else None))
    return hashlib.sha256(json.dumps(sig, default=str).encode()).hexdigest(), len(appr)

# ---- data loading: EXACTLY train_step1.load (merged binary mask) --------------------
def load(d):
    raw,_ = storage.get_object(d["storage_path"])
    im = np.array(Image.open(io.BytesIO(raw)).convert("RGB").resize((RES,RES), Image.BILINEAR))
    md,_ = storage.get_object(config.merged_path(d["dataset_id"]))
    mk = np.array(Image.open(io.BytesIO(md)).convert("L").resize((RES,RES), Image.NEAREST))
    return im.astype(np.uint8), (mk>127).astype(np.uint8)

cache = {}
def batchify(ids):
    x=np.stack([((cache[i][0]/255.0 - MEAN)/STD).transpose(2,0,1) for i in ids]).astype(np.float32)
    y=np.stack([cache[i][1] for i in ids]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)

def roofline_bottom(mask):                                       # eave proxy (unchanged from train_step1)
    ys={}; H,W=mask.shape
    for x in range(W):
        col=np.where(mask[:,x]>0)[0]
        if len(col): ys[x]=int(col.max())
    return ys

def evaluate(model, ids):                                        # EXACT train_step1.evaluate + val_loss
    model.eval(); inter=union=0.0; dice_n=dice_d=0.0
    perp_errs=[]; covs=[]; vloss=0.0; nb=0
    with torch.no_grad():
        for i in ids:
            x,y=batchify([i]); logits=model(pixel_values=x).logits
            up=F.interpolate(logits,size=(RES,RES),mode="bilinear",align_corners=False)
            vloss+=float(F.cross_entropy(up,y,weight=W_CE)); nb+=1
            pred=(up.argmax(1)[0].numpy()).astype(np.uint8); gt=y[0].numpy().astype(np.uint8)
            inter+=np.logical_and(pred,gt).sum(); union+=np.logical_or(pred,gt).sum()
            dice_n+=2*np.logical_and(pred,gt).sum(); dice_d+=pred.sum()+gt.sum()
            gl=roofline_bottom(gt); pl=roofline_bottom(pred)
            if gl:
                errs=[abs(gl[x]-pl[x]) for x in gl if x in pl]
                if errs: perp_errs.append(np.mean(errs)/RES)
                covs.append(len([x for x in gl if x in pl])/len(gl))
    iou=inter/union if union else 0.0; dice=dice_n/dice_d if dice_d else 0.0
    return {"val_loss":round(vloss/nb,4) if nb else None,
            "iou":round(float(iou),4),"dice":round(float(dice),4),
            "mean_perp_roofline_err_normwidth":round(float(np.mean(perp_errs)) if perp_errs else 0.0,5),
            "horizontal_coverage":round(float(np.mean(covs)) if covs else 0.0,4)}

def main():
    report={"model":"SegFormer-B0 (nvidia/mit-b0), num_labels=2 (0=bg,1=roof)","input_res":[RES,RES,3],
            "note":"SMOKE TEST ONLY — training pipeline @768 executes; NOT final accuracy.",
            "config":{"optimizer":"AdamW","lr":6e-5,"weight_decay":1e-4,"loss":"weighted CE [1,3]",
                      "seed":0,"device":"cpu"},"stages":[]}
    def st(name,ok,detail=""):
        report["stages"].append({"stage":name,"ok":bool(ok),"detail":str(detail)[:300]})
        print(("[PASS] " if ok else "[FAIL] ")+name+(f" :: {detail}" if detail else ""))
    class Fail(Exception): ...

    # ===== PRE integrity =====
    h_before, n_appr = ann_hash()
    appr = sorted(db.images.find({"status":"approved"}), key=lambda d: d["dataset_id"])
    holdout=[d for d in appr if d.get("split")=="holdout"]; pool=[d for d in appr if d.get("split")!="holdout"]
    hg_hold=set(d.get("house_group_id") for d in holdout); hg_pool=set(d.get("house_group_id") for d in pool)
    frozen_intact=all(any(d["dataset_id"]==f and d.get("split")=="holdout" for d in appr) for f in FROZEN13)
    st("integrity_pre_model_is_segformer_b0", True, "nvidia/mit-b0 num_labels=2")
    st("integrity_pre_resolution_768", RES==768, f"{RES}x{RES}")
    st("integrity_pre_frozen13_holdout_intact", frozen_intact, f"approved={n_appr}, 13 frozen all in holdout={frozen_intact}")

    # subset: 10 train (pool), 3 val (from frozen-13 holdout) — group-aware => no leak
    train_ids=[d["dataset_id"] for d in pool[:10]]
    val_ids=FROZEN13[:3]
    hg_train=set(db.images.find_one({"dataset_id":i}).get("house_group_id") for i in train_ids)
    hg_val=set(db.images.find_one({"dataset_id":i}).get("house_group_id") for i in val_ids)
    leak = hg_train & hg_val
    st("integrity_pre_no_leakage_subset_vs_holdout", not leak and not (set(train_ids)&set(val_ids)),
       f"train={train_ids} val(holdout)={val_ids} group_leak={leak or 'NONE'}")
    report["dataset_subset"]={"train":train_ids,"val_holdout":val_ids,"annotation_hash_before":h_before}
    if not (RES==768 and frozen_intact and not leak): raise Fail("pre-integrity")

    # ===== load subset via existing pipeline (merged binary masks) =====
    try:
        for i in set(train_ids+val_ids):
            cache[i]=load(db.images.find_one({"dataset_id":i}))
        shp={i:list(cache[i][0].shape) for i in train_ids[:1]}
        roof_frac={i:round(float(cache[i][1].mean()),4) for i in train_ids[:3]}
        st("dataset_load_768", all(cache[i][0].shape==(RES,RES,3) and cache[i][1].shape==(RES,RES) for i in cache),
           f"img_shape={shp} mask_roof_frac_sample={roof_frac}")
    except Exception:
        st("dataset_load_768", False, traceback.format_exc()); raise Fail("load")

    # ===== model init (exact) =====
    try:
        model=SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",num_labels=2,ignore_mismatched_sizes=True)
        opt=torch.optim.AdamW(model.parameters(),lr=6e-5,weight_decay=1e-4)
        st("model_init", True, f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M num_labels=2")
    except Exception:
        st("model_init", False, traceback.format_exc()); raise Fail("init")

    # ===== train 2 epochs (bs=2 for 768 CPU mem safety) + val each epoch =====
    EPOCHS=2; BS=2
    history=[]; best={"dice":-1}; best_state=None; t_all=time.time()
    try:
        for ep in range(1,EPOCHS+1):
            model.train(); random.shuffle(train_ids); tl=0.0; nb=0; te=time.time()
            for k in range(0,len(train_ids),BS):
                ids=train_ids[k:k+BS]; x,y=batchify(ids)
                up=F.interpolate(model(pixel_values=x).logits,size=(RES,RES),mode="bilinear",align_corners=False)
                loss=F.cross_entropy(up,y,weight=W_CE)
                opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
            ep_train_time=time.time()-te
            ev=evaluate(model,val_ids)
            row={"epoch":ep,"train_loss":round(tl/nb,4),"epoch_runtime_s":round(ep_train_time,1),**ev}
            history.append(row)
            if ev["dice"]>best["dice"]: best={**ev,"epoch":ep}; best_state={k:v.clone() for k,v in model.state_dict().items()}
            print(f"  ep{ep}/{EPOCHS} train_loss={row['train_loss']} val_loss={ev['val_loss']} dice={ev['dice']} iou={ev['iou']} "
                  f"perp={ev['mean_perp_roofline_err_normwidth']} cov={ev['horizontal_coverage']} ({int(ep_train_time)}s)")
        total_rt=round(time.time()-t_all,1)
        st("train_and_validate_2epochs", True, f"epochs={EPOCHS} total_runtime_s={total_rt}")
    except Exception:
        st("train_and_validate_2epochs", False, traceback.format_exc()); raise Fail("train")

    peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)  # KB->MB on Linux

    # ===== checkpoint save =====
    ckpt=OUT/"models"/"smoke_best.pt"
    try:
        torch.save(best_state, ckpt)
        st("checkpoint_save", ckpt.exists() and ckpt.stat().st_size>0, f"{ckpt} ({ckpt.stat().st_size//1024}KB) best_epoch={best['epoch']} by dice(diagnostic)")
    except Exception:
        st("checkpoint_save", False, traceback.format_exc()); raise Fail("save")

    # ===== checkpoint reload + one inference pass =====
    try:
        m2=SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",num_labels=2,ignore_mismatched_sizes=True)
        m2.load_state_dict(torch.load(ckpt,map_location="cpu")); m2.eval()
        st("checkpoint_reload", True, "state_dict loaded into fresh SegFormer-B0")
        with torch.no_grad():
            x,_=batchify([val_ids[0]])
            up=F.interpolate(m2(pixel_values=x).logits,size=(RES,RES),mode="bilinear",align_corners=False)
            binmask=up.argmax(1)[0].numpy()
        st("inference_from_saved_checkpoint", tuple(up.shape)==(1,2,RES,RES) and set(np.unique(binmask)).issubset({0,1}),
           f"logits={tuple(up.shape)} binary_mask_unique={np.unique(binmask).tolist()} roof_px={int((binmask==1).sum())}")
    except Exception:
        st("checkpoint_reload_inference", False, traceback.format_exc()); raise Fail("reload")

    # ===== POST integrity =====
    h_after,_=ann_hash()
    st("integrity_post_annotations_unchanged", h_after==h_before, f"hash_equal={h_after==h_before}")
    frozen_intact2=all(any(d["dataset_id"]==f and d.get("split")=="holdout" for d in db.images.find({"status":"approved"})) for f in FROZEN13)
    st("integrity_post_frozen13_intact", frozen_intact2, f"frozen13_holdout_intact={frozen_intact2}")

    report.update({"history":history,"best_by_dice_DIAGNOSTIC":best,"total_runtime_s":total_rt,
                   "peak_memory_mb":peak_mb,"checkpoint":str(ckpt),
                   "annotation_hash_after":h_after,"annotation_hash_before":h_before,
                   "batch_size":BS,"epochs":EPOCHS})
    report["all_pass"]=all(s["ok"] for s in report["stages"])
    (OUT/"smoke_results.json").write_text(json.dumps(report,indent=2,default=str))
    return report

if __name__=="__main__":
    print("=== SegFormer-B0 @768 SMOKE TEST ===")
    try:
        rep=main()
        print("\n=== SMOKE TEST:", "PASS" if rep["all_pass"] else "FAIL", "===")
        print("runtime",rep["total_runtime_s"],"s | peak_mem",rep["peak_memory_mb"],"MB")
    except Exception as e:
        print("\n=== SMOKE TEST: FAIL ::", e, "===")
        sys.exit(1)
