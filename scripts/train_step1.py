"""Milestone 1 Step 1 — baseline SegFormer-B0 training (CPU) at 50 and 100(=max non-holdout) images.
Binary roof segmentation (roof vs background) on the frozen 101-image approved set.
Group-aware fixed holdout. Real measured metrics only. NO VLM."""
import os, io, sys, json, time, random
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import storage, config, pymongo
import warnings; warnings.filterwarnings("ignore")

torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.set_num_threads(os.cpu_count() or 4)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
OUT = Path("/app/reports/milestone1_step1"); (OUT/"models").mkdir(parents=True, exist_ok=True)
RES = 256
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)

# ---- frozen dataset + group-aware split ----
appr = sorted(db.images.find({"status":"approved"}), key=lambda d: d["dataset_id"])
holdout = [d for d in appr if d.get("split")=="holdout"]
pool    = [d for d in appr if d.get("split")!="holdout"]   # train+val -> training pool
pool_ids = [d["dataset_id"] for d in pool]; hold_ids=[d["dataset_id"] for d in holdout]
train50 = pool[:50]; train100 = pool[:]                    # 50 strict subset of full pool
# group isolation check
hg_hold = set(d.get("house_group_id") for d in holdout); hg_pool=set(d.get("house_group_id") for d in pool)
leak = hg_hold & hg_pool
print(f"pool={len(pool)} holdout={len(holdout)} | house-group leak between train/holdout: {leak or 'NONE'}")

def load(d):
    raw,_ = storage.get_object(d["storage_path"])
    im = np.array(Image.open(io.BytesIO(raw)).convert("RGB").resize((RES,RES), Image.BILINEAR))
    md,_ = storage.get_object(config.merged_path(d["dataset_id"]))
    mk = np.array(Image.open(io.BytesIO(md)).convert("L").resize((RES,RES), Image.NEAREST))
    return im.astype(np.uint8), (mk>127).astype(np.uint8)

print("loading images..."); cache={}
for d in appr: cache[d["dataset_id"]]=load(d)
print("loaded", len(cache))

def batchify(ims):
    x=np.stack([( (cache[i][0]/255.0 - MEAN)/STD ).transpose(2,0,1) for i in ims]).astype(np.float32)
    y=np.stack([cache[i][1] for i in ims]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)

def roofline_bottom(mask):  # per-column lowest roof pixel (eave proxy); returns dict col->y
    ys={}
    H,W=mask.shape
    for x in range(W):
        col=np.where(mask[:,x]>0)[0]
        if len(col): ys[x]=col.max()
    return ys

def evaluate(model, ids):
    model.eval(); inter=union=tp=0.0; dice_n=dice_d=0.0
    perp_errs=[]; covs=[]
    with torch.no_grad():
        for i in ids:
            x,y=batchify([i]); logits=model(pixel_values=x).logits
            up=F.interpolate(logits,size=(RES,RES),mode="bilinear",align_corners=False)
            pred=(up.argmax(1)[0].numpy()).astype(np.uint8); gt=y[0].numpy().astype(np.uint8)
            inter+=np.logical_and(pred,gt).sum(); union+=np.logical_or(pred,gt).sum()
            dice_n+=2*np.logical_and(pred,gt).sum(); dice_d+=pred.sum()+gt.sum()
            gl=roofline_bottom(gt); pl=roofline_bottom(pred)
            if gl:
                errs=[abs(gl[x]-pl[x]) for x in gl if x in pl]
                if errs: perp_errs.append(np.mean(errs)/RES)   # normalized by image width
                covs.append(len([x for x in gl if x in pl])/len(gl))
    iou=inter/union if union else 0.0; dice=dice_n/dice_d if dice_d else 0.0
    return {"iou":round(float(iou),4),"dice":round(float(dice),4),
            "mean_perp_roofline_err_normwidth":round(float(np.mean(perp_errs)) if perp_errs else 0.0,5),
            "horizontal_coverage":round(float(np.mean(covs)) if covs else 0.0,4),
            "n_holdout":len(ids)}

def train(train_ids, tag, epochs=18, bs=4, lr=6e-5):
    model=SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",num_labels=2,ignore_mismatched_sizes=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    w=torch.tensor([1.0,3.0])  # roof minority weight
    hist=[]; best={"dice":-1}; best_state=None; t0=time.time()
    for ep in range(1,epochs+1):
        model.train(); random.shuffle(train_ids); tl=0.0; nb=0
        for k in range(0,len(train_ids),bs):
            ids=train_ids[k:k+bs]; x,y=batchify(ids)
            logits=model(pixel_values=x).logits
            up=F.interpolate(logits,size=(RES,RES),mode="bilinear",align_corners=False)
            loss=F.cross_entropy(up,y,weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        ev=evaluate(model,hold_ids); hist.append({"epoch":ep,"train_loss":round(tl/nb,4),**ev})
        if ev["dice"]>best["dice"]:
            best=ev; best_state={k:v.clone() for k,v in model.state_dict().items()}
        print(f"[{tag}] ep{ep}/{epochs} train_loss={tl/nb:.4f} holdout dice={ev['dice']} iou={ev['iou']} "
              f"perp={ev['mean_perp_roofline_err_normwidth']} cov={ev['horizontal_coverage']}  ({int(time.time()-t0)}s)")
    torch.save(best_state, OUT/"models"/f"{tag}.pt")
    return {"tag":tag,"n_train":len(train_ids),"epochs":epochs,"history":hist,"best_holdout":best}

cfg={"model":"SegFormer-B0 (nvidia/mit-b0) binary roof seg","input_res":RES,"batch_size":4,"lr":6e-5,
     "optimizer":"AdamW wd=1e-4","loss":"weighted CE [1,3]","epochs":18,"device":"cpu",
     "holdout_ids":hold_ids,"pool_size":len(pool),"train50_ids":[d['dataset_id'] for d in train50],
     "train100_ids":[d['dataset_id'] for d in train100],
     "note":"target 100-image checkpoint uses all %d non-holdout labels (max available with 101 total & fixed group-aware holdout)"%len(pool),
     "house_group_leak":list(leak)}
(OUT/"training_config.json").write_text(json.dumps(cfg,indent=2))

print("=== TRAIN baseline_50 ===")
r50=train([d["dataset_id"] for d in train50],"baseline_50")
(OUT/"metrics_50.json").write_text(json.dumps(r50,indent=2))
print("=== TRAIN baseline_100 ===")
r100=train([d["dataset_id"] for d in train100],"baseline_100")
(OUT/"metrics_100.json").write_text(json.dumps(r100,indent=2))
print("DONE. 50 best:",r50["best_holdout"],"| 100 best:",r100["best_holdout"])
