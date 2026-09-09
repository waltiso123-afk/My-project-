"""TASK 2 — per-plane roofline evaluation pipeline (parapet-aware).
Evaluates EACH roof plane separately (not only the merged mask).
- normal plane   -> LOWER eave boundary (max-y per column)
- parapet plane  -> UPPER parapet boundary (min-y per column)  [from -parapet filename / meta.parapets]
Metrics per plane: mean perpendicular roofline error (normalized by image width), horizontal coverage.
Dry-run uses the archived baseline_100 binary model (interim: prediction restricted to each GT plane's
column span). NOTE: true per-plane PREDICTION needs an instance/multi-class model (v2 to add); the
GROUND-TRUTH per-plane extraction + metric framework below is what the 150-checkpoint will use."""
import os, io, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import storage, config, pymongo
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
RES = 256
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

def boundary(mask, upper):
    """per-column boundary y; upper=True -> min-y (parapet top), else max-y (eave bottom)."""
    ys={}
    for x in range(mask.shape[1]):
        col=np.where(mask[:,x]>0)[0]
        if len(col): ys[x]= int(col.min()) if upper else int(col.max())
    return ys

def plane_metrics(gt_mask, pred_roof, upper, W):
    gb=boundary(gt_mask, upper)
    if not gb: return None
    cols=sorted(gb); x0,x1=cols[0],cols[-1]
    pred_span=pred_roof[:, x0:x1+1]
    pb=boundary(pred_span, upper); pb={x0+k:v for k,v in pb.items()}
    errs=[abs(gb[x]-pb[x]) for x in gb if x in pb]
    perp=(np.mean(errs)/W) if errs else None
    cov=len([x for x in gb if x in pb])/len(gb)
    return {"perp_err_normwidth": round(float(perp),5) if perp is not None else None,
            "coverage": round(float(cov),4), "gt_cols": len(gb)}

def load_img(did):
    raw,_=storage.get_object(db.images.find_one({"dataset_id":did})["storage_path"])
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB").resize((RES,RES)))

def predict_roof(net, im):
    x=torch.from_numpy(((im/255.0-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32))
    with torch.no_grad():
        up=F.interpolate(net(pixel_values=x).logits,size=(RES,RES),mode="bilinear",align_corners=False)
    return up.argmax(1)[0].numpy().astype(np.uint8)

def eval_dataset(ids, net):
    per_img={}; all_perp=[]; all_cov=[]; n_planes=0; n_parapet=0
    for did in ids:
        ann=db.annotations.find_one({"dataset_id":did})
        im=load_img(did); pred=predict_roof(net, im)
        rows=[]
        for g in ann["generated"]["planes"]:
            md,_=storage.get_object(g)
            gm=np.array(Image.open(io.BytesIO(md)).convert("L").resize((RES,RES),Image.NEAREST)); gm=(gm>127).astype(np.uint8)
            is_par = g.endswith("-parapet.png")
            m=plane_metrics(gm, pred, upper=is_par, W=RES)
            if m is None: continue
            n_planes+=1; n_parapet+= int(is_par)
            if m["perp_err_normwidth"] is not None: all_perp.append(m["perp_err_normwidth"])
            all_cov.append(m["coverage"])
            rows.append({"plane":g.split("/")[-1],"is_parapet":is_par,**m})
        per_img[did]={"n_planes":len(rows),"planes":rows}
    return {"per_image":per_img,
            "dataset_per_plane":{"mean_perp_err_normwidth":round(float(np.mean(all_perp)),5) if all_perp else None,
                                 "mean_horizontal_coverage":round(float(np.mean(all_cov)),4) if all_cov else None,
                                 "n_valid_plane_instances":n_planes,"n_parapet_planes":n_parapet}}

if __name__=="__main__":
    hold=[d["dataset_id"] for d in db.images.find({"status":"approved","split":"holdout"})]
    net=SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",num_labels=2,ignore_mismatched_sizes=True)
    net.load_state_dict(torch.load("/app/reports/milestone1_step1/models/baseline_100.pt",map_location="cpu")); net.eval()
    res=eval_dataset(sorted(hold), net)
    out=Path("/app/reports/milestone1_checkpoint_v2_768_per_plane/per_plane_eval_dryrun.json")
    out.write_text(json.dumps(res,indent=2))
    print("holdout:",sorted(hold))
    print("DATASET per-plane:",json.dumps(res["dataset_per_plane"]))
    print("saved",out)
