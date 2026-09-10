"""REAL GPU training run — SegFormer-B0 AND SegFormer-B1 @768 binary roof/bg.
Reuses the validated train_step1.py methodology (AdamW 6e-5 / wd 1e-4, weighted CE [1,3],
seed 0, 18 epochs, bs 4, group-aware fixed holdout). Self-contained: consumes the frozen
dataset bundle (no Mongo/object-storage). Device-agnostic (cuda if available).

Per client spec:
 - both architectures, SAME dataset/split/schedule/eval protocol
 - single-pass eval AND horizontal-flip averaging eval
 - save raw logits + probability maps for ALL holdout images
 - checkpoint SELECTION by MEAN PERPENDICULAR ROOFLINE ERROR (minimize) = PRIMARY
   (coverage secondary; Dice/IoU diagnostic).  <-- client-directed change vs train_step1's
   dice-based selection; documented, not silent.
 - ONNX export (opset17, static 1x3x768x768 -> 1x2x768x768, in-graph upsample).
"""
import os, io, json, time, random, argparse, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from export_onnx import export_onnx
import warnings; warnings.filterwarnings("ignore")

torch.manual_seed(0); np.random.seed(0); random.seed(0)
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)
W_CE = [1.0, 3.0]
MODELS = {"b0": "nvidia/mit-b0", "b1": "nvidia/mit-b1"}

def load_bundle(data_dir, res):
    data_dir = Path(data_dir)
    split = json.loads((data_dir/"split.json").read_text())
    cache = {}
    for did in split["holdout"] + split["pool"]:
        im = np.array(Image.open(data_dir/"images"/f"{did}.png").convert("RGB").resize((res,res), Image.BILINEAR))
        mk = np.array(Image.open(data_dir/"masks"/f"{did}.png").convert("L").resize((res,res), Image.NEAREST))
        cache[did] = (im.astype(np.uint8), (mk>127).astype(np.uint8))
    return split, cache

def batchify(cache, ids, dev):
    x = np.stack([((cache[i][0]/255.0 - MEAN)/STD).transpose(2,0,1) for i in ids]).astype(np.float32)
    y = np.stack([cache[i][1] for i in ids]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

def roofline_bottom(mask):
    ys = {}
    for x in range(mask.shape[1]):
        col = np.where(mask[:, x] > 0)[0]
        if len(col): ys[x] = int(col.max())
    return ys

def metrics_from_pred(pred, gt, res):
    inter = np.logical_and(pred, gt).sum(); union = np.logical_or(pred, gt).sum()
    dice = (2*inter)/(pred.sum()+gt.sum()) if (pred.sum()+gt.sum()) else 0.0
    iou = inter/union if union else 0.0
    gl = roofline_bottom(gt); pl = roofline_bottom(pred)
    perp = cov = None
    if gl:
        errs = [abs(gl[x]-pl[x]) for x in gl if x in pl]
        perp = (np.mean(errs)/res) if errs else None
        cov = len([x for x in gl if x in pl])/len(gl)
    return dice, iou, perp, cov

def _probs(model, x, res):
    up = F.interpolate(model(pixel_values=x).logits, size=(res,res), mode="bilinear", align_corners=False)
    return torch.softmax(up, dim=1)

def evaluate(model, cache, ids, dev, res, tta=False, save_dir=None):
    model.eval(); dl=[]; il=[]; pl=[]; cl=[]
    logits_manifest = {}
    with torch.no_grad():
        for i in ids:
            x, y = batchify(cache, [i], dev)
            up = F.interpolate(model(pixel_values=x).logits, size=(res,res), mode="bilinear", align_corners=False)
            prob = torch.softmax(up, 1)
            if tta:
                up_f = F.interpolate(model(pixel_values=torch.flip(x, dims=[3])).logits, size=(res,res),
                                     mode="bilinear", align_corners=False)
                prob = (prob + torch.flip(torch.softmax(up_f, 1), dims=[3])) / 2.0
            pred = prob.argmax(1)[0].cpu().numpy().astype(np.uint8)
            gt = y[0].cpu().numpy().astype(np.uint8)
            d,iou,perp,cov = metrics_from_pred(pred, gt, res)
            dl.append(d); il.append(iou)
            if perp is not None: pl.append(perp)
            if cov is not None: cl.append(cov)
            if save_dir is not None:
                np.save(Path(save_dir)/f"{i}_logits.npy", up[0].cpu().numpy().astype(np.float16))
                np.save(Path(save_dir)/f"{i}_probs.npy", prob[0].cpu().numpy().astype(np.float16))
                logits_manifest[i] = {"logits": f"{i}_logits.npy", "probs": f"{i}_probs.npy",
                                      "shape": list(up[0].shape)}
    out = {"n": len(ids),
           "dice": round(float(np.mean(dl)),4), "iou": round(float(np.mean(il)),4),
           "mean_perp_roofline_err_normwidth": round(float(np.mean(pl)),5) if pl else None,
           "horizontal_coverage": round(float(np.mean(cl)),4) if cl else None}
    if save_dir is not None: out["logits_manifest"] = logits_manifest
    return out

def train_one(hf_name, tag, cache, split, dev, res, epochs, bs, lr, out):
    model = SegformerForSemanticSegmentation.from_pretrained(hf_name, num_labels=2, ignore_mismatched_sizes=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor(W_CE, device=dev)
    train_ids = list(split["pool"]); hold = split["holdout"]
    hist=[]; best={"perp": float("inf")}; best_state=None; t0=time.time()
    for ep in range(1, epochs+1):
        model.train(); random.shuffle(train_ids); tl=0.0; nb=0
        for k in range(0, len(train_ids), bs):
            ids = train_ids[k:k+bs]; x,y = batchify(cache, ids, dev)
            up = F.interpolate(model(pixel_values=x).logits, size=(res,res), mode="bilinear", align_corners=False)
            loss = F.cross_entropy(up, y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        ev = evaluate(model, cache, hold, dev, res, tta=False)     # single-pass for selection
        row = {"epoch": ep, "train_loss": round(tl/nb,4), **ev, "elapsed_s": int(time.time()-t0)}
        hist.append(row)
        # PRIMARY selection: MIN mean perpendicular roofline error
        cur = ev["mean_perp_roofline_err_normwidth"]
        if cur is not None and cur < best["perp"]:
            best = {"perp": cur, "epoch": ep, **ev}
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"[{tag}] ep{ep}/{epochs} loss={row['train_loss']} perp={cur} cov={ev['horizontal_coverage']} "
              f"dice={ev['dice']} iou={ev['iou']} ({row['elapsed_s']}s)")
    (out/"models").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out/"models"/f"{tag}_best.pt")
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, out/"models"/f"{tag}_last.pt")
    # reload BEST for final eval + logits dump + onnx
    model.load_state_dict({k: v.to(dev) for k,v in best_state.items()}); model.eval()
    hold_dir = out/"holdout_outputs"/tag; hold_dir.mkdir(parents=True, exist_ok=True)
    single = evaluate(model, cache, hold, dev, res, tta=False, save_dir=hold_dir)
    hflip  = evaluate(model, cache, hold, dev, res, tta=True)
    onnx_path = out/"onnx"/f"segformer_{tag}_768_binary.onnx"; onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(hf_name, str(out/"models"/f"{tag}_best.pt"), str(onnx_path), opset=17)
    result = {"tag": tag, "hf_name": hf_name, "n_train": len(train_ids), "epochs": epochs, "bs": bs, "lr": lr,
              "selected_epoch_by_MIN_perp": best.get("epoch"),
              "best_single_pass": {k: best[k] for k in ["mean_perp_roofline_err_normwidth","horizontal_coverage","dice","iou"] if k in best},
              "final_single_pass_eval": {k:v for k,v in single.items() if k!="logits_manifest"},
              "final_hflip_avg_eval": hflip,
              "onnx": str(onnx_path), "onnx_size_mb": round(onnx_path.stat().st_size/1e6,2),
              "history": hist}
    (out/f"metrics_{tag}.json").write_text(json.dumps(result, indent=2))
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_bundle")
    ap.add_argument("--out", default="run_outputs")
    ap.add_argument("--models", default="b0,b1")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device=="auto" else a.device
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print("device:", dev, "| torch:", torch.__version__, "| cuda_avail:", torch.cuda.is_available())
    if dev.startswith("cuda"): print("GPU:", torch.cuda.get_device_name(0))
    split, cache = load_bundle(a.data, a.res)
    print("holdout:", len(split["holdout"]), "pool:", len(split["pool"]))
    run = {"device": dev, "res": a.res, "epochs": a.epochs, "bs": a.bs, "lr": a.lr,
           "optimizer": "AdamW wd=1e-4", "loss": "weighted CE [1,3]", "seed": 0,
           "selection_metric": "mean_perpendicular_roofline_error_normwidth (MINIMIZE, PRIMARY)",
           "secondary": "horizontal_coverage", "diagnostics": "dice,iou",
           "holdout": split["holdout"], "pool_size": len(split["pool"]), "results": {}}
    t0 = time.time()
    for m in a.models.split(","):
        m = m.strip(); assert m in MODELS, m
        if dev.startswith("cuda"): torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        r = train_one(MODELS[m], m, cache, split, dev, a.res, a.epochs, a.bs, a.lr, out)
        if dev.startswith("cuda"): r["peak_vram_mb"] = round(torch.cuda.max_memory_allocated()/1e6,1)
        run["results"][m] = r
    run["total_runtime_s"] = round(time.time()-t0,1)
    (out/"run_summary.json").write_text(json.dumps(run, indent=2))
    print("DONE. total", run["total_runtime_s"], "s ->", out/"run_summary.json")

if __name__ == "__main__":
    main()
