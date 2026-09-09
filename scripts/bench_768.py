"""TASK 1 — short 768 benchmark for SegFormer-B0 on current CPU (no long training)."""
import os, sys, time, resource, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
torch.set_num_threads(os.cpu_count() or 4)
print("cpu threads:", torch.get_num_threads(), "| cores:", os.cpu_count())

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0  # KB->MB on linux

def bench(res, bs, iters=4):
    m = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True)
    opt = torch.optim.AdamW(m.parameters(), lr=6e-5); m.train()
    x = torch.randn(bs,3,res,res); y = torch.randint(0,2,(bs,res,res))
    # warmup
    o=F.interpolate(m(pixel_values=x).logits,size=(res,res),mode="bilinear"); F.cross_entropy(o,y).backward(); opt.step(); opt.zero_grad()
    t=time.time()
    for _ in range(iters):
        o=F.interpolate(m(pixel_values=x).logits,size=(res,res),mode="bilinear")
        loss=F.cross_entropy(o,y); opt.zero_grad(); loss.backward(); opt.step()
    train_s=(time.time()-t)/iters
    m.eval(); t=time.time()
    with torch.no_grad():
        for _ in range(iters): F.interpolate(m(pixel_values=x).logits,size=(res,res),mode="bilinear")
    inf_s=(time.time()-t)/iters
    return train_s, inf_s

for (res,bs) in [(256,2),(768,1),(768,2)]:
    ts,inf = bench(res,bs)
    print(f"res={res} bs={bs}: train {ts:.2f}s/iter ({ts/bs:.2f}s/img) | infer {inf:.2f}s/iter ({inf/bs:.2f}s/img) | peakRSS {rss_mb():.0f}MB")
