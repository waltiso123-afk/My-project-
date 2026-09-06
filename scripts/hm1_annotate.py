"""Milestone 1 (HUMAN + SAM2 workflow, NO VLM) — annotation pass.
The human (agent) identifies each roof plane and places SAM2 point prompts from direct visual
inspection of the original photo. Real local SAM2 (CPU, Hiera-Tiny) turns each point into a mask;
the human clips/cleans it and will visually validate the overlay before approval. NO Gemini / no
VLM / no automatic roof detection is used anywhere. For 0169's genuinely ambiguous flat/parapet
areas the human draws provisional parapet-cap polygons and flags them for client clarification.
"""
import os, io, sys, json
sys.path.insert(0, "/app/backend")
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import cv2, torch
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage, ml   # ml = real SAM2; no VLM anywhere

ROOT = Path("/app/reports/MILESTONE_1_HUMAN")
FINAL = ROOT / "final"; FINAL.mkdir(parents=True, exist_ok=True)
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]

# HUMAN annotation plan — planes and SAM2 point prompts chosen by the human annotator (agent) from
# direct visual inspection. method 'sam2' = human point prompt -> SAM2; method 'poly' = human polygon
# correction (used for 0169 parapet caps where SAM2 cannot separate white-on-white parapet edges).
HUMAN = {
 "0001": {"view":"oblique","planes":[
    {"name":"left_wing_front","method":"sam2","point":[665,452],"clip":[560,415,790,505],"is_parapet":False},
    {"name":"central_upper_hip","method":"sam2","point":[955,378],"clip":[820,325,1065,470],"is_parapet":False},
    {"name":"garage_front_hip","method":"sam2","point":[1085,460],"clip":[880,398,1250,522],"is_parapet":False}],
   "occlusions":[{"type":"palm","bbox":[0.32,0.15,0.52,0.55],"hides":["plane-02","plane-03"]},
                 {"type":"palm","bbox":[0.79,0.30,0.86,0.45],"hides":["plane-03"]}],
   "notes":"Subject house only; left and right neighbour houses excluded. Palms partially cross planes (<15% image width each) so planes continue underneath per the 15% rule."},
 "0050": {"view":"frontal","planes":[
    {"name":"gable_left_slope","method":"sam2","point":[560,232],"clip":[470,206,652,264],"is_parapet":False},
    {"name":"gable_right_slope","method":"sam2","point":[705,238],"clip":[645,200,815,300],"is_parapet":False},
    {"name":"garage_front_hip","method":"sam2","point":[523,361],"clip":[360,320,665,412],"is_parapet":False}],
   "occlusions":[{"type":"tree","bbox":[0.00,0.30,0.28,0.60],"hides":["plane-01"]}],
   "notes":"Central front gable (both visible slopes) + front garage hip of the subject villa. Attached villa to the right is a neighbour and excluded. A tree partly occludes the left gable slope."},
 "0138": {"view":"oblique","planes":[
    {"name":"left_front_hip","method":"sam2","point":[263,439],"clip":[120,408,472,478],"is_parapet":False},
    {"name":"right_garage_hip","method":"sam2","point":[1097,470],"clip":[955,432,1252,508],"is_parapet":False}],
   "occlusions":[{"type":"palm","bbox":[0.40,0.35,0.62,0.75],"hides":["plane-01","plane-02"]}],
   "notes":"Long front hip roof. A large palm occludes the centre (>15% image width) so the readable front plane is delivered as two segments (left of palm and right/garage side). Neighbour house at far right excluded."},
 "0173": {"view":"oblique","planes":[
    {"name":"left_upper_hip","method":"sam2","point":[1142,1530],"clip":[600,1350,1720,1690],"is_parapet":False},
    {"name":"main_lower_hip","method":"sam2","point":[1632,2081],"clip":[700,1880,3260,2390],"is_parapet":False},
    {"name":"central_tower_front","method":"sam2","point":[2850,1020],"clip":[2450,880,3390,1190],"is_parapet":False},
    {"name":"right_upper_hip","method":"sam2","point":[4855,1000],"clip":[4100,780,5712,1320],"is_parapet":False}],
   "occlusions":[{"type":"palm","bbox":[0.54,0.29,0.86,0.85],"hides":["plane-04"]}],
   "notes":"Tile hip roof: left upper, main lower (over garage), central tower front, right upper. Palm crosses the right upper plane (~12% image width, <15%) so that plane continues underneath."},
 "0169": {"view":"frontal","planes":[
    {"name":"highest_block_parapet","method":"poly","polygon":[[1300,1055],[1955,1055],[1955,1140],[1300,1140]],"is_parapet":True},
    {"name":"left_block_parapet","method":"poly","polygon":[[305,1285],[775,1285],[775,1360],[305,1360]],"is_parapet":True},
    {"name":"main_upper_parapet","method":"poly","polygon":[[900,1258],[2650,1250],[2650,1330],[900,1345]],"is_parapet":True},
    {"name":"lower_canopy_front","method":"poly","polygon":[[700,1600],[2900,1585],[2900,1662],[700,1682]],"is_parapet":False}],
   "occlusions":[{"type":"palm","bbox":[0.00,0.05,0.13,0.40],"hides":["plane-02"]}],
   "notes":"PROVISIONAL / AMBIGUOUS — flagged for client clarification. Modern flat-roof home behind parapet walls; from ground only parapet caps and a lower canopy slab are visible, so real local SAM2 cannot separate the white-on-white parapet edges and the human drew provisional parapet-cap polygons. Open questions for the client: (a) does each flat roof behind a parapet count as its own lit plane; (b) is the louvered pergola/trellis a lit element (currently excluded); (c) the lower canopy is lit along the TOP of its dark fascia (top = drip edge), not the fascia bottom. These masks require client sign-off before scaling."},
}
COLORS=[(255,64,64),(64,160,255),(64,220,120),(255,205,40),(200,90,255),(255,130,0)]

def largest_cc(m):
    n,lab,st,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
    if n<=1: return m
    k=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA])); return np.where(lab==k,255,0).astype(np.uint8)

predictor=ml._load_sam2("cpu")
summary={}
for did,spec in HUMAN.items():
    doc=db.images.find_one({"dataset_id":did})
    raw,_=storage.get_object(doc["storage_path"])
    disp=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    W,H=disp.size; arr=np.array(disp)
    if any(p["method"]=="sam2" for p in spec["planes"]):
        with torch.inference_mode(): predictor.set_image(arr)
    fd=FINAL/did; fd.mkdir(exist_ok=True); tint=np.zeros((H,W,4),np.uint8); recs=[]
    for i,pl in enumerate(spec["planes"],start=1):
        if pl["method"]=="sam2":
            px,py=pl["point"]
            with torch.inference_mode():
                mk,sc,_=predictor.predict(point_coords=np.array([[px,py]],np.float32),
                                          point_labels=np.array([1],np.int32),multimask_output=True)
            m=(mk[int(np.argmax(sc))]>0).astype(np.uint8)*255
            cb=np.zeros((H,W),np.uint8); x0,y0,x1,y1=pl["clip"]; cb[y0:y1,x0:x1]=255
            m=cv2.bitwise_and(m,cb); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8)); m=largest_cc(m)
        else:
            m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[np.array(pl["polygon"],np.int32)],255)
        Image.fromarray(m).save(fd/f"plane_{i:02d}.png")
        col=COLORS[(i-1)%len(COLORS)]; tint[m>0]=(*col,110)
        recs.append({"plane":f"plane-{i:02d}","name":pl["name"],"is_parapet":pl["is_parapet"],
                     "method":pl["method"],"area_frac":round(float(m.sum()/255/(H*W)),4)})
    ov=Image.alpha_composite(disp.convert("RGBA"),Image.fromarray(tint)); o=np.array(ov)
    for i in range(1,len(spec["planes"])+1):
        m=np.array(Image.open(fd/f"plane_{i:02d}.png").convert("L")); col=COLORS[(i-1)%len(COLORS)]
        c,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(o,c,-1,(*col,255),max(2,W//600))
    oi=Image.fromarray(o).convert("RGB"); oi.thumbnail((1600,1600)); oi.save(FINAL/f"{did}_overlay.jpg",quality=74)
    summary[did]={"display_size":[W,H],"n_planes":len(recs),"planes":recs}
    print(f"{did}: {len(recs)} planes areas={[r['area_frac'] for r in recs]}")
(FINAL/"summary.json").write_text(json.dumps(summary,indent=2))
print("HUMAN annotation masks ->",FINAL)
