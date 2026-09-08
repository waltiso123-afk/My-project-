"""Build contact-sheet montages of all approved images (upright) to spot flat-roof/parapet homes."""
import os, io, sys, math
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("backend/.env")
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import storage, pymongo
db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
OUT = Path("/app/reports/contact"); OUT.mkdir(parents=True, exist_ok=True)

imgs = sorted(db.images.find({"status": "approved"}), key=lambda d: d["dataset_id"])
def font(s):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, s)
    return ImageFont.load_default()
F = font(20)
CELL, BAR, COLS = 240, 26, 7
PER = 49  # per sheet (7x7)
thumbs = []
for d in imgs:
    try:
        raw, _ = storage.get_object(d["storage_path"])
        im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((CELL, CELL))
        thumbs.append((d["dataset_id"], d.get("batch", ""), im))
    except Exception as e:
        print("skip", d["dataset_id"], str(e)[:40])
print("thumbs:", len(thumbs))
for s in range(math.ceil(len(thumbs) / PER)):
    chunk = thumbs[s*PER:(s+1)*PER]
    rows = math.ceil(len(chunk) / COLS)
    W = COLS * CELL; H = rows * (CELL + BAR)
    sheet = Image.new("RGB", (W, H), (18, 22, 30)); dr = ImageDraw.Draw(sheet)
    for i, (did, batch, im) in enumerate(chunk):
        cx, cy = (i % COLS) * CELL, (i // COLS) * (CELL + BAR)
        sheet.paste(im, (cx + (CELL - im.width)//2, cy + BAR + (CELL - im.height)//2))
        dr.rectangle([cx, cy, cx+CELL, cy+BAR], fill=(30, 40, 55))
        dr.text((cx+4, cy+3), f"{did} [{batch[:2]}]", fill=(120, 230, 255), font=F)
    sheet.save(OUT / f"contact_{s+1}.jpg", quality=72)
    print("saved", OUT / f"contact_{s+1}.jpg", f"({len(chunk)} imgs)")
