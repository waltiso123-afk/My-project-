"""Fetch the 5 pilot originals (EXIF-applied display space) to /app/reports/pilot/orig."""
import os, io, sys
sys.path.insert(0, "/app/backend")
from pathlib import Path
from PIL import Image, ImageOps
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage

OUT = Path("/app/reports/pilot/orig"); OUT.mkdir(parents=True, exist_ok=True)
c = pymongo.MongoClient(os.environ["MONGO_URL"]); db = c[os.environ["DB_NAME"]]

IDS = ["0001", "0169", "0050", "0138", "0173"]
for did in IDS:
    img = db.images.find_one({"dataset_id": did})
    data, ct = storage.get_object(img["storage_path"])
    disp = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    # save a viewing copy scaled to max 1400px for inspection
    view = disp.copy()
    view.thumbnail((1400, 1400))
    view.save(OUT / f"{did}_view.png")
    print(f"{did}: batch={img['batch']} exif={img.get('exif_orientation')} "
          f"raw=({img['raw_width']}x{img['raw_height']}) db_size=({img.get('width')}x{img.get('height')}) "
          f"display=({disp.width}x{disp.height}) view=({view.width}x{view.height}) split={img['split']}")
print("done")
