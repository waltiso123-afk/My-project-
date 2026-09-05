"""Finalize: zip the client-review package, checksum, upload to object storage, register manifest."""
import os, sys, json, hashlib, zipfile
sys.path.insert(0, "/app/backend")
from pathlib import Path
from datetime import datetime, timezone
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage

ROOT = Path("/app/reports/client_review/PILOT_5_CLIENT_REVIEW")
ZIP = Path("/app/reports/client_review/PILOT_5_CLIENT_REVIEW.zip")
STORAGE_PATH = "roofline/deliverables/PILOT_5_CLIENT_REVIEW.zip"

if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(ROOT.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(ROOT.parent))  # keep top folder PILOT_5_CLIENT_REVIEW/

data = ZIP.read_bytes()
size = len(data)
sha = hashlib.sha256(data).hexdigest()
storage.put_object(STORAGE_PATH, data, "application/zip")

n_images = sum(1 for _ in (ROOT / "originals").iterdir())
n_planes = sum(1 for _ in (ROOT / "planes").rglob("plane-*.png"))
specials = sorted(p.name for p in (ROOT / "special_cases").iterdir() if p.is_dir())

backend_url = None
for line in Path("/app/frontend/.env").read_text().splitlines():
    if line.startswith("REACT_APP_BACKEND_URL="):
        backend_url = line.split("=", 1)[1].strip()

manifest = {
    "_id": "pilot-5-client-review",
    "filename": "PILOT_5_CLIENT_REVIEW.zip",
    "storage_path": STORAGE_PATH,
    "size_bytes": size,
    "sha256": sha,
    "image_count": n_images,
    "planes_total": n_planes,
    "special_cases": specials,
    "download_api": "GET /api/deliverables/pilot-5/download",
    "info_api": "GET /api/deliverables/pilot-5/info",
    "download_url": (backend_url.rstrip("/") + "/api/deliverables/pilot-5/download") if backend_url else None,
    "built_at": datetime.now(timezone.utc).isoformat(),
    "note": "SAM2-assisted pilot proposals pending human review + client confirmation. Not approved/final.",
}
c = pymongo.MongoClient(os.environ["MONGO_URL"]); dbm = c[os.environ["DB_NAME"]]
dbm.deliverables.replace_one({"_id": manifest["_id"]}, manifest, upsert=True)

print(json.dumps({k: manifest[k] for k in
      ["filename","storage_path","size_bytes","sha256","image_count","planes_total","special_cases","download_url"]}, indent=2))
