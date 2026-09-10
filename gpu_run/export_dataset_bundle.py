"""Build a FROZEN, self-contained dataset bundle for the RunPod GPU run.
Runs in THIS environment only (reads Mongo + object storage). READ-ONLY w.r.t. production
annotations. Output: /app/gpu_run/dataset_bundle/ (images/, masks/, split.json, manifest.json).

Holdout = the FIXED 13 ids (client mandate). Training pool = all other approved images whose
house_group_id does NOT collide with any holdout group (group-aware, no leakage)."""
import os, io, sys, json, hashlib
sys.path.insert(0, "/app/backend")
from dotenv import load_dotenv; load_dotenv("/app/backend/.env")
from pathlib import Path
from PIL import Image
import numpy as np
import storage, config, pymongo

db = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]]
FROZEN13 = ["0016","0023","0031","0032","0037","0044","0051","0054","0058","0064","0085","0120","0138"]
OUT = Path("/app/gpu_run/dataset_bundle")
(OUT/"images").mkdir(parents=True, exist_ok=True)
(OUT/"masks").mkdir(parents=True, exist_ok=True)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    appr = sorted(db.images.find({"status":"approved"}), key=lambda d: d["dataset_id"])
    by_id = {d["dataset_id"]: d for d in appr}
    assert all(f in by_id for f in FROZEN13), "a frozen holdout id is missing from approved set!"
    hold_groups = set(by_id[f].get("house_group_id") for f in FROZEN13)

    holdout, pool, excluded = list(FROZEN13), [], []
    for d in appr:
        did = d["dataset_id"]
        if did in FROZEN13: continue
        if d.get("house_group_id") in hold_groups:
            excluded.append(did)          # would leak into holdout groups -> exclude from training
        else:
            pool.append(did)

    manifest = {"files": {}, "counts": {}}
    def save(did):
        d = by_id[did]
        raw,_ = storage.get_object(d["storage_path"])
        Image.open(io.BytesIO(raw)).convert("RGB").save(OUT/"images"/f"{did}.png")
        md,_ = storage.get_object(config.merged_path(did))
        m = (np.array(Image.open(io.BytesIO(md)).convert("L")) > 127).astype(np.uint8) * 255
        Image.fromarray(m, "L").save(OUT/"masks"/f"{did}.png")
        manifest["files"][f"images/{did}.png"] = sha(OUT/"images"/f"{did}.png")
        manifest["files"][f"masks/{did}.png"]  = sha(OUT/"masks"/f"{did}.png")

    for did in holdout + pool:
        save(did)

    split = {
        "holdout": holdout,                 # FIXED 13 (client mandate)
        "pool": sorted(pool),               # training pool (group-aware, no holdout leakage)
        "excluded_group_overlap": sorted(excluded),
        "house_groups": {d["dataset_id"]: d.get("house_group_id") for d in appr},
        "note": "Holdout=frozen 13. Pool excludes any image sharing a house_group with the holdout.",
    }
    (OUT/"split.json").write_text(json.dumps(split, indent=2))
    manifest["files"]["split.json"] = sha(OUT/"split.json")
    manifest["counts"] = {"approved_total": len(appr), "holdout": len(holdout),
                          "pool_train": len(pool), "excluded_group_overlap": len(excluded)}
    manifest["bundle_sha256"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2))

    print("counts:", manifest["counts"])
    print("excluded (group overlap w/ holdout):", excluded)
    print("bundle_sha256:", manifest["bundle_sha256"])
    print("bundle at:", OUT)

if __name__ == "__main__":
    main()
