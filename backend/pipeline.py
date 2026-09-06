"""Ingestion + indexation + house grouping + anti-leakage split.

Real, resumable pipeline (idempotent per dataset_id). Uses pymongo (sync) so it can be
run both from a CLI script and imported. No fabrication: every value is measured from the
actual image files and stored in MongoDB + dataset_index.csv.
"""
import csv
import hashlib
import io
import os
from datetime import datetime, timezone

import imagehash
import cv2
import numpy as np
from PIL import Image, ImageOps
from pymongo import MongoClient

import config
import storage

Image.MAX_IMAGE_PIXELS = None  # large aerial images


def _db():
    client = MongoClient(os.environ["MONGO_URL"])
    return client[os.environ["DB_NAME"]]


def _collect_files():
    """Return ordered list of (batch, original_path) preserving original names."""
    files = []
    b1 = sorted([p for p in config.STAGING_BATCH1.glob("*") if p.is_file()],
                key=lambda p: p.name)
    b2 = sorted([p for p in config.STAGING_BATCH2.glob("*") if p.is_file()],
                key=lambda p: p.name)
    for p in b1:
        files.append(("batch1", p))
    for p in b2:
        files.append(("batch2", p))
    return files


def _validate_image(path):
    """Open + verify. Returns dict with raw + EXIF-corrected (display) dimensions.
    Browsers auto-apply EXIF orientation, so display dims are the canonical coordinate
    space for annotation + masks. Backend must exif-transpose to match."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            raw_w, raw_h = im.size
            mode = im.mode
            exif = im.getexif()
            orient_tag = exif.get(274) if exif else None  # 274 = Orientation
            corrected = ImageOps.exif_transpose(im)
            disp_w, disp_h = corrected.size
        transposed = (disp_w, disp_h) != (raw_w, raw_h)
        return {"ok": True, "raw_w": raw_w, "raw_h": raw_h,
                "disp_w": disp_w, "disp_h": disp_h, "mode": mode,
                "exif_orientation": int(orient_tag) if orient_tag else None,
                "exif_transposed": transposed, "reason": None}
    except Exception as e:
        return {"ok": False, "raw_w": None, "raw_h": None, "disp_w": None, "disp_h": None,
                "mode": None, "exif_orientation": None, "exif_transposed": False, "reason": str(e)}


def _to_upright(raw: bytes, ext: str):
    """Phase 1: return EXIF-applied, EXIF-stripped UPRIGHT image bytes + dims so the stored
    production file matches the display/mask coordinate space (file space == display space)."""
    fmt = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}.get(ext.lower(), "JPEG")
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    if fmt in ("JPEG", "WEBP"):
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format=fmt, **({"quality": 95} if fmt in ("JPEG", "WEBP") else {}))
    return buf.getvalue(), im.width, im.height


def run_ingest(log=print):
    db = _db()
    files = _collect_files()
    log(f"[ingest] discovered {len(files)} files "
        f"(batch1={sum(1 for b,_ in files if b=='batch1')}, "
        f"batch2={sum(1 for b,_ in files if b=='batch2')})")

    seen_hashes = {}
    for i, (batch, path) in enumerate(files, start=1):
        dataset_id = f"{i:04d}"
        ext = path.suffix.lower().lstrip(".")
        existing = db.images.find_one({"dataset_id": dataset_id}, {"_id": 0, "storage_uploaded": 1})
        v = _validate_image(path)
        ok = v["ok"]
        w, h = v["disp_w"], v["disp_h"]
        corrupt = v["reason"]
        raw = path.read_bytes()
        sha = hashlib.sha1(raw).hexdigest()
        duplicate_of = seen_hashes.get(sha)
        seen_hashes.setdefault(sha, dataset_id)

        phash = None
        if ok:
            try:
                with Image.open(path) as im:
                    phash = str(imagehash.phash(ImageOps.exif_transpose(im).convert("RGB"), hash_size=8))
            except Exception:
                phash = None

        orientation = None
        if ok:
            orientation = "portrait" if h > w else ("landscape" if w > h else "square")

        # Upload original to object storage (resumable: skip if already uploaded)
        uploaded = existing.get("storage_uploaded") if existing else False
        sp = config.img_path(dataset_id, ext)
        upload_error = None
        store_w, store_h, exif_norm = w, h, 1
        orig_raw_size = [v["raw_w"], v["raw_h"]]
        if ok and not uploaded:
            ct = config.MIME_TYPES.get(ext, "application/octet-stream")
            try:
                up_bytes, store_w, store_h = _to_upright(raw, ext)  # Phase 1: store upright
                storage.put_object(sp, up_bytes, ct)
                uploaded = True
            except Exception as e:
                upload_error = str(e)
                log(f"[ingest] upload FAILED {dataset_id}: {e}")

        doc = {
            "dataset_id": dataset_id,
            "batch": batch,
            "original_filename": path.name,
            "ext": ext,
            "content_type": config.MIME_TYPES.get(ext, "application/octet-stream"),
            "storage_path": sp,
            "storage_uploaded": uploaded,
            "upload_error": upload_error,
            "width": w, "height": h, "mode": v["mode"],
            "raw_width": store_w, "raw_height": store_h,
            "exif_orientation": exif_norm,
            "exif_transposed": v["exif_transposed"],
            "coordinate_policy": "upright_normalized",
            "original_raw_size": orig_raw_size,
            "orientation": orientation,
            "file_size_bytes": len(raw),
            "sha1": sha,
            "duplicate_of": duplicate_of,
            "phash": phash,
            "corrupted": (not ok),
            "corrupt_reason": corrupt,
            "status": (existing or {}).get("status", "pending") if existing else "pending",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        db.images.update_one({"dataset_id": dataset_id}, {"$set": doc}, upsert=True)
        if i % 25 == 0:
            log(f"[ingest] {i}/{len(files)} processed")

    log(f"[ingest] done. corrupted={db.images.count_documents({'corrupted': True})}, "
        f"duplicates={db.images.count_documents({'duplicate_of': {'$ne': None}})}")
    return db


def _orb_features(path, max_dim=1000, nfeatures=1200):
    """Detect ORB keypoints/descriptors on the EXIF-corrected, downscaled grayscale image."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    w, h = im.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    arr = np.array(im)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(arr, None)
    return kp, des


# Same-property decision thresholds (RANSAC-homography inliers on ORB matches)
ORB_STRONG_INLIERS = 22   # >= -> auto-group (still flagged uncertain for human confirm)
ORB_CANDIDATE_INLIERS = 12  # >= -> surface as review candidate
ORB_RATIO = 0.75


def build_house_groups(db=None, log=print):
    """Detect images showing the SAME physical property (multi-view) using ORB local
    features + RANSAC homography geometric verification. This is robust to viewpoint /
    framing changes (unlike perceptual hashing). Strong geometric matches are auto-grouped
    and flagged uncertain=True for human confirmation; borderline pairs are stored as
    review candidates. No silent decisions."""
    if db is None:
        db = _db()
    imgs = list(db.images.find({"corrupted": False}, {"_id": 0, "dataset_id": 1, "batch": 1,
                                                      "original_filename": 1}))
    imgs.sort(key=lambda d: d["dataset_id"])
    ids = [d["dataset_id"] for d in imgs]
    path_by_id = {}
    for d in imgs:
        sub = "batch1" if d["batch"] == "batch1" else "batch2"
        path_by_id[d["dataset_id"]] = (config.DATA_DIR / "staging" / sub / d["original_filename"])

    log(f"[groups] extracting ORB features for {len(ids)} images…")
    feats = {}
    for k, did in enumerate(ids, 1):
        try:
            feats[did] = _orb_features(path_by_id[did])
        except Exception as e:
            feats[did] = (None, None)
            log(f"[groups] ORB failed {did}: {e}")
        if k % 40 == 0:
            log(f"[groups] features {k}/{len(ids)}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    db.group_pairs.delete_many({})
    n = len(ids)
    pair_count = 0
    for i in range(n):
        kp_i, des_i = feats[ids[i]]
        if des_i is None or len(des_i) < 8:
            continue
        for j in range(i + 1, n):
            kp_j, des_j = feats[ids[j]]
            if des_j is None or len(des_j) < 8:
                continue
            matches = bf.knnMatch(des_i, des_j, k=2)
            good = [m for m, nn in (pair for pair in matches if len(pair) == 2)
                    if m.distance < ORB_RATIO * nn.distance]
            if len(good) < 10:
                continue
            src = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_j[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
            if inliers < ORB_CANDIDATE_INLIERS:
                continue
            a, b = ids[i], ids[j]
            db.group_pairs.insert_one({
                "image_a": a, "image_b": b, "inliers": inliers,
                "good_matches": len(good),
                "batch_a": imgs[i]["batch"], "batch_b": imgs[j]["batch"],
                "strong": inliers >= ORB_STRONG_INLIERS,
            })
            pair_count += 1
            if inliers >= ORB_STRONG_INLIERS:
                union(a, b)
        if (i + 1) % 25 == 0:
            log(f"[groups] matched rows {i + 1}/{n} (pairs found={pair_count})")

    clusters = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)

    db.house_groups.delete_many({})
    group_records = []
    for gi, (root, members) in enumerate(sorted(clusters.items()), start=1):
        members = sorted(members)
        gid = f"HG-{gi:04d}"
        multi = len(members) > 1
        evid = {}
        if multi:
            for gp in db.group_pairs.find({"image_a": {"$in": members},
                                           "image_b": {"$in": members}}, {"_id": 0}):
                evid[f"{gp['image_a']}|{gp['image_b']}"] = gp["inliers"]
        max_in = max(evid.values()) if evid else None
        rec = {
            "group_id": gid,
            "image_ids": members,
            "size": len(members),
            "uncertain": multi,
            "human_confirmed": False,
            "confidence": max_in,
            "method": "orb_features + ransac_homography_inliers",
            "strong_inlier_threshold": ORB_STRONG_INLIERS,
            "evidence_pair_inliers": evid,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        group_records.append(rec)
        db.house_groups.insert_one(dict(rec))
        for m in members:
            db.images.update_one({"dataset_id": m}, {"$set": {"house_group_id": gid}})

    multi_groups = [g for g in group_records if g["size"] > 1]
    log(f"[groups] {len(group_records)} house_groups "
        f"({len(multi_groups)} multi-view groups auto-detected, "
        f"{len(group_records) - len(multi_groups)} singletons); "
        f"{db.group_pairs.count_documents({})} evidence pairs stored")
    return group_records


def build_split(db=None, log=print):
    """Assign each house_group deterministically to train/val/holdout. Split is at
    house_group level -> a group can never span splits -> zero leakage by construction.
    Holdout is FIXED (deterministic seed hash)."""
    if db is None:
        db = _db()
    groups = list(db.house_groups.find({}, {"_id": 0, "group_id": 1, "image_ids": 1}))

    def bucket(g):
        # Seed on the stable smallest member dataset_id (content-based), NOT the group
        # sequence number, so the holdout stays FIXED even if group ids get renumbered.
        key = min(g["image_ids"])
        h = hashlib.sha256(f"{config.SPLIT_SEED}:{key}".encode()).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        if frac < config.SPLIT_RATIOS["train"]:
            return "train"
        if frac < config.SPLIT_RATIOS["train"] + config.SPLIT_RATIOS["val"]:
            return "val"
        return "holdout"

    counts = {"train": 0, "val": 0, "holdout": 0}
    img_counts = {"train": 0, "val": 0, "holdout": 0}
    for g in groups:
        s = bucket(g)
        counts[s] += 1
        img_counts[s] += len(g["image_ids"])
        db.house_groups.update_one({"group_id": g["group_id"]},
                                   {"$set": {"split": s, "split_key": min(g["image_ids"])}})
        for m in g["image_ids"]:
            db.images.update_one({"dataset_id": m}, {"$set": {"split": s}})

    split_doc = {
        "id": "split_config",
        "seed": config.SPLIT_SEED,
        "ratios": config.SPLIT_RATIOS,
        "level": "house_group",
        "holdout_fixed": True,
        "split_key": "min(image_ids) per group (content-stable)",
        "group_counts": counts,
        "image_counts": img_counts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.splits.update_one({"id": "split_config"}, {"$set": split_doc}, upsert=True)
    log(f"[split] groups={counts} images={img_counts}")
    return split_doc


def write_index_csv(db=None, log=print):
    if db is None:
        db = _db()
    rows = list(db.images.find({}, {"_id": 0}).sort("dataset_id", 1))
    cols = ["dataset_id", "batch", "original_filename", "ext", "storage_path",
            "width", "height", "orientation", "corrupted", "corrupt_reason",
            "file_size_bytes", "sha1", "duplicate_of", "phash",
            "house_group_id", "split", "status"]
    with open(config.DATASET_INDEX_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    log(f"[index] wrote {config.DATASET_INDEX_CSV} ({len(rows)} rows)")
    return str(config.DATASET_INDEX_CSV)


def compute_candidates(db=None, min_inliers=12, top=120):
    """Return ORB+homography evidence pairs (same-property candidates) sorted by inlier
    count. Strong pairs (>= ORB_STRONG_INLIERS) were auto-grouped; borderline pairs are
    for human confirmation. Real geometric evidence, not perceptual hashing."""
    if db is None:
        db = _db()
    grp = {d["dataset_id"]: d.get("house_group_id")
           for d in db.images.find({}, {"_id": 0, "dataset_id": 1, "house_group_id": 1})}
    pairs = []
    for gp in db.group_pairs.find({"inliers": {"$gte": min_inliers}}, {"_id": 0}).sort("inliers", -1).limit(top):
        pairs.append({
            "image_a": gp["image_a"], "image_b": gp["image_b"],
            "inliers": gp["inliers"], "good_matches": gp.get("good_matches"),
            "strong": gp.get("strong", False),
            "batch_a": gp.get("batch_a"), "batch_b": gp.get("batch_b"),
            "group_a": grp.get(gp["image_a"]), "group_b": grp.get(gp["image_b"]),
            "same_group": grp.get(gp["image_a"]) == grp.get(gp["image_b"]),
        })
    return pairs


def merge_images_into_group(image_ids, db=None, human=True):
    """Merge the given images into one human-confirmed house_group, then recompute split."""
    if db is None:
        db = _db()
    image_ids = sorted(set(image_ids))
    old_groups = set()
    for im in db.images.find({"dataset_id": {"$in": image_ids}}, {"_id": 0, "house_group_id": 1}):
        if im.get("house_group_id"):
            old_groups.add(im["house_group_id"])

    n_merged = db.house_groups.count_documents({"method": "human_merge"})
    new_gid = f"HG-M{n_merged + 1:03d}"
    from datetime import datetime, timezone as _tz
    db.house_groups.insert_one({
        "group_id": new_gid, "image_ids": image_ids, "size": len(image_ids),
        "uncertain": False, "human_confirmed": True, "confidence": 1.0,
        "method": "human_merge",
        "created_at": datetime.now(_tz.utc).isoformat(),
    })
    db.images.update_many({"dataset_id": {"$in": image_ids}},
                          {"$set": {"house_group_id": new_gid}})
    for og in old_groups:
        g = db.house_groups.find_one({"group_id": og}, {"_id": 0, "image_ids": 1})
        if not g:
            continue
        remaining = [i for i in g["image_ids"] if i not in image_ids]
        if remaining:
            db.house_groups.update_one({"group_id": og},
                                       {"$set": {"image_ids": remaining, "size": len(remaining)}})
        else:
            db.house_groups.delete_one({"group_id": og})
    split = build_split(db=db, log=lambda *a: None)
    write_index_csv(db=db, log=lambda *a: None)
    return {"group_id": new_gid, "image_ids": image_ids, "split": split}


def run_all(log=print):
    db = run_ingest(log=log)
    build_house_groups(db=db, log=log)
    build_split(db=db, log=log)
    write_index_csv(db=db, log=log)
    log("[pipeline] complete")
