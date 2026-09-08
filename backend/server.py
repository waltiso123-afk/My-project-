from fastapi import FastAPI, APIRouter, HTTPException, Query, Body
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from datetime import datetime, timezone

import config
import storage
import ml
import envreport
import pipeline
import spec_checklist

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="Roofline Phase-1 Foundation API")
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("roofline")

CLEAN = {"_id": 0}


@app.on_event("startup")
async def _startup():
    try:
        await run_in_threadpool(storage.init_storage)
        logger.info("Object storage initialized")
    except Exception as e:
        logger.error(f"Storage init failed: {e}")


@api.get("/")
async def root():
    return {"service": "roofline-phase1", "status": "ok"}


# ---------------- Environment ----------------
@api.get("/environment")
async def environment():
    report = await run_in_threadpool(envreport.build_report)
    report["id"] = "env_report"
    await db.environment_reports.update_one({"id": "env_report"}, {"$set": report}, upsert=True)
    return report


@api.get("/ml/status")
async def ml_status():
    return await run_in_threadpool(ml.sam2_status)


@api.get("/spec/checklist")
async def spec_check():
    return {
        "spec_loaded": True,
        "spec_version": spec_checklist.SPEC_VERSION,
        "reference_file": "/app/PROJECT_SPEC.md",
        "checklist": spec_checklist.CHECKLIST,
        "rules_meta": spec_checklist.RULES_META,
        "occlusion_types": spec_checklist.OCCLUSION_TYPES,
        "generation_methods": spec_checklist.GENERATION_METHODS,
    }


# ---------------- Dataset ----------------
@api.get("/dataset/summary")
async def dataset_summary():
    total = await db.images.count_documents({})
    b1 = await db.images.count_documents({"batch": "batch1"})
    b2 = await db.images.count_documents({"batch": "batch2"})
    corrupted = await db.images.count_documents({"corrupted": True})
    duplicates = await db.images.count_documents({"duplicate_of": {"$ne": None}})
    status_counts = {}
    for s in ("pending", "approved", "needs_review"):
        status_counts[s] = await db.images.count_documents({"status": s})
    orient = {}
    for o in ("portrait", "landscape", "square"):
        orient[o] = await db.images.count_documents({"orientation": o})
    groups = await db.house_groups.count_documents({})
    multi = await db.house_groups.count_documents({"size": {"$gt": 1}})
    split = await db.splits.find_one({"id": "split_config"}, CLEAN)
    return {
        "total_images": total, "batch1": b1, "batch2": b2,
        "corrupted": corrupted, "duplicates": duplicates,
        "status_counts": status_counts, "orientation": orient,
        "house_groups": groups, "multi_view_groups": multi,
        "split": split,
        "index_csv_exists": config.DATASET_INDEX_CSV.exists(),
    }


@api.get("/dataset")
async def list_dataset(status: str = Query(None), batch: str = Query(None),
                       house_group: str = Query(None), search: str = Query(None),
                       limit: int = Query(500)):
    q = {}
    if status:
        q["status"] = status
    if batch:
        q["batch"] = batch
    if house_group:
        q["house_group_id"] = house_group
    if search:
        q["$or"] = [{"dataset_id": {"$regex": search, "$options": "i"}},
                    {"original_filename": {"$regex": search, "$options": "i"}},
                    {"house_group_id": {"$regex": search, "$options": "i"}}]
    rows = await db.images.find(q, CLEAN).sort("dataset_id", 1).to_list(limit)
    # attach plane counts
    ids = [r["dataset_id"] for r in rows]
    anns = await db.annotations.find({"dataset_id": {"$in": ids}}, CLEAN).to_list(len(ids))
    ann_map = {a["dataset_id"]: len(a.get("planes", [])) for a in anns}
    for r in rows:
        r["plane_count"] = ann_map.get(r["dataset_id"], 0)
    return rows


@api.get("/dataset/representative")
async def representative():
    rows = await db.images.find({"representative": True}, CLEAN).sort("representative_rank", 1).to_list(20)
    return rows


@api.get("/dataset/{dataset_id}")
async def get_image(dataset_id: str):
    img = await db.images.find_one({"dataset_id": dataset_id}, CLEAN)
    if not img:
        raise HTTPException(404, "image not found")
    ann = await db.annotations.find_one({"dataset_id": dataset_id}, CLEAN)
    img["annotation"] = ann
    return img


@api.get("/images/{dataset_id}/raw")
async def raw_image(dataset_id: str):
    img = await db.images.find_one({"dataset_id": dataset_id}, {"_id": 0, "storage_path": 1, "content_type": 1})
    if not img:
        raise HTTPException(404, "image not found")
    data, ct = await run_in_threadpool(storage.get_object, img["storage_path"])
    return Response(content=data, media_type=img.get("content_type") or ct)


@api.get("/masks/{dataset_id}/{plane_file}")
async def raw_mask(dataset_id: str, plane_file: str):
    path = config.plane_path(dataset_id, plane_file)
    try:
        data, ct = await run_in_threadpool(storage.get_object, path)
    except Exception:
        raise HTTPException(404, "mask not found")
    return Response(content=data, media_type="image/png")


@api.get("/merged/{dataset_id}/raw")
async def raw_merged(dataset_id: str):
    try:
        data, ct = await run_in_threadpool(storage.get_object, config.merged_path(dataset_id))
    except Exception:
        raise HTTPException(404, "merged mask not found")
    return Response(content=data, media_type="image/png")


# ---------------- Deliverables (client review package) ----------------
@api.get("/deliverables/pilot-5/info")
async def pilot5_info():
    doc = await db.deliverables.find_one({"_id": "pilot-5-client-review"}, CLEAN)
    if not doc:
        raise HTTPException(404, "pilot-5 package not built")
    return doc


@api.get("/deliverables/pilot-5/download")
async def pilot5_download():
    doc = await db.deliverables.find_one({"_id": "pilot-5-client-review"})
    if not doc:
        raise HTTPException(404, "pilot-5 package not built")
    data, _ = await run_in_threadpool(storage.get_object, doc["storage_path"])
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{doc["filename"]}"'})


@api.get("/deliverables/milestone-1/info")
async def milestone1_info():
    doc = await db.deliverables.find_one({"_id": "pilot-milestone-1"}, CLEAN)
    if not doc:
        raise HTTPException(404, "milestone-1 package not built")
    return doc


@api.get("/deliverables/milestone-1/download")
async def milestone1_download():
    doc = await db.deliverables.find_one({"_id": "pilot-milestone-1"})
    if not doc:
        raise HTTPException(404, "milestone-1 package not built")
    data, _ = await run_in_threadpool(storage.get_object, doc["storage_path"])
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{doc["filename"]}"'})


@api.get("/deliverables/milestone1-step1/info")
async def m1s1_info():
    doc = await db.deliverables.find_one({"_id": "milestone1-step1"}, CLEAN)
    if not doc:
        raise HTTPException(404, "milestone1-step1 not built")
    return doc


@api.get("/deliverables/milestone1-step1/download")
async def m1s1_download():
    doc = await db.deliverables.find_one({"_id": "milestone1-step1"})
    if not doc:
        raise HTTPException(404, "milestone1-step1 not built")
    data, _ = await run_in_threadpool(storage.get_object, doc["storage_path"])
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{doc["filename"]}"'})


# ---------------- House groups ----------------
@api.get("/housegroups")
async def house_groups(only_multi: bool = Query(False)):
    q = {"size": {"$gt": 1}} if only_multi else {}
    return await db.house_groups.find(q, CLEAN).sort("group_id", 1).to_list(1000)


@api.post("/housegroups/{group_id}/confirm")
async def confirm_group(group_id: str):
    r = await db.house_groups.update_one({"group_id": group_id}, {"$set": {"uncertain": False, "human_confirmed": True}})
    if r.matched_count == 0:
        raise HTTPException(404, "group not found")
    return {"group_id": group_id, "uncertain": False, "human_confirmed": True}


@api.get("/housegroups/candidates")
async def group_candidates(max_distance: int = Query(12), top: int = Query(120)):
    pairs = await run_in_threadpool(pipeline.compute_candidates, None, max_distance, top)
    return {"min_inliers": max_distance, "count": len(pairs), "candidates": pairs,
            "note": "ORB local-feature + RANSAC homography evidence pairs (same-property "
                    "candidates). Strong geometric matches were auto-grouped; borderline "
                    "pairs await your confirmation. Not perceptual hashing."}


@api.post("/housegroups/merge")
async def merge_groups(payload: dict = Body(...)):
    image_ids = payload.get("image_ids") or []
    if len(image_ids) < 2:
        raise HTTPException(400, "provide at least 2 image_ids to merge")
    result = await run_in_threadpool(pipeline.merge_images_into_group, image_ids, None, True)
    return result


_orb_kp_cache = {}


def _orb_keypoint_count(dataset_id, batch, filename):
    if dataset_id in _orb_kp_cache:
        return _orb_kp_cache[dataset_id]
    sub = "batch1" if batch == "batch1" else "batch2"
    path = config.DATA_DIR / "staging" / sub / filename
    try:
        kp, des = pipeline._orb_features(path)
        n = len(kp) if kp else 0
    except Exception:
        n = None
    _orb_kp_cache[dataset_id] = n
    return n


@api.get("/housegroups/pairs")
async def group_pairs_full():
    """All ORB+RANSAC candidate pairs with full geometric metrics + any human ruling.
    ORB/RANSAC is candidate evidence, NOT ground truth — the human rules on each pair."""
    pairs = await db.group_pairs.find({}, CLEAN).sort("inliers", -1).to_list(500)
    out = []
    for gp in pairs:
        a, b = gp["image_a"], gp["image_b"]
        ia = await db.images.find_one({"dataset_id": a}, {"_id": 0, "batch": 1, "original_filename": 1, "orientation": 1, "house_group_id": 1})
        ib = await db.images.find_one({"dataset_id": b}, {"_id": 0, "batch": 1, "original_filename": 1, "orientation": 1, "house_group_id": 1})
        kpa = await run_in_threadpool(_orb_keypoint_count, a, ia["batch"], ia["original_filename"])
        kpb = await run_in_threadpool(_orb_keypoint_count, b, ib["batch"], ib["original_filename"])
        good = gp.get("good_matches") or 0
        out.append({
            "image_a": a, "image_b": b,
            "file_a": ia["original_filename"], "file_b": ib["original_filename"],
            "batch_a": ia["batch"], "batch_b": ib["batch"],
            "orientation_a": ia["orientation"], "orientation_b": ib["orientation"],
            "group_a": ia.get("house_group_id"), "group_b": ib.get("house_group_id"),
            "keypoints_a": kpa, "keypoints_b": kpb,
            "raw_good_matches": good, "inliers": gp["inliers"],
            "inlier_ratio": round(gp["inliers"] / good, 3) if good else None,
            "strong": gp.get("strong", False),
            "ruling": gp.get("ruling"),
        })
    return {"count": len(out), "pairs": out,
            "note": "ORB/RANSAC is candidate evidence only. Rule SAME / DIFFERENT / UNSURE per pair."}


@api.post("/housegroups/pairs/ruling")
async def pair_ruling(payload: dict = Body(...)):
    a, b, ruling = payload.get("image_a"), payload.get("image_b"), payload.get("ruling")
    if ruling not in ("same", "different", "unsure"):
        raise HTTPException(400, "ruling must be same|different|unsure")
    await db.group_pairs.update_one(
        {"$or": [{"image_a": a, "image_b": b}, {"image_a": b, "image_b": a}]},
        {"$set": {"ruling": ruling}})
    result = {"image_a": a, "image_b": b, "ruling": ruling, "merged": False}
    if ruling == "same":
        merged = await run_in_threadpool(pipeline.merge_images_into_group, [a, b], None, True)
        result["merged"] = True
        result["group"] = merged
    return result


# ---------------- Split ----------------
@api.get("/split")
async def split():
    doc = await db.splits.find_one({"id": "split_config"}, CLEAN)
    if not doc:
        raise HTTPException(404, "split not built")
    groups = await db.house_groups.find({}, {"_id": 0, "group_id": 1, "split": 1, "size": 1}).to_list(2000)
    doc["groups"] = groups
    return doc


@api.get("/split/leakage")
async def leakage_check():
    """Live leakage check: verify no house_group appears in more than one split."""
    imgs = await db.images.find({}, {"_id": 0, "dataset_id": 1, "house_group_id": 1, "split": 1}).to_list(5000)
    group_splits = {}
    for im in imgs:
        g = im.get("house_group_id")
        s = im.get("split")
        if g is None or s is None:
            continue
        group_splits.setdefault(g, set()).add(s)
    leaks = {g: sorted(list(ss)) for g, ss in group_splits.items() if len(ss) > 1}
    return {
        "passed": len(leaks) == 0,
        "groups_checked": len(group_splits),
        "images_checked": len(imgs),
        "leaks": leaks,
        "message": "No house_group spans multiple splits — zero leakage." if not leaks
                   else f"LEAKAGE DETECTED in {len(leaks)} group(s).",
    }


# ---------------- ML propose ----------------
@api.post("/ml/propose")
async def ml_propose(payload: dict = Body(...)):
    dataset_id = payload["dataset_id"]
    method = payload.get("method", "sam2")  # 'sam2' | 'colour_region'
    positive = payload.get("positive_points") or []
    negative = payload.get("negative_points") or []
    if not positive:
        raise HTTPException(400, "at least one positive point required")
    img = await db.images.find_one({"dataset_id": dataset_id}, {"_id": 0, "storage_path": 1})
    if not img:
        raise HTTPException(404, "image not found")
    data, _ = await run_in_threadpool(storage.get_object, img["storage_path"])

    if method == "colour_region":
        tol = int(payload.get("tolerance", 22))
        return await run_in_threadpool(ml.colour_region, data, positive, tol)

    # SAM2 — surface errors explicitly, never silently fall back to flood-fill
    try:
        return await run_in_threadpool(ml.sam2_propose, data, dataset_id, positive, negative)
    except Exception as e:
        return {"backend": "sam2_error", "polygons": [], "error": str(e),
                "note": "SAM2 failed. Switch to manual annotation or the Colour Region helper. "
                        "No silent flood-fill fallback was performed."}


# ---------------- Annotations ----------------
@api.get("/annotations/{dataset_id}")
async def get_annotation(dataset_id: str):
    ann = await db.annotations.find_one({"dataset_id": dataset_id}, CLEAN)
    if not ann:
        return {"dataset_id": dataset_id, "planes": [], "occlusions": [], "notes": "",
                "checklist": {}, "status": "pending"}
    return ann


@api.put("/annotations/{dataset_id}")
async def save_annotation(dataset_id: str, payload: dict = Body(...)):
    img = await db.images.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not img:
        raise HTTPException(404, "image not found")
    doc = {
        "dataset_id": dataset_id,
        "planes": payload.get("planes", []),
        "occlusions": payload.get("occlusions", []),
        "notes": payload.get("notes", ""),
        "checklist": payload.get("checklist", {}),
        "view_type": payload.get("view_type"),
        "status": payload.get("status", img.get("status", "pending")),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.annotations.update_one({"dataset_id": dataset_id}, {"$set": doc}, upsert=True)
    return {"ok": True, "saved_at": doc["updated_at"], "planes": len(doc["planes"])}


@api.post("/annotations/{dataset_id}/needs_review")
async def mark_needs_review(dataset_id: str, payload: dict = Body(default={})):
    reason = payload.get("reason", "")
    await db.annotations.update_one({"dataset_id": dataset_id},
                                    {"$set": {"status": "needs_review", "review_reason": reason,
                                              "updated_at": datetime.now(timezone.utc).isoformat()}},
                                    upsert=True)
    await db.images.update_one({"dataset_id": dataset_id}, {"$set": {"status": "needs_review"}})
    return {"dataset_id": dataset_id, "status": "needs_review", "reason": reason}


@api.post("/annotations/{dataset_id}/approve")
async def approve(dataset_id: str, payload: dict = Body(...)):
    img = await db.images.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not img:
        raise HTTPException(404, "image not found")
    planes = payload.get("planes", [])
    if not planes:
        raise HTTPException(400, "cannot approve with zero planes")
    size = (img["width"], img["height"])
    result = await run_in_threadpool(
        ml.generate_and_store, dataset_id, img["ext"], size, planes,
        payload.get("occlusions", []), payload.get("notes", ""),
        payload.get("view_type"),
    )
    if payload.get("view_type"):
        await db.images.update_one({"dataset_id": dataset_id}, {"$set": {"view_type": payload["view_type"]}})
    now = datetime.now(timezone.utc).isoformat()
    ann_doc = {
        "dataset_id": dataset_id, "planes": planes,
        "occlusions": payload.get("occlusions", []),
        "notes": payload.get("notes", ""),
        "checklist": payload.get("checklist", {}),
        "status": "approved", "approved_at": now, "updated_at": now,
        "generated": {"merged": result["merged"], "planes": result["planes"],
                      "meta": config.meta_path(dataset_id)},
    }
    await db.annotations.update_one({"dataset_id": dataset_id}, {"$set": ann_doc}, upsert=True)
    await db.images.update_one({"dataset_id": dataset_id}, {"$set": {"status": "approved"}})
    return {"dataset_id": dataset_id, "status": "approved",
            "generated": ann_doc["generated"], "meta": result["meta"]}


# ---------------- QA scripts ----------------
@api.get("/qa/scripts")
async def qa_scripts():
    return {
        "scripts": [
            {"name": "validate_dataset.py",
             "path": "scripts/validate_dataset.py",
             "purpose": "Dataset integrity, formats, numbering consistency, -parapet suffix check"},
            {"name": "check_split_leakage.py",
             "path": "scripts/check_split_leakage.py",
             "purpose": "Verify no house_group leaks across train/val/holdout"},
        ]
    }


app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def _shutdown():
    client.close()
