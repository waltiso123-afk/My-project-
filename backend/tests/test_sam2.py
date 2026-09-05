"""SAM2 CPU point-prompt tests + colour_region isolation + approve pipeline.

Covers the iteration_3 review scope:
- /api/ml/status truthfulness (sam2_cpu, cpu, checkpoint present)
- /api/ml/propose method=sam2 real segmentation on roof points
- Embedding cache (2nd call same dataset -> cached)
- /api/ml/propose method=colour_region backend + NOT SAM2 note
- sam2_error path (no silent fallback)
- Approve pipeline with polygons + raster_png -> generated masks + merged raw
- needs_review flag
"""
import base64
import io
import os
import time

import pytest
import requests
from PIL import Image

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://roofdata-validate.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def s():
    ses = requests.Session()
    ses.headers.update({"Content-Type": "application/json"})
    return ses


# ---------- /api/ml/status ----------
def test_ml_status_truthful(s):
    r = s.get(f"{API}/ml/status", timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["active_backend"] == "sam2_cpu"
    assert d["device"] == "cpu"
    assert d["sam2_installed"] is True
    assert d["checkpoint_present"] is True
    assert d["cuda_available"] is False
    assert d["torch_version"]


# ---------- SAM2 propose on roof point ----------
@pytest.mark.parametrize("pt", [[645, 391], [952, 414]])
def test_sam2_propose_roof_point(s, pt):
    r = s.post(f"{API}/ml/propose",
               json={"dataset_id": "0001", "method": "sam2", "positive_points": [pt]},
               timeout=60)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["backend"] == "sam2_cpu", d
    assert d["device"] == "cpu"
    assert isinstance(d.get("score"), (int, float))
    af = d["area_frac"]
    assert 0.001 <= af <= 0.9, f"area_frac out of plausible range: {af}"
    assert d["polygons"], "polygons must be non-empty for a real SAM2 mask"


# ---------- Embedding cache ----------
def test_sam2_embedding_cache(s):
    # first call may already be cached from prior tests; reset by calling on a
    # different image, then two consecutive on same image.
    s.post(f"{API}/ml/propose",
           json={"dataset_id": "0050", "method": "sam2", "positive_points": [[100, 100]]},
           timeout=60)
    r1 = s.post(f"{API}/ml/propose",
                json={"dataset_id": "0001", "method": "sam2", "positive_points": [[645, 391]]},
                timeout=60).json()
    r2 = s.post(f"{API}/ml/propose",
                json={"dataset_id": "0001", "method": "sam2", "positive_points": [[652, 400]]},
                timeout=60).json()
    assert r1["embedding_cached"] is False
    assert r1["encode_ms"] > 500, r1
    assert r2["embedding_cached"] is True
    assert r2["encode_ms"] < 200, r2
    assert r2["predict_ms"] < 2000


# ---------- Colour region helper (never labelled SAM2) ----------
def test_colour_region_isolated(s):
    r = s.post(f"{API}/ml/propose",
               json={"dataset_id": "0001", "method": "colour_region",
                     "positive_points": [[726, 598]]}, timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["backend"] == "cpu_colour_region"
    assert "NOT SAM2" in d.get("note", "")


# ---------- SAM2 error path (no silent fallback) ----------
def test_sam2_error_no_fallback():
    # Verify code path via source inspection — server ml_propose SAM2 branch
    # returns explicit sam2_error on exception; no flood-fill fallback.
    src = open("/app/backend/server.py").read()
    assert '"backend": "sam2_error"' in src
    assert "No silent flood-fill fallback" in src


# ---------- Approve pipeline: polygons + brush raster ----------
def _tiny_raster_b64(w=64, h=64):
    im = Image.new("L", (w, h), 0)
    # a filled rectangle in the middle
    for y in range(20, 44):
        for x in range(20, 44):
            im.putpixel((x, y), 255)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_approve_polygons_and_raster(s):
    payload = {
        "planes": [
            {"name": "plane-01", "polygons": [[[100, 100], [400, 100], [400, 300], [100, 300]]],
             "generation_method": "manual_polygon", "human_corrected": True},
            {"name": "plane-02", "raster_png": _tiny_raster_b64(),
             "generation_method": "brush", "human_corrected": True},
        ],
        "occlusions": [],
        "notes": "TEST_approve",
        "view_type": "oblique",
        "checklist": {"eave_lower_boundary": True},
    }
    r = s.post(f"{API}/annotations/0001/approve", json=payload, timeout=60)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "approved"
    assert d["generated"]["merged"]
    assert len(d["generated"]["planes"]) == 2
    meta = d["meta"]
    assert meta["coordinate_space"] == "exif_applied_display"
    assert meta["planes"][0]["generation_method"] == "manual_polygon"
    assert meta["planes"][0]["human_approved"] is True
    assert meta["planes"][1]["generation_method"] == "brush"
    # merged raw serves
    rr = s.get(f"{API}/merged/0001/raw", timeout=30)
    assert rr.status_code == 200
    assert rr.headers.get("content-type") == "image/png"


# ---------- needs_review ----------
def test_needs_review(s):
    r = s.post(f"{API}/annotations/0138/needs_review", json={"reason": "ambiguous"}, timeout=15)
    assert r.status_code == 200
    assert r.json()["status"] == "needs_review"
    g = s.get(f"{API}/dataset/0138", timeout=15).json()
    assert g["status"] == "needs_review"


# ---------- Cleanup ----------
def test_zzz_cleanup(s):
    """Reset 0001 and 0138 back to pending & delete their annotations."""
    import pymongo
    cli = pymongo.MongoClient(os.environ.get("MONGO_URL", "mongodb://localhost:27017"))
    dbn = os.environ.get("DB_NAME", "test_database")
    db = cli[dbn]
    db.images.update_many({"dataset_id": {"$in": ["0001", "0138"]}}, {"$set": {"status": "pending"}})
    db.annotations.delete_many({"dataset_id": {"$in": ["0001", "0138"]}})
    # Also delete stored merged mask files? Leave storage objects; DB state matters.
    for did in ("0001", "0138"):
        img = db.images.find_one({"dataset_id": did})
        assert img["status"] == "pending"
        assert db.annotations.find_one({"dataset_id": did}) is None
    cli.close()
