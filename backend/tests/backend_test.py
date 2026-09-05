"""Roofline Phase-1 Foundation backend tests."""
import os
import pytest
import requests

BASE = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") if os.environ.get("REACT_APP_BACKEND_URL") else None
if not BASE:
    # Fallback: read from frontend/.env
    with open("/app/frontend/.env") as f:
        for line in f:
            if line.startswith("REACT_APP_BACKEND_URL="):
                BASE = line.split("=", 1)[1].strip().rstrip("/")
API = f"{BASE}/api"


# ---------- Dataset summary ----------
def test_dataset_summary():
    r = requests.get(f"{API}/dataset/summary", timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["total_images"] == 217
    assert d["batch1"] == 168
    assert d["batch2"] == 49
    assert d["corrupted"] == 0


def test_batch2_naming():
    r = requests.get(f"{API}/dataset", params={"batch": "batch2"}, timeout=30)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 49
    import re
    pat = re.compile(r"^b2-0\d{2}")
    for row in rows:
        assert pat.match(row["original_filename"]), row["original_filename"]


# ---------- House grouping (bug fix) ----------
def test_housegroups_method_orb():
    r = requests.get(f"{API}/housegroups", timeout=30)
    assert r.status_code == 200
    groups = r.json()
    assert len(groups) >= 1
    methods = {g.get("method") for g in groups}
    assert "orb_features + ransac_homography_inliers" in methods, f"methods={methods}"


def test_housegroup_candidates():
    r = requests.get(f"{API}/housegroups/candidates", params={"max_distance": 12}, timeout=60)
    assert r.status_code == 200
    d = r.json()
    cands = d["candidates"]
    assert isinstance(cands, list) and len(cands) >= 1
    for c in cands:
        assert isinstance(c.get("inliers"), int)
        assert isinstance(c.get("good_matches"), int)
        assert isinstance(c.get("strong"), bool)


# ---------- Split ----------
def test_split_structure():
    r = requests.get(f"{API}/split", timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["level"] == "house_group"
    assert d["holdout_fixed"] is True
    ic = d["image_counts"]
    assert ic["train"] + ic["val"] + ic["holdout"] == 217


def test_split_leakage_passes():
    r = requests.get(f"{API}/split/leakage", timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["passed"] is True


# ---------- Environment ----------
def test_environment():
    r = requests.get(f"{API}/environment", timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["verdict"] == "NO_GPU_CPU_ONLY"
    assert d["gpu_present"] is False
    assert d["torch"]["cuda_available"] is False
    assert d["cpu"]["logical_cores"]
    assert d["memory"]["total_gb"]


def test_ml_status():
    r = requests.get(f"{API}/ml/status", timeout=30)
    assert r.status_code == 200
    d = r.json()
    assert d["active_backend"] == "cpu_assist_fallback"
    assert d["gpu_ready"] is False


# ---------- Representative ----------
def test_representative_5():
    r = requests.get(f"{API}/dataset/representative", timeout=30)
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 5
    ids = sorted([x["dataset_id"] for x in rows])
    assert ids == ["0001", "0050", "0138", "0169", "0173"]
    for row in rows:
        assert row.get("representative_reason")


# ---------- ML propose ----------
def test_ml_propose():
    r = requests.post(f"{API}/ml/propose",
                      json={"dataset_id": "0001", "positive_points": [[600, 400]]},
                      timeout=60)
    assert r.status_code == 200
    d = r.json()
    assert d["backend"] == "cpu_assist_fallback"
    assert isinstance(d.get("polygons"), list) and len(d["polygons"]) > 0


# ---------- Approve flow generates artifacts ----------
def test_approve_and_artifacts():
    # Get propose polygons first
    p = requests.post(f"{API}/ml/propose",
                      json={"dataset_id": "0001", "positive_points": [[600, 400]]},
                      timeout=60).json()
    polys = p["polygons"]
    planes = [{"plane_id": "P1", "label": "roof", "polygon": polys[0], "has_parapet": False}]
    r = requests.post(f"{API}/annotations/0001/approve",
                      json={"planes": planes, "occlusions": [], "notes": "test approve"},
                      timeout=90)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "approved"

    g = requests.get(f"{API}/dataset/0001", timeout=30).json()
    assert g["status"] == "approved"
    assert g["annotation"]["generated"]["merged"]

    m = requests.get(f"{API}/merged/0001/raw", timeout=30)
    assert m.status_code == 200
    assert len(m.content) > 100


# ---------- Needs review ----------
def test_needs_review():
    r = requests.post(f"{API}/annotations/0169/needs_review",
                      json={"reason": "ambiguous"}, timeout=30)
    assert r.status_code == 200
    g = requests.get(f"{API}/dataset/0169", timeout=30).json()
    assert g["status"] == "needs_review"


# ---------- Merge house groups + leakage still passes ----------
def test_merge_housegroups_and_leakage():
    r = requests.post(f"{API}/housegroups/merge",
                      json={"image_ids": ["0005", "0195"]}, timeout=60)
    assert r.status_code == 200, r.text
    d = r.json()
    gid = d.get("group_id") or d.get("merged_group", {}).get("group_id")
    assert gid and gid.startswith("HG-M"), d
    lk = requests.get(f"{API}/split/leakage", timeout=30).json()
    assert lk["passed"] is True
