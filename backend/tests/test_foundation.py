"""Regression tests for the Roofline foundation.

Isolation: these tests NEVER touch the production dataset. DB writes go to a dedicated
'roofline_regression' database and object-storage writes use a 'roofline_regression/' prefix,
proving test data cannot mutate production ('test_database' / 'roofline/').
"""
import io
import os
import sys

sys.path.insert(0, "/app/backend")
import config  # noqa: E402
import ml  # noqa: E402
import pipeline  # noqa: E402
from PIL import Image  # noqa: E402
import numpy as np  # noqa: E402
from pymongo import MongoClient  # noqa: E402

REGRESSION_DB = "roofline_regression"


def _exif_image(orientation, size=(400, 300)):
    im = Image.new("RGB", size, (120, 130, 140))
    exif = im.getexif()
    exif[274] = orientation  # 274 = Orientation tag
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return buf


def _write_temp(buf, name):
    p = config.DATA_DIR / "tmp_tests"
    p.mkdir(parents=True, exist_ok=True)
    fp = p / name
    fp.write_bytes(buf.getvalue())
    return fp


# ---------- Orientation / EXIF ----------

def test_landscape_pixels_with_exif6_is_portrait():
    """4:3 landscape pixels + EXIF orientation 6 (rotate 90) => effective PORTRAIT."""
    fp = _write_temp(_exif_image(6, (400, 300)), "exif6.jpg")
    v = pipeline._validate_image(fp)
    assert v["ok"]
    assert (v["raw_w"], v["raw_h"]) == (400, 300)          # raw landscape
    assert v["disp_h"] > v["disp_w"]                        # display portrait
    assert v["exif_transposed"] is True
    assert v["exif_orientation"] == 6


def test_raw_portrait_no_exif_is_portrait():
    fp = _write_temp(_exif_image(1, (300, 500)), "portrait_normal.jpg")
    v = pipeline._validate_image(fp)
    assert v["disp_h"] > v["disp_w"]
    assert v["exif_transposed"] is False


def test_ordinary_landscape():
    fp = _write_temp(_exif_image(1, (500, 300)), "landscape_normal.jpg")
    v = pipeline._validate_image(fp)
    assert v["disp_w"] > v["disp_h"]
    assert v["exif_transposed"] is False


def test_propose_applies_exif_orientation():
    """propose() must operate in the EXIF-applied display space (portrait), so returned
    polygon coords fit portrait dims — proving image/mask coordinate alignment."""
    buf = _exif_image(6, (400, 300))
    res = ml.propose(buf.getvalue(), [[150, 200]], tolerance=40)
    assert res["backend"] == "cpu_assist_fallback"  # no GPU here
    # display space is 300x400 (portrait); any polygon point must fit within it
    for poly in res["polygons"]:
        for x, y in poly:
            assert 0 <= x <= 300 and 0 <= y <= 400


# ---------- Mask generation (255/0, merged == union, -parapet) ----------

def test_polygons_to_mask_binary():
    m = ml.polygons_to_mask([[[10, 10], [90, 10], [90, 90], [10, 90]]], (100, 100))
    arr = np.array(m)
    assert set(np.unique(arr).tolist()).issubset({0, 255})
    assert arr[50, 50] == 255 and arr[2, 2] == 0


def test_generate_and_store_isolated(monkeypatch):
    """Generate masks under an isolated storage prefix; verify -parapet suffix, merged union,
    and provenance in meta. Uses 'roofline_regression/' prefix -> cannot touch production."""
    monkeypatch.setattr(config, "APP_NAME", REGRESSION_DB)
    did = "test0001"
    planes = [
        {"name": "main", "is_parapet": False, "polygons": [[[0, 0], [60, 0], [60, 60], [0, 60]]],
         "generation_method": "manual_polygon"},
        {"name": "flat", "is_parapet": True, "polygons": [[[70, 70], [100, 70], [100, 100], [70, 100]]],
         "generation_method": "cpu_fallback_assist"},
    ]
    res = ml.generate_and_store(did, "jpg", (100, 100), planes,
                                [{"type": "palm", "bbox": [0.1, 0.1, 0.2, 0.2], "hides": ["plane-01"]}],
                                "note", view_type="oblique")
    assert res["planes"][0].endswith("plane-01.png")
    assert res["planes"][1].endswith("plane-02-parapet.png")
    assert res["meta"]["parapets"] == ["plane-02"]
    assert res["meta"]["planes"][1]["generation_method"] == "cpu_fallback_assist"
    assert res["meta"]["planes"][1]["human_approved"] is True
    assert res["meta"]["view_type"] == "oblique"
    assert res["meta"]["coordinate_space"] == "exif_applied_display"
    assert "roofline_regression/" in res["merged"]


# ---------- Production isolation ----------

def test_regression_db_is_not_production():
    prod = os.environ["DB_NAME"]
    assert prod != REGRESSION_DB
    client = MongoClient(os.environ["MONGO_URL"])
    client[REGRESSION_DB].sentinel.insert_one({"marker": "regression_only"})
    # writing to regression DB must NOT appear in production DB
    assert client[prod].sentinel.count_documents({"marker": "regression_only"}) == 0
    client[REGRESSION_DB].sentinel.delete_many({})


def test_production_dataset_intact():
    client = MongoClient(os.environ["MONGO_URL"])
    prod = client[os.environ["DB_NAME"]]
    assert prod.images.count_documents({}) == 217  # production untouched by tests
