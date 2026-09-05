"""Mini-Milestone 1 — STEP 1: VLM reasoning/planning layer.
For each of the 5 production images, an advanced vision model reads the scene per
/app/PROJECT_SPEC.md and returns a STRUCTURED PLAN: which surfaces are roof planes,
how to separate them, exclusions, eave notes, parapet flags, occlusions (15% rule),
and a point prompt + coarse polygon per plane (normalized 0..1 to the display image).
It plans SAM2 prompts; it does NOT produce final pixel masks. Output -> plans/{id}.json
"""
import os, io, sys, json, asyncio, re
sys.path.insert(0, "/app/backend")
from pathlib import Path
from PIL import Image, ImageOps
import pymongo
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")
import storage
from emergentintegrations.llm.chat import LlmChat, UserMessage, FileContentWithMimeType

MODEL = ("gemini", "gemini-3.1-pro-preview")
OUT = Path("/app/reports/PILOT_MILESTONE_1/plans"); OUT.mkdir(parents=True, exist_ok=True)
TMP = Path("/app/reports/PILOT_MILESTONE_1/_vlm_input"); TMP.mkdir(parents=True, exist_ok=True)
IDS = ["0001", "0050", "0138", "0169", "0173"]
KEY = os.environ["EMERGENT_LLM_KEY"]

SPEC = Path("/app/PROJECT_SPEC.md").read_text()

SYSTEM = (
    "You are an expert roof-plane annotation planner for a Christmas-light roofline extraction "
    "project. You must follow the client's labeling specification EXACTLY. The specification is the "
    "absolute source of truth. You reason about the scene and PLAN the annotation; a separate SAM2 "
    "engine will cut the pixels from your point prompts, and a human QA step will correct them.\n\n"
    "=== CLIENT LABELING SPECIFICATION (verbatim) ===\n" + SPEC + "\n=== END SPEC ===\n\n"
    "KEY RULES YOU MUST APPLY:\n"
    "- A roof plane = one continuous visible roof SURFACE. Each distinct plane is its own label. Never "
    "merge separate planes just because they touch. Separate: each front section, each visible side of a "
    "gable, garage roofs, porch/entry roofs, dormers, lower/stepped sections, flat/parapet sections.\n"
    "- The ONLY scored boundary is the LOWER boundary = the EAVE (gutter lip, or drip edge if no gutter). "
    "NOT the ridge, NOT the tile cap/shingle edge above, NOT the fascia lower edge, NOT the soffit.\n"
    "- DO NOT label: gable wall triangles, ridges/peaks, the neighbor's roof or any non-subject structure, "
    "walls, trim, window heads, tile courses, vegetation, ground/road. The SUBJECT is the main listed home.\n"
    "- PARAPET: a flat roof behind a parapet wall is lit along the TOP of the parapet (its UPPER boundary). "
    "Mark such a plane is_parapet=true.\n"
    "- OCCLUSION 15% rule: if an obstruction (palm/tree/shrub/wire/vehicle/neighbor_structure/other) crosses "
    "a roof and its width <=15% of image width, the plane CONTINUES underneath (one plane). If >15%, split "
    "into separate planes. Report each obstruction with normalized bbox and which planes it hides.\n\n"
    "COORDINATES: all coordinates are NORMALIZED floats 0..1 relative to the image I give you "
    "(x=left→right, y=top→bottom).\n\n"
    "Return ONLY strict minified JSON, no prose, no code fences, with this schema:\n"
    "{\"planes\":[{\"name\":str,\"is_parapet\":bool,\"point\":[x,y],"
    "\"polygon\":[[x,y],...>=4 pts tracing the plane, LOWER edge on the eave],"
    "\"eave_note\":str,\"why_roof\":str}],"
    "\"excluded_surfaces\":[{\"what\":str,\"bbox\":[x0,y0,x1,y1]}],"
    "\"occlusions\":[{\"type\":str,\"bbox\":[x0,y0,x1,y1],\"hides\":[plane_name...]}],"
    "\"ambiguities\":str}"
)

def parse_json(t):
    t = t.strip()
    t = re.sub(r"^```(json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    a, b = t.find("{"), t.rfind("}")
    return json.loads(t[a:b+1])

async def plan_one(did):
    doc = pymongo.MongoClient(os.environ["MONGO_URL"])[os.environ["DB_NAME"]].images.find_one({"dataset_id": did})
    raw, _ = storage.get_object(doc["storage_path"])
    disp = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    W, H = disp.size
    vlm = disp.copy(); vlm.thumbnail((1536, 1536))
    p = TMP / f"{did}.jpg"; vlm.save(p, quality=90)
    chat = LlmChat(api_key=KEY, session_id=f"mm1-plan-{did}", system_message=SYSTEM).with_model(*MODEL)
    msg = UserMessage(
        text=(f"Plan the roof-plane annotation for this property photo (image id {did}). "
              "Identify EVERY distinct visible roof plane of the SUBJECT house only, per the spec. "
              "Give one interior point and a polygon (lower edge on the eave) per plane, list excluded "
              "surfaces, occlusions with the 15% rule, and any ambiguity. Strict JSON only."),
        file_contents=[FileContentWithMimeType(file_path=str(p), mime_type="image/jpeg")],
    )
    resp = await chat.send_message(msg)
    text = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
    try:
        plan = parse_json(text)
    except Exception as e:
        plan = {"_parse_error": str(e), "_raw": text}
    plan["_display_size"] = [W, H]
    (OUT / f"{did}.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    n = len(plan.get("planes", [])) if "planes" in plan else "ERR"
    print(f"{did}: planes={n} occl={len(plan.get('occlusions',[]))} parapet={[p['name'] for p in plan.get('planes',[]) if p.get('is_parapet')]} amb={str(plan.get('ambiguities',''))[:60]}")

async def main():
    for did in IDS:
        try:
            await plan_one(did)
        except Exception as e:
            print(f"{did}: ERROR {e}")

asyncio.run(main())
print("PLANS ->", OUT)
