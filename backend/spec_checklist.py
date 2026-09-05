"""Client labeling spec — operational rules (source of truth = /app/PROJECT_SPEC.md).
This module exposes the machine-usable checklist derived verbatim from that document.
The written spec wins over any chat summary."""

SPEC_VERSION = "roofline-labeling-spec-v1"

OCCLUSION_TYPES = ["palm", "tree", "shrub", "wire", "vehicle", "neighbor_structure", "other"]
OCCLUSION_WIDTH_THRESHOLD = 0.15  # 15% of image width

GENERATION_METHODS = [
    "sam2_point_prompt", "manual_polygon", "manual_brush",
    "cpu_fallback_assist", "sam2_plus_manual_correction",
]

CHECKLIST = [
    {"id": "one_plane_one_mask", "text": "One mask = one continuous flat visible roof plane (per-plane masks are primary)"},
    {"id": "eave_lower_boundary", "text": "Normal plane scored boundary = lower eave"},
    {"id": "gutter_lip", "text": "Gutter present → boundary on the gutter lip"},
    {"id": "drip_edge", "text": "No gutter → drip edge at equivalent height"},
    {"id": "not_shingle", "text": "NOT the shingle/tile edge above"},
    {"id": "not_fascia_soffit", "text": "NOT the fascia lower edge, NOT the soffit line"},
    {"id": "no_gable_wall", "text": "Do NOT label the gable wall triangle (it is wall, not roof)"},
    {"id": "sections_separate", "text": "Separate planes: each front face, each visible gable side, garage, porch/entry, dormers, lower/stepped sections"},
    {"id": "parapet_upper", "text": "Parapet exception: boundary along the TOP of the parapet (upper), not lower"},
    {"id": "parapet_filename", "text": "Parapet plane filename contains -parapet (e.g. plane-02-parapet.png)"},
    {"id": "parapet_meta", "text": "Parapet listed in metadata parapets: ['plane-02']"},
    {"id": "occlusion_continue", "text": "Occlusion ≤15% image width → continue the plane straight through"},
    {"id": "occlusion_break", "text": "Occlusion >15% → stop at last readable point, restart where readable, flag the gap"},
    {"id": "no_neighbor", "text": "Exclude the neighbor's roof / non-subject structures"},
    {"id": "stop_at_frame", "text": "A plane stops at the image edge (no roof outside frame)"},
    {"id": "visual_check", "text": "Every mask visually checked over the original photo before done"},
]

RULES_META = {
    "spec_version": SPEC_VERSION,
    "boundary_scored": "lower eave (gutter lip, or drip edge at same height); parapet = upper edge",
    "occlusion_threshold_frac_image_width": OCCLUSION_WIDTH_THRESHOLD,
    "occlusion_types": OCCLUSION_TYPES,
    "labeling_method": "SAM point prompts per roof plane, merged, then manually corrected (not auto/box mode)",
    "delivery": "images/<id>.<ext>, planes/<id>/plane-NN[-parapet].png (255/0), merged/<id>.png, meta/<id>.json",
    "coordinate_policy": "masks in EXIF-applied display coordinate space; original file preserved unchanged",
    "acceptance_metric": "mean perpendicular error ≤1% image width AND ≥90% horizontal coverage (client hidden 30-photo set)",
    "ambiguity_rule": "if ambiguous, ask — never resolve silently",
}
