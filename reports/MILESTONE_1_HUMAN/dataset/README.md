# Roofline Labeling — Milestone 1 (first 5-image delivery)

This is the first delivery of the roof-plane labeling work, covering 5 house photos. It uses the exact same
specification, format and quality standard planned for the full Milestone 1 dataset — only the number of
images differs.

## Contents (per image)
- `images/<id>.<ext>` — original photo, unmodified.
- `planes/<id>/plane-01.png, plane-02.png, ...` — one binary mask per roof plane (255 = plane, 0 = background). Primary output.
- `merged/<id>.png` — union of the planes (convenience only).
- `meta/<id>.json` — occlusion records and parapet information.

## Reading it
- Each distinct roof plane is a separate mask at the exact pixel size of the original.
- Flat roofs behind a parapet use `-parapet` in the filename and appear under `"parapets"` in metadata (lit along the parapet top).
- Trees/palms/wires crossing a roof are recorded under `"occlusions"`.

## Image 0169
0169 is a modern flat-roof / parapet-style home containing an ambiguous situation we would like you to confirm
before scaling. The specific open questions are in `meta/0169.json` under `notes`; its masks are provisional.

## Purpose
Please confirm the roof-plane separation, boundaries and metadata match your expectation. The same process and
format will then be applied to the full dataset.
