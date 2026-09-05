# Roofline Labeling — Milestone 1 (first 5-image delivery)

This is the first delivery of the roof-plane labeling work, covering 5 house photos. It uses the
exact same specification, format and quality standard that will be used for the full Milestone 1
dataset — only the number of images differs.

## What is in this delivery
For each image you receive:
- `images/<id>.<ext>` — the original photo, unmodified.
- `planes/<id>/plane-01.png, plane-02.png, ...` — one binary mask per roof plane (255 = plane, 0 = background).
  These per-plane masks are the primary annotation output.
- `merged/<id>.png` — the union of that image's planes (provided for convenience only).
- `meta/<id>.json` — metadata for the image, including occlusion records and parapet information.

## How to read it
- Each distinct roof plane is delivered as its own separate mask, at the exact pixel dimensions of the
  original image.
- Flat roofs behind a parapet wall are delivered with `-parapet` in the filename and listed under
  `"parapets"` in the metadata; these are lit along the top of the parapet.
- Where a tree, palm, wire or similar crosses a roof, the metadata records it under `"occlusions"`.

## Note on image 0169
Image 0169 is a modern flat-roof / parapet-style home. It contains an ambiguous situation that we would
like you to confirm before we scale to the full dataset. The specific open questions are written in
`meta/0169.json` under `notes`. The masks provided for 0169 are provisional pending your confirmation.

## Purpose
Please review these 5 images and confirm that the roof-plane separation, boundaries and metadata match
your expectation. Once confirmed, the same process and format will be applied to the full dataset.
