# Roofline labeling spec  (IMMUTABLE PROJECT REFERENCE — SOURCE OF TRUTH)

> This is the client's written labeling specification, preserved verbatim.
> Where this document and anything said in chat disagree, THIS DOCUMENT WINS.
> Operational, machine-usable form: `/app/backend/spec_checklist.py` (spec_version roofline-labeling-spec-v1).

## What one label is
A **roof plane**: one continuous, flat surface of roof visible in the photo. Not the whole roof as one
shape. Not the roofline as a thin line. A region, one per plane, each plane kept separately.

## The boundary that matters
The **lower boundary of each plane is the eave**, and it is the only boundary that is scored. It sits
where the lights physically hang:
- on the **gutter lip** where there is a gutter
- on the **drip edge at the same height** where there is no gutter

Not on the shingle or tile edge above it. Not on the tile cap. Not on the fascia board's lower edge,
and not on the soffit line below that. C9 bulbs clip to the gutter lip, so that is where the boundary goes.

The other boundaries of a plane (the ridge at the top, and where a plane meets another plane) are not
scored and do not need to be precise.

### The one exception: parapets
A flat roof behind a parapet wall is lit **along the top of the parapet**, which is the plane's **upper**
boundary, not its lower one. Common in South Florida. A parapet plane is delivered with `-parapet` in its
filename: `planes/0007/plane-02-parapet.png`. That suffix tells extraction to take the upper boundary. A
parapet plane without it is measured along the bottom of the wall and scores as a total failure. Also list
in metadata as `"parapets": ["plane-02"]`. Everything else applies to parapet planes unchanged.

## What counts as its own plane
Each is a separate plane with its own mask: each front facing roof section; each side of a gable where both
visible; garage roof sections; **porch and entry roofs** (including below/in front of the main roof);
**dormers**; **lower and stepped roof sections**; flat roof sections with a parapet (parapet top edge is the line).

## Do not label
- the **gable wall triangle** (that is wall, not roof)
- **ridges and peaks** as their own thing
- **the neighbor's roof** or any non-subject structure
- roof **outside the frame** (a plane stops at the image edge)
- walls, trim bands, window heads, tile courses

## Occlusions
Where a palm frond, tree branch, wire or vehicle crosses in front of the roof, **the plane continues
underneath as if the obstruction were not there**. **The threshold is 15 percent of image width.** Continue
straight through any gap up to that. Beyond it, stop the plane at the last readable point, start a new plane
where readable again, and flag the gap. Each image gets an occlusion metadata record:

    {
      "file": "0042.jpg",
      "occlusions": [
        {"type": "palm",  "bbox": [0.61, 0.28, 0.79, 0.55], "hides": ["plane-03"]},
        {"type": "wire",  "bbox": [0.00, 0.31, 1.00, 0.36], "hides": ["plane-01"]}
      ],
      "parapets": [],
      "notes": ""
    }

`type` ∈ palm, tree, shrub, wire, vehicle, neighbor_structure, other. `bbox` is normalized x0,y0,x1,y1.

## Labeling method
SAM with **point prompts per roof plane**, merged, then manually corrected. Not SAM automatic mode and not
box mode. Every mask gets checked over the original photo before it is called done.

## Delivery structure
    dataset/
      images/0001.jpg                 the original, unmodified
      planes/0001/plane-01.png        one binary mask per plane, 255 on, 0 off
      planes/0001/plane-02.png
      merged/0001.png                 the union of that image's planes
      meta/0001.json                  the occlusion record above
Same pixel dimensions as the source image, same numbering across every folder. Per-plane masks are the
primary artifact; merged is convenience (merged alone is not acceptable — porch and main roof overlap).

## Scoring
**Your checkpoints:** scored on your own held-out split; report holdout error curve at 50/100/150/200.
**Acceptance:** 30 photos marked separately, never shared, never in training/validation. Metric = mean
perpendicular distance between extracted predicted polyline and marked polyline, normalized by image width,
plus fraction of marked roofline horizontally covered. Bar: ≤1% and ≥90% coverage. The acceptance marks are
made under this same document.

## Consistency beats cleverness
If a case is ambiguous, ask rather than deciding quietly.
