# Mini-Milestone 1 — INTERNAL QA report (not for client)

Generated 2026-09-05T21:58:25.907064+00:00

Pipeline: VLM reasoning/plan (gemini-3.1-pro-preview) -> SAM2 (CPU, hiera-tiny) point prompts ->
clip/reconcile -> human (agent) QA & correction -> ml.generate_and_store (production path).

QA PASSED: True   |   images: 5   |   total planes: 16

## 0001  (1536x1152, 3 planes, parapets=[])
- corrections/observations: 3 planes (left wing, central upper hip, garage front hip). VLM over-segmented to 11; QA consolidated to 3 real planes. SAM2 points repositioned off palms/walls. Neighbours excluded.
- per-plane px areas: [4764, 8495, 15573]

## 0050  (1376x768, 3 planes, parapets=[])
- corrections/observations: 3 planes (gable left slope, gable right slope, garage hip). Right attached villa = neighbour, excluded. VLM falsely proposed parapet planes; corrected to pitched. Left gable slope partly tree-occluded.
- per-plane px areas: [10782, 11571, 12091]

## 0138  (1536x831, 2 planes, parapets=[])
- corrections/observations: 2 planes (left front hip, right/garage hip). Centre front plane is occluded by a large palm (>15% width) so front is delivered as two segments per the 15% rule. Neighbour at right excluded.
- per-plane px areas: [15019, 4281]

## 0169  (3024x4032, 4 planes, parapets=['plane-01', 'plane-02', 'plane-03'])
- corrections/observations: 4 planes (3 parapet caps + 1 lower canopy). PROVISIONAL/AMBIGUOUS: flat-roof/parapet home; only parapet caps visible from ground. Automatic SAM2 grabbed walls, so parapet planes were hand-drawn along the caps. Requires client ruling (pergola? each flat roof? fascia top). Documented in meta notes.
- per-plane px areas: [56416, 35796, 147962, 177188]

## 0173  (5712x4284, 4 planes, parapets=[])
- corrections/observations: 4 planes (left upper, main lower, central tower front, right upper). Clean tile roofs; SAM2 lower edge sits at tile bottom ~ drip edge. Palm over right upper ~12% (<15%), plane continued.
- per-plane px areas: [87090, 329471, 163929, 403254]

## File integrity
- issues found: none
- all masks binary and at image dimensions; merged == union of planes; originals unchanged; parapet filename/metadata consistent; exactly 5 images, no extras.