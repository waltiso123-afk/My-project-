# Milestone 1 (HUMAN + SAM2, NO VLM) — INTERNAL QA report (not for client)

Generated 2026-09-06T03:20:45.382120+00:00

Workflow: human identifies plane -> human places SAM2 point prompt -> real local SAM2 (CPU, Hiera-Tiny) mask ->
human visual validation -> human correction (clip/polygon) -> approve -> ml.generate_and_store (production path).
NO Gemini / NO VLM / NO automatic roof detection / NO automatic point selection anywhere.

QA PASSED: True | images: 5 | total planes: 16 | issues: none

## 0001 (1536x1152, 3 planes, parapets=[])
- 3 planes (left wing, central upper hip, garage front hip). Human-placed SAM2 points; neighbours excluded.

## 0050 (1376x768, 3 planes, parapets=[])
- 3 planes (gable left slope, gable right slope, garage hip). Left slope clip tightened by human to exclude the gable wall. Right attached villa = neighbour, excluded. Tree partly occludes left slope.

## 0138 (1536x831, 2 planes, parapets=[])
- 2 planes (left front hip, right/garage hip). Centre occluded by a large palm (>15%): front delivered as two segments per the 15% rule. Neighbour at right excluded.

## 0169 (3024x4032, 4 planes, parapets=['plane-01', 'plane-02', 'plane-03'])
- 4 planes (3 parapet caps + 1 lower canopy). PROVISIONAL/AMBIGUOUS: real local SAM2 grabbed walls (white-on-white), so human drew parapet-cap polygons. Requires client ruling (pergola? each flat roof? fascia). Flagged in meta notes.

## 0173 (5712x4284, 4 planes, parapets=[])
- 4 planes (left upper, main lower, central tower front, right upper). Clean tile roofs; SAM2 lower edge at tile bottom ~ drip edge.
