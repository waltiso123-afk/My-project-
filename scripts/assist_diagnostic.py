#!/usr/bin/env python3
"""AUDIT diagnostic (read-only): probe the current CPU flood-fill assist on image 0001.

Does NOT modify any annotation or dataset state. Picks color-classified sample points
(road / grass / wall / red-tile roof / vegetation / window-dark / sky) and runs the exact
propose() flood-fill at several tolerances, reporting region area% and returned-polygon count.
"""
import sys
sys.path.insert(0, "/app/backend")
import numpy as np
import cv2
from PIL import Image, ImageOps
import config

PATH = config.DATA_DIR / "staging" / "batch1" / "0001.webp"  # dataset_id 0001


def floodfill_region(img, seed, tol):
    h, w = img.shape[:2]
    seed = (int(np.clip(seed[0], 0, w - 1)), int(np.clip(seed[1], 0, h - 1)))
    fm = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(img.copy(), fm, seed, 0, (tol,) * 3, (tol,) * 3, flags)
    m = fm[1:-1, 1:-1]
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in cnts if cv2.contourArea(c) >= 50]
    return int(m.sum() / 255), len(kept)


def local_variance(gray, seed, r=7):
    x, y = seed
    patch = gray[max(0, y - r):y + r, max(0, x - r):x + r]
    return round(float(np.std(patch)), 1)


def main():
    img = np.array(ImageOps.exif_transpose(Image.open(PATH)).convert("RGB"))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    R, G, B = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    total = h * w
    print(f"image 0001 EXIF-applied dims = {w}x{h} ({total} px)\n")

    # color-classified masks
    classes = {
        "red_tile_roof": (R > 120) & (R - G > 25) & (R - B > 25),
        "grass":         (G > 80) & (G - R > 15) & (G - B > 10),
        "road_gray":     (np.abs(R - G) < 18) & (np.abs(G - B) < 18) & (gray > 60) & (gray < 150),
        "wall_bright":   (gray > 180) & (np.abs(R - G) < 30) & (np.abs(G - B) < 40),
        "sky_blue":      (B - R > 20) & (B > 120),
        "dark_window":   (gray < 55),
    }
    print(f"{'class':14} {'seed(x,y)':>12} {'localStd':>8}   tol22           tol40           tol60")
    print(f"{'':14} {'':>12} {'':>8}   area%/polys     area%/polys     area%/polys")
    for name, mask in classes.items():
        ys, xs = np.where(mask)
        if len(xs) == 0:
            print(f"{name:14} {'(none)':>12}")
            continue
        # pick the densest point: centroid clamped to an actual class pixel
        cx, cy = int(np.median(xs)), int(np.median(ys))
        # snap to nearest class pixel
        d = (xs - cx) ** 2 + (ys - cy) ** 2
        k = int(np.argmin(d)); cx, cy = int(xs[k]), int(ys[k])
        lv = local_variance(gray, (cx, cy))
        cells = []
        for tol in (22, 40, 60):
            area, polys = floodfill_region(img, (cx, cy), tol)
            cells.append(f"{100*area/total:5.1f}/{polys:<2}")
        print(f"{name:14} {f'({cx},{cy})':>12} {lv:>8}   " + "   ".join(f"{c:14}" for c in cells))

    print("\nInterpretation:")
    print("- Homogeneous surfaces (road/grass/sky/wall) => low localStd, large flood region.")
    print("- Textured red-tile roof => high localStd; flood stops at grout/shadow edges =>")
    print("  tiny/empty region at tol22 ('No region found'). Higher tol leaks into non-roof.")


if __name__ == "__main__":
    main()
