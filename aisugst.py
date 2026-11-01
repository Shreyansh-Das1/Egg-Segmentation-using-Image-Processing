#!/usr/bin/env python3
"""
Complete Classical Image Processing Pipeline for Egg Segmentation.
Incorporates:
  1. Bilateral edge-preserving filtering
  2. Multi-space chrominance thresholding (YCrCb skin suppression + Lab/HSV)
  3. Border component clearance (removes table edges, clothing, and walls)
  4. Morphological opening and closing
  5. Distance transform with multi-tier adaptive seed extraction
  6. Marker-controlled watershed to separate touching eggs
  7. Strict geometric validation (Scale, Solidity, Aspect Ratio, Circularity)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def clear_border_components(binary_mask: np.ndarray, border_margin: int = 4) -> np.ndarray:
    """Eliminates any foreground component touching the image boundaries."""
    H, W = binary_mask.shape[:2]
    cleared = binary_mask.copy()
    contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= border_margin or y <= border_margin or (x + w) >= (W - border_margin) or (y + h) >= (H - border_margin):
            cv2.drawContours(cleared, [cnt], -1, 0, thickness=cv2.FILLED)

    return cleared


def build_candidate_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Separates eggshell reflectance from hand skin and background clutter
    via cross-color space analysis (YCrCb, HSV, L*a*b*).
    """
    H, W = img_bgr.shape[:2]
    img_area = H * W

    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    _, Cr, Cb = cv2.split(ycrcb)
    _, S, _ = cv2.split(hsv)
    L, a, b = cv2.split(lab)

    # 1. Skin suppression locus (standard YCrCb human skin boundaries)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 175, 127]))

    # 2. White Egg Profile: High Lightness, Low Saturation, Non-Skin
    _, l_thresh = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, s_inv = cv2.threshold(S, 80, 255, cv2.THRESH_BINARY_INV)

    white_egg_cand = cv2.bitwise_and(l_thresh, s_inv)
    white_egg_cand = cv2.bitwise_and(white_egg_cand, cv2.bitwise_not(skin_mask))

    # 3. Brown Egg Profile: Yellowish chrominance separation (b* - a*)
    diff = cv2.subtract(b, a)
    _, brown_egg_cand = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    brown_egg_cand = cv2.bitwise_and(brown_egg_cand, cv2.bitwise_not(skin_mask))

    # Check which color distribution captures plausible egg area
    if np.sum(white_egg_cand > 0) >= 0.005 * img_area:
        candidate = white_egg_cand
    elif np.sum(brown_egg_cand > 0) >= 0.005 * img_area:
        candidate = brown_egg_cand
    else:
        # Fallback if both chromatic filters over-suppressed
        candidate = cv2.bitwise_and(l_thresh, cv2.bitwise_not(skin_mask))

    return candidate


def segment_eggs_classical(image_bgr: np.ndarray) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    img_area = H * W

    # 1. Edge-preserving smoothing
    smoothed = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=60, sigmaSpace=60)

    # 2. Extract chromatic candidate mask
    candidate = build_candidate_mask(smoothed)

    # 3. Morphological opening to cut thin skin bridges and hair
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    cleaned = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k_open, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close, iterations=1)

    # 4. Remove elements touching image borders
    cleaned = clear_border_components(cleaned, border_margin=5)

    if np.sum(cleaned > 0) < 0.003 * img_area:
        return np.zeros((H, W), dtype=np.uint8)

    # 5. Sure Background Area
    sure_bg = cv2.dilate(cleaned, k_open, iterations=3)

    # 6. Euclidean Distance Transform
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    max_dist = dist.max()

    if max_dist <= 0:
        return np.zeros((H, W), dtype=np.uint8)

    # 7. Adaptive Multi-Tier Foreground Seed Extraction
    sure_fg = np.zeros((H, W), dtype=np.uint8)
    for ratio in [0.45, 0.32, 0.20]:
        _, sure_fg = cv2.threshold(dist, ratio * max_dist, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        num_markers, _ = cv2.connectedComponents(sure_fg)
        if num_markers > 1:
            break

    # 8. Marker Formulation & Watershed
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(smoothed, markers)

    # 9. Morphological & Geometric Validation
    final_mask = np.zeros((H, W), dtype=np.uint8)

    for label in range(2, num_markers + 1):
        comp = (markers == label).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k_open)

        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Scale Constraint: 0.6% to 45% of total frame
        if not (0.006 * img_area < area < 0.45 * img_area):
            continue

        # Solidity Constraint: Area / Convex Hull Area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.84:
            continue

        # Circularity / Form Factor: 4 * pi * Area / Perimeter^2
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.45:
            continue

        # Aspect Ratio via Fitted Ellipse
        if len(cnt) >= 5:
            (_, _), (d1, d2), _ = cv2.fitEllipse(cnt)
            major = max(d1, d2)
            minor = min(d1, d2)
            aspect_ratio = major / (minor + 1e-5)
            if not (1.01 <= aspect_ratio <= 1.90):
                continue

        # Valid egg instance confirmed
        cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return final_mask


def main():
    parser = argparse.ArgumentParser(
        description="Classical Image Processing Pipeline for Egg Segmentation."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output binary masks")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [p for p in input_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    )

    if not image_files:
        print(f"No image files found in {input_path}")
        return

    print(f"Found {len(image_files)} images. Executing classical segmentation pipeline...")

    for idx, path in enumerate(image_files, 1):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue

        mask = segment_eggs_classical(img_bgr)
        out_name = output_path / f"{path.stem}_mask.png"
        cv2.imwrite(str(out_name), mask)

        if idx % 10 == 0 or idx == len(image_files):
            print(f"[{idx}/{len(image_files)}] Processed: {path.name}")

    print(f"\nPipeline finished. Segmented masks stored at: {output_path.resolve()}")


if __name__ == "__main__":
    main()
