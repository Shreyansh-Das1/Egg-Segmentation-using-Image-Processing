#!/usr/bin/env python3
"""
Baseline Classical Segmentation Pipeline for Task 2.
Combines Canny Edge Detection, Otsu Thresholding, Morphological Operations,
and Geometric Constraints (Area, Solidity, Aspect Ratio).
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def segment_egg_classical(image_bgr: np.ndarray) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    img_area = H * W

    # 1. Preprocessing & Color conversion
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 2. Otsu Thresholding (Region Proposal)
    _, otsu_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert if the background is brighter than the object
    if np.mean(otsu_mask) > 127:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    # 3. Canny Edge Detection
    # Derive adaptive threshold bounds based on median intensity
    median_val = np.median(gray)
    lower_thresh = int(max(0, (1.0 - 0.33) * median_val))
    upper_thresh = int(min(255, (1.0 + 0.33) * median_val))
    edges = cv2.Canny(gray, lower_thresh, upper_thresh)

    # 4. Morphological Boundary Closing
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge)

    # Fill closed contours formed by Canny
    edge_filled = np.zeros((H, W), dtype=np.uint8)
    contours_edge, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours_edge:
        if cv2.contourArea(c) > 0.005 * img_area:
            cv2.drawContours(edge_filled, [c], -1, 255, thickness=cv2.FILLED)

    # 5. Fusion: Constrain Otsu region proposal with Canny edge containment
    combined_raw = cv2.bitwise_or(otsu_mask, edge_filled)

    # Morphological Clean-up
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(combined_raw, cv2.MORPH_OPEN, kernel_clean)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_clean)

    # 6. Geometric Property Filtering
    final_mask = np.zeros((H, W), dtype=np.uint8)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Scale Constraint: 0.5% to 40% of image size
        if not (0.005 * img_area < area < 0.40 * img_area):
            continue

        # Solidity Constraint: Area / Convex Hull Area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.85:
            continue

        # Circularity / Form Factor: 4 * pi * Area / Perimeter^2
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if not (0.55 <= circularity <= 0.95):
            continue

        # Aspect Ratio via Fitted Ellipse
        if len(cnt) >= 5:
            (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(cnt)
            aspect_ratio = max(major_axis, minor_axis) / (
                min(major_axis, minor_axis) + 1e-5
            )
            if not (1.05 <= aspect_ratio <= 1.85):
                continue
        else:
            continue

        # Valid candidate: write to final mask
        cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return final_mask


def main():
    parser = argparse.ArgumentParser(
        description="Run Canny + Otsu + Aspect Ratio Classical Egg Segmentation."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to development/eval split images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save output binary masks",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        [p for p in input_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    )
    if not img_files:
        print(f"No valid image files located in {input_path}")
        return

    print(f"Processing {len(img_files)} images via classical edge-otsu fusion...")
    for idx, path in enumerate(img_files, 1):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue

        mask = segment_egg_classical(img_bgr)
        out_name = output_path / f"{path.stem}_classical_mask.png"
        cv2.imwrite(str(out_name), mask)

        if idx % 10 == 0 or idx == len(img_files):
            print(f"[{idx}/{len(img_files)}] Processed {path.name}")

    print(f"Classical inference finished. Output masks stored in: {output_path}")


if __name__ == "__main__":
    main()
