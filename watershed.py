#!/usr/bin/env python3
"""
Trial 3: Marker-Controlled Distance-Transform Watershed Segmentation.
Designed for Task 2 to separate touching eggs and clean boundary contacts.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def segment_egg_watershed(image_bgr: np.ndarray) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    img_area = H * W

    # 1. Edge-preserving smoothing
    smoothed = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. Color-space candidate masking (Separating shell from hand skin)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)

    L, a, b = cv2.split(lab)
    _, S, _ = cv2.split(hsv)

    # White egg prompt: High Luminance + Low Saturation
    _, l_mask = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, s_inv_mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidate = cv2.bitwise_and(l_mask, s_inv_mask)

    # Brown egg fallback (b* - a* difference)
    if np.sum(candidate > 0) < 0.005 * img_area:
        diff = cv2.subtract(b, a)
        _, candidate = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological opening to eliminate fine finger edges & creases
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Sure Background Area (Dilate candidate mask)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. Sure Foreground Area via Euclidean Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_val = dist_transform.max()
    if max_val <= 0:
        return np.zeros((H, W), dtype=np.uint8)

    # Threshold distance transform to locate distinct egg core centers
    _, sure_fg = cv2.threshold(dist_transform, 0.42 * max_val, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 6. Unknown Boundary Region (between sure foreground and sure background)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Marker Labeling
    num_markers, markers = cv2.connectedComponents(sure_fg)

    if num_markers <= 1:
        # Fallback if distance thresholding eliminated weak detections
        _, sure_fg = cv2.threshold(dist_transform, 0.25 * max_val, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        num_markers, markers = cv2.connectedComponents(sure_fg)

    # Shift markers so background is 1, unknown is 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # 8. Run Watershed
    # Watershed expects a 3-channel BGR image to compute topographic gradients
    cv2.watershed(smoothed, markers)

    # 9. Extract and Filter Individual Segmented Egg Components
    final_mask = np.zeros((H, W), dtype=np.uint8)

    # Marker indices 2..num_markers correspond to foreground objects
    for label in range(2, num_markers + 1):
        component_mask = (markers == label).astype(np.uint8) * 255
        
        # Morphological close to bridge boundary pixels (-1 watershed ridge lines)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Scale Constraint: 0.8% to 45% of total frame
        if not (0.008 * img_area < area < 0.45 * img_area):
            continue

        # Solidity Constraint (Eggs are highly convex ovoids)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.82:
            continue

        # Aspect Ratio Validation
        if len(cnt) >= 5:
            (_, _), (d1, d2), _ = cv2.fitEllipse(cnt)
            major = max(d1, d2)
            minor = min(d1, d2)
            aspect_ratio = major / (minor + 1e-5)
            if not (1.02 <= aspect_ratio <= 1.95):
                continue

        # Merge validated egg mask
        cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return final_mask


def main():
    parser = argparse.ArgumentParser(
        description="Run Marker-Controlled Watershed Egg Segmentation."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input images folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output masks folder")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    print(f"Running Watershed segmentation on {len(image_paths)} images...")

    for idx, p in enumerate(image_paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        mask = segment_egg_watershed(img)
        out_file = output_dir / f"{p.stem}_watershed_mask.png"
        cv2.imwrite(str(out_file), mask)

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] Processed {p.name}")

    print(f"Processing complete. Masks stored in: {output_dir}")


if __name__ == "__main__":
    main()
