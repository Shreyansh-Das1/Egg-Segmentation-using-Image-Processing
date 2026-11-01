#!/usr/bin/env python3
"""
Task 3: Comparative Quantitative Evaluation Script.
Computes Mean IoU and Pixel Accuracy across three classical pipelines
against Ground Truth (Task 1) masks, producing publication-ready styled tables
saved via Matplotlib.
"""

import argparse
import os
from pathlib import Path
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def extract_image_index(filename: str) -> int | None:
  """Extracts numeric index 'i' from titles like 'Not Damaged (i)...'"""
  match = re.search(r"Not\s*Damaged\s*\((\d+)\)", filename, re.IGNORECASE)
  if match:
    return int(match.group(1))
  return None


def index_directory(dir_path: Path) -> dict[int, Path]:
  """Maps integer index to file path for all valid mask files in directory."""
  mapping = {}
  if not dir_path.is_dir():
    return mapping
  for p in dir_path.iterdir():
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
      idx = extract_image_index(p.stem)
      if idx is not None:
        mapping[idx] = p
  return mapping


def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
  """Computes Intersection-over-Union (IoU), Pixel Accuracy, and Dice (F1).

  pred_bin and gt_bin are binary boolean arrays (True for foreground).
  """
  intersection = np.logical_and(pred_bin, gt_bin).sum()
  union = np.logical_or(pred_bin, gt_bin).sum()

  # Standard IoU definition with empty-empty handling
  if union == 0:
    iou = 1.0 if intersection == 0 else 0.0
  else:
    iou = float(intersection) / float(union)

  # Pixel Accuracy: (TP + TN) / Total
  total_pixels = pred_bin.size
  correct_pixels = (pred_bin == gt_bin).sum()
  pixel_acc = float(correct_pixels) / float(total_pixels)

  # Dice / F1 Score
  denom = pred_bin.sum() + gt_bin.sum()
  dice = (2.0 * intersection) / denom if denom > 0 else 1.0

  return iou, pixel_acc, dice


def render_table(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    highlight_col: str = None,
    ascending: bool = False,
):
  """Renders and saves a clean, publication-ready table graphic using Matplotlib."""
  fig, ax = plt.subplots(figsize=(10, len(df) * 0.7 + 1.8), dpi=300)
  ax.axis("off")
  ax.axis("tight")

  col_labels = list(df.columns)
  cell_text = []
  cell_colors = []

  df_sorted = (
      df.sort_values(by=highlight_col, ascending=ascending)
      if highlight_col
      else df
  )

  for _, row in df_sorted.iterrows():
    row_text = []
    row_colors = []
    for col in col_labels:
      val = row[col]
      if isinstance(val, float):
        row_text.append(f"{val:.4f}")
      else:
        row_text.append(str(val))
      row_colors.append("#F8F9FA")
    cell_text.append(row_text)
    cell_colors.append(row_colors)

  table = ax.table(
      cellText=cell_text,
      colLabels=col_labels,
      cellColours=cell_colors,
      loc="center",
      cellLoc="center",
  )

  table.auto_set_font_size(False)
  table.set_fontsize(11)
  table.scale(1.2, 1.8)

  # Header styling
  for (row_idx, _), cell in table.get_celld().items():
    if row_idx == 0:
      cell.set_facecolor("#2B3A42")
      cell.set_text_props(color="white", weight="bold")
    else:
      cell.set_edgecolor("#DEE2E6")

  plt.title(title, fontsize=14, weight="bold", pad=15, color="#1A252C")
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches="tight", dpi=300)
  plt.close()
  print(f"Saved table image: {output_path}")


def main():
  parser = argparse.ArgumentParser(
      description="Compute IoU & Pixel Accuracy across 3 Classical Methods vs"
      " GT."
  )
  parser.add_argument(
      "--gt_dir",
      type=str,
      required=True,
      help="Task 1 Ground Truth Masks dir",
  )
  parser.add_argument(
      "--otsucanny_dir", type=str, required=True, help="Otsu+Canny Masks dir"
  )
  parser.add_argument(
      "--watershed_dir",
      type=str,
      required=True,
      help="Standard Watershed Masks dir",
  )
  parser.add_argument(
      "--pipeline_dir",
      type=str,
      required=True,
      help="Suggested Refined Pipeline Masks dir",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="./eval_metrics",
      help="Directory for output tables",
  )
  args = parser.parse_args()

  gt_dir = Path(args.gt_dir)
  oc_dir = Path(args.otsucanny_dir)
  ws_dir = Path(args.watershed_dir)
  pl_dir = Path(args.pipeline_dir)
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  print("Indexing directories...")
  gt_map = index_directory(gt_dir)
  oc_map = index_directory(oc_dir)
  ws_map = index_directory(ws_dir)
  pl_map = index_directory(pl_dir)

  common_indices = sorted(
      list(
          set(gt_map.keys())
          & (set(oc_map.keys()) | set(ws_map.keys()) | set(pl_map.keys()))
      )
  )
  print(
      f"Found {len(common_indices)} paired evaluation samples with Ground"
      " Truth."
  )

  methods = {
      "Trial 1: Otsu + Canny": oc_map,
      "Trial 2: Watershed": ws_map,
      "Trial 3: AI-Watershed Refined": pl_map,
  }

  records = []

  for idx in common_indices:
    gt_path = gt_map[idx]
    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
      continue
    gt_bin = gt_img > 127
    H, W = gt_bin.shape

    for name, m_map in methods.items():
      if idx in m_map:
        pred_img = cv2.imread(str(m_map[idx]), cv2.IMREAD_GRAYSCALE)
        if pred_img is not None:
          if pred_img.shape != (H, W):
            pred_img = cv2.resize(
                pred_img, (W, H), interpolation=cv2.INTER_NEAREST
            )
          pred_bin = pred_img > 127
        else:
          pred_bin = np.zeros_like(gt_bin)
      else:
        pred_bin = np.zeros_like(gt_bin)

      iou, pix_acc, dice = compute_metrics(pred_bin, gt_bin)
      records.append({
          "Sample_ID": idx,
          "Method": name,
          "IoU": iou,
          "Pixel_Accuracy": pix_acc,
          "Dice": dice,
      })

  df_all = pd.DataFrame(records)

  # -------------------------------------------------------------
  # Table 1: Global Summary Metrics (Mean IoU, Dice, Pixel Acc)
  # -------------------------------------------------------------
  summary_list = []
  for name in methods.keys():
    sub = df_all[df_all["Method"] == name]
    summary_list.append({
        "Method / Trial Pipeline": name,
        "Mean IoU (Jaccard)": sub["IoU"].mean(),
        "Std IoU": sub["IoU"].std(),
        "Mean Pixel Accuracy": sub["Pixel_Accuracy"].mean(),
        "Mean Dice (F1)": sub["Dice"].mean(),
    })

  df_summary = pd.DataFrame(summary_list)
  df_summary.to_csv(out_dir / "summary_metrics.csv", index=False)
  print("\n=== Global Evaluation Summary ===")
  print(df_summary.to_string(index=False))

  render_table(
      df_summary,
      title=(
          "Table 1: Quantitative Segmentation Performance vs Ground Truth"
          " (Task 1)"
      ),
      output_path=out_dir / "table1_overall_iou_metrics.png",
      highlight_col="Mean IoU (Jaccard)",
      ascending=False,
  )

  # -------------------------------------------------------------
  # Table 2: Pixel Accuracy Distribution & Dropout Diagnostics
  # -------------------------------------------------------------
  breakdown_list = []
  for name in methods.keys():
    sub = df_all[df_all["Method"] == name]
    total = len(sub)
    dropouts = (sub["IoU"] == 0.0).sum()
    high_quality = (sub["IoU"] >= 0.70).sum()

    breakdown_list.append({
        "Method Pipeline": name,
        "Mean Pixel Acc (%)": sub["Pixel_Accuracy"].mean() * 100.0,
        "Median Pixel Acc (%)": sub["Pixel_Accuracy"].median() * 100.0,
        "Min Pixel Acc (%)": sub["Pixel_Accuracy"].min() * 100.0,
        "Complete Dropouts (IoU=0)": (
            f"{dropouts} / {total} ({dropouts/total*100:.1f}%)"
        ),
        "High Overlap (IoU ≥ 0.70)": (
            f"{high_quality} / {total} ({high_quality/total*100:.1f}%)"
        ),
    })

  df_breakdown = pd.DataFrame(breakdown_list)
  df_breakdown.to_csv(out_dir / "pixel_accuracy_breakdown.csv", index=False)
  print("\n=== Pixel Accuracy & Failure Distribution ===")
  print(df_breakdown.to_string(index=False))

  render_table(
      df_breakdown,
      title="Table 2: Pixel Accuracy Distribution & Dropout Diagnostics",
      output_path=out_dir / "table2_pixel_accuracy_breakdown.png",
  )

  print(f"\nAll tables generated successfully in: {out_dir.resolve()}")


if __name__ == "__main__":
  main()
