#!/usr/bin/env python3


"""
Dataset Splitter Script (4:1 Dev/Eval Split).
Splits matched pairs from 'Eggs' and 'Masks' folders while preventing data leakage.
"""

import argparse
import os
from pathlib import Path
import random
import re
import shutil

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def extract_image_index(filename: str) -> int | None:
  """Extracts the integer index 'i' from titles like 'Not Damaged (i)'."""
  match = re.search(r"Not\s*Damaged\s*\((\d+)\)", filename, re.IGNORECASE)
  if match:
    return int(match.group(1))
  return None


def split_dataset(
    eggs_dir: Path, masks_dir: Path, output_dir: Path, seed: int = 42
):
  egg_files = [
      p
      for p in eggs_dir.iterdir()
      if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
  ]
  mask_files = [
      p
      for p in masks_dir.iterdir()
      if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
  ]

  # Map index -> Path
  egg_map = {}
  for p in egg_files:
    idx = extract_image_index(p.stem)
    if idx is not None:
      egg_map[idx] = p

  mask_map = {}
  for p in mask_files:
    idx = extract_image_index(p.stem)
    if idx is not None:
      mask_map[idx] = p

  # Pair on identical numeric indices
  common_indices = sorted(list(set(egg_map.keys()) & set(mask_map.keys())))

  if not common_indices:
    print(
        "Error: No matching pairs of 'Not Damaged (i)' images and masks found."
    )
    return

  print(f"Total matching pairs found: {len(common_indices)}")

  # Deterministic shuffle to ensure reproducibility across runs
  rng = random.Random(seed)
  shuffled_indices = common_indices.copy()
  rng.shuffle(shuffled_indices)

  # 4:1 Split (80% Dev, 20% Eval)
  num_total = len(shuffled_indices)
  num_eval = int(round(num_total * 0.20))
  num_dev = num_total - num_eval

  dev_indices = shuffled_indices[:num_dev]
  eval_indices = shuffled_indices[num_dev:]

  print(
      f"Split breakdown: Dev = {len(dev_indices)}"
      f" ({len(dev_indices)/num_total*100:.1f}%), Eval = {len(eval_indices)}"
      f" ({len(eval_indices)/num_total*100:.1f}%)"
  )

  splits = {"dev": dev_indices, "eval": eval_indices}

  for split_name, indices in splits.items():
    split_eggs_dir = output_dir / split_name / "Eggs"
    split_masks_dir = output_dir / split_name / "Masks"
    split_eggs_dir.mkdir(parents=True, exist_ok=True)
    split_masks_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
      egg_src = egg_map[idx]
      mask_src = mask_map[idx]

      shutil.copy2(egg_src, split_eggs_dir / egg_src.name)
      shutil.copy2(mask_src, split_masks_dir / mask_src.name)

  print(f"Dataset successfully split and saved to: {output_dir.resolve()}")


def main():
  parser = argparse.ArgumentParser(
      description="Split dataset into 4:1 Dev/Eval splits."
  )
  parser.add_argument(
      "--eggs_dir",
      type=str,
      default="Eggs",
      help="Path to source Eggs directory",
  )
  parser.add_argument(
      "--masks_dir",
      type=str,
      default="Masks",
      help="Path to source Masks directory",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="dataset_split",
      help="Target directory for output splits",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for deterministic split",
  )
  args = parser.parse_args()

  split_dataset(
      Path(args.eggs_dir),
      Path(args.masks_dir),
      Path(args.output_dir),
      seed=args.seed,
  )


if __name__ == "__main__":
  main()
