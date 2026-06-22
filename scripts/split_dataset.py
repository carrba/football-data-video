"""
Reorganise an existing flat YOLO dataset into proper train/val subdirectories.

A flat dataset has:
    images/frame_000001.jpg  ...
    labels/frame_000001.txt  ...
    data.yaml  (train: images, val: images  <-- wrong)

After running, it becomes:
    images/train/frame_000001.jpg  ...
    images/val/frame_000050.jpg    ...
    labels/train/frame_000001.txt  ...
    labels/val/frame_000050.txt    ...
    data.yaml  (train: images/train, val: images/val  <-- correct)

Usage:
    # Fix a single dataset in place (80/20 split, seed 42):
    python scripts/split_dataset.py datasets/ball_detector_060628

    # Fix multiple datasets and also create a merged combined dataset:
    python scripts/split_dataset.py datasets/ball_detector_060628 datasets/ball_detector_clip_14_30 --merge datasets/ball_detector_combined

    # Custom split ratio:
    python scripts/split_dataset.py datasets/ball_detector_060628 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


def is_flat(dataset_dir: Path) -> bool:
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        return False
    train_subdir = images_dir / "train"
    return not train_subdir.exists()


def split_dataset(dataset_dir: Path, val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    """
    Reorganise a flat dataset into images/train, images/val, labels/train, labels/val.
    Returns (train_image_paths, val_image_paths).
    """
    dataset_dir = dataset_dir.resolve()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not is_flat(dataset_dir):
        print(f"  {dataset_dir.name}: already split, skipping reorganisation")
        return [], []

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        print(f"  {dataset_dir.name}: no images found, skipping")
        return [], []

    rng = random.Random(seed)
    shuffled = image_files[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_ratio))
    val_images = shuffled[:n_val]
    train_images = shuffled[n_val:]

    for split, imgs in [("train", train_images), ("val", val_images)]:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)
        for img_path in imgs:
            label_path = labels_dir / img_path.with_suffix(".txt").name
            shutil.move(str(img_path), str(images_dir / split / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(labels_dir / split / label_path.name))

    # Remove labels.cache if present (stale after reorganisation)
    cache = labels_dir / "labels.cache"
    if cache.exists():
        cache.unlink()

    _write_data_yaml(dataset_dir, dataset_dir)

    print(
        f"  {dataset_dir.name}: {len(train_images)} train / {len(val_images)} val"
        f"  ({len(image_files)} total)"
    )
    return train_images, val_images


def _write_data_yaml(dataset_dir: Path, path_root: Path) -> None:
    data_yaml = dataset_dir / "data.yaml"
    existing: dict = {}
    if data_yaml.exists():
        with open(data_yaml, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    config = {
        "path": str(path_root),
        "train": "images/train",
        "val": "images/val",
        "nc": existing.get("nc", 1),
        "names": existing.get("names", ["ball"]),
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def create_merged(sources: list[Path], merged_dir: Path, val_ratio: float, seed: int) -> None:
    """
    Create a new combined dataset by copying files from multiple already-split sources.
    Each source must already have images/train, images/val structure.
    """
    merged_dir = merged_dir.resolve()
    for split in ("train", "val"):
        (merged_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"train": 0, "val": 0}
    for src in sources:
        src = src.resolve()
        for split in ("train", "val"):
            img_src = src / "images" / split
            lbl_src = src / "labels" / split
            if not img_src.exists():
                continue
            for img in sorted(img_src.iterdir()):
                if img.suffix.lower() not in {".jpg", ".png"}:
                    continue
                # Prefix with source dir name to avoid name collisions
                stem = f"{src.name}__{img.name}"
                shutil.copy2(str(img), str(merged_dir / "images" / split / stem))
                lbl = lbl_src / img.with_suffix(".txt").name
                if lbl.exists():
                    shutil.copy2(str(lbl), str(merged_dir / "labels" / split / Path(stem).with_suffix(".txt").name))
                counts[split] += 1

    _write_data_yaml(merged_dir, merged_dir)
    print(f"  {merged_dir.name}: {counts['train']} train / {counts['val']} val (merged)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix YOLO dataset train/val split")
    parser.add_argument("datasets", nargs="+", help="Dataset directory paths to fix")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--merge", metavar="DIR", help="Also create a merged dataset at this path")
    args = parser.parse_args()

    print("Splitting datasets:")
    dataset_dirs = [Path(d) for d in args.datasets]
    for d in dataset_dirs:
        if not d.exists():
            print(f"  {d}: directory not found, skipping")
            continue
        split_dataset(d, val_ratio=args.val_ratio, seed=args.seed)

    if args.merge:
        print("\nCreating merged dataset:")
        create_merged(dataset_dirs, Path(args.merge), val_ratio=args.val_ratio, seed=args.seed)

    print("\nDone. Update your data.yaml paths if training on a specific dataset.")


if __name__ == "__main__":
    main()
