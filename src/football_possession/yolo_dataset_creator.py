"""
Convert annotated frames to YOLO format dataset.

YOLO format expects:
- images/ folder with .jpg files
- labels/ folder with corresponding .txt files
- Each .txt file contains: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)

For ball detection, we use class_id = 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def create_yolo_dataset(
    video_path: str,
    annotations_path: str,
    output_dir: str,
    train_split: float = 0.8,
    skip_frames: int = 1,
) -> None:
    """
    Create YOLO-format dataset from annotated video.
    
    Args:
        video_path: Path to video file
        annotations_path: Path to annotations.json
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data for training (rest goes to validation)
        skip_frames: Extract every Nth frame (1 = every frame, 2 = every other frame, etc.)
    """
    video_path = Path(video_path)
    annotations_path = Path(annotations_path)
    output_dir = Path(output_dir)
    
    # Load annotations
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    # Convert string keys to integers
    annotations = {int(k): v for k, v in annotations.items()}
    
    if not annotations:
        print("No annotations found!")
        return
    
    print(f"Found {len(annotations)} annotated frames")
    
    # Create directory structure (train/val subdirs built after split is known)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Collect all frames first so we can assign train/val before writing
    raw_frames: list[tuple[int, float, float]] = []
    for frame_idx, (x, y) in sorted(annotations.items()):
        if frame_idx % skip_frames != 0:
            continue
        raw_frames.append((frame_idx, x, y))

    num_train = int(len(raw_frames) * train_split)
    train_set = {fi for fi, _, _ in raw_frames[:num_train]}

    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    dataset: list[tuple[str, str]] = []
    for frame_idx, x, y in raw_frames:
        split = "train" if frame_idx in train_set else "val"

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx}")
            continue

        image_filename = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(images_dir / split / image_filename), frame)

        # x, y are normalized centers; ball_size is fixed at 2% of image
        ball_size = 0.02
        label_filename = f"frame_{frame_idx:06d}.txt"
        with open(labels_dir / split / label_filename, "w") as f:
            f.write(f"0 {x:.6f} {y:.6f} {ball_size:.6f} {ball_size:.6f}\n")

        dataset.append((image_filename, label_filename))
        print(f"Frame {frame_idx} [{split}]: ({x:.3f}, {y:.3f}) -> {image_filename}")

    cap.release()

    train_count = len(train_set)
    val_count = len(dataset) - train_count
    print(f"\nDataset split: {train_count} train, {val_count} validation")

    # Create data.yaml for YOLO training
    import yaml as _yaml
    data_yaml = output_dir / "data.yaml"
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["ball"],
    }
    with open(data_yaml, "w") as f:
        _yaml.safe_dump(config, f, sort_keys=False)

    print(f"\nCreated data.yaml at {data_yaml}")
    print(f"Images saved to: {images_dir}/{{train,val}}/")
    print(f"Labels saved to: {labels_dir}/{{train,val}}/")
    print("\nYOLO dataset is ready for training!")
    print(f"Use: yolo detect train data={data_yaml} model=yolov8n.pt epochs=100")


def main():
    parser = argparse.ArgumentParser(description="Convert annotations to YOLO format")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--annotations", required=True, help="Path to annotations.json")
    parser.add_argument("--output", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--skip-frames", type=int, default=1, help="Extract every Nth frame")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split ratio")
    
    args = parser.parse_args()
    
    create_yolo_dataset(
        args.video,
        args.annotations,
        args.output,
        train_split=args.train_split,
        skip_frames=args.skip_frames,
    )


if __name__ == "__main__":
    main()
