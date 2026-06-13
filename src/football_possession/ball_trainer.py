"""
Train a custom YOLO model for ball detection.

Usage:
    python -m football_possession.ball_trainer --data-yaml path/to/data.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is required. Install with: pip install ultralytics")
    print(f"Active Python: {sys.executable}")
    print("Tip: install into this interpreter with:")
    print(f"  {sys.executable} -m pip install ultralytics")
    exit(1)


def _resolve_device(device: int | str) -> int | str:
    """Resolve requested device, supporting automatic CPU fallback."""
    if isinstance(device, str) and device.lower() == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return 0 if torch.cuda.is_available() else "cpu"
    return device


def _prepare_data_yaml_for_runtime(data_yaml: Path) -> Path:
    """Create a runtime-safe YOLO data.yaml with a normalized absolute dataset path."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        return data_yaml

    dataset_root = data_yaml.parent.resolve()
    path_value = config.get("path")
    normalized_path = None

    if not isinstance(path_value, str) or not path_value.strip() or path_value.strip() == ".":
        normalized_path = dataset_root
    else:
        path_text = path_value.strip()
        is_windows_abs = bool(re.match(r"^[A-Za-z]:[\\/]", path_text))
        if is_windows_abs and os.name != "nt":
            normalized_path = dataset_root
        else:
            candidate = Path(path_text)
            if not candidate.is_absolute():
                normalized_path = (dataset_root / candidate).resolve()

    if normalized_path is None:
        return data_yaml

    config["path"] = str(normalized_path)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="ball_train_data_", delete=False, encoding="utf-8"
    ) as temp_file:
        yaml.safe_dump(config, temp_file, sort_keys=False)
        return Path(temp_file.name)


def train_ball_detector(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    device: int | str = "auto",
    patience: int = 20,
    output_dir: str = "runs/detect/ball_detector",
) -> None:
    """
    Train a custom YOLO model for ball detection.
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Image size for training
        device: 'auto', GPU device ID, or 'cpu'
        patience: Early stopping patience (0 = disabled)
        output_dir: Output directory for model and results
    """
    data_yaml = Path(data_yaml)
    
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return

    runtime_data_yaml = _prepare_data_yaml_for_runtime(data_yaml)
    
    resolved_device = _resolve_device(device)

    print(f"Starting training:")
    print(f"  Data YAML: {data_yaml}")
    if runtime_data_yaml != data_yaml:
        print(f"  Runtime data YAML: {runtime_data_yaml}")
    print(f"  Model size: yolov8{model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: {resolved_device} (requested: {device})")
    print(f"  Early stopping patience: {patience}")
    print()
    
    # Load base model
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Train
    results = model.train(
        data=str(runtime_data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device=resolved_device,
        patience=patience,
        save=True,
        project="runs/detect",
        name="ball_detector",
        # Hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        # Other
        batch=16,
        workers=4,
        verbose=True,
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: runs/detect/ball_detector/weights/best.pt")
    print(f"Last model saved to: runs/detect/ball_detector/weights/last.pt")
    print("\nTo use trained model:")
    print("  model = YOLO('runs/detect/ball_detector/weights/best.pt')")
    print("  results = model.predict('image.jpg')")
    
    return results


def evaluate_model(model_path: str, data_yaml: str) -> None:
    """Evaluate a trained model."""
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)
    
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        return
    
    if not data_yaml.exists():
        print(f"Error: {data_yaml} not found")
        return

    runtime_data_yaml = _prepare_data_yaml_for_runtime(data_yaml)
    
    print(f"Evaluating model: {model_path}")
    if runtime_data_yaml != data_yaml:
        print(f"Runtime data YAML: {runtime_data_yaml}")
    model = YOLO(str(model_path))
    metrics = model.val(data=str(runtime_data_yaml))
    
    print("\nEvaluation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLO model for ball detection")
    parser.add_argument("--data-yaml", required=True, help="Path to data.yaml")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="n",
                       help="YOLO model size (default: n for faster training)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: 'auto' (default), GPU id like 0, or 'cpu'",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--eval", action="store_true", help="Evaluate model instead of training")
    parser.add_argument("--model", help="Model path for evaluation")
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model:
            print("Error: --model required for evaluation")
            return
        evaluate_model(args.model, args.data_yaml)
    else:
        train_ball_detector(
            args.data_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
