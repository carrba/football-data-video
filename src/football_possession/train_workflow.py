"""
Quick-start workflow for ball tracking model training.

This script guides you through the entire training pipeline in order.

Usage:
    python -m football_possession.train_workflow --step 1 --video video/clip_14_30.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def find_latest_best_model() -> Path | None:
    """Return the most recently trained best.pt, accounting for YOLO's auto-incremented run dirs."""
    candidates = sorted(
        Path("runs/detect").glob("ball_detector*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def step_1_annotate(video_path: str):
    """Step 1: Annotate video frames."""
    print_header("STEP 1: Annotate Ball Positions")
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return False
    
    output_dir = Path("annotations") / video_path.stem
    
    print(f"Video: {video_path}")
    print(f"Annotations will be saved to: {output_dir}")
    print()
    print("Instructions:")
    print("  1. Click on the ball in each frame")
    print("  2. Press SPACE to save and move to next frame")
    print("  3. Press N/P to navigate without annotating")
    print("  4. Press S to save progress")
    print("  5. Press Q when done")
    print()
    print("Starting annotation tool...\n")
    
    cmd = [
        sys.executable, "-m", "football_possession.ball_annotation",
        "--video", str(video_path),
        "--output", str(output_dir),
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("ERROR: Annotation tool failed")
        return False
    
    annotations_file = output_dir / "annotations.json"
    if not annotations_file.exists():
        print("ERROR: No annotations were saved")
        return False
    
    print(f"✓ Annotations saved to: {annotations_file}")
    return True


def step_2_create_dataset(video_path: str, skip_frames: int = 1):
    """Step 2: Create YOLO dataset."""
    print_header("STEP 2: Create YOLO Dataset")
    
    video_path = Path(video_path)
    annotations_dir = Path("annotations") / video_path.stem
    annotations_file = annotations_dir / "annotations.json"
    
    if not annotations_file.exists():
        print(f"ERROR: Annotations not found: {annotations_file}")
        print("Run step 1 first!")
        return False
    
    output_dir = Path("datasets") / f"ball_detector_{video_path.stem}"
    
    print(f"Converting annotations to YOLO format...")
    print(f"  Input: {annotations_file}")
    print(f"  Output: {output_dir}")
    print(f"  Skip frames: {skip_frames} (use every {skip_frames}th frame)")
    print()
    
    cmd = [
        sys.executable, "-m", "football_possession.yolo_dataset_creator",
        "--video", str(video_path),
        "--annotations", str(annotations_file),
        "--output", str(output_dir),
        "--skip-frames", str(skip_frames),
        "--train-split", "0.8",
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("ERROR: Dataset creation failed")
        return False
    
    data_yaml = output_dir / "data.yaml"
    if not data_yaml.exists():
        print("ERROR: data.yaml not created")
        return False
    
    print(f"✓ Dataset created: {output_dir}")
    print(f"✓ Configuration: {data_yaml}")
    return True


def step_3_train_model(
    data_yaml_path: str,
    model_size: str = "n",
    epochs: int = 100,
):
    """Step 3: Train YOLO model."""
    print_header("STEP 3: Train Ball Detection Model")
    
    data_yaml = Path(data_yaml_path)
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found: {data_yaml}")
        print("Run step 2 first!")
        return False
    
    print(f"Training configuration:")
    print(f"  Dataset: {data_yaml}")
    print(f"  Model size: yolov8{model_size} (nano, small, medium, large, xlarge)")
    print(f"  Epochs: {epochs}")
    print()
    print("This may take 5-30 minutes depending on dataset size and GPU...")
    print()
    
    cmd = [
        sys.executable, "-m", "football_possession.ball_trainer",
        "--data-yaml", str(data_yaml),
        "--model-size", model_size,
        "--epochs", str(epochs),
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("ERROR: Training failed")
        return False
    
    best_model = find_latest_best_model()
    if best_model is None:
        print("ERROR: Trained model not found")
        return False

    print(f"✓ Training complete!")
    print(f"✓ Best model: {best_model}")
    return True


def step_4_evaluate(data_yaml_path: str = "datasets/ball_detector_clip_14_30/data.yaml"):
    """Step 4: Evaluate trained model."""
    print_header("STEP 4: Evaluate Model")
    
    model_path = find_latest_best_model()
    if model_path is None:
        print("ERROR: No trained model found under runs/detect/")
        return False

    print(f"Evaluating model...")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {data_yaml_path}")
    print()
    
    cmd = [
        sys.executable, "-m", "football_possession.ball_trainer",
        "--eval",
        "--model", str(model_path),
        "--data-yaml", data_yaml_path,
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("WARNING: Evaluation had issues")
        return False
    
    print(f"✓ Evaluation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ball tracking model training workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate a video
  python -m football_possession.train_workflow --step 1 --video video/clip_14_30.mp4
  
  # Create dataset
  python -m football_possession.train_workflow --step 2 --video video/clip_14_30.mp4
  
  # Train model
  python -m football_possession.train_workflow --step 3 --data-yaml datasets/ball_detector_clip_14_30/data.yaml
  
  # Run all steps
  python -m football_possession.train_workflow --all --video video/clip_14_30.mp4
        """)
    
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                       help="Run specific step (1=annotate, 2=dataset, 3=train, 4=eval)")
    parser.add_argument("--all", action="store_true",
                       help="Run all steps in sequence")
    parser.add_argument("--video", help="Path to video file (for steps 1-2)")
    parser.add_argument("--data-yaml", help="Path to data.yaml (for steps 3-4)")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                       help="Model size for training (default: n)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--skip-frames", type=int, default=1,
                       help="Extract every Nth frame (default: 1)")
    
    args = parser.parse_args()
    
    if not args.step and not args.all:
        parser.print_help()
        return
    
    # Show introduction
    print_header("Ball Tracking Model Training Workflow")
    print("This workflow will guide you through training a custom YOLO model")
    print("specifically optimized for detecting balls in your video footage.\n")
    print("For detailed instructions, see: BALL_TRAINING_GUIDE.md\n")
    
    if args.all:
        if not args.video:
            print("ERROR: --video required when using --all")
            return
        
        print(f"Running complete workflow for: {args.video}\n")
        
        if not step_1_annotate(args.video):
            return
        
        if not step_2_create_dataset(args.video, skip_frames=args.skip_frames):
            return
        
        # Derive data.yaml path
        video_stem = Path(args.video).stem
        data_yaml = f"datasets/ball_detector_{video_stem}/data.yaml"
        
        if not step_3_train_model(data_yaml, model_size=args.model_size, epochs=args.epochs):
            return
        
        step_4_evaluate(data_yaml)
        
        print_header("Workflow Complete!")
        best_model = find_latest_best_model()
        print(f"Trained model: {best_model or 'runs/detect/ball_detector*/weights/best.pt'}")
        print("\nNext steps:")
        print("1. Test on new video clips")
        print("2. Annotate more data to improve accuracy")
        print("3. Integrate trained model into detection pipeline")
        
    elif args.step == 1:
        if not args.video:
            print("ERROR: --video required for step 1")
            return
        step_1_annotate(args.video)
        
    elif args.step == 2:
        if not args.video:
            print("ERROR: --video required for step 2")
            return
        step_2_create_dataset(args.video, skip_frames=args.skip_frames)
        
    elif args.step == 3:
        if not args.data_yaml:
            print("ERROR: --data-yaml required for step 3")
            return
        step_3_train_model(args.data_yaml, model_size=args.model_size, epochs=args.epochs)
        
    elif args.step == 4:
        if not args.data_yaml:
            print("ERROR: --data-yaml required for step 4")
            return
        step_4_evaluate(args.data_yaml)


if __name__ == "__main__":
    main()
