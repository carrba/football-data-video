# Ball Training Quick Reference

## Complete Workflow (All Steps at Once)

```powershell
# Activate virtual environment first
python -m football_possession.train_workflow --all --video video/clip_14_30.mp4 --epochs 100
```

This will:
1. Open annotation tool - mark ball positions
2. Create YOLO dataset
3. Train model
4. Evaluate results

## Individual Steps

### Step 1: Annotate Video
```powershell
python -m football_possession.ball_annotation `
    --video "video/clip_14_30.mp4" `
    --output "annotations/clip_14_30"
```
**Controls:** CLICK ball + SPACE=save, N=next, P=prev, S=save all, D=delete, Q=quit

### Step 2: Create YOLO Dataset
```powershell
python -m football_possession.yolo_dataset_creator `
    --video "video/clip_14_30.mp4" `
    --annotations "annotations/clip_14_30/annotations.json" `
    --output "datasets/ball_detector_v1"
```

### Step 3: Train Model
```powershell
# Quick training (nano model, 50 epochs)
python -m football_possession.ball_trainer `
    --data-yaml "datasets/ball_detector_v1/data.yaml" `
    --model-size "n" `
    --epochs 50

# Better quality (medium model, 100 epochs)
python -m football_possession.ball_trainer `
    --data-yaml "datasets/ball_detector_v1/data.yaml" `
    --model-size "m" `
    --epochs 100
```

### Step 4: Evaluate Model
```powershell
python -m football_possession.ball_trainer `
    --eval `
    --model "runs/detect/runs/detect/ball_detectorweights/best.pt" `
    --data-yaml "datasets/ball_detector_v1/data.yaml"
```

### Step 5: Auto-Set Latest Trained Checkpoint in Config
```powershell
python -m football_possession.model_checkpoint_util `
    --config "config/custom_ball_two_stage.yaml"
```

Optional dry run:
```powershell
python -m football_possession.model_checkpoint_util `
    --config "config/custom_ball_two_stage.yaml" `
    --dry-run
```

## Using Trained Model

```python
from football_possession.custom_ball_detector import CustomBallDetector
import cv2

# Load model
detector = CustomBallDetector("runs/detect/ball_detector/weights/best.pt")

# Detect in frame
frame = cv2.imread("frame.jpg")
detections = detector.detect(frame, confidence=0.5)

for bbox, confidence in detections:
    x1, y1, x2, y2 = bbox
    print(f"Ball found at ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f}) with confidence {confidence:.3f}")
```

## Tips for Success

### For Faster Training
- Use model size `n` (nano)
- Reduce epochs (50 instead of 100)
- Use fewer frames `--skip-frames 2`

### For Better Accuracy
- Annotate 200+ frames (not just 50)
- Use model size `m` (medium) or `l` (large)
- Train 100-150 epochs
- Annotate varied footage (different angles, lighting, distances)

### Annotation Tips
- Be consistent clicking the ball center
- Skip very blurry or partially occluded balls
- Aim for 1-2 annotations per second of video

### If Model Performs Poorly
1. Check annotation accuracy
2. Annotate more frames
3. Try larger model size
4. Train longer
5. Check data covers different scenarios

## Output Locations

- **Annotations:** `annotations/<video_name>/annotations.json`
- **Dataset:** `datasets/ball_detector_v1/` (images/ and labels/)
- **Trained model:** `runs/detect/ball_detector/weights/best.pt`
- **Training logs:** `runs/detect/ball_detector/`

## File Reference

- Annotation tool: [src/football_possession/ball_annotation.py](src/football_possession/ball_annotation.py)
- Dataset creation: [src/football_possession/yolo_dataset_creator.py](src/football_possession/yolo_dataset_creator.py)
- Training: [src/football_possession/ball_trainer.py](src/football_possession/ball_trainer.py)
- Workflow runner: [src/football_possession/train_workflow.py](src/football_possession/train_workflow.py)
- Model integration: [src/football_possession/custom_ball_detector.py](src/football_possession/custom_ball_detector.py)
- Full guide: [BALL_TRAINING_GUIDE.md](BALL_TRAINING_GUIDE.md)

## Expected Performance

With ~100 annotated frames:
- Nano model (yolov8n): Fast, ~80% accuracy, ~5-10 min training
- Small model (yolov8s): Balanced, ~85% accuracy, ~10-20 min training
- Medium model (yolov8m): Better, ~88% accuracy, ~20-40 min training

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model doesn't detect ball | Annotate more frames, ensure consistent labeling |
| Training very slow | Use smaller model or fewer epochs |
| Out of memory | Reduce batch size in ball_trainer.py |
| Invalid CUDA device=0 | Re-run with `--device "cpu"` or omit `--device` to auto-detect |
| Annotations won't save | Press S manually in annotation tool |
| Model path not found | Check path exists: `runs/detect/ball_detector/weights/best.pt` |
