# Ball Tracking Model Training Guide

This guide walks you through training a custom YOLO model specifically for detecting and tracking the football/ball in your video clips.

## Why Train a Custom Model?

The pre-trained YOLO models (yolov8n, yolov8m, etc.) are trained on general objects from the COCO dataset. They detect a "sports ball" category, but they're optimized for average cases. By training on your specific video content, you can achieve much better detection rates for:
- Different lighting conditions in your footage
- Specific ball appearances (color, texture, size in frame)
- Eliminating false positives from similar-looking objects

## Overview of the Process

1. **Annotate a video clip** - Mark where the ball is in each frame
2. **Create YOLO dataset** - Convert annotations to YOLO format
3. **Train the model** - Use YOLO's training pipeline
4. **Test and iterate** - Evaluate performance and refine

## Step-by-Step Instructions

### Step 1: Prepare a Test Video Clip

Choose a short video clip (30-60 seconds) that's representative of your footage. Ideally, it should contain:
- Various lighting conditions
- Different ball positions (close-up, far away)
- Different angles and perspectives
- Both clear and partially obscured ball views

For example: `video/clip_14_30.mp4` (from your project structure)

### Step 2: Annotate the Video

Run the annotation tool to mark ball positions:

```powershell
# From project root with virtual environment activated
python -m football_possession.ball_annotation `
    --video "video/clip_14_30.mp4" `
    --output "annotations/clip_14_30"
```

**Annotation Controls:**
- **CLICK on the ball** in the video frame
- **SPACE** to save the annotation and move to next frame
- **N** to move to next frame without annotating
- **P** to go to previous frame
- **D** to delete annotation for current frame
- **S** to save progress to file
- **Q** to quit

**Tips for good annotations:**
- Be consistent - click the center of the ball
- Skip frames where the ball is not clearly visible (too small, blocked, etc.)
- Aim for at least 50-100 labeled frames (can use every other frame for faster annotation)
- Save frequently with **S** to avoid losing work

### Step 3: Convert to YOLO Format

Convert your annotations to YOLO dataset format:

```powershell
python -m football_possession.yolo_dataset_creator `
    --video "video/clip_14_30.mp4" `
    --annotations "annotations/clip_14_30/annotations.json" `
    --output "datasets/ball_detector_v1" `
    --skip-frames 1 `
    --train-split 0.8
```

**Options:**
- `--skip-frames 1`: Use every frame (set to 2 to use every other frame)
- `--train-split 0.8`: 80% training, 20% validation

This creates:
- `datasets/ball_detector_v1/images/` - Extracted frames
- `datasets/ball_detector_v1/labels/` - YOLO format annotations
- `datasets/ball_detector_v1/data.yaml` - Dataset configuration

### Step 4: Train the Model

Start training:

```powershell
python -m football_possession.ball_trainer `
    --data-yaml "datasets/ball_detector_v1/data.yaml" `
    --model-size "n" `
    --epochs 100
```

If you want to force CPU explicitly, add `--device "cpu"`.

**Model sizes (trade-off between speed and accuracy):**
- `n` (nano) - Fastest, best for quick training/testing
- `s` (small) - Balanced
- `m` (medium) - Better accuracy, slower
- `l` (large) - High accuracy, slow

**Training output:**
- `runs/detect/ball_detector/weights/best.pt` - Best model (use this)
- `runs/detect/ball_detector/weights/last.pt` - Last checkpoint
- `runs/detect/ball_detector/` - Training metrics and plots

### Step 5: Test Your Model

Once training completes, test on new frames:

```powershell
# From Python REPL or script
from ultralytics import YOLO

model = YOLO("runs/detect/ball_detector/weights/best.pt")
results = model.predict("path/to/test/image.jpg", conf=0.5)
```

Or evaluate on validation set:

```powershell
python -m football_possession.ball_trainer `
    --eval `
    --model "runs/detect/ball_detector/weights/best.pt" `
    --data-yaml "datasets/ball_detector_v1/data.yaml"
```

## Integration with Your Pipeline

Once you have a trained model, update your `detector.py` to use it:

```python
# In detector.py, update YoloDetector to support custom models:
def _predict(self, frame, *, model_path: str, ...):
    # If model_path points to your trained model:
    # model = self._model_for("runs/detect/ball_detector/weights/best.pt")
    model = self._model_for(model_path)
    # ... rest of prediction logic
```

## Troubleshooting

**Model isn't detecting the ball well:**
- Annotate more frames (200+ for better results)
- Try different model sizes (medium or large)
- Increase training epochs
- Check that annotations are consistent and accurate

**Training is very slow:**
- Use smaller model (`--model-size n`)
- Use fewer epochs or enable early stopping
- Check GPU is being used (monitor with `nvidia-smi`)

**Out of memory:**
- Reduce batch size in `ball_trainer.py`
- Use smaller image size (default 640, try 416 or 512)
- Use smaller model

**Annotations look wrong after saving:**
- Coordinates are normalized [0, 1], so check the annotation tool is placing circles correctly
- Review `annotations/clip_14_30/annotations.json` to verify coordinates

## Next Steps for Production

Once happy with results:

1. **Annotate more clips** to build a larger dataset (500+ frames)
2. **Train with more data** for better generalization
3. **Fine-tune** on different field types, lighting, etc.
4. **Monitor performance** on real matches
5. **Iterate** - retrain with corrected annotations when needed

## Advanced: Custom Hyperparameters

Edit `ball_trainer.py` to adjust training parameters for your specific needs:

- `batch`: Larger batches (32, 64) for better gradient estimates
- `lr0`: Initial learning rate (higher = faster but less stable)
- `augmentation`: Color/rotation/scale changes
- `patience`: Early stopping (stops if no improvement after N epochs)

## Reference Files

- Annotation tool: `src/football_possession/ball_annotation.py`
- Dataset creator: `src/football_possession/yolo_dataset_creator.py`
- Training script: `src/football_possession/ball_trainer.py`
