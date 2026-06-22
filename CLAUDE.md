# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python computer vision pipeline that estimates football (soccer) possession from match video (e.g., Veo footage). Secondary features include shot analysis (xG coordinate extraction), custom ball detector training, clip generation, and AWS S3/EC2 sync.

Package name: `football-possession`. Installed as an editable package via `pip install -e .`.

## Commands

### Install / Setup
```bash
pip install -e .
```

### Run the main possession pipeline
```bash
football-possession --video path/to/match.mp4 --config config/default.yaml
# Two-stage custom ball model:
football-possession --video path/to/match.mp4 --config config/custom_ball_two_stage.yaml
```

### Run tests
```bash
python -m unittest discover tests
# Single test file:
python -m unittest tests.test_detector
```

### Custom ball training workflow (4-step interactive)
```bash
python -m football_possession.train_workflow
```

### Clip generation
```bash
football-clips --video path/to/match.mp4 --manifest clips_manifest.csv
```

### S3 sync
```bash
football-s3 push   # upload data to S3
football-s3 pull   # download from S3
```

### Shot analysis (single image)
```bash
python -m football_possession.shot_example --image path/to/frame.jpg
```

## Architecture

### Core possession pipeline (`src/football_possession/`)

`PossessionPipeline.run()` in `pipeline.py` is the main loop:

1. **Detection** (`detector.py` — `YoloDetector`) — runs YOLO on each frame; optionally uses two-stage tiled detection for small ball detection with a custom model (`two_stage_ball_detection: true` in config)
2. **Tracking** (`tracker.py` — `PlayerTracker`) — wraps `supervision.ByteTrack` to maintain stable `track_id` per player across frames
3. **Team classification** (`team_classifier.py` — `TeamColorClassifier`) — KMeans clustering on mean upper-body BGR pixel features; assigns `team_id` 0 or 1 to each track
4. **Possession** (`possession.py` — `PossessionEstimator`) — nearest player within `control_radius_px` claims possession; team must hold for `min_control_frames` consecutive frames before a switch is committed

Outputs go to `outputs/<video_stem>/`:
- `frame_possession.csv` — per-frame records
- `summary.json` — possession % for team_0, team_1, unknown
- `<video_stem>_annotated.mp4` — annotated video

### Configuration system

`config.py` defines a dataclass hierarchy (`AppConfig` → `ModelConfig`, `VideoConfig`, `TeamsConfig`, `PossessionConfig`, `OutputConfig`). Config is loaded from YAML, then CLI flags override fields. Key presets in `config/`:
- `default.yaml` — standard YOLO `yolov8m.pt`, every 2nd frame, `control_radius_px: 85`
- `custom_ball_two_stage.yaml` — enables tiled ball detection with a fine-tuned model
- `fast.yaml`, `high_recall_*.yaml` — model-specific tuning presets

### Shot analysis path (separate feature)

`ShotAnalyzer` (`shot_analyzer.py`) maps ball + goalkeeper pixel positions into standard pitch coordinates (Statsbomb/Opta: 0-120 × 0-80). Uses `HomographyTransform` (`coordinate_system.py`) via RANSAC when `LandmarkDetector` (`landmark_detector.py`) finds ≥4 pitch line intersections; falls back to linear bounding-box scaling otherwise.

### Custom ball training path

4-step workflow orchestrated by `train_workflow.py`:
1. **Annotate** — `ball_annotation.py` interactive OpenCV GUI for click-annotating ball positions
2. **Dataset** — `yolo_dataset_creator.py` converts `annotations.json` → YOLO-format `images/` + `labels/` + `data.yaml`
3. **Train** — `ball_trainer.py` fine-tunes YOLO; auto-detects CPU/GPU
4. **Evaluate** — runs model evaluation; `model_checkpoint_util.py` writes the latest `best.pt` path back into the config YAML

### Infrastructure

- `infra/terraform/s3-bucket/` — persistent S3 bucket (`prevent_destroy = true`)
- `infra/terraform/ec2-instance/` — disposable GPU EC2 (AWS Deep Learning AMI via SSM)
- `scripts/bootstrap_amazon_linux.sh` — installs Python 3.11, NVIDIA drivers, creates `.venv`, installs package

## Key Conventions

- `frame_stride` (default 2) skips every other frame — affects both speed and possession smoothing
- Player class ID is `0` (COCO person); standard ball class ID is `32` (COCO sports ball); custom ball models use class `0`
- Two virtual environments exist (`.venv/` and `myenv/`); `.venv/` is the canonical one used by the bootstrap script
- `annotations/`, `datasets/`, `video/`, `outputs/` are gitignored — large data lives in S3
