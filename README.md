# Football Possession MVP

This project provides a practical starting point for estimating football possession from recorded match video such as Veo footage. The first version is intentionally based on existing models rather than custom training:

- YOLO for player and ball detection
- ByteTrack for player tracking
- Jersey-color clustering for team assignment
- Rule-based possession inference with temporal smoothing

## What this MVP does

- Reads a recorded video file offline
- Detects `person` and `sports ball` objects frame by frame
- Tracks players across frames
- Clusters player tracks into two teams using jersey color features
- Assigns possession to the team nearest the detected ball when the ball is within a configurable control radius
- Writes:
  - an annotated output video
  - a frame-level possession CSV
  - a JSON summary with possession percentages

## Why this is the right first step

Possession is an inferred event, not a directly visible label. A strong first version comes from combining stable detections and tracking with explicit rules. That lets you validate the approach on real Veo clips before deciding whether to fine-tune a detector or train a custom model.

## Setup

Create and activate a virtual environment, then install the package in editable mode.

```powershell
../football-data-video/.venv/Scripts/python.exe -m pip install --upgrade pip
../football-data-video/.venv/Scripts/python.exe -m pip install -e .
```

On Amazon Linux, you can bootstrap the instance with the helper script in this repo:

```bash
bash scripts/bootstrap_amazon_linux.sh
```

It installs Python 3.11+ if needed, creates `.venv`, upgrades packaging tools, and installs the project in editable mode.

## Run

```powershell
../football-data-video/.venv/Scripts/python.exe -m football_possession.main --video path\to\veo_clip.mp4
```

If you run the command from the repository root, a clip in the `video` folder should be passed as `video\clip_10_30.mp4`.

If you run the command from inside the `video` folder, pass just the filename:

```powershell
..\.venv\Scripts\python.exe -m football_possession.main --video clip_10_30.mp4
```

The first run also downloads the default YOLO model weights, so an initial delay is expected.

Optional arguments:

- `--config config/default.yaml`
- `--output-dir outputs/sample-run`
- `--no-video`

## EC2 + S3 data sync workflow

When running this repo on an EC2 instance, keep large runtime data in S3 and sync it on demand.

This repo now includes `football-s3` with two commands:

- `push`: upload selected local paths to S3
- `pull`: download selected S3 paths to local repo paths

Bucket and prefix can be set either with command flags or environment variables:

- `FOOTBALL_S3_BUCKET` for bucket name
- `FOOTBALL_S3_PREFIX` for key prefix (optional, defaults to `football-data-video`)

Default synced paths are:

- `video`
- `datasets`
- `annotations`
- `outputs`
- `results.json`

### 1) AWS auth on EC2

Recommended: attach an IAM role to the EC2 instance with at least:

- `s3:ListBucket` on the target bucket
- `s3:GetObject` and `s3:PutObject` on the target prefix

### 2) Install dependencies

```powershell
../football-data-video/.venv/Scripts/python.exe -m pip install -e .
```

### 3) Pull data to EC2 before running

```powershell
football-s3 pull --bucket your-bucket-name --prefix football-data-video --region us-east-1
```

Or with env vars:

```powershell
$env:FOOTBALL_S3_BUCKET = "your-bucket-name"
$env:FOOTBALL_S3_PREFIX = "football-data-video"
football-s3 pull --region us-east-1
```

Preview only (no writes):

```powershell
football-s3 pull --bucket your-bucket-name --prefix football-data-video --dry-run
```

### 4) Run processing on EC2

```powershell
football-possession --video video\clip_10_30.mp4
```

### 5) Push outputs back to S3

```powershell
football-s3 push --bucket your-bucket-name --prefix football-data-video --paths outputs results.json
```

You can sync custom paths with `--paths`, for example:

```powershell
football-s3 push --bucket your-bucket-name --prefix football-data-video --paths video outputs
```

To include local YOLO model files too:

```powershell
football-s3 push --paths "*.pt" "video/*.pt" video datasets annotations outputs results.json
```

You can preview any command first:

```powershell
football-s3 push --paths "*.pt" "video/*.pt" video datasets annotations outputs results.json --dry-run
```

## Generate clips from a full match

Yes. The standard free CLI tool for this is `ffmpeg`, and it is already installed in your environment.

For quick manual clipping, this is enough:

```powershell
ffmpeg -y -ss 00:10:00 -i video\full_match.mp4 -t 00:00:30 -c copy video\clip_001.mp4
```

That extracts a 30-second clip starting at 10 minutes. Use `-c copy` for speed. If you need frame-accurate cuts, re-encode instead:

```powershell
ffmpeg -y -ss 00:10:00 -i video\full_match.mp4 -t 00:00:30 -c:v libx264 -preset fast -crf 18 -c:a aac video\clip_001.mp4
```

This repo now also includes a helper command to generate a batch of fixed-length clips:

```powershell
../football-data-video/.venv/Scripts/python.exe -m football_possession.clip_generator --input video\full_match.mp4 --output-dir video\clips --clip-duration 30 --step 60 --start-offset 300 --end-padding 300 --max-clips 10
```

That example will:

- create 30-second clips
- start 5 minutes into the match
- create a new clip every 60 seconds
- avoid the final 5 minutes
- stop after 10 clips

It also writes a manifest CSV with clip start times.

After reinstalling the package, you can use the shorter entrypoint too:

```powershell
football-clips --input video\full_match.mp4 --output-dir video\clips --clip-duration 30 --step 60
```

## Project layout

- `src/football_possession/detector.py`: YOLO wrapper for player and ball detections
- `src/football_possession/clip_generator.py`: FFmpeg-based fixed-duration clip generation
- `src/football_possession/tracker.py`: ByteTrack wrapper for player IDs
- `src/football_possession/team_classifier.py`: jersey-color clustering
- `src/football_possession/possession.py`: possession inference and smoothing
- `src/football_possession/pipeline.py`: pipeline orchestration and export
- `config/default.yaml`: tunable thresholds and model settings

## Current limitations

- COCO `sports ball` detection is a baseline and may miss the ball in wide Veo footage
- Team classification assumes the two teams have visually distinct kits
- This version uses image-space distances, not calibrated pitch coordinates
- Goalkeepers, referees, and close-contact situations can still produce ambiguous ownership

## Recommended next improvements

1. Test this baseline on short Veo clips and review the annotated output.
2. Replace the default ball detector with a football-specific fine-tuned model if recall is weak.
3. Add pitch calibration so control distances are measured in field coordinates.
4. Label a few difficult sequences and tune thresholds against those clips.
