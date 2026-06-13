# Shot Analysis Feature - Implementation Summary

## Overview
The shot analysis feature extracts ball and goalkeeper coordinates from still images of shots on goal, converting them to a standard coordinate system suitable for xG calculation.

## Files Created

### 1. `src/football_possession/coordinate_system.py`
Core coordinate transformation module with:
- **PitchBoundaries**: Dataclass representing pitch boundaries in pixel coordinates
- **StandardCoordinate**: Dataclass representing coordinates in standard format (0-120 x, 0-80 y)
- **pixel_to_standard()**: Converts pixel coordinates to standard format
- **standard_to_pixel()**: Converts standard format back to pixel coordinates
- **get_bbox_center()**: Extracts center point from bounding box

**Standard Format:**
- X-axis: 0-120 (120 = goal line of the defending goal)
- Y-axis: 0-80 (40 = center of goal mouth)

### 2. `src/football_possession/shot_analyzer.py`
Main shot analysis module with:
- **ShotAnalysis**: Dataclass holding analysis results
- **ShotAnalyzer**: Main analyzer class with these methods:
  - `analyze(image_path)`: Analyze a shot image from file
  - `analyze_frame(frame)`: Analyze a numpy array frame directly
  - `_detect_players()`: Reuses existing YoloDetector
  - `_detect_balls()`: Reuses existing YoloDetector
  - `_extract_ball_coordinate()`: Extracts ball position
  - `_extract_goalkeeper_coordinate()`: Identifies goalkeeper as player closest to goal line (minimum x)

**Process:**
1. Load image and detect pitch boundaries using existing green mask detection
2. Detect all players using YOLO person detector
3. Detect ball using YOLO sports ball detector
4. Identify goalkeeper as player with minimum x coordinate
5. Convert both detections to standard coordinates

### 3. `src/football_possession/shot_example.py`
CLI example script for running shot analysis with:
- Command-line argument parsing for image path
- Config loading for YOLO model
- JSON output export capability
- Human-readable result summary

## Usage

### Basic Usage (CLI)
```bash
.venv/Scripts/python -m football_possession.shot_example --image path/to/shot.jpg --output results.json
```

### Programmatic Usage
```python
from football_possession.config import load_config
from football_possession.shot_analyzer import ShotAnalyzer

config = load_config('config/default.yaml')
analyzer = ShotAnalyzer(config)
analysis = analyzer.analyze('shot.jpg')

if analysis.is_valid():
    print(f"Ball: x={analysis.ball_coordinate.x:.1f}, y={analysis.ball_coordinate.y:.1f}")
    print(f"GK: x={analysis.goalkeeper_coordinate.x:.1f}, y={analysis.goalkeeper_coordinate.y:.1f}")
```

## Output Format

### JSON Output Example
```json
{
  "image": "path/to/shot.jpg",
  "valid": true,
  "error": null,
  "ball": {
    "standard_format": {
      "x": 110.5,
      "y": 42.3
    },
    "pixel_format": {
      "x": 1240.2,
      "y": 487.5
    }
  },
  "goalkeeper": {
    "standard_format": {
      "x": 5.2,
      "y": 40.0
    },
    "pixel_format": {
      "x": 120.5,
      "y": 460.0
    }
  },
  "pitch_bounds": {
    "x_min": 50,
    "x_max": 1270,
    "y_min": 100,
    "y_max": 700,
    "width": 1220,
    "height": 600
  }
}
```

## Key Design Decisions

1. **Goalkeeper Identification**: Goalkeeper is identified as the player with the smallest x coordinate (closest to goal line they defend). This assumes the shot is taken on the goal where x=120 is the target.

2. **Reuse of Existing Code**: Leverages existing YoloDetector, pitch boundary detection, and type definitions to minimize duplication.

3. **Error Handling**: Comprehensive error handling with detailed error messages for debugging.

4. **Flexible Input**: Supports both file paths and numpy arrays for frame input.

## Next Steps / Future Enhancements

1. **Goalkeeper Validation**: Add more sophisticated goalkeeper identification (e.g., jersey number, position in frame)
2. **Shot Angle Calculation**: Add utility functions to calculate shot angle and distance
3. **Goalkeeper Distance to Ball**: Calculate distance from goalkeeper to ball
4. **Player Positions**: Return positions of all players within certain distance of ball
5. **Defensive Line Analysis**: Identify and analyze defensive line positions
6. **Model Confidence Filtering**: Add configurable confidence thresholds for detections

## Integration with xG Calculation

The `ShotAnalysis` object provides:
- Ball position in standard coordinates (for shot location)
- Goalkeeper position in standard coordinates (for goalkeeper placement)
- Pixel coordinates for visualization/debugging
- Pitch boundaries for coordinate system reference

These can be passed to an independent xG calculation application as needed.
