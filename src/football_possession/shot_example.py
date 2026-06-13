"""Example script for analyzing shots on goal to extract xG-relevant coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_possession.config import load_config
from football_possession.shot_analyzer import ShotAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a shot on goal image to extract ball and goalkeeper coordinates."
    )
    parser.add_argument("--image", required=True, help="Path to the shot image.")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save analysis results as JSON. If not provided, prints to stdout.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Optional JSON file with known landmark correspondences. "
            "Format: [{\"pixel\": [x, y], \"standard\": [sx, sy]}, ...]"
        ),
    )
    
    args = parser.parse_args()

    # Load config and create analyzer
    config = load_config(args.config)
    analyzer = ShotAnalyzer(config.model)

    calibration_landmarks = None
    if args.calibration:
        calibration_path = Path(args.calibration)
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_landmarks = [
            (
                (float(item["pixel"][0]), float(item["pixel"][1])),
                (float(item["standard"][0]), float(item["standard"][1])),
            )
            for item in payload
        ]

    # Analyze image
    print(f"Analyzing shot image: {args.image}")
    analysis = analyzer.analyze(args.image, calibration_landmarks=calibration_landmarks)

    # Prepare output
    result = {
        "image": str(args.image),
        "valid": analysis.is_valid(),
        "error": analysis.error,
        "landmarks_used": analysis.landmarks_used,
        "coordinate_method": analysis.coordinate_method,
        "ball": None,
        "goalkeeper": None,
        "pitch_bounds": None,
    }

    if analysis.ball_coordinate:
        result["ball"] = {
            "standard_format": {
                "x": analysis.ball_coordinate.x,
                "y": analysis.ball_coordinate.y,
            },
            "pixel_format": {
                "x": analysis.ball_pixel_coordinate[0],
                "y": analysis.ball_pixel_coordinate[1],
            } if analysis.ball_pixel_coordinate else None,
        }

    if analysis.goalkeeper_coordinate:
        result["goalkeeper"] = {
            "standard_format": {
                "x": analysis.goalkeeper_coordinate.x,
                "y": analysis.goalkeeper_coordinate.y,
            },
            "pixel_format": {
                "x": analysis.goalkeeper_pixel_coordinate[0],
                "y": analysis.goalkeeper_pixel_coordinate[1],
            } if analysis.goalkeeper_pixel_coordinate else None,
        }

    if analysis.pitch_bounds:
        result["pitch_bounds"] = {
            "x_min": analysis.pitch_bounds.x_min,
            "x_max": analysis.pitch_bounds.x_max,
            "y_min": analysis.pitch_bounds.y_min,
            "y_max": analysis.pitch_bounds.y_max,
            "width": analysis.pitch_bounds.width,
            "height": analysis.pitch_bounds.height,
        }

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output_path}")
    else:
        print(json.dumps(result, indent=2))

    # Print summary
    print("\n" + "=" * 60)
    if analysis.is_valid():
        method = f"{analysis.coordinate_method} ({analysis.landmarks_used} landmarks)" if analysis.landmarks_used else analysis.coordinate_method
        print(f"✓ Analysis successful! [{method}]")
        print(f"  Ball position (standard): x={analysis.ball_coordinate.x:.1f}, y={analysis.ball_coordinate.y:.1f}")
        print(f"  Goalkeeper position (standard): x={analysis.goalkeeper_coordinate.x:.1f}, y={analysis.goalkeeper_coordinate.y:.1f}")
    else:
        print("✗ Analysis failed!")
        print(f"  Error: {analysis.error}")
    print("=" * 60)


if __name__ == "__main__":
    main()
