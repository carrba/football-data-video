"""Shot analysis for extracting xG-relevant information from still images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from football_possession.config import AppConfig, ModelConfig
from football_possession.coordinate_system import (
    HomographyTransform,
    PitchBoundaries,
    StandardCoordinate,
    get_bbox_center,
    pixel_to_standard,
)
from football_possession.detector import YoloDetector, build_pitch_mask, pitch_bbox
from football_possession.landmark_detector import detect_landmarks
from football_possession.detection_types import BallDetection, PlayerDetection


@dataclass(slots=True)
class ShotAnalysis:
    """Analysis results for a shot on goal."""
    ball_coordinate: StandardCoordinate | None
    goalkeeper_coordinate: StandardCoordinate | None
    ball_pixel_coordinate: tuple[float, float] | None
    goalkeeper_pixel_coordinate: tuple[float, float] | None
    pitch_bounds: PitchBoundaries | None
    landmarks_used: int = 0
    coordinate_method: str = "pitch_boundary_fallback"
    error: str | None = None

    def is_valid(self) -> bool:
        """Check if analysis found both ball and goalkeeper."""
        return (
            self.ball_coordinate is not None
            and self.goalkeeper_coordinate is not None
            and self.error is None
        )


class ShotAnalyzer:
    """Analyzer for extracting ball and goalkeeper coordinates from shot images."""

    def __init__(self, config: ModelConfig | AppConfig) -> None:
        """Initialize the shot analyzer.
        
        Args:
            config: Model configuration for YOLO detector, or full app config
        """
        model_config = config.model if isinstance(config, AppConfig) else config
        self._detector = YoloDetector(model_config)
        self._config = model_config

    def analyze(
        self,
        image_path: str | Path,
        calibration_landmarks: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> ShotAnalysis:
        """Analyze a shot on goal image.
        
        Args:
            image_path: Path to the shot image
        
        Returns:
            ShotAnalysis with ball and goalkeeper coordinates
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return ShotAnalysis(
                ball_coordinate=None,
                goalkeeper_coordinate=None,
                ball_pixel_coordinate=None,
                goalkeeper_pixel_coordinate=None,
                pitch_bounds=None,
                error=f"Image not found: {image_path}",
            )

        # Load image
        frame = cv2.imread(str(image_path))
        if frame is None:
            return ShotAnalysis(
                ball_coordinate=None,
                goalkeeper_coordinate=None,
                ball_pixel_coordinate=None,
                goalkeeper_pixel_coordinate=None,
                pitch_bounds=None,
                error=f"Failed to load image: {image_path}",
            )

        return self.analyze_frame(frame, calibration_landmarks=calibration_landmarks)

    def analyze_frame(
        self,
        frame: np.ndarray,
        calibration_landmarks: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> ShotAnalysis:
        """Analyze a shot on goal frame.

        Coordinate mapping priority:
        1. Landmark homography — if >= 4 pitch markings are detected, a perspective-
           correct homography is computed from pixel to standard coordinates.
        2. Pitch-boundary fallback — if landmark detection fails, a simple linear
           scaling from the detected green-pitch bounding box is used instead.

        Args:
            frame: Image frame (BGR format from cv2)

        Returns:
            ShotAnalysis with ball and goalkeeper coordinates
        """
        try:
            # --- Step 1: build coordinate transform ---
            transform: HomographyTransform | None = None
            pitch_bounds: PitchBoundaries | None = None
            landmarks_used = 0
            coordinate_method = "pitch_boundary_fallback"

            if calibration_landmarks and len(calibration_landmarks) >= 4:
                transform = HomographyTransform(calibration_landmarks)
                landmarks_used = len(calibration_landmarks)
                coordinate_method = "manual_homography"

            if transform is None:
                landmarks = detect_landmarks(frame)
                if len(landmarks) >= 4:
                    try:
                        transform = HomographyTransform(landmarks)
                        landmarks_used = len(landmarks)
                        coordinate_method = "auto_homography"
                    except ValueError:
                        pass  # fall through to bounding-box approach

            if transform is None:
                # Fall back to green-pitch bounding box
                mask = build_pitch_mask(frame)
                bbox_coords = pitch_bbox(mask)
                if bbox_coords is None:
                    return ShotAnalysis(
                        ball_coordinate=None,
                        goalkeeper_coordinate=None,
                        ball_pixel_coordinate=None,
                        goalkeeper_pixel_coordinate=None,
                        pitch_bounds=None,
                        landmarks_used=0,
                        coordinate_method=coordinate_method,
                        error="Could not detect pitch boundaries or landmarks in image",
                    )
                pitch_bounds = PitchBoundaries(
                    x_min=float(bbox_coords[0]),
                    y_min=float(bbox_coords[1]),
                    x_max=float(bbox_coords[2]),
                    y_max=float(bbox_coords[3]),
                )

            # --- Step 2: detect players and ball ---
            players = self._detect_players(frame)
            balls = self._detect_balls_tiled(frame, players)

            # --- Step 3: convert detections to standard coordinates ---
            ball_coord, ball_pixel = self._extract_ball_coordinate(
                balls, transform=transform, pitch_bounds=pitch_bounds
            )
            gk_coord, gk_pixel = self._extract_goalkeeper_coordinate(
                players, transform=transform, pitch_bounds=pitch_bounds
            )

            error: str | None = None
            if ball_coord is None and gk_coord is None:
                error = "Could not detect ball or goalkeeper"
            elif ball_coord is None:
                error = "Could not detect ball"
            elif gk_coord is None:
                error = "Could not detect goalkeeper"

            return ShotAnalysis(
                ball_coordinate=ball_coord,
                goalkeeper_coordinate=gk_coord,
                ball_pixel_coordinate=ball_pixel,
                goalkeeper_pixel_coordinate=gk_pixel,
                pitch_bounds=pitch_bounds,
                landmarks_used=landmarks_used,
                coordinate_method=coordinate_method,
                error=error,
            )

        except Exception as e:  # pylint: disable=broad-except
            return ShotAnalysis(
                ball_coordinate=None,
                goalkeeper_coordinate=None,
                ball_pixel_coordinate=None,
                goalkeeper_pixel_coordinate=None,
                pitch_bounds=None,
                landmarks_used=0,
                coordinate_method="pitch_boundary_fallback",
                error=f"Analysis error: {str(e)}",
            )

    def _detect_players(self, frame: np.ndarray) -> list[PlayerDetection]:
        """Detect players in frame."""
        result = self._detector._predict(
            frame,
            model_path=self._detector._player_model_path(),
            confidence=self._detector._player_confidence(),
            iou=self._config.iou_threshold,
            class_id=0,  # COCO class 0 = person
        )
        return self._detector._parse_players(result)

    def _detect_balls_tiled(self, frame: np.ndarray, players: list[PlayerDetection]) -> list[BallDetection]:
        """Detect balls using two-stage tiled search, constrained to player area."""
        return self._detector._detect_balls_two_stage(frame, players)

    # ------------------------------------------------------------------
    # Coordinate extraction helpers
    # ------------------------------------------------------------------

    def _to_standard(
        self,
        pixel_x: float,
        pixel_y: float,
        *,
        transform: HomographyTransform | None,
        pitch_bounds: PitchBoundaries | None,
    ) -> StandardCoordinate:
        """Convert a pixel position to standard coordinates using whichever
        transform is available (homography preferred over bounding-box)."""
        if transform is not None:
            return transform.pixel_to_standard(pixel_x, pixel_y)
        if pitch_bounds is not None:
            return pixel_to_standard(pixel_x, pixel_y, pitch_bounds)
        raise RuntimeError("No coordinate transform available")

    def _extract_ball_coordinate(
        self,
        balls: list[BallDetection],
        *,
        transform: HomographyTransform | None,
        pitch_bounds: PitchBoundaries | None,
    ) -> tuple[StandardCoordinate | None, tuple[float, float] | None]:
        if not balls:
            return None, None
        ball = max(balls, key=lambda b: b.confidence)
        pixel_x, pixel_y = get_bbox_center(ball.bbox)
        standard_coord = self._to_standard(
            pixel_x, pixel_y, transform=transform, pitch_bounds=pitch_bounds
        )
        return standard_coord, (pixel_x, pixel_y)

    def _extract_goalkeeper_coordinate(
        self,
        players: list[PlayerDetection],
        *,
        transform: HomographyTransform | None,
        pitch_bounds: PitchBoundaries | None,
    ) -> tuple[StandardCoordinate | None, tuple[float, float] | None]:
        """Identify the goalkeeper and return their standard coordinate.

        When a homography transform is available, the goalkeeper is the player
        whose mapped standard x is highest (closest to the goal line at x=120)
        and whose standard y is within a plausible range (10 < y < 70).
        This is robust to side-on camera views.

        Falls back to minimum pixel-x when no homography is available
        (frontal camera assumption).
        """
        if not players:
            return None, None

        if transform is not None:
            # Map all players to standard coords and pick the one nearest the goal line
            best_player = None
            best_x = -1.0
            for player in players:
                px, py = get_bbox_center(player.bbox)
                try:
                    std = transform.pixel_to_standard(px, py)
                except Exception:  # noqa: BLE001
                    continue
                if 10.0 < std.y < 70.0 and std.x > best_x:
                    best_x = std.x
                    best_player = player

            if best_player is None:
                # Loosen y constraint and try again
                for player in players:
                    px, py = get_bbox_center(player.bbox)
                    try:
                        std = transform.pixel_to_standard(px, py)
                    except Exception:  # noqa: BLE001
                        continue
                    if std.x > best_x:
                        best_x = std.x
                        best_player = player

            if best_player is None:
                return None, None

            pixel_x, pixel_y = get_bbox_center(best_player.bbox)
            standard_coord = transform.pixel_to_standard(pixel_x, pixel_y)
            return standard_coord, (pixel_x, pixel_y)

        # No homography — fall back to minimum pixel x (frontal camera)
        goalkeeper = min(players, key=lambda p: get_bbox_center(p.bbox)[0])
        pixel_x, pixel_y = get_bbox_center(goalkeeper.bbox)
        standard_coord = self._to_standard(
            pixel_x, pixel_y, transform=None, pitch_bounds=pitch_bounds
        )
        return standard_coord, (pixel_x, pixel_y)
