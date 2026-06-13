"""Coordinate system transformations between pixel and standard formats.

Standard format (e.g., Statsbomb/Opta):
- X-axis: 0-120 (120 = goal line of the defensive goal)
- Y-axis: 0-80 (40 = center of goal mouth)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PitchBoundaries:
    """Pitch boundaries in pixel coordinates."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(slots=True)
class StandardCoordinate:
    """Coordinate in standard format (0-120 x, 0-80 y)."""
    x: float
    y: float


def pixel_to_standard(
    pixel_x: float,
    pixel_y: float,
    pitch_bounds: PitchBoundaries,
) -> StandardCoordinate:
    """Convert pixel coordinates to standard format.
    
    Args:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
        pitch_bounds: Pitch boundaries in pixel coordinates
    
    Returns:
        StandardCoordinate with x in [0, 120] and y in [0, 80]
    """
    # Normalize to pitch area
    norm_x = (pixel_x - pitch_bounds.x_min) / pitch_bounds.width
    norm_y = (pixel_y - pitch_bounds.y_min) / pitch_bounds.height
    
    # Clamp to [0, 1] to handle detections slightly outside pitch
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))
    
    # Scale to standard format
    standard_x = norm_x * 120.0
    standard_y = norm_y * 80.0
    
    return StandardCoordinate(x=standard_x, y=standard_y)


def standard_to_pixel(
    standard_x: float,
    standard_y: float,
    pitch_bounds: PitchBoundaries,
) -> tuple[float, float]:
    """Convert standard format coordinates to pixel coordinates.
    
    Args:
        standard_x: X coordinate in standard format [0, 120]
        standard_y: Y coordinate in standard format [0, 80]
        pitch_bounds: Pitch boundaries in pixel coordinates
    
    Returns:
        Tuple of (pixel_x, pixel_y)
    """
    # Normalize from standard format
    norm_x = standard_x / 120.0
    norm_y = standard_y / 80.0
    
    # Scale to pixel coordinates
    pixel_x = pitch_bounds.x_min + norm_x * pitch_bounds.width
    pixel_y = pitch_bounds.y_min + norm_y * pitch_bounds.height
    
    return (pixel_x, pixel_y)


def get_bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Get center point of a bounding box.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
    
    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class HomographyTransform:
    """Perspective-correct coordinate transform using a homography matrix.

    Computed from a set of (pixel, standard) point correspondences via
    cv2.findHomography with RANSAC, so a few outlier landmarks are tolerated.
    """

    def __init__(
        self,
        landmarks: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> None:
        """Initialise from matched landmark pairs.

        Args:
            landmarks: List of ((pixel_x, pixel_y), (standard_x, standard_y)) pairs.
                       At least 4 pairs required.

        Raises:
            ValueError: If fewer than 4 pairs supplied or homography cannot be computed.
        """
        if len(landmarks) < 4:
            raise ValueError(f"At least 4 landmarks required, got {len(landmarks)}")

        src = np.array([p[0] for p in landmarks], dtype=np.float32)
        dst = np.array([p[1] for p in landmarks], dtype=np.float32)

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("Could not compute homography — landmarks may be degenerate")
        self._H: np.ndarray = H
        self._landmark_count = len(landmarks)

    @property
    def landmark_count(self) -> int:
        return self._landmark_count

    def pixel_to_standard(self, pixel_x: float, pixel_y: float) -> StandardCoordinate:
        """Map a pixel coordinate to standard pitch format using the homography."""
        src = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self._H)[0][0]
        return StandardCoordinate(x=float(dst[0]), y=float(dst[1]))
