from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PlayerDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None
    team_id: int | None = None


@dataclass(slots=True)
class BallDetection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(slots=True)
class FrameRecord:
    frame_index: int
    timestamp_s: float
    team_in_possession: int | None
    ball_visible: bool
    player_count: int
