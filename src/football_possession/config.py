from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class VideoConfig:
    frame_stride: int = 2
    write_annotated_video: bool = True


@dataclass(slots=True)
class ModelConfig:
    model_path: str = "yolov8m.pt"
    player_model_path: str | None = None
    ball_model_path: str | None = None
    detection_confidence: float = 0.25
    player_detection_confidence: float | None = None
    ball_detection_confidence: float | None = None
    iou_threshold: float = 0.45
    ball_iou_threshold: float | None = None
    player_class_id: int = 0
    ball_class_id: int = 32
    device: str | None = None
    two_stage_ball_detection: bool = False
    pitch_mask_enabled: bool = True
    ball_tile_size: int = 640
    ball_tile_overlap: int = 160


@dataclass(slots=True)
class TrackingConfig:
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 1


@dataclass(slots=True)
class TeamConfig:
    enabled: bool = True
    team_count: int = 2
    min_track_samples: int = 5
    upper_body_ratio: float = 0.55


@dataclass(slots=True)
class PossessionConfig:
    control_radius_px: float = 85.0
    min_control_frames: int = 8


@dataclass(slots=True)
class AppConfig:
    video: VideoConfig
    model: ModelConfig
    tracking: TrackingConfig
    teams: TeamConfig
    possession: PossessionConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        return cls(
            video=VideoConfig(**raw.get("video", {})),
            model=ModelConfig(**raw.get("model", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
            teams=TeamConfig(**raw.get("teams", {})),
            possession=PossessionConfig(**raw.get("possession", {})),
        )


def load_config(config_path: str | Path) -> AppConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.from_dict(raw)
