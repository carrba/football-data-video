from __future__ import annotations

import numpy as np

from football_possession.config import TrackingConfig
from football_possession.types import PlayerDetection

try:
    import supervision as sv
except ImportError as exc:  # pragma: no cover
    sv = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class PlayerTracker:
    def __init__(self, config: TrackingConfig, frame_rate: float) -> None:
        if sv is None:
            raise ImportError(
                "supervision is required for tracking. Install project dependencies first."
            ) from IMPORT_ERROR

        self._tracker = sv.ByteTrack(
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
            frame_rate=frame_rate,
        )

    def update(self, players: list[PlayerDetection]) -> list[PlayerDetection]:
        if not players:
            empty = sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
            self._tracker.update_with_detections(empty)
            return []

        detections = sv.Detections(
            xyxy=np.array([player.bbox for player in players], dtype=np.float32),
            confidence=np.array([player.confidence for player in players], dtype=np.float32),
            class_id=np.zeros((len(players),), dtype=np.int32),
        )
        tracked = self._tracker.update_with_detections(detections)

        tracked_players: list[PlayerDetection] = []
        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        confidences = tracked.confidence if tracked.confidence is not None else []

        for bbox, score, track_id in zip(tracked.xyxy, confidences, tracker_ids, strict=True):
            tracked_players.append(
                PlayerDetection(
                    bbox=tuple(float(value) for value in bbox),
                    confidence=float(score),
                    track_id=int(track_id),
                )
            )

        return tracked_players
