from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from football_possession.config import TeamConfig
from football_possession.detection_types import PlayerDetection


class TeamColorClassifier:
    def __init__(self, config: TeamConfig) -> None:
        self._config = config
        self._track_samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._track_features: dict[int, np.ndarray] = {}
        self._model: KMeans | None = None

    def assign(self, frame: np.ndarray, players: list[PlayerDetection]) -> list[PlayerDetection]:
        if not self._config.enabled:
            return players

        for player in players:
            if player.track_id is None:
                continue
            feature = self._extract_feature(frame, player.bbox)
            if feature is None:
                continue
            self._track_samples[player.track_id].append(feature)
            self._track_features[player.track_id] = np.mean(
                self._track_samples[player.track_id], axis=0
            )

        self._fit_if_ready()
        if self._model is None:
            return players

        for player in players:
            if player.track_id is None:
                continue
            feature = self._track_features.get(player.track_id)
            if feature is None:
                continue
            player.team_id = int(self._model.predict([feature])[0])
        return players

    def _fit_if_ready(self) -> None:
        enough_tracks = [
            track_id
            for track_id, samples in self._track_samples.items()
            if len(samples) >= self._config.min_track_samples
        ]
        if len(enough_tracks) < self._config.team_count:
            return

        features = np.array([self._track_features[track_id] for track_id in enough_tracks])
        if len(features) < self._config.team_count:
            return

        self._model = KMeans(n_clusters=self._config.team_count, n_init="auto", random_state=7)
        self._model.fit(features)

    def _extract_feature(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(value) for value in bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None

        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return None

        upper_cut = max(1, int(player_crop.shape[0] * self._config.upper_body_ratio))
        upper_crop = player_crop[:upper_cut, :]
        return upper_crop.mean(axis=(0, 1))
