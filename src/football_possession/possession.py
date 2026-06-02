from __future__ import annotations

import math

from football_possession.config import PossessionConfig
from football_possession.types import BallDetection, PlayerDetection


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PossessionEstimator:
    def __init__(self, config: PossessionConfig) -> None:
        self._config = config
        self._active_team: int | None = None
        self._pending_team: int | None = None
        self._pending_count = 0

    def update(
        self, players: list[PlayerDetection], ball: BallDetection | None
    ) -> int | None:
        candidate = self._candidate_team(players, ball)
        if candidate == self._active_team:
            self._pending_team = None
            self._pending_count = 0
            return self._active_team

        if candidate == self._pending_team:
            self._pending_count += 1
        else:
            self._pending_team = candidate
            self._pending_count = 1

        if self._pending_count >= self._config.min_control_frames:
            self._active_team = self._pending_team
            self._pending_team = None
            self._pending_count = 0

        return self._active_team

    def _candidate_team(
        self, players: list[PlayerDetection], ball: BallDetection | None
    ) -> int | None:
        if ball is None:
            return None

        ball_center = _bbox_center(ball.bbox)
        best_player: PlayerDetection | None = None
        best_distance = float("inf")

        for player in players:
            if player.team_id is None:
                continue
            player_center = _bbox_center(player.bbox)
            distance = math.dist(player_center, ball_center)
            if distance < best_distance:
                best_distance = distance
                best_player = player

        if best_player is None or best_distance > self._config.control_radius_px:
            return None
        return best_player.team_id
