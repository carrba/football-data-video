from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from football_possession.config import ModelConfig
from football_possession.types import BallDetection, PlayerDetection

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    YOLO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@dataclass(slots=True)
class DetectionBatch:
    players: list[PlayerDetection]
    balls: list[BallDetection]


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0

    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    denominator = left_area + right_area - inter_area
    if denominator <= 0.0:
        return 0.0
    return inter_area / denominator


def deduplicate_ball_detections(
    balls: list[BallDetection], iou_threshold: float = 0.35
) -> list[BallDetection]:
    kept: list[BallDetection] = []
    for candidate in sorted(balls, key=lambda item: item.confidence, reverse=True):
        if all(_bbox_iou(candidate.bbox, existing.bbox) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def build_pitch_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 25, 25], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def pitch_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        return None
    x, y, width, height = cv2.boundingRect(nonzero)
    if width <= 0 or height <= 0:
        return None
    return (x, y, x + width, y + height)


def build_ball_tiles(
    frame_shape: tuple[int, int, int],
    search_bbox: tuple[int, int, int, int] | None,
    tile_size: int,
    tile_overlap: int,
) -> list[tuple[int, int, int, int]]:
    frame_height, frame_width = frame_shape[:2]
    if search_bbox is None:
        search_x1, search_y1, search_x2, search_y2 = (0, 0, frame_width, frame_height)
    else:
        search_x1, search_y1, search_x2, search_y2 = search_bbox

    search_x1 = max(0, min(frame_width, search_x1))
    search_y1 = max(0, min(frame_height, search_y1))
    search_x2 = max(search_x1 + 1, min(frame_width, search_x2))
    search_y2 = max(search_y1 + 1, min(frame_height, search_y2))

    effective_tile = max(64, tile_size)
    step = max(32, effective_tile - max(0, tile_overlap))
    x_starts = list(range(search_x1, search_x2, step))
    y_starts = list(range(search_y1, search_y2, step))
    if not x_starts:
        x_starts = [search_x1]
    if not y_starts:
        y_starts = [search_y1]

    tiles: list[tuple[int, int, int, int]] = []
    for y1 in y_starts:
        tile_y2 = min(search_y2, y1 + effective_tile)
        tile_y1 = max(search_y1, tile_y2 - effective_tile)
        for x1 in x_starts:
            tile_x2 = min(search_x2, x1 + effective_tile)
            tile_x1 = max(search_x1, tile_x2 - effective_tile)
            tile = (tile_x1, tile_y1, tile_x2, tile_y2)
            if tile not in tiles:
                tiles.append(tile)
    return tiles


class YoloDetector:
    def __init__(self, config: ModelConfig) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for detection. Install project dependencies first."
            ) from IMPORT_ERROR

        self._config = config
        self._models: dict[str, YOLO] = {}

    def _model_for(self, path: str) -> YOLO:
        model = self._models.get(path)
        if model is None:
            model = YOLO(path)
            self._models[path] = model
        return model

    def _predict(self, frame: np.ndarray, *, model_path: str, confidence: float, iou: float, class_id: int):
        model = self._model_for(model_path)
        return model.predict(
            source=frame,
            conf=confidence,
            iou=iou,
            device=self._config.device,
            classes=[class_id],
            verbose=False,
        )[0]

    def _parse_players(self, result) -> list[PlayerDetection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        return [
            PlayerDetection(
                bbox=tuple(float(value) for value in bbox),
                confidence=float(score),
            )
            for bbox, score in zip(xyxy, confidence, strict=True)
        ]

    def _parse_balls(self, result, *, offset_x: int = 0, offset_y: int = 0) -> list[BallDetection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        balls: list[BallDetection] = []
        for bbox, score in zip(xyxy, confidence, strict=True):
            x1, y1, x2, y2 = bbox
            balls.append(
                BallDetection(
                    bbox=(
                        float(x1 + offset_x),
                        float(y1 + offset_y),
                        float(x2 + offset_x),
                        float(y2 + offset_y),
                    ),
                    confidence=float(score),
                )
            )
        return balls

    def _player_model_path(self) -> str:
        return self._config.player_model_path or self._config.model_path

    def _ball_model_path(self) -> str:
        return self._config.ball_model_path or self._config.model_path

    def _player_confidence(self) -> float:
        return self._config.player_detection_confidence or self._config.detection_confidence

    def _ball_confidence(self) -> float:
        return self._config.ball_detection_confidence or self._config.detection_confidence

    def _ball_iou(self) -> float:
        return self._config.ball_iou_threshold or self._config.iou_threshold

    def _detect_players(self, frame: np.ndarray) -> list[PlayerDetection]:
        result = self._predict(
            frame,
            model_path=self._player_model_path(),
            confidence=self._player_confidence(),
            iou=self._config.iou_threshold,
            class_id=self._config.player_class_id,
        )
        return self._parse_players(result)

    def _ball_search_bbox(self, frame: np.ndarray, players: list[PlayerDetection]) -> tuple[int, int, int, int] | None:
        bbox = pitch_bbox(build_pitch_mask(frame)) if self._config.pitch_mask_enabled else None
        if not players:
            return bbox

        player_x1 = min(int(player.bbox[0]) for player in players)
        player_y1 = min(int(player.bbox[1]) for player in players)
        player_x2 = max(int(player.bbox[2]) for player in players)
        player_y2 = max(int(player.bbox[3]) for player in players)
        margin = max(32, self._config.ball_tile_overlap)
        player_bbox = (
            max(0, player_x1 - margin),
            max(0, player_y1 - margin),
            min(frame.shape[1], player_x2 + margin),
            min(frame.shape[0], player_y2 + margin),
        )
        if bbox is None:
            return player_bbox

        clipped = (
            max(bbox[0], player_bbox[0]),
            max(bbox[1], player_bbox[1]),
            min(bbox[2], player_bbox[2]),
            min(bbox[3], player_bbox[3]),
        )
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            return player_bbox
        return clipped

    def _detect_balls_two_stage(
        self, frame: np.ndarray, players: list[PlayerDetection]
    ) -> list[BallDetection]:
        search_bbox = self._ball_search_bbox(frame, players)
        tiles = build_ball_tiles(
            frame.shape,
            search_bbox,
            self._config.ball_tile_size,
            self._config.ball_tile_overlap,
        )

        balls: list[BallDetection] = []
        for x1, y1, x2, y2 in tiles:
            tile = frame[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            result = self._predict(
                tile,
                model_path=self._ball_model_path(),
                confidence=self._ball_confidence(),
                iou=self._ball_iou(),
                class_id=self._config.ball_class_id,
            )
            balls.extend(self._parse_balls(result, offset_x=x1, offset_y=y1))
        return deduplicate_ball_detections(balls)

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        if self._config.two_stage_ball_detection:
            players = self._detect_players(frame)
            balls = self._detect_balls_two_stage(frame, players)
            return DetectionBatch(players=players, balls=balls)

        result = self._model_for(self._config.model_path).predict(
            source=frame,
            conf=self._config.detection_confidence,
            iou=self._config.iou_threshold,
            device=self._config.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionBatch(players=[], balls=[])

        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        players: list[PlayerDetection] = []
        balls: list[BallDetection] = []

        for bbox, score, class_id in zip(xyxy, confidence, class_ids, strict=True):
            box = tuple(float(value) for value in bbox)
            if class_id == self._config.player_class_id:
                players.append(PlayerDetection(bbox=box, confidence=float(score)))
            elif class_id == self._config.ball_class_id:
                balls.append(BallDetection(bbox=box, confidence=float(score)))

        return DetectionBatch(players=players, balls=balls)
