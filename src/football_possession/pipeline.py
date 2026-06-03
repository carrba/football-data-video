from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd

from football_possession.config import AppConfig
from football_possession.detector import YoloDetector
from football_possession.possession import PossessionEstimator
from football_possession.team_classifier import TeamColorClassifier
from football_possession.tracker import PlayerTracker
from football_possession.detection_types import BallDetection, FrameRecord, PlayerDetection
from football_possession.video_io import build_video_writer, get_video_metadata, iter_video_frames

TEAM_COLORS = {
    0: (44, 160, 44),
    1: (214, 39, 40),
    None: (160, 160, 160),
}


class PossessionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def run(self, video_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = get_video_metadata(video_path)
        detector = YoloDetector(self._config.model)
        tracker = PlayerTracker(self._config.tracking, frame_rate=metadata.fps)
        team_classifier = TeamColorClassifier(self._config.teams)
        possession_estimator = PossessionEstimator(self._config.possession)

        records: list[FrameRecord] = []
        video_writer = None
        annotated_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
        if self._config.video.write_annotated_video:
            video_writer = build_video_writer(annotated_video_path, metadata)

        total_processed = 0
        estimated_frames = max(metadata.frame_count // max(self._config.video.frame_stride, 1), 1)

        for frame in iter_video_frames(video_path, self._config.video.frame_stride):
            detections = detector.detect(frame.image)
            players = tracker.update(detections.players)
            players = team_classifier.assign(frame.image, players)
            ball = self._select_ball(detections.balls)
            team_in_possession = possession_estimator.update(players, ball)

            records.append(
                FrameRecord(
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    team_in_possession=team_in_possession,
                    ball_visible=ball is not None,
                    player_count=len(players),
                )
            )

            if video_writer is not None:
                annotated = self._annotate_frame(frame.image, players, ball, team_in_possession)
                video_writer.write(annotated)

            total_processed += 1
            if total_processed == 1 or total_processed % 100 == 0:
                print(
                    f"Processed {total_processed}/{estimated_frames} sampled frames for {video_path.name}",
                    flush=True,
                )

        if video_writer is not None:
            video_writer.release()

        frame_csv_path = output_dir / "frame_possession.csv"
        summary_json_path = output_dir / "summary.json"

        dataframe = pd.DataFrame(asdict(record) for record in records)
        dataframe.to_csv(frame_csv_path, index=False)

        summary = self._build_summary(records)
        summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "frame_csv": frame_csv_path,
            "summary_json": summary_json_path,
            "annotated_video": annotated_video_path,
        }

    def _select_ball(self, balls: list[BallDetection]) -> BallDetection | None:
        if not balls:
            return None
        return max(balls, key=lambda item: item.confidence)

    def _annotate_frame(
        self,
        frame,
        players: list[PlayerDetection],
        ball: BallDetection | None,
        team_in_possession: int | None,
    ):
        annotated = frame.copy()
        for player in players:
            color = TEAM_COLORS.get(player.team_id, TEAM_COLORS[None])
            x1, y1, x2, y2 = (int(value) for value in player.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"id={player.track_id} team={player.team_id if player.team_id is not None else '?'}"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        if ball is not None:
            x1, y1, x2, y2 = (int(value) for value in ball.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(
                annotated,
                "ball",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 215, 255),
                1,
                cv2.LINE_AA,
            )

        possession_text = (
            f"Possession: Team {team_in_possession}"
            if team_in_possession is not None
            else "Possession: unknown"
        )
        cv2.putText(
            annotated,
            possession_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _build_summary(self, records: list[FrameRecord]) -> dict[str, object]:
        totals = Counter(record.team_in_possession for record in records)
        total_frames = max(len(records), 1)
        percentages = {
            "team_0": round((totals.get(0, 0) / total_frames) * 100, 2),
            "team_1": round((totals.get(1, 0) / total_frames) * 100, 2),
            "unknown": round((totals.get(None, 0) / total_frames) * 100, 2),
        }
        return {
            "frames_processed": len(records),
            "ball_visible_frames": sum(1 for record in records if record.ball_visible),
            "possession_percentages": percentages,
        }
