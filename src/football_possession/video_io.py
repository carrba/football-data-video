from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(slots=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(slots=True)
class VideoFrame:
    index: int
    timestamp_s: float
    image: np.ndarray


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    metadata = VideoMetadata(
        fps=max(capture.get(cv2.CAP_PROP_FPS), 1.0),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return metadata


def iter_video_frames(video_path: str | Path, frame_stride: int) -> Iterator[VideoFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = max(capture.get(cv2.CAP_PROP_FPS), 1.0)
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_stride == 0:
            yield VideoFrame(
                index=frame_index,
                timestamp_s=frame_index / fps,
                image=frame,
            )
        frame_index += 1

    capture.release()


def build_video_writer(output_path: str | Path, metadata: VideoMetadata) -> cv2.VideoWriter:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, metadata.fps, (metadata.width, metadata.height))
