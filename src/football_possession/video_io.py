from __future__ import annotations

import shutil
import struct
import subprocess
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


def iter_video_frames(
    video_path: str | Path,
    frame_stride: int,
    start_frame_index: int = 0,
) -> Iterator[VideoFrame]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = max(capture.get(cv2.CAP_PROP_FPS), 1.0)

    if start_frame_index > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

    frame_index = start_frame_index

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


class _FfmpegPipeWriter:
    """Writes raw BGR frames to an H.264 mp4 via a piped ffmpeg process.

    cv2.VideoWriter's mp4v codec produces huge, poorly-compressed files, and its
    muxer writes an invalid container (missing moov atom) once output exceeds
    ~4GiB because it doesn't use 64-bit box sizes. Piping frames into ffmpeg's
    own muxer avoids both problems and bakes in faststart from the start.
    """

    def __init__(self, output_path: Path, metadata: VideoMetadata) -> None:
        self._process = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{metadata.width}x{metadata.height}",
                "-r", str(metadata.fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray) -> None:
        self._process.stdin.write(frame.tobytes())

    def release(self) -> None:
        self._process.stdin.close()
        returncode = self._process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {returncode} while writing annotated video")


def build_video_writer(
    output_path: str | Path, metadata: VideoMetadata
) -> "_FfmpegPipeWriter | cv2.VideoWriter":
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg"):
        return _FfmpegPipeWriter(output_path, metadata)

    print("ffmpeg not found on PATH; falling back to cv2.VideoWriter (mp4v, no faststart, no compression).")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, metadata.fps, (metadata.width, metadata.height))


def _is_already_faststart(video_path: Path, max_boxes: int = 12) -> bool:
    try:
        with open(video_path, "rb") as handle:
            offset = 0
            for _ in range(max_boxes):
                handle.seek(offset)
                header = handle.read(8)
                if len(header) < 8:
                    return False

                size, box_type = struct.unpack(">I4s", header)
                box_type = box_type.decode("ascii", errors="replace")
                if box_type == "moov":
                    return True
                if box_type == "mdat":
                    return False

                if size == 1:
                    largesize = handle.read(8)
                    if len(largesize) < 8:
                        return False
                    size = struct.unpack(">Q", largesize)[0]
                elif size == 0:
                    return False

                offset += size
    except OSError:
        return False

    return False


def remux_faststart(video_path: str | Path) -> None:
    """Move the moov atom to the front so the file is seekable/streamable over HTTP.

    No-op if the file already has moov before mdat (e.g. it was written by the
    ffmpeg pipe writer, which bakes in faststart already).
    """
    video_path = Path(video_path)

    if _is_already_faststart(video_path):
        return

    if not shutil.which("ffmpeg"):
        print(f"ffmpeg not found on PATH; skipping faststart remux for {video_path}")
        return

    temp_path = video_path.with_suffix(".faststart.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(temp_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        temp_path.unlink(missing_ok=True)
        print(f"ffmpeg faststart remux failed for {video_path}:\n{result.stderr}")
        return

    temp_path.replace(video_path)
