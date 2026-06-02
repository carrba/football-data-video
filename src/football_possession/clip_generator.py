from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(slots=True)
class ClipSpec:
    index: int
    start_s: float
    duration_s: float


def get_video_duration(video_path: str | Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = max(capture.get(cv2.CAP_PROP_FPS), 1.0)
    frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    capture.release()
    return frame_count / fps


def plan_clips(
    total_duration_s: float,
    clip_duration_s: float,
    step_s: float,
    start_offset_s: float,
    end_padding_s: float,
    max_clips: int | None,
) -> list[ClipSpec]:
    if clip_duration_s <= 0:
        raise ValueError("clip_duration_s must be positive")
    if step_s <= 0:
        raise ValueError("step_s must be positive")

    usable_end = max(0.0, total_duration_s - end_padding_s)
    start = max(0.0, start_offset_s)
    clips: list[ClipSpec] = []
    index = 0

    while start + clip_duration_s <= usable_end:
        clips.append(ClipSpec(index=index, start_s=start, duration_s=clip_duration_s))
        index += 1
        if max_clips is not None and len(clips) >= max_clips:
            break
        start += step_s

    return clips


def run_ffmpeg(input_path: Path, output_path: Path, clip: ClipSpec, reencode: bool) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{clip.start_s:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{clip.duration_s:.3f}",
    ]

    if reencode:
        command.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac"])
    else:
        command.extend(["-c", "copy"])

    command.append(str(output_path))
    subprocess.run(command, check=True)


def write_manifest(output_dir: Path, clips: list[ClipSpec]) -> Path:
    manifest_path = output_dir / "clips_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_index", "start_s", "duration_s", "filename"])
        writer.writeheader()
        for clip in clips:
            writer.writerow(
                {
                    "clip_index": clip.index,
                    "start_s": f"{clip.start_s:.3f}",
                    "duration_s": f"{clip.duration_s:.3f}",
                    "filename": f"clip_{clip.index:03d}.mp4",
                }
            )
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed-duration clips from a full match video using FFmpeg.")
    parser.add_argument("--input", required=True, help="Path to the full match MP4.")
    parser.add_argument("--output-dir", default="video/clips", help="Directory for generated clips.")
    parser.add_argument("--clip-duration", type=float, default=30.0, help="Length of each clip in seconds.")
    parser.add_argument("--step", type=float, default=60.0, help="Seconds between clip start times.")
    parser.add_argument("--start-offset", type=float, default=0.0, help="Skip this many seconds from the start.")
    parser.add_argument("--end-padding", type=float, default=0.0, help="Do not generate clips that enter the final padded seconds.")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional maximum number of clips to generate.")
    parser.add_argument("--reencode", action="store_true", help="Re-encode clips for frame-accurate cutting instead of stream copy.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed or not available on PATH.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_duration_s = get_video_duration(input_path)
    clips = plan_clips(
        total_duration_s=total_duration_s,
        clip_duration_s=args.clip_duration,
        step_s=args.step,
        start_offset_s=args.start_offset,
        end_padding_s=args.end_padding,
        max_clips=args.max_clips,
    )
    if not clips:
        raise SystemExit("No clips fit within the selected duration and offsets.")

    for clip in clips:
        output_path = output_dir / f"clip_{clip.index:03d}.mp4"
        run_ffmpeg(input_path=input_path, output_path=output_path, clip=clip, reencode=args.reencode)

    manifest_path = write_manifest(output_dir, clips)
    print(f"Created {len(clips)} clips in {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()