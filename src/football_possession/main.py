from __future__ import annotations

import argparse
import signal
from pathlib import Path

from football_possession.config import load_config
from football_possession.pipeline import PossessionPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GracefulShutdown(BaseException):
    """Raised from a signal handler so in-flight work can finalize cleanly."""


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        raise GracefulShutdown(f"Received signal {signum}")

    # SIGTERM: a plain kill. SIGHUP: what a dropped SSH session sends to its
    # foreground process group -- without a handler, the default action for both
    # is to terminate the process immediately with no chance to finalize the
    # in-progress annotated video.
    signal.signal(signal.SIGTERM, _handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handler)


def resolve_project_path(path_value: str, *, must_exist: bool = False) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    cwd_path = Path.cwd() / candidate
    if cwd_path.exists() or not must_exist:
        return cwd_path.resolve()

    project_path = PROJECT_ROOT / candidate
    if project_path.exists() or not must_exist:
        return project_path.resolve()

    return cwd_path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate football possession from recorded match video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV, summary, and annotated video outputs.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video generation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run from the last processed frame.",
    )
    return parser


def main() -> None:
    _install_signal_handlers()
    args = build_parser().parse_args()
    config_path = resolve_project_path(args.config, must_exist=True)
    config = load_config(config_path)
    config.model.model_path = str(resolve_project_path(config.model.model_path))
    if config.model.player_model_path:
        config.model.player_model_path = str(resolve_project_path(config.model.player_model_path))
    if config.model.ball_model_path:
        config.model.ball_model_path = str(resolve_project_path(config.model.ball_model_path))
    if args.no_video:
        config.video.write_annotated_video = False

    video_path = resolve_project_path(args.video, must_exist=True)
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else (PROJECT_ROOT / "outputs" / video_path.stem)
    )

    pipeline = PossessionPipeline(config)
    try:
        outputs = pipeline.run(video_path=video_path, output_dir=output_dir, resume=args.resume)
    except (GracefulShutdown, KeyboardInterrupt) as exc:
        print(f"Interrupted ({exc}); annotated video finalized up to the last processed frame.")
        print("No _SUCCESS marker was written since the run did not complete. Re-run with --resume.")
        raise SystemExit(1) from None

    print("Run complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
