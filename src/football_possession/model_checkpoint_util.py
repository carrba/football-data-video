from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def find_latest_checkpoint(search_root: Path, glob_pattern: str) -> Path | None:
    candidates = sorted(
        search_root.glob(glob_pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def to_preferred_path(path: Path) -> str:
    """Prefer a workspace-relative path for config readability."""
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
        return relative.as_posix()
    except ValueError:
        return str(path.resolve())


def update_ball_model_path(config_path: Path, model_path: str, dry_run: bool) -> tuple[str | None, bool]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    model_cfg = raw.setdefault("model", {})
    previous = model_cfg.get("ball_model_path")
    changed = previous != model_path
    model_cfg["ball_model_path"] = model_path

    if not dry_run:
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(raw, handle, sort_keys=False)

    return previous, changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find latest YOLO best.pt and update model.ball_model_path in a config file."
    )
    parser.add_argument(
        "--config",
        default="config/custom_ball_two_stage.yaml",
        help="Path to YAML config file to update (default: config/custom_ball_two_stage.yaml)",
    )
    parser.add_argument(
        "--search-root",
        default="runs/detect",
        help="Directory to search for checkpoints (default: runs/detect)",
    )
    parser.add_argument(
        "--glob",
        default="**/ball_detector*/weights/best.pt",
        help="Glob pattern under --search-root (default: **/ball_detector*/weights/best.pt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing the config file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    search_root = Path(args.search_root)
    if not search_root.exists():
        print(f"Error: search root not found: {search_root}")
        return 1

    latest = find_latest_checkpoint(search_root, args.glob)
    if latest is None:
        print(
            "Error: no checkpoint found. "
            "Expected something like runs/detect/**/ball_detector*/weights/best.pt"
        )
        return 1

    model_path = to_preferred_path(latest)
    config_path = Path(args.config)

    try:
        previous, changed = update_ball_model_path(config_path, model_path, args.dry_run)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    mode_label = "Dry run" if args.dry_run else "Updated"
    print(f"{mode_label} config: {config_path}")
    print(f"Latest checkpoint: {model_path}")
    print(f"Previous ball_model_path: {previous}")
    print(f"Changed: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
