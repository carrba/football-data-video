from __future__ import annotations

import argparse
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import boto3
from botocore.client import BaseClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = ["video", "datasets", "annotations", "outputs", "config", "results.json"]
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
ENV_BUCKET = "FOOTBALL_S3_BUCKET"
ENV_PREFIX = "FOOTBALL_S3_PREFIX"
ENV_PROFILE = "FOOTBALL_AWS_PROFILE"


@dataclass(frozen=True)
class SyncConfig:
    bucket: str
    prefix: str
    region: str | None
    profile: str | None
    paths: list[str]
    dry_run: bool


def _normalize_prefix(prefix: str) -> str:
    value = prefix.strip("/")
    return f"{value}/" if value else ""


def _is_glob_spec(spec: str) -> bool:
    return any(ch in spec for ch in "*?[]")


def _expand_local_targets(root: Path, spec: str) -> list[Path]:
    if _is_glob_spec(spec):
        return [p.resolve() for p in root.glob(spec)]

    return [(root / spec).resolve()]


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _iter_local_files(root: Path, path_specs: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()

    for spec in path_specs:
        targets = _expand_local_targets(root, spec)
        existing_targets = [t for t in targets if t.exists()]
        if not existing_targets:
            print(f"Skipping missing path: {spec}")
            continue

        for target in existing_targets:
            if _is_excluded(target):
                continue

            if target.is_file():
                if target not in seen:
                    seen.add(target)
                    yield target
                continue

            for file_path in target.rglob("*"):
                if not file_path.is_file():
                    continue
                if _is_excluded(file_path):
                    continue
                if file_path not in seen:
                    seen.add(file_path)
                    yield file_path


def _path_spec_matches(relative_path: str, path_specs: Iterable[str]) -> bool:
    for spec in path_specs:
        cleaned = spec.strip("/")
        if not cleaned:
            continue

        if _is_glob_spec(cleaned):
            if fnmatch.fnmatch(relative_path, cleaned):
                return True
            continue

        if relative_path == cleaned or relative_path.startswith(f"{cleaned}/"):
            return True

    return False


def _relative_key(path: Path, root: Path, prefix: str) -> str:
    rel = path.relative_to(root).as_posix()
    return f"{prefix}{rel}"


def _s3_client(region: str | None, profile: str | None) -> BaseClient:
    if profile:
        session = boto3.session.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.session.Session(region_name=region)
    return session.client("s3")


def _list_s3_sizes(client: BaseClient, bucket: str, prefix: str) -> dict[str, int]:
    """Return {key: size_bytes} for all objects under prefix."""
    sizes: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def push_to_s3(config: SyncConfig) -> None:
    root = PROJECT_ROOT
    prefix = _normalize_prefix(config.prefix)
    client = _s3_client(config.region, config.profile)
    s3_sizes = _list_s3_sizes(client, config.bucket, prefix)

    uploaded = skipped = 0
    for file_path in _iter_local_files(root, config.paths):
        key = _relative_key(file_path, root, prefix)
        if s3_sizes.get(key) == file_path.stat().st_size:
            print(f"SKIP {file_path.relative_to(root)} (same size)")
            skipped += 1
            continue
        print(f"PUSH {file_path.relative_to(root)} -> s3://{config.bucket}/{key}")
        if not config.dry_run:
            client.upload_file(str(file_path), config.bucket, key)
        uploaded += 1

    print(f"Upload complete. Uploaded: {uploaded}, skipped: {skipped}")


def pull_from_s3(config: SyncConfig) -> None:
    root = PROJECT_ROOT
    prefix = _normalize_prefix(config.prefix)
    client = _s3_client(config.region, config.profile)

    downloaded = skipped = 0

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            if not rel:
                continue

            # Keep only selected paths (supports both prefixes and glob specs).
            if not _path_spec_matches(rel, config.paths):
                continue

            destination = root / rel
            if _is_excluded(destination):
                continue

            if destination.exists() and destination.stat().st_size == obj["Size"]:
                print(f"SKIP {destination.relative_to(root)} (same size)")
                skipped += 1
                continue

            print(f"PULL s3://{config.bucket}/{key} -> {destination.relative_to(root)}")
            if not config.dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(config.bucket, key, str(destination))
            downloaded += 1

    print(f"Download complete. Downloaded: {downloaded}, skipped: {skipped}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push/pull project data files between local repo and S3."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("push", "pull"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "--bucket",
            default=os.getenv(ENV_BUCKET),
            help=f"S3 bucket name. Defaults to {ENV_BUCKET}.",
        )
        sub.add_argument(
            "--prefix",
            default=os.getenv(ENV_PREFIX, "football-data-video"),
            help=(
                "S3 key prefix used as a project namespace. "
                f"Defaults to {ENV_PREFIX} or 'football-data-video'."
            ),
        )
        sub.add_argument(
            "--region",
            default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            help="AWS region. Defaults to AWS_REGION/AWS_DEFAULT_REGION if set.",
        )
        sub.add_argument(
            "--profile",
            default=os.getenv(ENV_PROFILE),
            help=(
                f"AWS named profile for authentication. Defaults to {ENV_PROFILE} "
                "env var if set, otherwise the default credential chain is used "
                "(EC2 IAM roles work automatically when this is unset)."
            ),
        )
        sub.add_argument(
            "--paths",
            nargs="+",
            default=DEFAULT_PATHS,
            help=(
                "Top-level files/folders to sync relative to repo root. "
                f"Default: {' '.join(DEFAULT_PATHS)}"
            ),
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Print actions without uploading/downloading files.",
        )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.bucket:
        raise SystemExit(
            "Missing S3 bucket. Pass --bucket or set FOOTBALL_S3_BUCKET."
        )

    cfg = SyncConfig(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        profile=args.profile,
        paths=args.paths,
        dry_run=args.dry_run,
    )

    if args.command == "push":
        push_to_s3(cfg)
    elif args.command == "pull":
        pull_from_s3(cfg)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
