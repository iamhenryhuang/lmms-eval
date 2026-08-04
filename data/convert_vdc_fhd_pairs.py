#!/usr/bin/env python3
"""Create matched 1080p/240p H.264 pairs from a VDC JSONL manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of source videos to process; 0 means all.",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def probe_dimensions(path: Path) -> tuple[int, int] | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        width, height = result.stdout.strip().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError:
        return None


def ffmpeg_command(
    source: Path,
    output: Path,
    crf: int,
    preset: str,
    low: bool,
) -> list[str]:
    command = [
        "ffmpeg",
        "-n",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if low:
        command.extend(
            [
                "-vf",
                "scale=432:240:flags=lanczos,setsar=1",
            ]
        )
    command.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return command


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe must be available in PATH")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    high_dir = args.output_dir / "high"
    low_dir = args.output_dir / "low"
    if not args.dry_run:
        high_dir.mkdir(parents=True, exist_ok=True)
        low_dir.mkdir(parents=True, exist_ok=True)

    converted_high = converted_low = existing_high = existing_low = failures = 0

    for index, row in enumerate(rows, start=1):
        source = Path(row["source_path"])
        name = Path(row["video_name"]).stem + ".mp4"
        high_output = high_dir / name
        low_output = low_dir / name

        if not source.is_file():
            print(f"Missing source: {source}")
            failures += 1
            continue
        if probe_dimensions(source) != (1920, 1080):
            print(f"Not 1920x1080: {source}")
            failures += 1
            continue

        print(f"[{index}/{len(rows)}] {source.name}")
        for quality, output, low in (
            ("High", high_output, False),
            ("Low", low_output, True),
        ):
            if output.exists():
                print(f"  Skip existing {quality}: {output.name}")
                if low:
                    existing_low += 1
                else:
                    existing_high += 1
                continue
            if args.dry_run:
                print(f"  Would create {quality}: {output}")
                continue
            result = subprocess.run(
                ffmpeg_command(source, output, args.crf, args.preset, low),
                check=False,
            )
            if result.returncode != 0:
                print(f"  Failed {quality}: {source}")
                failures += 1
            elif low:
                converted_low += 1
            else:
                converted_high += 1

    print("\nFinished")
    print(f"  Source videos: {len(rows)}")
    print(f"  High converted/existing: {converted_high}/{existing_high}")
    print(f"  Low converted/existing:  {converted_low}/{existing_low}")
    print(f"  Failures: {failures}")
    print(f"  Output directory: {args.output_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
