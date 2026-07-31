#!/usr/bin/env python3
"""Convert 1920x1080 videos into the low-quality evaluation condition.

The script keeps the source videos untouched and skips output files that
already exist. FFmpeg and FFprobe must be available in PATH.

Examples:
    python data/convert_fhd_to_low.py
    python data/convert_fhd_to_low.py --limit 2
    python data/convert_fhd_to_low.py --source-dir /path/to/videos
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "VideoDetailCaption" / "extracted" / "Test_Videos"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "VideoDetailCaption" / "paired" / "low"
VIDEO_EXTENSIONS = {".mp4", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 1920x1080 videos to 432x240 H.264 videos at 400 kb/s."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Source video directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of new videos to convert; 0 means all (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which videos would be converted without writing output files.",
    )
    return parser.parse_args()


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"Error: {command} is not available in PATH.")


def probe_dimensions(video_path: Path) -> tuple[int, int] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None

    try:
        width, height = result.stdout.strip().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError:
        return None


def build_ffmpeg_command(source: Path, output: Path) -> list[str]:
    scale_filter = (
        "scale=432:240:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "pad=432:240:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    return [
        "ffmpeg",
        "-n",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-b:v",
        "400k",
        "-minrate",
        "400k",
        "-maxrate",
        "400k",
        "-bufsize",
        "800k",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(output),
    ]


def main() -> int:
    args = parse_args()
    require_command("ffmpeg")
    require_command("ffprobe")

    if args.limit < 0:
        raise SystemExit("Error: --limit must be a non-negative integer.")
    if not args.source_dir.is_dir():
        raise SystemExit(f"Error: source directory does not exist: {args.source_dir}")

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path
        for path in args.source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    converted = 0
    existing = 0
    non_fhd = 0
    probe_failed = 0
    conversion_failed = 0

    for source in videos:
        dimensions = probe_dimensions(source)
        if dimensions is None:
            print(f"Probe failed: {source.name}", file=sys.stderr)
            probe_failed += 1
            continue
        if dimensions != (1920, 1080):
            non_fhd += 1
            continue

        output = args.output_dir / f"{source.stem}.mp4"
        if output.exists():
            print(f"Skip existing: {output}")
            existing += 1
            continue
        if args.limit and converted >= args.limit:
            break

        action = "Would convert" if args.dry_run else "Convert"
        print(f"{action}: {source.name} (1920x1080) -> {output.name}")

        if args.dry_run:
            converted += 1
            continue

        result = subprocess.run(build_ffmpeg_command(source, output), check=False)
        if result.returncode == 0:
            converted += 1
        else:
            print(f"Conversion failed: {source}", file=sys.stderr)
            conversion_failed += 1

    print("\nFinished")
    print(f"  {'Would convert' if args.dry_run else 'Converted'}:       {converted}")
    print(f"  Existing outputs: {existing}")
    print(f"  Non-1920x1080:    {non_fhd}")
    print(f"  Probe failures:    {probe_failed}")
    print(f"  Convert failures:  {conversion_failed}")
    print(f"  Output directory:  {args.output_dir}")

    return 1 if probe_failed or conversion_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
