#!/usr/bin/env python3
"""Select a reproducible, stratified Full-HD subset from local VDC videos."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path


QUOTAS = {
    "people_daily": 25,
    "sports_animals_activities": 19,
    "other_closeups": 12,
    "city_transport": 19,
    "nature_scenery": 25,
}

# Categories are assigned in this order so scarce categories are filled before
# the abundant nature/scenery category. Each video can be selected only once.
PATTERNS = {
    "people_daily": re.compile(
        r"\b(person|people|woman|women|man|men|girl|girls|boy|boys|family|"
        r"friends?|child|children|couple|workers?|tourists?)\b",
        re.IGNORECASE,
    ),
    "sports_animals_activities": re.compile(
        r"\b(sports?|exercise|fitness|soccer|football|basketball|tennis|"
        r"cycling|cyclist|running|runner|workout|skating|skateboard|surfing|"
        r"skiing|snowboard|hiking|swimming|yoga|animals?|dogs?|cats?|birds?|"
        r"horses?|wildlife|fish|lion|monkey|deer)\b",
        re.IGNORECASE,
    ),
    "other_closeups": re.compile(
        r"\b(food|cooking|cook|kitchen|chef|meal|coffee|artist|painting|"
        r"concert|music|musician|factory|repair|machine|machinery|tools?|"
        r"product|device|computer|phone|close-up|close up)\b",
        re.IGNORECASE,
    ),
    "city_transport": re.compile(
        r"\b(urban|city|cityscape|streets?|buildings?|architecture|bridges?|"
        r"town|cars?|vehicles?|trains?|airplanes?|boats?|ships?|motorcycles?|"
        r"bicycles?|roads?|driving|traffic|sailing|port|freeway)\b",
        re.IGNORECASE,
    ),
    "nature_scenery": re.compile(
        r"\b(nature|natural|landscape|forests?|mountains?|ocean|sea|beach|"
        r"river|waterfall|sunset|meadow|valley|desert|lake|coast|coastline|"
        r"clouds?|snow|fields?|canyon|island|sky)\b",
        re.IGNORECASE,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-fhd",
        action="store_true",
        help="Select every matched 1920x1080 video instead of the stratified subset.",
    )
    return parser.parse_args()


def is_full_hd(path: Path) -> bool:
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
    return result.returncode == 0 and result.stdout.strip() == "1920x1080"


def searchable_text(row: dict) -> str:
    fields = (
        row.get("video_name", "").replace("-", " "),
        row.get("short_caption", ""),
        row.get("main_object_caption", ""),
        row.get("camera_caption", ""),
    )
    return " ".join(fields)


def primary_category(row: dict) -> str:
    text = searchable_text(row)
    for category, pattern in PATTERNS.items():
        if pattern.search(text):
            return category
    return "unclassified"


def output_row(row: dict, category: str, videos_dir: Path) -> dict:
    result = dict(row)
    result["selection_category"] = category
    result["source_path"] = str((videos_dir / row["video_name"]).resolve())
    return result


def main() -> None:
    args = parse_args()
    annotations = {}
    with args.annotations.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            annotations[row["video_name"]] = row

    candidates = []
    for path in sorted(args.videos_dir.glob("*.mp4")):
        row = annotations.get(path.name)
        if row is not None and is_full_hd(path):
            candidates.append(row)

    selected = []
    summary = {}

    if args.all_fhd:
        for row in candidates:
            category = primary_category(row)
            selected.append(output_row(row, category, args.videos_dir))
            summary[category] = summary.get(category, 0) + 1
    else:
        rng = random.Random(args.seed)
        used = set()
        for category, quota in QUOTAS.items():
            pool = [
                row
                for row in candidates
                if row["video_name"] not in used
                and PATTERNS[category].search(searchable_text(row))
            ]
            rng.shuffle(pool)
            chosen = pool[:quota]
            if len(chosen) != quota:
                raise RuntimeError(
                    f"Not enough unique videos for {category}: "
                    f"needed {quota}, found {len(chosen)}"
                )
            for row in chosen:
                selected.append(output_row(row, category, args.videos_dir))
                used.add(row["video_name"])
            summary[category] = len(chosen)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Full-HD candidates: {len(candidates)}")
    for category, count in summary.items():
        print(f"{category}: {count}")
    print(f"Selected: {len(selected)}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
