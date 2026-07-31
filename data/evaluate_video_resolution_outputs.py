#!/usr/bin/env python3
"""Evaluate saved High/Low descriptions with lmms-eval's VDD GPT judge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Score existing video-resolution JSONL outputs with the evaluator " "from lmms_eval.tasks.video_detail_description."))
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Experiment directory containing High and Low sample JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <input_dir>/gpt_eval_scores.json).",
    )
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern!r} below {root}, found {len(matches)}")
    return matches[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = (args.output or input_dir / "gpt_eval_scores.json").resolve()

    # Existing shell variables take precedence over values in the local .env.
    load_dotenv(REPO_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is empty. Add it to the repository's .env file "
            "before running."
        )

    # The repository evaluator expects HF_HOME during import even though this
    # offline script does not load its dataset or videos.
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    from lmms_eval.tasks.video_detail_description.utils import (  # noqa: PLC0415
        GPT_EVAL_MODEL_NAME,
        get_eval_generic,
        parse_score,
    )

    sample_paths = {
        "high": find_one(input_dir, "*_samples_video_resolution_high.jsonl"),
        "low": find_one(input_dir, "*_samples_video_resolution_low.jsonl"),
    }
    samples = {quality: read_jsonl(path) for quality, path in sample_paths.items()}

    if len(samples["high"]) != len(samples["low"]):
        raise RuntimeError("High and Low sample counts do not match")

    high_ids = [sample["doc_id"] for sample in samples["high"]]
    low_ids = [sample["doc_id"] for sample in samples["low"]]
    if high_ids != low_ids:
        raise RuntimeError("High and Low doc_id order does not match")

    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("judge_model") != GPT_EVAL_MODEL_NAME:
            raise RuntimeError("Existing output uses a different judge model; choose another " "--output path or restore the original MODEL_VERSION.")
    else:
        payload = {
            "judge_model": GPT_EVAL_MODEL_NAME,
            "source_files": {key: str(value) for key, value in sample_paths.items()},
            "results": [],
            "averages": {},
        }

    completed = {(item["quality"], item["doc_id"]) for item in payload["results"]}

    for quality in ("high", "low"):
        for sample in samples[quality]:
            result_key = (quality, sample["doc_id"])
            if result_key in completed:
                print(f"{quality} doc_id={sample['doc_id']}: already evaluated")
                continue

            review, response_model = get_eval_generic(
                question=sample["input"],
                answer=sample["target"],
                pred=sample["filtered_resps"],
                max_tokens=64,
            )
            result = {
                "quality": quality,
                "doc_id": sample["doc_id"],
                "score": parse_score(review),
                "review": review,
                "response_model": response_model,
            }
            payload["results"].append(result)
            completed.add(result_key)

            for current_quality in ("high", "low"):
                scores = [item["score"] for item in payload["results"] if item["quality"] == current_quality]
                if scores:
                    payload["averages"][current_quality] = sum(scores) / len(scores)

            save_progress(output_path, payload)
            print(f"{quality} doc_id={sample['doc_id']}: " f"score={result['score']} review={review}")

    print(f"High average: {payload['averages']['high']:.3f}")
    print(f"Low average:  {payload['averages']['low']:.3f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
