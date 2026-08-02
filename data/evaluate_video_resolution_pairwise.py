#!/usr/bin/env python3
"""Two-pass blind pairwise evaluation for saved High/Low descriptions."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4o-2024-11-20"
DEFAULT_SEED = 20260802
RETRIES = 5
RETRY_DELAY_SECONDS = 5


SYSTEM_PROMPT = """You are evaluating two descriptions of the same video.
Compare them against the reference answer and choose which description is better.

Use only these criteria:
1. Correctness and consistency with the reference answer.
2. Coverage of the important subjects, actions, and background information in the reference answer.
3. Useful, specific detail rather than vague statements.

Penalize clear contradictions, factual errors, and excessive repetition. Do not prefer an
answer merely because it is longer. The reference answer may be incomplete, so do not assume
that every additional detail is false unless it contradicts the reference. Choose "tie" when
neither candidate is meaningfully better. Candidate labels are randomized and contain no
information about video quality."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blindly compare saved High and Low descriptions with an API judge."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Experiment directory containing High and Low sample JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <input_dir>/gpt_pairwise_scores.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N pairs; useful for a pilot run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed used to balance A/B assignment (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def response_text(sample: dict) -> str:
    response = sample["filtered_resps"]
    if isinstance(response, list):
        if len(response) != 1:
            raise ValueError(
                f"doc_id={sample['doc_id']} has {len(response)} filtered responses"
            )
        response = response[0]
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"doc_id={sample['doc_id']} has an empty response")
    return response


def index_samples(samples: list[dict], quality: str) -> tuple[list, dict]:
    ordered_ids = [sample["doc_id"] for sample in samples]
    indexed = {sample["doc_id"]: sample for sample in samples}
    if len(indexed) != len(samples):
        raise RuntimeError(f"Duplicate doc_id found in {quality} samples")
    return ordered_ids, indexed


def make_balanced_assignment(doc_ids: list, seed: int) -> dict:
    """Assign High to A for half the documents and Low to A for the rest."""
    shuffled = list(doc_ids)
    random.Random(seed).shuffle(shuffled)
    high_as_a = set(shuffled[: (len(shuffled) + 1) // 2])
    return {doc_id: ("high" if doc_id in high_as_a else "low") for doc_id in doc_ids}


def make_user_prompt(question: str, reference: str, candidate_a: str, candidate_b: str) -> str:
    return f"""Question:
{question}

Reference answer:
{reference}

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Return the better candidate and a brief reason."""


def parse_review(content: str) -> dict:
    try:
        review = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Judge returned invalid JSON: {content!r}") from error

    winner = review.get("winner")
    reason = review.get("reason")
    if winner not in {"A", "B", "tie"}:
        raise ValueError(f"Judge returned invalid winner: {winner!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Judge returned an empty reason")
    return {"winner": winner, "reason": reason.strip()}


def call_judge(
    *,
    api_url: str,
    api_key: str,
    model: str,
    question: str,
    reference: str,
    candidate_a: str,
    candidate_b: str,
) -> tuple[dict, str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": make_user_prompt(
                    question, reference, candidate_a, candidate_b
                ),
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "video_description_pairwise_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["winner", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    }

    # GPT-5 models use max_completion_tokens. Keep max_tokens for older
    # judges such as GPT-4o so existing evaluation commands remain usable.
    if model.lower().startswith("gpt-5"):
        payload["max_completion_tokens"] = 200
        payload["reasoning_effort"] = "none"
    else:
        payload["max_tokens"] = 200

    errors = []
    for attempt in range(1, RETRIES + 1):
        response = None
        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"].strip()
            review = parse_review(content)
            return review, content, response_data.get("model", model)
        except requests.HTTPError as error:
            status_code = response.status_code if response is not None else None
            response_text = response.text.strip() if response is not None else ""
            detail = f"attempt {attempt}: {error}"
            if response_text:
                detail += f"; response={response_text}"
            errors.append(detail)

            # Retry rate limits and server-side failures, but configuration
            # errors such as an unsupported parameter will not fix themselves.
            if status_code is not None and status_code != 429 and status_code < 500:
                break
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"attempt {attempt}: {error}")
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError("Judge failed after retries: " + " | ".join(errors))


def winner_quality(winner: str, candidate_a_quality: str) -> str:
    if winner == "tie":
        return "tie"
    if winner == "A":
        return candidate_a_quality
    return "low" if candidate_a_quality == "high" else "high"


def summarize(results: list[dict]) -> dict:
    counts = {
        "high_wins": sum(item["winner_quality"] == "high" for item in results),
        "low_wins": sum(item["winner_quality"] == "low" for item in results),
        "ties": sum(item["winner_quality"] == "tie" for item in results),
    }
    decisive = counts["high_wins"] + counts["low_wins"]
    return {
        "evaluated_pairs": len(results),
        **counts,
        "high_win_rate_excluding_ties": (
            counts["high_wins"] / decisive if decisive else None
        ),
    }


def build_consensus(first_pass: list[dict], swapped_pass: list[dict]) -> list[dict]:
    """Keep a winner only when both A/B orders choose the same quality."""
    first_by_id = {item["doc_id"]: item for item in first_pass}
    swapped_by_id = {item["doc_id"]: item for item in swapped_pass}
    consensus = []
    for doc_id in first_by_id.keys() & swapped_by_id.keys():
        first_winner = first_by_id[doc_id]["winner_quality"]
        swapped_winner = swapped_by_id[doc_id]["winner_quality"]
        winner = (
            first_winner
            if first_winner == swapped_winner and first_winner in {"high", "low"}
            else "tie"
        )
        consensus.append(
            {
                "doc_id": doc_id,
                "first_pass_winner": first_winner,
                "swapped_pass_winner": swapped_winner,
                "winner_quality": winner,
            }
        )
    return sorted(consensus, key=lambda item: item["doc_id"])


def save_progress(path: Path, payload: dict) -> None:
    first_pass = payload["results"]
    swapped_pass = payload.get("swapped_results", [])
    consensus = build_consensus(first_pass, swapped_pass)
    # Keep the original summary field for compatibility with first-pass outputs.
    payload["summary"] = summarize(first_pass)
    payload["first_pass_summary"] = summarize(first_pass)
    payload["swapped_pass_summary"] = summarize(swapped_pass)
    payload["consensus_results"] = consensus
    payload["consensus_summary"] = summarize(consensus)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    input_dir = args.input_dir.resolve()
    output_path = (args.output or input_dir / "gpt_pairwise_scores.json").resolve()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is empty. Add it to the repository's .env file before running."
        )
    api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    judge_model = os.getenv("MODEL_VERSION", DEFAULT_MODEL)

    sample_paths = {
        "high": find_one(input_dir, "*_samples_video_resolution_high.jsonl"),
        "low": find_one(input_dir, "*_samples_video_resolution_low.jsonl"),
    }
    raw_samples = {
        quality: read_jsonl(path) for quality, path in sample_paths.items()
    }
    high_ids, samples_high = index_samples(raw_samples["high"], "high")
    low_ids, samples_low = index_samples(raw_samples["low"], "low")
    if high_ids != low_ids:
        raise RuntimeError("High and Low doc_id order does not match")

    assignment = make_balanced_assignment(high_ids, args.seed)
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("judge_model") != judge_model:
            raise RuntimeError(
                "Existing output uses a different judge model; choose another --output path."
            )
        if payload.get("assignment_seed") != args.seed:
            raise RuntimeError(
                "Existing output uses a different assignment seed; choose another --output path."
            )
    else:
        payload = {
            "evaluation_type": "two_pass_blind_reference_based_pairwise",
            "judge_model": judge_model,
            "assignment_seed": args.seed,
            "source_files": {
                quality: str(path) for quality, path in sample_paths.items()
            },
            "results": [],
            "swapped_results": [],
            "summary": {},
        }

    # Migrate an existing one-pass output without discarding any API results.
    payload["evaluation_type"] = "two_pass_blind_reference_based_pairwise"
    payload.setdefault("swapped_results", [])

    selected_ids = high_ids[: args.limit] if args.limit is not None else high_ids
    passes = (
        ("first", payload["results"], False),
        ("swapped", payload["swapped_results"], True),
    )
    for pass_name, pass_results, reverse_assignment in passes:
        completed = {item["doc_id"] for item in pass_results}
        for doc_id in selected_ids:
            if doc_id in completed:
                print(f"{pass_name} pass doc_id={doc_id}: already evaluated")
                continue

            high_sample = samples_high[doc_id]
            low_sample = samples_low[doc_id]
            if high_sample["input"] != low_sample["input"]:
                raise RuntimeError(f"Question mismatch for doc_id={doc_id}")
            if high_sample["target"] != low_sample["target"]:
                raise RuntimeError(f"Reference mismatch for doc_id={doc_id}")

            candidate_a_quality = assignment[doc_id]
            if reverse_assignment:
                candidate_a_quality = (
                    "low" if candidate_a_quality == "high" else "high"
                )
            candidate_b_quality = (
                "low" if candidate_a_quality == "high" else "high"
            )
            samples_by_quality = {"high": high_sample, "low": low_sample}
            candidate_a = response_text(samples_by_quality[candidate_a_quality])
            candidate_b = response_text(samples_by_quality[candidate_b_quality])

            review, raw_review, response_model = call_judge(
                api_url=api_url,
                api_key=api_key,
                model=judge_model,
                question=high_sample["input"],
                reference=high_sample["target"],
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )
            resolved_winner = winner_quality(review["winner"], candidate_a_quality)
            result = {
                "doc_id": doc_id,
                "candidate_a_quality": candidate_a_quality,
                "candidate_b_quality": candidate_b_quality,
                "winner_label": review["winner"],
                "winner_quality": resolved_winner,
                "reason": review["reason"],
                "raw_review": raw_review,
                "response_model": response_model,
            }
            pass_results.append(result)
            completed.add(doc_id)
            save_progress(output_path, payload)
            print(
                f"{pass_name} pass doc_id={doc_id}: winner={resolved_winner} "
                f"({review['winner']}) reason={review['reason']}"
            )

    save_progress(output_path, payload)
    summary = payload["consensus_summary"]
    win_rate = summary["high_win_rate_excluding_ties"]
    win_rate_text = "N/A" if win_rate is None else f"{win_rate:.1%}"
    print("Consensus (same winner required in both A/B orders)")
    print(f"Evaluated pairs: {summary['evaluated_pairs']}")
    print(f"High wins:       {summary['high_wins']}")
    print(f"Low wins:        {summary['low_wins']}")
    print(f"Ties/conflicts:  {summary['ties']}")
    print(f"High win rate:   {win_rate_text} (excluding ties/conflicts)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
