#!/usr/bin/env python3
"""Two-pass blind evaluation of the magnitude of the High/Low caption gap."""

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
DEFAULT_MODEL = "gpt-5.1-2025-11-13"
DEFAULT_SEED = 20260802
RETRIES = 5
RETRY_DELAY_SECONDS = 5
BOOTSTRAP_SAMPLES = 10_000


SYSTEM_PROMPT = """You are evaluating two descriptions of the same video.
Compare them against the reference answer and judge both the direction and magnitude of
the quality difference.

Use only these criteria:
1. Correctness and consistency with the reference answer.
2. Coverage of important subjects, actions, events, and background information.
3. Useful, specific detail rather than vague statements.

Penalize clear contradictions, factual errors, important omissions, unsupported
specific claims, and excessive repetition. Do not prefer a description merely because
it is longer. The reference may be incomplete, so an additional detail is not an error
unless it is implausible or contradicts the reference.

Use this gap scale:
0 = no meaningful quality difference; outcome must be "tie".
1 = slight difference; both descriptions are similarly useful, with only a small edge.
2 = meaningful difference; one description has notably better correctness, coverage,
    or specificity.
3 = large difference; one description is substantially better on the core video content,
    or the other has major errors or omissions.

Return exactly one outcome: "tie", "A_slight", "A_meaningful", "A_large",
"B_slight", "B_meaningful", or "B_large". Use "tie" only when there is no
meaningful quality difference. The outcome combines the winner and gap magnitude so
they cannot contradict each other.

Candidate labels are randomized and contain no information about video quality."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blindly score the direction and 0-3 magnitude of the quality gap "
            "between saved High and Low video descriptions."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Experiment directory containing High and Low sample JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <input_dir>/gpt_paired_gap_scores.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N pairs; useful for a smoke test.",
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
    shuffled = list(doc_ids)
    random.Random(seed).shuffle(shuffled)
    high_as_a = set(shuffled[: (len(shuffled) + 1) // 2])
    return {doc_id: ("high" if doc_id in high_as_a else "low") for doc_id in doc_ids}


def make_user_prompt(
    question: str,
    reference: str,
    candidate_a: str,
    candidate_b: str,
) -> str:
    return f"""Question:
{question}

Reference answer:
{reference}

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Return one outcome and a brief reason."""


def parse_review(content: str) -> dict:
    try:
        review = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Judge returned invalid JSON: {content!r}") from error

    outcome = review.get("outcome")
    reason = review.get("reason")
    outcome_map = {
        "tie": ("tie", 0),
        "A_slight": ("A", 1),
        "A_meaningful": ("A", 2),
        "A_large": ("A", 3),
        "B_slight": ("B", 1),
        "B_meaningful": ("B", 2),
        "B_large": ("B", 3),
    }
    if outcome not in outcome_map:
        raise ValueError(f"Judge returned invalid outcome: {outcome!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Judge returned an empty reason")
    winner, gap = outcome_map[outcome]
    return {
        "outcome": outcome,
        "winner": winner,
        "gap": gap,
        "reason": reason.strip(),
    }


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
                "name": "video_description_paired_gap_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "enum": [
                                "tie",
                                "A_slight",
                                "A_meaningful",
                                "A_large",
                                "B_slight",
                                "B_meaningful",
                                "B_large",
                            ],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["outcome", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    }

    if model.lower().startswith("gpt-5"):
        payload["max_completion_tokens"] = 250
        payload["reasoning_effort"] = "none"
    else:
        payload["max_tokens"] = 250

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
            response_body = response.text.strip() if response is not None else ""
            detail = f"attempt {attempt}: {error}"
            if response_body:
                detail += f"; response={response_body}"
            errors.append(detail)
            if status_code is not None and status_code != 429 and status_code < 500:
                break
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
        except (
            requests.RequestException,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
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


def signed_gap(winner: str, gap: float) -> float:
    if winner == "high":
        return gap
    if winner == "low":
        return -gap
    return 0.0


def build_consensus(first_pass: list[dict], swapped_pass: list[dict]) -> list[dict]:
    """Accept a score only when both A/B orders agree on the winner quality."""
    first_by_id = {item["doc_id"]: item for item in first_pass}
    swapped_by_id = {item["doc_id"]: item for item in swapped_pass}
    consensus = []
    for doc_id in sorted(first_by_id.keys() & swapped_by_id.keys()):
        first = first_by_id[doc_id]
        swapped = swapped_by_id[doc_id]
        first_winner = first["winner_quality"]
        swapped_winner = swapped["winner_quality"]
        agrees = first_winner == swapped_winner

        if agrees:
            winner = first_winner
            gap = (first["gap"] + swapped["gap"]) / 2
            status = "agreement"
            signed = signed_gap(winner, gap)
        else:
            winner = "conflict"
            gap = None
            status = "conflict"
            signed = None

        consensus.append(
            {
                "doc_id": doc_id,
                "first_pass_winner": first_winner,
                "first_pass_gap": first["gap"],
                "swapped_pass_winner": swapped_winner,
                "swapped_pass_gap": swapped["gap"],
                "gap_disagreement": abs(first["gap"] - swapped["gap"]),
                "status": status,
                "winner_quality": winner,
                "gap": gap,
                "signed_gap": signed,
            }
        )
    return consensus


def bootstrap_mean_ci(
    values: list[float], seed: int, samples: int = BOOTSTRAP_SAMPLES
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    lower = means[int(samples * 0.025)]
    upper = means[min(samples - 1, int(samples * 0.975))]
    return [lower, upper]


def summarize(consensus: list[dict], seed: int) -> dict:
    scored = [item for item in consensus if item["status"] == "agreement"]
    signed_values = [item["signed_gap"] for item in scored]
    absolute_values = [item["gap"] for item in scored]
    disagreements = [item["gap_disagreement"] for item in scored]
    return {
        "evaluated_pairs": len(consensus),
        "scored_pairs": len(scored),
        "conflicts": len(consensus) - len(scored),
        "high_better": sum(item["winner_quality"] == "high" for item in scored),
        "low_better": sum(item["winner_quality"] == "low" for item in scored),
        "equivalent": sum(item["winner_quality"] == "tie" for item in scored),
        "mean_signed_gap": (
            sum(signed_values) / len(signed_values) if signed_values else None
        ),
        "mean_absolute_gap": (
            sum(absolute_values) / len(absolute_values) if absolute_values else None
        ),
        "mean_signed_gap_bootstrap_95_ci": bootstrap_mean_ci(
            signed_values, seed
        ),
        "mean_gap_disagreement": (
            sum(disagreements) / len(disagreements) if disagreements else None
        ),
    }


def save_progress(path: Path, payload: dict) -> None:
    consensus = build_consensus(
        payload["results"], payload.get("swapped_results", [])
    )
    payload["consensus_results"] = consensus
    payload["summary"] = summarize(consensus, payload["assignment_seed"])
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
    output_path = (
        args.output or input_dir / "gpt_paired_gap_scores.json"
    ).resolve()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is empty. Add it to the repository's .env file before running."
        )
    api_url = os.getenv(
        "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
    )
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
            "evaluation_type": "two_pass_blind_reference_based_paired_gap_0_to_3",
            "judge_model": judge_model,
            "assignment_seed": args.seed,
            "gap_scale": {
                "0": "no meaningful difference",
                "1": "slight difference",
                "2": "meaningful difference",
                "3": "large difference",
            },
            "source_files": {
                quality: str(path) for quality, path in sample_paths.items()
            },
            "results": [],
            "swapped_results": [],
            "consensus_results": [],
            "summary": {},
        }

    selected_ids = high_ids[: args.limit] if args.limit is not None else high_ids
    failures: list[dict] = []
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

            try:
                review, raw_review, response_model = call_judge(
                    api_url=api_url,
                    api_key=api_key,
                    model=judge_model,
                    question=high_sample["input"],
                    reference=high_sample["target"],
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                )
            except RuntimeError as error:
                # Leave the pair unrecorded so a later run retries it, rather
                # than losing every remaining document to one bad judgement.
                failures.append({"pass": pass_name, "doc_id": doc_id, "error": str(error)})
                print(f"{pass_name} pass doc_id={doc_id}: FAILED, skipping ({error})")
                continue

            resolved_winner = winner_quality(review["winner"], candidate_a_quality)
            result = {
                "doc_id": doc_id,
                "candidate_a_quality": candidate_a_quality,
                "candidate_b_quality": candidate_b_quality,
                "outcome": review["outcome"],
                "winner_label": review["winner"],
                "winner_quality": resolved_winner,
                "gap": review["gap"],
                "reason": review["reason"],
                "raw_review": raw_review,
                "response_model": response_model,
            }
            pass_results.append(result)
            completed.add(doc_id)
            save_progress(output_path, payload)
            print(
                f"{pass_name} pass doc_id={doc_id}: winner={resolved_winner} "
                f"gap={review['gap']} reason={review['reason']}"
            )

    save_progress(output_path, payload)
    summary = payload["summary"]
    signed = summary["mean_signed_gap"]
    absolute = summary["mean_absolute_gap"]
    print("Consensus (same winner required in both A/B orders)")
    print(f"Evaluated pairs:  {summary['evaluated_pairs']}")
    print(f"Scored pairs:     {summary['scored_pairs']}")
    print(f"High better:      {summary['high_better']}")
    print(f"Low better:       {summary['low_better']}")
    print(f"Equivalent:       {summary['equivalent']}")
    print(f"Conflicts:        {summary['conflicts']}")
    print(f"Mean signed gap:  {signed:.4f}" if signed is not None else "Mean signed gap:  N/A")
    print(f"Mean absolute gap:{absolute: .4f}" if absolute is not None else "Mean absolute gap: N/A")
    print(f"95% bootstrap CI: {summary['mean_signed_gap_bootstrap_95_ci']}")
    print(f"Mean gap disagreement: {summary['mean_gap_disagreement']}")
    if failures:
        print(f"Failed judgements: {len(failures)} (rerun to retry)")
        for failure in failures:
            print(f"  {failure['pass']} pass doc_id={failure['doc_id']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
