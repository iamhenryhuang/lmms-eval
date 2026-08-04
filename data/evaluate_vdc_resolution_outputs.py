#!/usr/bin/env python3
"""Evaluate saved VDC High/Low captions with the official two-stage VDCScore idea.

The script reads lmms-eval ``--predict_only`` JSONL files, answers VDC QA pairs
from each generated caption, and asks the local Llama-3.1-8B SGLang server to
compare each predicted answer with the reference answer.

Results are saved after every video/quality combination, so an interrupted run
can be resumed by executing the same command again.
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "VDC" / "paired" / "annotations_all.jsonl"
DEFAULT_ENDPOINT = "http://127.0.0.1:30000"
DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CATEGORIES = ("short", "background", "main_object", "camera", "detailed")


ANSWER_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for providing accurate answers "
    "to questions related to the content based on a detailed description of a "
    "video or image. Here's how you can accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Read the detailed description carefully.\n"
    "- Answer the question only based on the detailed description.\n"
    "- The answer should be a short sentence or phrase.\n"
)


SCORE_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness "
    "of generative outputs for question-answer pairs. Your task is to compare "
    "the predicted answer with the correct answer and determine if they match "
    "meaningfully. Here's how you can accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on the meaningful match between the predicted answer and the "
    "correct answer.\n"
    "- Consider synonyms or paraphrases as valid matches.\n"
    "- Evaluate the correctness of the prediction compared to the answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved VDC High/Low descriptions through a local SGLang "
            "Llama-3.1-8B judge."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Experiment directory containing High and Low sample JSONL files.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f"Paired VDC annotation JSONL (default: {DEFAULT_ANNOTATIONS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON (default: <input_dir>/vdc_eval_<categories>.json).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=(*CATEGORIES, "all"),
        default=["detailed"],
        help="QA categories to evaluate (default: detailed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N paired videos; useful for a smoke test.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N paired videos (default: 0).",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"SGLang server URL (default: {DEFAULT_ENDPOINT}).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Tokenizer/model ID used by the judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge generation temperature (default: 0 for reproducibility).",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for answering each QA question (default: 256).",
    )
    parser.add_argument(
        "--score-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for each yes/no and score judgment (default: 256).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout per generation request in seconds (default: 180).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum attempts for a failed request or malformed score (default: 5).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds between retries (default: 2).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.retries <= 0:
        parser.error("--retries must be greater than zero")
    if "all" in args.categories:
        if len(args.categories) != 1:
            parser.error("Use --categories all by itself")
        args.categories = list(CATEGORIES)
    else:
        # Preserve the user's order while rejecting accidental duplicates.
        args.categories = list(dict.fromkeys(args.categories))

    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
    return rows


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0].resolve()


def response_text(sample: dict[str, Any]) -> str:
    response = sample.get("filtered_resps")
    while isinstance(response, list) and len(response) == 1:
        response = response[0]
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"doc_id={sample.get('doc_id')} has an empty response")
    return response.strip()


def index_samples(
    samples: list[dict[str, Any]], quality: str
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    ordered_ids: list[int] = []
    indexed: dict[int, dict[str, Any]] = {}
    for sample in samples:
        doc_id = sample.get("doc_id")
        if not isinstance(doc_id, int):
            raise TypeError(f"{quality} sample has non-integer doc_id={doc_id!r}")
        if doc_id in indexed:
            raise RuntimeError(f"Duplicate doc_id={doc_id} in {quality} samples")
        ordered_ids.append(doc_id)
        indexed[doc_id] = sample
    return ordered_ids, indexed


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def make_answer_messages(caption: str, question: str) -> list[dict[str, str]]:
    user_prompt = (
        "Please provide accurate answers to questions related to the content "
        "based on a detailed description of a video or image:\n\n"
        f"detailed description: {caption}, question: {question}\n"
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide "
        "short but accurate answer."
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def make_score_messages(
    question: str, reference_answer: str, predicted_answer: str
) -> list[dict[str, str]]:
    user_prompt = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {reference_answer}\n"
        f"Predicted Answer: {predicted_answer}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is "
        "an integer value between 0 and 5, with 5 indicating the highest "
        "meaningful match. Please generate the response in the form of a Python "
        "dictionary string with keys 'pred' and 'score', where the value of "
        "'pred' is a string of 'yes' or 'no' and the value of 'score' is a "
        "number. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. For "
        "example: {'pred': 'yes', 'score': 5}."
    )
    return [
        {"role": "system", "content": SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class SGLangJudge:
    def __init__(
        self,
        *,
        endpoint: str,
        tokenizer: Any,
        temperature: float,
        timeout: float,
        retries: int,
        retry_delay: float,
    ) -> None:
        self.generate_url = endpoint.rstrip("/") + "/generate"
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        prompt = self.format_messages(messages)
        errors: list[str] = []
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.post(
                    self.generate_url,
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": self.temperature,
                            "max_new_tokens": max_new_tokens,
                        },
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                text = body.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Empty generation response: {body!r}")
                return text.strip()
            except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
                errors.append(f"attempt {attempt}: {error}")
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
        raise RuntimeError("SGLang generation failed: " + " | ".join(errors))


def parse_score_response(text: str) -> tuple[str, float]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"No dictionary found in judge response: {text!r}")

    try:
        parsed = ast.literal_eval(cleaned[start : end + 1])
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid score dictionary: {text!r}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"Score response is not a dictionary: {parsed!r}")

    pred = str(parsed.get("pred", "")).strip().lower()
    if pred not in {"yes", "no"}:
        raise ValueError(f"Invalid pred value: {pred!r}")
    try:
        score = float(parsed["score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid score value: {parsed.get('score')!r}") from error
    if not 0.0 <= score <= 5.0:
        raise ValueError(f"Score is outside [0, 5]: {score}")
    return pred, score


def judge_score(
    judge: SGLangJudge,
    *,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    max_new_tokens: int,
) -> tuple[str, float, str]:
    errors: list[str] = []
    for attempt in range(1, judge.retries + 1):
        raw = judge.generate(
            make_score_messages(question, reference_answer, predicted_answer),
            max_new_tokens=max_new_tokens,
        )
        try:
            pred, score = parse_score_response(raw)
            return pred, score, raw
        except ValueError as error:
            errors.append(f"attempt {attempt}: {error}")
            if attempt < judge.retries:
                time.sleep(judge.retry_delay)
    raise RuntimeError("Judge returned malformed scores: " + " | ".join(errors))


def evaluate_category(
    judge: SGLangJudge,
    *,
    caption: str,
    qa_pairs: list[dict[str, Any]],
    answer_max_tokens: int,
    score_max_tokens: int,
) -> dict[str, Any]:
    qa_results: list[dict[str, Any]] = []
    for qa_index, qa in enumerate(qa_pairs):
        question = qa.get("question")
        reference_answer = qa.get("answer")
        if not isinstance(question, str) or not isinstance(reference_answer, str):
            raise ValueError(f"Invalid QA pair at index {qa_index}: {qa!r}")

        predicted_answer = judge.generate(
            make_answer_messages(caption, question),
            max_new_tokens=answer_max_tokens,
        )
        pred, score, raw_score_response = judge_score(
            judge,
            question=question,
            reference_answer=reference_answer,
            predicted_answer=predicted_answer,
            max_new_tokens=score_max_tokens,
        )
        qa_results.append(
            {
                "qa_index": qa_index,
                "question": question,
                "reference_answer": reference_answer,
                "predicted_answer": predicted_answer,
                "pred": pred,
                "score": score,
                "raw_score_response": raw_score_response,
            }
        )

    if not qa_results:
        raise ValueError("QA category is empty")
    return {
        "qa_count": len(qa_results),
        "accuracy": sum(item["pred"] == "yes" for item in qa_results)
        / len(qa_results),
        "score": sum(item["score"] for item in qa_results) / len(qa_results),
        "qa_results": qa_results,
    }


def build_summary(
    records: dict[str, dict[str, dict[str, Any]]], categories: list[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {"categories": {}}
    for category in categories:
        category_summary: dict[str, Any] = {}
        for quality in ("high", "low"):
            category_results = [
                record["categories"][category]
                for record in records[quality].values()
                if category in record.get("categories", {})
            ]
            if category_results:
                category_summary[quality] = {
                    "videos": len(category_results),
                    "qa_count": sum(item["qa_count"] for item in category_results),
                    # Match official VDC aggregation: average the per-video values.
                    "accuracy": sum(item["accuracy"] for item in category_results)
                    / len(category_results),
                    "score": sum(item["score"] for item in category_results)
                    / len(category_results),
                }

        if "high" in category_summary and "low" in category_summary:
            category_summary["high_minus_low"] = {
                "accuracy": (
                    category_summary["high"]["accuracy"]
                    - category_summary["low"]["accuracy"]
                ),
                "score": (
                    category_summary["high"]["score"]
                    - category_summary["low"]["score"]
                ),
            }
        summary["categories"][category] = category_summary
    return summary


def load_or_initialize_output(
    output_path: Path,
    *,
    input_dir: Path,
    annotations_path: Path,
    sample_paths: dict[str, Path],
    categories: list[str],
    endpoint: str,
    judge_model: str,
    temperature: float,
) -> dict[str, Any]:
    expected_metadata = {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "annotations": str(annotations_path),
        "source_files": {key: str(value) for key, value in sample_paths.items()},
        "categories": categories,
        "judge_model": judge_model,
        "endpoint": endpoint.rstrip("/"),
        "temperature": temperature,
        "method": "VDC two-stage QA evaluation",
    }
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("metadata") != expected_metadata:
            raise RuntimeError(
                "Existing output metadata does not match this run. Choose another "
                "--output path or use the original settings."
            )
        if not isinstance(payload.get("records"), dict):
            raise RuntimeError("Existing output has no valid records object")
        payload["records"].setdefault("high", {})
        payload["records"].setdefault("low", {})
        return payload

    return {
        "metadata": expected_metadata,
        "records": {"high": {}, "low": {}},
        "summary": {"categories": {}},
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    annotations_path = args.annotations.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not annotations_path.is_file():
        raise SystemExit(f"Annotations file does not exist: {annotations_path}")

    sample_paths = {
        "high": find_one(input_dir, "*_samples_vdc_resolution_high.jsonl"),
        "low": find_one(input_dir, "*_samples_vdc_resolution_low.jsonl"),
    }
    samples_raw = {
        quality: read_jsonl(path) for quality, path in sample_paths.items()
    }
    high_ids, high_samples = index_samples(samples_raw["high"], "high")
    low_ids, low_samples = index_samples(samples_raw["low"], "low")
    if high_ids != low_ids:
        raise RuntimeError("High and Low sample doc_id order does not match")

    annotations = read_jsonl(annotations_path)
    for doc_id in high_ids:
        if not 0 <= doc_id < len(annotations):
            raise IndexError(
                f"doc_id={doc_id} is outside annotations range 0..{len(annotations) - 1}"
            )

    selected_ids = high_ids[args.offset :]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]
    if not selected_ids:
        raise SystemExit("No paired samples selected")

    category_label = "_".join(args.categories)
    output_path = (
        args.output or input_dir / f"vdc_eval_{category_label}.json"
    ).resolve()
    payload = load_or_initialize_output(
        output_path,
        input_dir=input_dir,
        annotations_path=annotations_path,
        sample_paths=sample_paths,
        categories=args.categories,
        endpoint=args.endpoint,
        judge_model=args.judge_model,
        temperature=args.temperature,
    )

    print(f"Loading judge tokenizer: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    judge = SGLangJudge(
        endpoint=args.endpoint,
        tokenizer=tokenizer,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )

    # Fail early with a clear connection error before starting a long evaluation.
    readiness = judge.generate(
        [{"role": "user", "content": "Reply with only OK."}],
        max_new_tokens=8,
    )
    print(f"Judge connection OK: {readiness!r}")

    total_jobs = len(selected_ids) * 2
    job_index = 0
    for doc_id in selected_ids:
        annotation = annotations[doc_id]
        qa_by_category = annotation.get("qa")
        if not isinstance(qa_by_category, dict):
            raise ValueError(f"Annotation doc_id={doc_id} has no qa object")

        for quality, indexed_samples in (
            ("high", high_samples),
            ("low", low_samples),
        ):
            job_index += 1
            record_key = str(doc_id)
            if record_key in payload["records"][quality]:
                print(
                    f"[{job_index}/{total_jobs}] {quality} doc_id={doc_id}: "
                    "already evaluated"
                )
                continue

            caption = response_text(indexed_samples[doc_id])
            print(f"[{job_index}/{total_jobs}] {quality} doc_id={doc_id}: evaluating")
            category_results: dict[str, Any] = {}
            for category in args.categories:
                qa_pairs = qa_by_category.get(category)
                if not isinstance(qa_pairs, list):
                    raise ValueError(
                        f"Annotation doc_id={doc_id} has no QA list for {category}"
                    )
                print(f"  {category}: {len(qa_pairs)} QA pairs")
                category_results[category] = evaluate_category(
                    judge,
                    caption=caption,
                    qa_pairs=qa_pairs,
                    answer_max_tokens=args.answer_max_tokens,
                    score_max_tokens=args.score_max_tokens,
                )
                print(
                    f"  {category}: accuracy="
                    f"{category_results[category]['accuracy']:.3f}, score="
                    f"{category_results[category]['score']:.3f}"
                )

            payload["records"][quality][record_key] = {
                "doc_id": doc_id,
                "video_id": annotation.get("video_id"),
                "video_name": annotation.get("video_name"),
                "categories": category_results,
            }
            payload["summary"] = build_summary(
                payload["records"], args.categories
            )
            save_progress(output_path, payload)

    payload["summary"] = build_summary(payload["records"], args.categories)
    save_progress(output_path, payload)

    print("\nSummary")
    for category in args.categories:
        result = payload["summary"]["categories"][category]
        print(f"{category}:")
        for quality in ("high", "low"):
            if quality in result:
                print(
                    f"  {quality}: videos={result[quality]['videos']}, "
                    f"accuracy={result[quality]['accuracy']:.4f}, "
                    f"score={result[quality]['score']:.4f}"
                )
        if "high_minus_low" in result:
            print(
                "  high-low: accuracy="
                f"{result['high_minus_low']['accuracy']:.4f}, score="
                f"{result['high_minus_low']['score']:.4f}"
            )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
