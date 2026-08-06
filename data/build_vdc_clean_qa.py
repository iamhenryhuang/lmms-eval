#!/usr/bin/env python3
"""Build a reproducible factual subset of VDC detailed QA annotations.

The filter only inspects the provided questions and reference answers. It never
reads model captions, High/Low labels, or evaluation scores. The original
annotation file is not modified.

An optional review JSONL can override individual automatic decisions. Each line
must contain ``doc_id``, ``qa_index``, and ``decision`` (``keep`` or
``exclude``), for example::

    {"doc_id": 0, "qa_index": 13, "decision": "keep", "note": "Visible expression"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


UNKNOWN_ANSWER_RE = re.compile(
    r"\b(unknown|unspecified|not specified|not mentioned|cannot be "
    r"determined|cannot tell|unclear)\b",
    re.IGNORECASE,
)
AUDIO_RE = re.compile(
    r"\b(audio|sound|music|hear|heard|noise|voice|voices|chirp|chirping|"
    r"spoken|speaking)\b",
    re.IGNORECASE,
)
SUBJECTIVE_RE = re.compile(
    r"\b(mood|atmosphere|feeling|feelings|evoke|evokes|evoked|emotion|"
    r"emotional|tone|symbolize|symbolizes|symbolism|sense of|impression|"
    r"inviting|heartwarming|captivating)\b",
    re.IGNORECASE,
)
NONVISUAL_META_RE = re.compile(
    r"\b(creator|audience|viewer|viewers|artistic intention|intended|"
    r"intention|purpose|message|meaning|created the video|video created)\b",
    re.IGNORECASE,
)
NEGATIVE_OR_ABSENCE_QUESTION_RE = re.compile(
    r"^(who|what|which)\s+(is|are|was|were|does|do)\s+not\b|"
    r"\bnot shown in the video\b",
    re.IGNORECASE,
)
ABSENCE_ANSWER_RE = re.compile(
    r"^(none|nothing|no one|nobody|not shown|not visible|absent)[.!]?$",
    re.IGNORECASE,
)
MALFORMED_TEXT_RE = re.compile(r"(?:\['|']|\{[^}]*$|\[[^]]*$)")

FINE_DETAIL_RE = re.compile(
    r"\b(color|colour|how many|number of|wearing|wears|wore|dressed|"
    r"clothing|clothes|shirt|jacket|coat|dress|pants|trousers|shorts|"
    r"shoes|hat|scarf|text|word|words|letter|letters|number|sign|logo|"
    r"label|brand|material|texture|pattern|striped|small|tiny|distant|"
    r"foreground|background|left|right|behind|in front of|object|objects)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter VDC detailed QA into an objective visual-factual subset "
            "without reading model outputs."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--review-file",
        type=Path,
        help="Optional JSONL decisions overriding automatic keep/exclude results.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Write only the report; do not create filtered annotations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of output/report files created by an earlier run.",
    )
    args = parser.parse_args()

    if not args.scan_only and args.output is None:
        parser.error("--output is required unless --scan-only is used")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_overrides(path: Path | None) -> dict[tuple[int, int], dict[str, str]]:
    if path is None:
        return {}
    overrides: dict[tuple[int, int], dict[str, str]] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        try:
            doc_id = int(row["doc_id"])
            qa_index = int(row["qa_index"])
            decision = str(row["decision"]).strip().lower()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid override at {path}:{line_number}") from error
        if decision not in {"keep", "exclude"}:
            raise ValueError(
                f"Invalid decision at {path}:{line_number}: {decision!r}"
            )
        key = (doc_id, qa_index)
        if key in overrides:
            raise ValueError(f"Duplicate override for doc_id/qa_index={key}")
        overrides[key] = {
            "decision": decision,
            "note": str(row.get("note", "")).strip(),
        }
    return overrides


def normalize_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def classify(question: Any, answer: Any, duplicate: bool) -> list[str]:
    reasons: list[str] = []
    if not isinstance(question, str) or not question.strip():
        reasons.append("invalid_question")
    if not isinstance(answer, str) or not answer.strip():
        reasons.append("invalid_answer")
    if reasons:
        return reasons

    question = question.strip()
    answer = answer.strip()
    combined = f"{question} {answer}"

    if re.match(r"^why\b", question, re.IGNORECASE):
        reasons.append("causal_or_intent_inference")
    if UNKNOWN_ANSWER_RE.search(answer):
        reasons.append("unknown_or_unspecified_answer")
    # Restrict this rule to the question. Words such as "music" in an answer
    # can still refer to a visibly present instrument or performer.
    if AUDIO_RE.search(question):
        reasons.append("audio_dependent")
    if SUBJECTIVE_RE.search(combined):
        reasons.append("subjective_or_affective")
    if NONVISUAL_META_RE.search(combined):
        reasons.append("nonvisual_metadata_or_intent")
    if NEGATIVE_OR_ABSENCE_QUESTION_RE.search(question) or ABSENCE_ANSWER_RE.fullmatch(
        answer
    ):
        reasons.append("negative_or_absence_question")
    if MALFORMED_TEXT_RE.search(question) or MALFORMED_TEXT_RE.search(answer):
        reasons.append("malformed_text")
    if duplicate:
        reasons.append("duplicate_question")
    return list(dict.fromkeys(reasons))


def ensure_writable(path: Path, input_path: Path, force: bool) -> None:
    if path.resolve() == input_path.resolve():
        raise ValueError(f"Refusing to overwrite the input annotation file: {path}")
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists; use --force to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input does not exist: {input_path}")

    rows = read_jsonl(input_path)
    overrides = load_overrides(args.review_file)
    used_overrides: set[tuple[int, int]] = set()
    reason_counts: Counter[str] = Counter()
    excluded_records: list[dict[str, Any]] = []
    kept_total = excluded_total = fine_total = 0
    factual_rows: list[dict[str, Any]] = []

    for doc_id, original_row in enumerate(rows):
        qa = original_row.get("qa")
        if not isinstance(qa, dict) or not isinstance(qa.get("detailed"), list):
            raise ValueError(f"doc_id={doc_id} has no qa.detailed list")

        seen_questions: set[str] = set()
        factual_detailed: list[dict[str, Any]] = []

        for qa_index, item in enumerate(qa["detailed"]):
            if not isinstance(item, dict):
                question = answer = None
            else:
                question = item.get("question")
                answer = item.get("answer")

            normalized = (
                normalize_question(question)
                if isinstance(question, str) and question.strip()
                else ""
            )
            duplicate = bool(normalized and normalized in seen_questions)
            if normalized:
                seen_questions.add(normalized)

            reasons = classify(question, answer, duplicate)
            automatic_decision = "exclude" if reasons else "keep"
            override = overrides.get((doc_id, qa_index))
            if override:
                decision = override["decision"]
                used_overrides.add((doc_id, qa_index))
            else:
                decision = automatic_decision

            combined = f"{question or ''} {answer or ''}"
            detail_type = "fine" if FINE_DETAIL_RE.search(combined) else "coarse"

            if decision == "keep" and isinstance(item, dict):
                factual_detailed.append(deepcopy(item))
                kept_total += 1
                if detail_type == "fine":
                    fine_total += 1
            else:
                excluded_total += 1
                reason_counts.update(reasons or ["manual_exclusion"])
                excluded_records.append(
                    {
                        "doc_id": doc_id,
                        "video_id": original_row.get("video_id"),
                        "qa_index": qa_index,
                        "question": question,
                        "answer": answer,
                        "automatic_decision": automatic_decision,
                        "final_decision": decision,
                        "reasons": reasons,
                        "detail_type": detail_type,
                        "override_note": override["note"] if override else "",
                    }
                )

        if not factual_detailed:
            raise ValueError(f"Filtering removed every detailed QA for doc_id={doc_id}")

        factual_row = deepcopy(original_row)
        factual_row["qa"]["detailed"] = factual_detailed
        factual_rows.append(factual_row)

    unused_overrides = sorted(set(overrides) - used_overrides)
    if unused_overrides:
        raise ValueError(f"Review overrides reference nonexistent QA indices: {unused_overrides}")

    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "method": "deterministic VDC detailed factual-QA filter",
        "input": str(input_path),
        "input_sha256": input_hash,
        "review_file": str(args.review_file.resolve()) if args.review_file else None,
        "summary": {
            "videos": len(rows),
            "original_detailed_qa": kept_total + excluded_total,
            "retained_factual_qa": kept_total,
            "excluded_qa": excluded_total,
            "retained_fine_detail_qa": fine_total,
            "retained_fraction": kept_total / (kept_total + excluded_total),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        },
        "rules": {
            "excluded": [
                "why/causal or intent inference",
                "unknown or unspecified reference answer",
                "audio-dependent content",
                "subjective mood, emotion, or viewer response",
                "creator, audience, purpose, or artistic intent",
                "negative/absence questions",
                "malformed text",
                "duplicate normalized questions within a video",
            ],
            "note": (
                "Rules were applied without reading model outputs, quality labels, "
                "or evaluation scores. Manual review overrides are recorded."
            ),
        },
        "excluded": excluded_records,
    }

    report_path = args.report.resolve()
    ensure_writable(report_path, input_path, args.force)
    write_json(report_path, report)

    if not args.scan_only:
        output_path = args.output.resolve()
        ensure_writable(output_path, input_path, args.force)
        write_jsonl(output_path, factual_rows)

    summary = report["summary"]
    print(f"Videos:                 {summary['videos']}")
    print(f"Original detailed QA:   {summary['original_detailed_qa']}")
    print(f"Retained factual QA:    {summary['retained_factual_qa']}")
    print(f"Excluded QA:            {summary['excluded_qa']}")
    print(f"Retained fine-detail:   {summary['retained_fine_detail_qa']}")
    print(f"Report:                 {report_path}")
    if not args.scan_only:
        print(f"Factual annotations:    {args.output.resolve()}")


if __name__ == "__main__":
    main()
