from __future__ import annotations

import json
from pathlib import Path

from microreasoner.eval.types import BenchmarkName, EvalExample


class EvalDataError(ValueError):
    """Raised when evaluation dataset loading fails."""


def _require_field(row: dict, field: str, path: Path, line_no: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or value.strip() == "":
        raise EvalDataError(f"{path}:{line_no} missing non-empty string field '{field}'")
    return value


def _parse_row(path: Path, line_no: int, line: str, benchmark: BenchmarkName) -> EvalExample:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise EvalDataError(f"{path}:{line_no} invalid JSON: {exc}") from exc

    if not isinstance(row, dict):
        raise EvalDataError(f"{path}:{line_no} expected JSON object")

    example_id = _require_field(row, "id", path, line_no)
    question = _require_field(row, "question", path, line_no)
    gold_answer = _require_field(row, "answer", path, line_no)

    mock_greedy = row.get("mock_greedy_response")
    if mock_greedy is not None and not isinstance(mock_greedy, str):
        raise EvalDataError(f"{path}:{line_no} field 'mock_greedy_response' must be string")

    sampled = row.get("mock_sampled_responses")
    mock_sampled: tuple[str, ...] | None = None
    if sampled is not None:
        if not isinstance(sampled, list) or not all(isinstance(item, str) for item in sampled):
            raise EvalDataError(
                f"{path}:{line_no} field 'mock_sampled_responses' must be list[str]"
            )
        mock_sampled = tuple(sampled)

    return EvalExample(
        example_id=example_id,
        benchmark=benchmark,
        question=question,
        gold_answer=gold_answer,
        mock_greedy_response=mock_greedy,
        mock_sampled_responses=mock_sampled,
    )


def load_jsonl_examples(path: Path, benchmark: BenchmarkName) -> list[EvalExample]:
    if not path.exists():
        raise EvalDataError(f"Dataset file not found: {path}")

    examples: list[EvalExample] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            example = _parse_row(path, line_no, line, benchmark)
            if example.example_id in seen_ids:
                raise EvalDataError(f"{path}:{line_no} duplicate id '{example.example_id}'")
            seen_ids.add(example.example_id)
            examples.append(example)

    examples.sort(key=lambda item: item.example_id)
    return examples

