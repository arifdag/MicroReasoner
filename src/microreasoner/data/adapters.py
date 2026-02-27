from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microreasoner.data.types import CanonicalExample
from microreasoner.eval.parser import parse_response
from microreasoner.runtime.models import DataSourceConfig


class DataAdapterError(ValueError):
    """Raised when raw data cannot be adapted into canonical records."""


def _require_str(row: dict[str, Any], field: str, file_path: Path, line_no: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or value.strip() == "":
        raise DataAdapterError(f"{file_path}:{line_no} missing non-empty string field '{field}'")
    return value


def _normalize_optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() != "":
        return value.strip()
    return None


def _parse_canonical_row(
    *,
    row: dict[str, Any],
    source: DataSourceConfig,
    file_path: Path,
    line_no: int,
) -> CanonicalExample:
    example_id = _normalize_optional_str(row.get("example_id")) or _normalize_optional_str(
        row.get("id")
    )
    if example_id is None:
        raise DataAdapterError(f"{file_path}:{line_no} missing example id")

    question = _require_str(row, "question", file_path, line_no).strip()
    benchmark = _normalize_optional_str(row.get("benchmark")) or "other"
    source_name = _normalize_optional_str(row.get("source_name")) or source.name

    think = _normalize_optional_str(row.get("think"))
    answer_boxed = _normalize_optional_str(row.get("answer_boxed")) or _normalize_optional_str(
        row.get("answer")
    )
    raw_answer = _normalize_optional_str(row.get("raw_answer"))
    metadata_raw = row.get("metadata", {})
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    response_text = _normalize_optional_str(row.get("response"))
    if response_text is not None:
        parsed = parse_response(response_text, strict_boxed_only=True)
        if parsed.parse_ok:
            think = parsed.think_text
            answer_boxed = parsed.boxed_answer
            raw_answer = parsed.answer_text
        else:
            metadata["parse_error"] = parsed.reason
    if raw_answer is None and answer_boxed is not None:
        raw_answer = answer_boxed

    return CanonicalExample(
        example_id=example_id,
        source_name=source_name,
        benchmark=benchmark,
        question=question,
        think=think,
        answer_boxed=answer_boxed,
        raw_answer=raw_answer,
        metadata=metadata,
    )


def _load_canonical_jsonl(source: DataSourceConfig, source_dir: Path | None) -> list[CanonicalExample]:
    path = Path(source.path)
    if source_dir is not None:
        path = source_dir / path.name
    if not path.exists():
        raise DataAdapterError(f"Source file not found: {path}")

    examples: list[CanonicalExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataAdapterError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DataAdapterError(f"{path}:{line_no} expected JSON object")
            examples.append(
                _parse_canonical_row(
                    row=row,
                    source=source,
                    file_path=path,
                    line_no=line_no,
                )
            )
    return examples


def load_sources(
    sources: tuple[DataSourceConfig, ...],
    source_dir: Path | None = None,
) -> list[CanonicalExample]:
    out: list[CanonicalExample] = []
    for source in sources:
        adapter = source.adapter.lower()
        if adapter == "canonical_jsonl":
            out.extend(_load_canonical_jsonl(source, source_dir))
            continue
        raise DataAdapterError(f"Unsupported source adapter: {source.adapter}")
    return out
