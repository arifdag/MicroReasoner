from __future__ import annotations

import hashlib
from typing import Iterable

from microreasoner.data.types import CanonicalExample


def normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    compact = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    compact = "\n".join(line.rstrip() for line in compact.split("\n"))
    return compact


def normalize_example(example: CanonicalExample) -> CanonicalExample:
    return CanonicalExample(
        example_id=example.example_id.strip(),
        source_name=example.source_name.strip(),
        benchmark=example.benchmark.strip().lower(),
        question=normalize_text(example.question) or "",
        think=normalize_text(example.think),
        answer_boxed=normalize_text(example.answer_boxed),
        raw_answer=normalize_text(example.raw_answer),
        metadata=dict(example.metadata),
    )


def normalize_examples(examples: Iterable[CanonicalExample]) -> list[CanonicalExample]:
    out = [normalize_example(item) for item in examples]
    out.sort(key=lambda item: item.example_id)
    return out


def example_fingerprint(example: CanonicalExample) -> str:
    key = "||".join(
        [
            example.benchmark.lower(),
            example.question.strip().lower(),
            (example.answer_boxed or "").strip().lower(),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

