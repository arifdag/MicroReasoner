from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CanonicalExample:
    example_id: str
    source_name: str
    benchmark: str
    question: str
    think: str | None
    answer_boxed: str | None
    raw_answer: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RejectedExample:
    example_id: str
    source_name: str
    benchmark: str
    reason: str
    detail: str | None
    question: str | None


@dataclass(frozen=True)
class SFTRecord:
    record_id: str
    prompt: str
    target_response: str
    split: str
    source_name: str
    benchmark: str
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class RLPromptRecord:
    record_id: str
    prompt: str
    gold_answer: str
    split: str
    source_name: str
    benchmark: str
    difficulty_tag: str
    curriculum_stage: str


@dataclass(frozen=True)
class BuildResult:
    dataset_type: str
    dataset_id: str
    output_dir: str
    manifest_path: str
    train_count: int
    val_count: int
    reject_count: int

