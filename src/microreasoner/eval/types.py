from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BenchmarkName = Literal["gsm8k", "math"]
EvalMode = Literal["greedy", "sampled"]


@dataclass(frozen=True)
class EvalExample:
    example_id: str
    benchmark: BenchmarkName
    question: str
    gold_answer: str
    mock_greedy_response: str | None = None
    mock_sampled_responses: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ParseResult:
    think_text: str | None
    answer_text: str | None
    boxed_answer: str | None
    parse_ok: bool
    schema_ok: bool
    reason: str | None


@dataclass(frozen=True)
class EvalPrediction:
    example_id: str
    benchmark: BenchmarkName
    mode: EvalMode
    sample_index: int
    prompt: str
    raw_text: str
    parsed_answer: str | None
    parse_ok: bool
    schema_ok: bool
    verified_correct: bool
    parse_reason: str | None
    think_token_count: int

