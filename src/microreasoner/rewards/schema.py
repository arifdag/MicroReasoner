from __future__ import annotations

from dataclasses import dataclass

from microreasoner.eval.parser import parse_response
from microreasoner.eval.types import ParseResult


@dataclass(frozen=True)
class SchemaRewardResult:
    score: float
    parse: ParseResult


def score_schema(response_text: str, *, strict_boxed_only: bool = True) -> SchemaRewardResult:
    parsed = parse_response(response_text, strict_boxed_only=strict_boxed_only)
    return SchemaRewardResult(score=1.0 if parsed.schema_ok else 0.0, parse=parsed)
