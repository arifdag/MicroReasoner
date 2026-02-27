from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LengthBand:
    min_tokens: int
    max_tokens: int


@dataclass(frozen=True)
class LengthRewardResult:
    score: float
    think_tokens: int


def _count_tokens(text: str | None) -> int:
    if text is None:
        return 0
    stripped = text.strip()
    if stripped == "":
        return 0
    return len(stripped.split())


def score_length(think_text: str | None, band: LengthBand) -> LengthRewardResult:
    token_count = _count_tokens(think_text)
    if band.max_tokens <= 0:
        return LengthRewardResult(score=0.0, think_tokens=token_count)

    if token_count < band.min_tokens:
        denom = max(1, band.min_tokens)
        overflow = float(band.min_tokens - token_count) / float(denom)
        return LengthRewardResult(score=-min(1.0, overflow), think_tokens=token_count)

    if token_count > band.max_tokens:
        denom = max(1, band.max_tokens)
        overflow = float(token_count - band.max_tokens) / float(denom)
        return LengthRewardResult(score=-min(1.0, overflow), think_tokens=token_count)

    return LengthRewardResult(score=0.0, think_tokens=token_count)
