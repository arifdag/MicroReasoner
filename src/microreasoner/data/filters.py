from __future__ import annotations

from collections import Counter

from microreasoner.data.normalize import example_fingerprint
from microreasoner.data.types import CanonicalExample, RejectedExample
from microreasoner.runtime.models import DataFilterConfig


def _token_count(text: str | None) -> int:
    if text is None or text.strip() == "":
        return 0
    return len(text.split())


def apply_filters(
    examples: list[CanonicalExample],
    config: DataFilterConfig,
) -> tuple[list[CanonicalExample], list[RejectedExample], dict[str, int]]:
    if config.max_think_tokens < config.min_think_tokens:
        raise ValueError("max_think_tokens must be >= min_think_tokens")

    accepted: list[CanonicalExample] = []
    rejected: list[RejectedExample] = []
    stats: Counter[str] = Counter()
    seen_fingerprints: set[str] = set()

    for example in examples:
        reason: str | None = None
        detail: str | None = None

        if example.question.strip() == "":
            reason = "missing_question"
        elif example.think is None or example.think.strip() == "":
            reason = "missing_think"
        elif config.require_single_boxed_answer and (
            example.answer_boxed is None or example.answer_boxed.strip() == ""
        ):
            reason = "missing_boxed_answer"
        else:
            think_tokens = _token_count(example.think)
            if think_tokens < config.min_think_tokens:
                reason = "below_min_think_tokens"
                detail = str(think_tokens)
            elif think_tokens > config.max_think_tokens:
                reason = "above_max_think_tokens"
                detail = str(think_tokens)

        if reason is None and config.drop_duplicates:
            fingerprint = example_fingerprint(example)
            if fingerprint in seen_fingerprints:
                reason = "duplicate"
            else:
                seen_fingerprints.add(fingerprint)

        if reason is not None:
            rejected.append(
                RejectedExample(
                    example_id=example.example_id,
                    source_name=example.source_name,
                    benchmark=example.benchmark,
                    reason=reason,
                    detail=detail,
                    question=example.question,
                )
            )
            stats[reason] += 1
            continue

        accepted.append(example)
        stats["accepted"] += 1

    stats["rejected_total"] = len(rejected)
    stats["accepted_total"] = len(accepted)
    return accepted, rejected, dict(stats)

