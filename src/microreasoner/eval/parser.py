from __future__ import annotations

import re

from microreasoner.eval.types import ParseResult


THINK_PATTERN = re.compile(r"(?s)<think>(.*?)</think>")
ANSWER_PATTERN = re.compile(r"(?s)<answer>(.*?)</answer>")
FULL_RESPONSE_PATTERN = re.compile(r"(?s)^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$")


def _extract_single_tag(text: str, pattern: re.Pattern[str], missing_reason: str, multi_reason: str) -> tuple[str | None, str | None]:
    matches = pattern.findall(text)
    if len(matches) == 0:
        return None, missing_reason
    if len(matches) > 1:
        return None, multi_reason
    return matches[0], None


def _extract_boxed(answer_text: str) -> tuple[list[str], str | None]:
    boxed_values: list[str] = []
    cursor = 0
    marker = r"\boxed{"
    while True:
        idx = answer_text.find(marker, cursor)
        if idx == -1:
            break
        i = idx + len(marker)
        depth = 1
        start = i
        while i < len(answer_text) and depth > 0:
            char = answer_text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            i += 1
        if depth != 0:
            return boxed_values, "malformed_boxed"
        content = answer_text[start : i - 1].strip()
        boxed_values.append(content)
        cursor = i
    return boxed_values, None


def parse_response(text: str, strict_boxed_only: bool = True) -> ParseResult:
    think_text, think_error = _extract_single_tag(
        text, THINK_PATTERN, "missing_think_tag", "multiple_think_tags"
    )
    answer_text, answer_error = _extract_single_tag(
        text, ANSWER_PATTERN, "missing_answer_tag", "multiple_answer_tags"
    )

    if think_error:
        return ParseResult(
            think_text=None,
            answer_text=answer_text,
            boxed_answer=None,
            parse_ok=False,
            schema_ok=False,
            reason=think_error,
        )

    if answer_error:
        return ParseResult(
            think_text=think_text,
            answer_text=None,
            boxed_answer=None,
            parse_ok=False,
            schema_ok=False,
            reason=answer_error,
        )

    if FULL_RESPONSE_PATTERN.fullmatch(text) is None:
        return ParseResult(
            think_text=think_text,
            answer_text=answer_text,
            boxed_answer=None,
            parse_ok=False,
            schema_ok=False,
            reason="extra_text_outside_tags",
        )

    assert answer_text is not None  # for type checkers
    boxed_values, boxed_error = _extract_boxed(answer_text)
    if boxed_error:
        return ParseResult(
            think_text=think_text,
            answer_text=answer_text,
            boxed_answer=None,
            parse_ok=False,
            schema_ok=False,
            reason=boxed_error,
        )
    if strict_boxed_only:
        if len(boxed_values) == 0:
            return ParseResult(
                think_text=think_text,
                answer_text=answer_text,
                boxed_answer=None,
                parse_ok=False,
                schema_ok=False,
                reason="missing_boxed",
            )
        if len(boxed_values) > 1:
            return ParseResult(
                think_text=think_text,
                answer_text=answer_text,
                boxed_answer=None,
                parse_ok=False,
                schema_ok=False,
                reason="multiple_boxed",
            )

    boxed_answer = boxed_values[-1] if boxed_values else answer_text.strip()
    if boxed_answer.strip() == "":
        return ParseResult(
            think_text=think_text,
            answer_text=answer_text,
            boxed_answer=None,
            parse_ok=False,
            schema_ok=False,
            reason="empty_boxed",
        )

    return ParseResult(
        think_text=think_text.strip() if think_text is not None else None,
        answer_text=answer_text.strip(),
        boxed_answer=boxed_answer.strip(),
        parse_ok=True,
        schema_ok=True,
        reason=None,
    )

