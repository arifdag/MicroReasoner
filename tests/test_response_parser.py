from __future__ import annotations

from microreasoner.eval.parser import parse_response


def test_parse_response_rejects_trailing_text_after_answer() -> None:
    text = "<think>reason</think><answer>\\boxed{2}</answer>junk"
    parsed = parse_response(text, strict_boxed_only=True)
    assert not parsed.parse_ok
    assert not parsed.schema_ok
    assert parsed.reason == "extra_text_outside_tags"


def test_parse_response_allows_whitespace_outside_tags() -> None:
    text = " \n<think>reason</think>\n<answer>\\boxed{2}</answer>\n"
    parsed = parse_response(text, strict_boxed_only=True)
    assert parsed.parse_ok
    assert parsed.schema_ok
    assert parsed.boxed_answer == "2"
