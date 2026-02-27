from __future__ import annotations

from microreasoner.final_eval.error_analysis import (
    BUCKET_MISSING_BOXED,
    BUCKET_PARSE_FAILURE,
    BUCKET_SCHEMA_FAILURE,
    BUCKET_WRONG_PARSED,
    analyze_predictions,
    classify_incorrect_sample,
)


def test_classify_incorrect_sample_buckets() -> None:
    assert (
        classify_incorrect_sample(
            {
                "mode": "sampled",
                "verified_correct": False,
                "parse_ok": False,
                "schema_ok": False,
                "parse_reason": "missing_boxed",
            }
        )
        == BUCKET_MISSING_BOXED
    )
    assert (
        classify_incorrect_sample(
            {
                "mode": "sampled",
                "verified_correct": False,
                "parse_ok": False,
                "schema_ok": False,
                "parse_reason": "missing_answer_tag",
            }
        )
        == BUCKET_PARSE_FAILURE
    )
    assert (
        classify_incorrect_sample(
            {
                "mode": "sampled",
                "verified_correct": False,
                "parse_ok": True,
                "schema_ok": False,
                "parse_reason": "schema_violation",
            }
        )
        == BUCKET_SCHEMA_FAILURE
    )
    assert (
        classify_incorrect_sample(
            {
                "mode": "sampled",
                "verified_correct": False,
                "parse_ok": True,
                "schema_ok": True,
                "parse_reason": None,
            }
        )
        == BUCKET_WRONG_PARSED
    )


def test_analyze_predictions_counts_buckets() -> None:
    rows = [
        {"mode": "sampled", "verified_correct": False, "parse_ok": False, "schema_ok": False, "parse_reason": "missing_boxed", "benchmark": "gsm8k", "example_id": "a"},
        {"mode": "sampled", "verified_correct": False, "parse_ok": False, "schema_ok": False, "parse_reason": "missing_answer_tag", "benchmark": "math", "example_id": "b"},
        {"mode": "sampled", "verified_correct": False, "parse_ok": True, "schema_ok": True, "parse_reason": None, "benchmark": "math", "example_id": "c"},
        {"mode": "sampled", "verified_correct": True, "parse_ok": True, "schema_ok": True, "parse_reason": None, "benchmark": "math", "example_id": "d"},
    ]
    analysis = analyze_predictions(rows)
    assert analysis["total_sampled_incorrect"] == 3
    assert analysis["buckets"][BUCKET_MISSING_BOXED]["count"] == 1
    assert analysis["buckets"][BUCKET_PARSE_FAILURE]["count"] == 1
    assert analysis["buckets"][BUCKET_WRONG_PARSED]["count"] == 1

