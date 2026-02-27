from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


BUCKET_PARSE_FAILURE = "parse_failure"
BUCKET_SCHEMA_FAILURE = "schema_failure"
BUCKET_MISSING_BOXED = "empty_or_missing_boxed"
BUCKET_WRONG_PARSED = "wrong_answer_parsed"
BUCKET_OTHER = "other"


_BOXED_REASONS = {"missing_boxed", "multiple_boxed", "empty_boxed", "malformed_boxed"}


def classify_incorrect_sample(row: dict[str, Any]) -> str:
    parse_ok = bool(row.get("parse_ok"))
    schema_ok = bool(row.get("schema_ok"))
    parse_reason = row.get("parse_reason")
    if isinstance(parse_reason, str) and parse_reason in _BOXED_REASONS:
        return BUCKET_MISSING_BOXED
    if not parse_ok:
        return BUCKET_PARSE_FAILURE
    if not schema_ok:
        return BUCKET_SCHEMA_FAILURE
    if parse_ok and schema_ok:
        return BUCKET_WRONG_PARSED
    return BUCKET_OTHER


def analyze_predictions(
    rows: list[dict[str, Any]],
    *,
    example_cap_per_bucket: int = 20,
) -> dict[str, Any]:
    sampled_incorrect = [
        row for row in rows if row.get("mode") == "sampled" and row.get("verified_correct") is False
    ]
    total = len(sampled_incorrect)
    bucket_counts: Counter[str] = Counter()
    by_benchmark: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()
    bucket_examples: dict[str, list[str]] = defaultdict(list)

    for row in sampled_incorrect:
        bucket = classify_incorrect_sample(row)
        bucket_counts[bucket] += 1
        benchmark = row.get("benchmark")
        if isinstance(benchmark, str):
            by_benchmark[bucket][benchmark] += 1
        reason = row.get("parse_reason")
        if isinstance(reason, str) and reason.strip() != "":
            reason_counts[reason] += 1

        example_id = row.get("example_id")
        if (
            isinstance(example_id, str)
            and len(bucket_examples[bucket]) < example_cap_per_bucket
            and example_id not in bucket_examples[bucket]
        ):
            bucket_examples[bucket].append(example_id)

    buckets: dict[str, dict[str, Any]] = {}
    ordered = [
        BUCKET_PARSE_FAILURE,
        BUCKET_SCHEMA_FAILURE,
        BUCKET_MISSING_BOXED,
        BUCKET_WRONG_PARSED,
        BUCKET_OTHER,
    ]
    for bucket in ordered:
        count = int(bucket_counts.get(bucket, 0))
        buckets[bucket] = {
            "count": count,
            "rate": (float(count) / float(total)) if total > 0 else 0.0,
            "by_benchmark": dict(sorted(by_benchmark.get(bucket, Counter()).items())),
            "example_ids": bucket_examples.get(bucket, []),
        }

    return {
        "total_sampled_incorrect": total,
        "bucket_order": ordered,
        "buckets": buckets,
        "top_parse_reasons": [
            {"reason": reason, "count": count} for reason, count in reason_counts.most_common(10)
        ],
    }


def build_error_analysis_markdown(
    by_model: dict[str, dict[str, Any]],
    *,
    comparison_order: tuple[str, ...] = ("base", "sft", "grpo"),
) -> str:
    lines: list[str] = [
        "# Error Analysis",
        "",
        "## Taxonomy",
        "- `parse_failure`: parser could not extract required structure.",
        "- `schema_failure`: structure parsed but violated schema constraints.",
        "- `empty_or_missing_boxed`: boxed answer extraction failed or empty.",
        "- `wrong_answer_parsed`: output parsed and schema-valid but incorrect.",
        "- `other`: uncategorized sampled incorrect outputs.",
        "",
    ]

    for model_id in comparison_order:
        analysis = by_model.get(model_id)
        if analysis is None:
            lines.extend([f"## {model_id}", "- No analysis available.", ""])
            continue

        total = int(analysis.get("total_sampled_incorrect", 0))
        lines.extend([f"## {model_id}", f"- sampled incorrect total: `{total}`", ""])
        lines.append("| bucket | count | rate |")
        lines.append("|---|---:|---:|")
        buckets = analysis.get("buckets", {})
        order = analysis.get("bucket_order", [])
        for bucket in order:
            data = buckets.get(bucket, {})
            lines.append(
                f"| {bucket} | {int(data.get('count', 0))} | {float(data.get('rate', 0.0)):.4f} |"
            )
        lines.append("")

        top_reasons = analysis.get("top_parse_reasons", [])
        lines.append("Top parse reasons:")
        if not top_reasons:
            lines.append("- None")
        else:
            for item in top_reasons:
                lines.append(f"- `{item['reason']}`: {item['count']}")
        lines.append("")

    lines.append("## Cross-Model Movement")
    for left, right in (("base", "sft"), ("sft", "grpo"), ("base", "grpo")):
        a = by_model.get(left)
        b = by_model.get(right)
        if a is None or b is None:
            lines.append(f"- `{right} - {left}`: unavailable")
            continue
        diffs: list[str] = []
        order = a.get("bucket_order", [])
        for bucket in order:
            count_a = int(a.get("buckets", {}).get(bucket, {}).get("count", 0))
            count_b = int(b.get("buckets", {}).get(bucket, {}).get("count", 0))
            diffs.append(f"{bucket}={count_b - count_a:+d}")
        lines.append(f"- `{right} - {left}`: " + ", ".join(diffs))
    lines.append("")
    return "\n".join(lines)

