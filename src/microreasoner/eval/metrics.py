from __future__ import annotations

from collections import defaultdict
from math import ceil

from microreasoner.eval.types import EvalPrediction


def _percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = max(0, ceil(q * len(sorted_values)) - 1)
    position = min(position, len(sorted_values) - 1)
    return float(sorted_values[position])


def _aggregate_mode_counts(predictions: list[EvalPrediction]) -> dict[str, dict[str, list[EvalPrediction]]]:
    grouped: dict[str, dict[str, list[EvalPrediction]]] = defaultdict(lambda: defaultdict(list))
    for pred in predictions:
        grouped[pred.benchmark][pred.mode].append(pred)
    return grouped


def _compute_pass_at_1(predictions: list[EvalPrediction], mode: str) -> float:
    by_example: dict[str, list[EvalPrediction]] = defaultdict(list)
    for pred in predictions:
        if pred.mode == mode:
            by_example[pred.example_id].append(pred)
    if not by_example:
        return 0.0

    correct = 0
    for _, preds in by_example.items():
        if mode == "greedy":
            if any(item.verified_correct for item in preds[:1]):
                correct += 1
        else:
            if any(item.verified_correct for item in preds):
                correct += 1
    return correct / len(by_example)


def build_metrics(predictions: list[EvalPrediction]) -> dict:
    grouped = _aggregate_mode_counts(predictions)

    accuracy: dict[str, dict[str, float]] = {}
    schema_total = 0
    parse_fail_total = 0
    prediction_count = len(predictions)
    think_lengths: list[int] = []

    for benchmark, mode_group in grouped.items():
        benchmark_preds = mode_group.get("greedy", []) + mode_group.get("sampled", [])
        greedy_score = _compute_pass_at_1(benchmark_preds, "greedy")
        sampled_score = _compute_pass_at_1(benchmark_preds, "sampled")
        accuracy[benchmark] = {
            "greedy_pass_at_1": greedy_score,
            "sampled_pass_at_1": sampled_score,
        }

        schema_total += sum(1 for pred in benchmark_preds if pred.schema_ok)
        parse_fail_total += sum(1 for pred in benchmark_preds if not pred.parse_ok)
        think_lengths.extend(pred.think_token_count for pred in benchmark_preds)

    compliance_rate = (schema_total / prediction_count) if prediction_count else 0.0
    extraction_failure_rate = (parse_fail_total / prediction_count) if prediction_count else 0.0
    sorted_lengths = sorted(think_lengths)
    mean_length = (sum(sorted_lengths) / len(sorted_lengths)) if sorted_lengths else 0.0

    return {
        "accuracy": accuracy,
        "schema": {"compliance_rate": compliance_rate},
        "parser": {"extraction_failure_rate": extraction_failure_rate},
        "length": {
            "think_tokens": {
                "mean": mean_length,
                "p95": _percentile(sorted_lengths, 0.95),
            }
        },
    }

