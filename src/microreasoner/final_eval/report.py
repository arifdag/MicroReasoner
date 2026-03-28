from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_final_metrics_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _coerce_model_entry(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _coerce_metrics_entry(model: dict[str, Any]) -> dict[str, Any]:
    metrics = model.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return {}


def _table_for_models(models: dict[str, Any]) -> list[str]:
    lines = [
        "| model | status | macro_greedy | macro_sampled | schema | parser | wallclock_s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_id in ("base", "sft", "grpo"):
        model = _coerce_model_entry(models.get(model_id))
        metrics = _coerce_metrics_entry(model)
        lines.append(
            "| "
            f"{model_id} | {model.get('status', 'missing')} | "
            f"{float(metrics.get('macro_greedy_pass_at_1', 0.0)):.4f} | "
            f"{float(metrics.get('macro_sampled_pass_at_1', 0.0)):.4f} | "
            f"{float(metrics.get('schema_compliance_rate', 0.0)):.4f} | "
            f"{float(metrics.get('parser_failure_rate', 0.0)):.4f} | "
            f"{float(model.get('wallclock_seconds', 0.0)):.1f} |"
        )
    return lines


def _table_for_comparisons(comparisons: dict[str, Any]) -> list[str]:
    lines = [
        "| comparison | d_macro_greedy | d_macro_sampled | d_schema | d_parser |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("sft_vs_base", "grpo_vs_sft", "grpo_vs_base"):
        comp = comparisons.get(key)
        if not isinstance(comp, dict):
            lines.append(f"| {key} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| "
            f"{key} | "
            f"{float(comp.get('delta_macro_greedy_pass_at_1', 0.0)):+.4f} | "
            f"{float(comp.get('delta_macro_sampled_pass_at_1', 0.0)):+.4f} | "
            f"{float(comp.get('delta_schema_compliance_rate', 0.0)):+.4f} | "
            f"{float(comp.get('delta_parser_failure_rate', 0.0)):+.4f} |"
        )
    return lines


def _table_for_cost(models: dict[str, Any]) -> list[str]:
    lines = [
        "| model | greedy_solved | sampled_solved | sec_per_greedy_solved | sec_per_sampled_solved |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in ("base", "sft", "grpo"):
        model = _coerce_model_entry(models.get(model_id))
        metrics = _coerce_metrics_entry(model)
        cost_g = metrics.get("cost_per_solved_greedy")
        cost_s = metrics.get("cost_per_solved_sampled")
        cost_g_text = "n/a" if cost_g is None else f"{float(cost_g):.4f}"
        cost_s_text = "n/a" if cost_s is None else f"{float(cost_s):.4f}"
        lines.append(
            f"| {model_id} | {int(metrics.get('greedy_solved', 0))} | "
            f"{int(metrics.get('sampled_solved', 0))} | {cost_g_text} | {cost_s_text} |"
        )
    return lines


def build_final_report_markdown(payload: dict[str, Any]) -> str:
    inputs = payload.get("inputs", {})
    models = payload.get("models", {})
    comparisons = payload.get("comparisons", {})
    failures = payload.get("failure_reasons", [])
    strict_ok = bool(payload.get("strict_claims_ok"))

    lines: list[str] = [
        "# Final Evaluation Report",
        "",
        "## Metadata",
        f"- `session_id`: `{payload.get('session_id', '')}`",
        f"- `status`: `{payload.get('status', '')}`",
        f"- `strict_claims_ok`: `{strict_ok}`",
        f"- `config`: `{inputs.get('config_path', '')}`",
        f"- `dataset_dir`: `{inputs.get('dataset_dir', '')}`",
        "",
        "## Core Metrics",
        *_table_for_models(models),
        "",
        "## Delta Comparisons",
        *_table_for_comparisons(comparisons),
        "",
        "## Cost Efficiency",
        *_table_for_cost(models),
        "",
        "## Reproducibility",
        f"- `base_checkpoint`: `{inputs.get('base_checkpoint', '')}`",
        f"- `sft_checkpoint`: `{inputs.get('sft_checkpoint', '')}`",
        f"- `grpo_checkpoint`: `{inputs.get('grpo_checkpoint', '')}`",
        f"- `seed`: `{inputs.get('seed', '')}`",
        f"- `max_items`: `{inputs.get('max_items', '')}`",
        "",
        "## Failures",
    ]
    if isinstance(failures, list) and failures:
        for item in failures:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_final_report_markdown(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_final_report_markdown(payload), encoding="utf-8")


def write_error_analysis_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

