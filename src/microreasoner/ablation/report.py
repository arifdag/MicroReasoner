from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any

from microreasoner.ablation.types import AblationRow, ExperimentOutcome


def _path_text(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path)


def _to_float(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(value)


def build_rows(
    outcomes: list[ExperimentOutcome],
    *,
    backend_mode: str,
    sft_baseline_id: str,
    sft_run_dir: Path | None,
) -> list[AblationRow]:
    baseline = next((item for item in outcomes if item.experiment_id == sft_baseline_id), None)
    baseline_metrics = baseline.metrics if baseline is not None else None
    baseline_greedy = baseline_metrics.greedy_pass_at_1 if baseline_metrics else 0.0
    baseline_sampled = baseline_metrics.sampled_pass_at_1 if baseline_metrics else 0.0
    baseline_schema = baseline_metrics.schema_compliance_rate if baseline_metrics else 0.0
    baseline_parser = baseline_metrics.parser_failure_rate if baseline_metrics else 0.0

    rows: list[AblationRow] = []
    for outcome in outcomes:
        metrics = outcome.metrics
        greedy = _to_float(metrics.greedy_pass_at_1 if metrics else None)
        sampled = _to_float(metrics.sampled_pass_at_1 if metrics else None)
        schema = _to_float(metrics.schema_compliance_rate if metrics else None)
        parser = _to_float(metrics.parser_failure_rate if metrics else None)
        eval_examples = int(metrics.eval_examples if metrics else 0)

        rows.append(
            AblationRow(
                experiment_id=outcome.experiment_id,
                family=outcome.family,
                backend_mode=backend_mode,
                status=outcome.status,
                sft_run_dir=_path_text(sft_run_dir),
                train_run_dir=_path_text(outcome.artifacts.train_run_dir),
                eval_run_dir=_path_text(outcome.artifacts.eval_run_dir),
                greedy_pass_at_1=greedy,
                sampled_pass_at_1=sampled,
                schema_compliance_rate=schema,
                parser_failure_rate=parser,
                delta_greedy_vs_sft=greedy - baseline_greedy,
                delta_sampled_vs_sft=sampled - baseline_sampled,
                delta_schema_vs_sft=schema - baseline_schema,
                delta_parser_vs_sft=parser - baseline_parser,
                wallclock_seconds=outcome.cost.wallclock_seconds,
                train_steps=outcome.cost.train_steps,
                eval_examples=eval_examples,
                notes=outcome.notes,
            )
        )
    return rows


def write_csv(rows: list[AblationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "family",
        "backend_mode",
        "status",
        "sft_run_dir",
        "train_run_dir",
        "eval_run_dir",
        "greedy_pass_at_1",
        "sampled_pass_at_1",
        "schema_compliance_rate",
        "parser_failure_rate",
        "delta_greedy_vs_sft",
        "delta_sampled_vs_sft",
        "delta_schema_vs_sft",
        "delta_parser_vs_sft",
        "wallclock_seconds",
        "train_steps",
        "eval_examples",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _render_table(rows: list[AblationRow]) -> str:
    header = (
        "| experiment_id | family | status | greedy | sampled | d_greedy | "
        "d_sampled | wallclock_s | train_steps |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| "
            f"{row.experiment_id} | {row.family} | {row.status} | "
            f"{row.greedy_pass_at_1:.4f} | {row.sampled_pass_at_1:.4f} | "
            f"{row.delta_greedy_vs_sft:+.4f} | {row.delta_sampled_vs_sft:+.4f} | "
            f"{row.wallclock_seconds:.1f} | {row.train_steps} |"
        )
    return "\n".join(lines)


def _best_rows(rows: list[AblationRow]) -> list[AblationRow]:
    ok = [item for item in rows if item.status == "success"]
    return sorted(
        ok,
        key=lambda item: (item.delta_sampled_vs_sft, item.delta_greedy_vs_sft, -item.wallclock_seconds),
        reverse=True,
    )


def write_markdown(
    rows: list[AblationRow],
    path: Path,
    *,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    best_ranked = _best_rows(rows)
    best_line = "No successful experiment rows."
    if best_ranked:
        best = best_ranked[0]
        best_line = (
            f"Best row: `{best.experiment_id}` "
            f"(d_sampled={best.delta_sampled_vs_sft:+.4f}, "
            f"d_greedy={best.delta_greedy_vs_sft:+.4f})."
        )

    failed = [item for item in rows if item.status != "success"]
    cost_success = [item.wallclock_seconds for item in rows if item.status == "success"]
    avg_cost = mean(cost_success) if cost_success else 0.0

    meta_lines = [f"- `{key}`: `{value}`" for key, value in metadata.items()]
    failed_lines = ["- None"] if not failed else [f"- `{item.experiment_id}`: {item.notes}" for item in failed]

    content = "\n".join(
        [
            "# Ablation Summary",
            "",
            "## Run Metadata",
            *meta_lines,
            "",
            "## Outcome",
            best_line,
            f"Average successful wallclock seconds: `{avg_cost:.1f}`",
            "",
            "## Result Table",
            _render_table(rows),
            "",
            "## Failures",
            *failed_lines,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")

