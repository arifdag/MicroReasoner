from __future__ import annotations

import json
import secrets
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microreasoner.cli.main import main as cli_main
from microreasoner.contracts.validation import validate_run_dir
from microreasoner.final_eval.error_analysis import (
    analyze_predictions,
    build_error_analysis_markdown,
)
from microreasoner.final_eval.report import (
    write_error_analysis_markdown,
    write_final_metrics_json,
    write_final_report_markdown,
)
from microreasoner.final_eval.types import FinalEvalResult, ModelMetrics, ModelOutcome
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root


MODEL_ORDER = ("base", "sft", "grpo")


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line == "":
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSON at {path}:{line_no}")
            rows.append(payload)
    return rows


def _parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _duration_seconds(summary_path: Path) -> float:
    if not summary_path.exists():
        return 0.0
    summary = _read_json(summary_path)
    started = summary.get("started_at")
    finished = summary.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return 0.0
    delta = _parse_iso(finished) - _parse_iso(started)
    return max(0.0, float(delta.total_seconds()))


def _failure_note(run_dir: Path) -> str:
    errors_path = run_dir / "errors.json"
    if not errors_path.exists():
        return "command_failed_without_errors_json"
    payload = _read_json(errors_path)
    message = payload.get("message")
    if isinstance(message, str) and message.strip() != "":
        return message
    return "command_failed"


def _run_command(
    *,
    argv: list[str],
    run_dir: Path,
    skip_existing: bool,
) -> tuple[int, bool]:
    summary_path = run_dir / "summary.json"
    if skip_existing and summary_path.exists():
        payload = _read_json(summary_path)
        if payload.get("status") == "success":
            return 0, True
    code = cli_main(argv)
    return int(code), False


def _extract_model_metrics(run_dir: Path, wallclock_seconds: float) -> ModelMetrics:
    metrics = _read_json(run_dir / "metrics.json")
    accuracy = metrics.get("accuracy")
    if not isinstance(accuracy, dict) or not accuracy:
        raise ValueError(f"Missing metrics.accuracy in {run_dir}")

    benchmark_scores: dict[str, dict[str, float]] = {}
    greedy_vals: list[float] = []
    sampled_vals: list[float] = []
    for benchmark, values in sorted(accuracy.items()):
        if not isinstance(values, dict):
            continue
        greedy = float(values.get("greedy_pass_at_1", 0.0))
        sampled = float(values.get("sampled_pass_at_1", 0.0))
        benchmark_scores[str(benchmark)] = {
            "greedy_pass_at_1": greedy,
            "sampled_pass_at_1": sampled,
        }
        greedy_vals.append(greedy)
        sampled_vals.append(sampled)
    if not greedy_vals or not sampled_vals:
        raise ValueError(f"No benchmark pass@1 values in {run_dir}")

    schema = metrics.get("schema", {})
    parser = metrics.get("parser", {})
    length = metrics.get("length", {})
    think = length.get("think_tokens", {}) if isinstance(length, dict) else {}
    schema_rate = float(schema.get("compliance_rate", 0.0)) if isinstance(schema, dict) else 0.0
    parser_rate = (
        float(parser.get("extraction_failure_rate", 0.0)) if isinstance(parser, dict) else 0.0
    )
    think_mean = float(think.get("mean", 0.0)) if isinstance(think, dict) else 0.0
    think_p95 = float(think.get("p95", 0.0)) if isinstance(think, dict) else 0.0

    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    greedy_correct = {
        row.get("example_id")
        for row in predictions
        if row.get("mode") == "greedy" and row.get("verified_correct") is True and isinstance(row.get("example_id"), str)
    }
    sampled_correct = {
        row.get("example_id")
        for row in predictions
        if row.get("mode") == "sampled" and row.get("verified_correct") is True and isinstance(row.get("example_id"), str)
    }
    all_examples = {
        row.get("example_id")
        for row in predictions
        if row.get("mode") == "greedy" and isinstance(row.get("example_id"), str)
    }
    greedy_solved = len(greedy_correct)
    sampled_solved = len(sampled_correct)

    cost_greedy = (
        (wallclock_seconds / float(greedy_solved)) if greedy_solved > 0 else None
    )
    cost_sampled = (
        (wallclock_seconds / float(sampled_solved)) if sampled_solved > 0 else None
    )

    return ModelMetrics(
        benchmarks=benchmark_scores,
        macro_greedy_pass_at_1=sum(greedy_vals) / len(greedy_vals),
        macro_sampled_pass_at_1=sum(sampled_vals) / len(sampled_vals),
        schema_compliance_rate=schema_rate,
        parser_failure_rate=parser_rate,
        think_tokens_mean=think_mean,
        think_tokens_p95=think_p95,
        eval_examples=len(all_examples),
        greedy_solved=greedy_solved,
        sampled_solved=sampled_solved,
        cost_per_solved_greedy=cost_greedy,
        cost_per_solved_sampled=cost_sampled,
    )


def _comparison(candidate: ModelOutcome, baseline: ModelOutcome) -> dict[str, Any] | None:
    if candidate.metrics is None or baseline.metrics is None:
        return None
    out: dict[str, Any] = {
        "delta_macro_greedy_pass_at_1": (
            candidate.metrics.macro_greedy_pass_at_1 - baseline.metrics.macro_greedy_pass_at_1
        ),
        "delta_macro_sampled_pass_at_1": (
            candidate.metrics.macro_sampled_pass_at_1 - baseline.metrics.macro_sampled_pass_at_1
        ),
        "delta_schema_compliance_rate": (
            candidate.metrics.schema_compliance_rate - baseline.metrics.schema_compliance_rate
        ),
        "delta_parser_failure_rate": (
            candidate.metrics.parser_failure_rate - baseline.metrics.parser_failure_rate
        ),
        "benchmark_deltas": {},
    }
    benchmarks: dict[str, dict[str, float]] = {}
    for bench in sorted(set(candidate.metrics.benchmarks) & set(baseline.metrics.benchmarks)):
        cand = candidate.metrics.benchmarks[bench]
        base = baseline.metrics.benchmarks[bench]
        benchmarks[bench] = {
            "delta_greedy_pass_at_1": cand.get("greedy_pass_at_1", 0.0)
            - base.get("greedy_pass_at_1", 0.0),
            "delta_sampled_pass_at_1": cand.get("sampled_pass_at_1", 0.0)
            - base.get("sampled_pass_at_1", 0.0),
        }
    out["benchmark_deltas"] = benchmarks
    return out


def _as_payload(outcome: ModelOutcome) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint": str(outcome.checkpoint),
        "run_dir": str(outcome.run_dir),
        "status": outcome.status,
        "notes": outcome.notes,
        "wallclock_seconds": outcome.wallclock_seconds,
        "validation_errors": list(outcome.validation_errors),
        "metrics": asdict(outcome.metrics) if outcome.metrics is not None else None,
    }
    return payload


def _validate_dataset_dir(config, dataset_dir: Path) -> tuple[Path, Path]:
    gsm_name = Path(config.evaluation.datasets.gsm8k.path).name
    math_name = Path(config.evaluation.datasets.math.path).name
    gsm = dataset_dir / gsm_name
    math = dataset_dir / math_name
    if not gsm.exists():
        raise ValueError(f"Missing eval dataset file: {gsm}")
    if not math.exists():
        raise ValueError(f"Missing eval dataset file: {math}")
    return gsm, math


def _fixture_eval_overrides() -> list[str]:
    return [
        "evaluation.inference.backend=fixture",
        "evaluation.sampled.num_samples=4",
    ]


def run_final_evaluation(
    *,
    config_path: Path,
    dataset_dir: Path,
    base_checkpoint: Path,
    sft_checkpoint: Path,
    grpo_checkpoint: Path,
    output_root: Path,
    report_dir: Path,
    seed: int | None,
    max_items: int | None,
    mode: str,
    skip_existing: bool,
    fail_fast: bool,
    strict_claims: bool,
    session_id: str | None = None,
) -> FinalEvalResult:
    mode_lower = mode.lower()
    if mode_lower not in {"fixture", "real"}:
        raise ValueError(f"Unsupported mode: {mode}")

    defaults_path = repo_root() / "configs" / "defaults.yaml"
    resolved = resolve_config(defaults_path, config_path)
    _validate_dataset_dir(resolved, dataset_dir)

    checkpoints = {
        "base": base_checkpoint,
        "sft": sft_checkpoint,
        "grpo": grpo_checkpoint,
    }
    for model_id, checkpoint in checkpoints.items():
        if not checkpoint.exists():
            raise ValueError(f"{model_id} checkpoint not found: {checkpoint}")

    selected_session_id = session_id or f"final-{_now_stamp()}-{secrets.token_hex(3)}"
    runs_root = (output_root / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    outcomes: dict[str, ModelOutcome] = {}
    reference_run_dir: Path | None = None
    failure_reasons: list[str] = []

    for model_id in MODEL_ORDER:
        checkpoint = checkpoints[model_id]
        run_id = f"{selected_session_id}-{model_id}-eval"
        run_dir = runs_root / run_id
        args = [
            "eval",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--run-id",
            run_id,
            "--output-dir",
            str(runs_root),
            "--dataset-dir",
            str(dataset_dir),
        ]
        if seed is not None:
            args.extend(["--seed", str(seed)])
        if max_items is not None:
            args.extend(["--max-items", str(max_items)])
        if mode_lower == "fixture":
            for item in _fixture_eval_overrides():
                args.extend(["--set", item])

        code, _ = _run_command(argv=args, run_dir=run_dir, skip_existing=skip_existing)
        wallclock = _duration_seconds(run_dir / "summary.json")
        if code != 0:
            notes = _failure_note(run_dir)
            outcome = ModelOutcome(
                model_id=model_id,
                checkpoint=checkpoint,
                run_dir=run_dir,
                status="failed",
                notes=notes,
                wallclock_seconds=wallclock,
                validation_errors=(),
                metrics=None,
            )
            outcomes[model_id] = outcome
            failure_reasons.append(f"{model_id}: {notes}")
            if fail_fast:
                break
            continue

        compare = reference_run_dir if reference_run_dir is not None else None
        validation = validate_run_dir(run_dir, compare_run_dir=compare)
        validation_errors = tuple(validation.errors)
        if not validation.ok:
            notes = "; ".join(validation.errors) if validation.errors else "run validation failed"
            outcome = ModelOutcome(
                model_id=model_id,
                checkpoint=checkpoint,
                run_dir=run_dir,
                status="failed",
                notes=notes,
                wallclock_seconds=wallclock,
                validation_errors=validation_errors,
                metrics=None,
            )
            outcomes[model_id] = outcome
            failure_reasons.append(f"{model_id}: {notes}")
            if fail_fast:
                break
            continue

        if reference_run_dir is None:
            reference_run_dir = run_dir

        try:
            metrics = _extract_model_metrics(run_dir, wallclock)
            outcome = ModelOutcome(
                model_id=model_id,
                checkpoint=checkpoint,
                run_dir=run_dir,
                status="success",
                notes="ok",
                wallclock_seconds=wallclock,
                validation_errors=validation_errors,
                metrics=metrics,
            )
            outcomes[model_id] = outcome
        except Exception as exc:
            notes = str(exc)
            outcome = ModelOutcome(
                model_id=model_id,
                checkpoint=checkpoint,
                run_dir=run_dir,
                status="failed",
                notes=notes,
                wallclock_seconds=wallclock,
                validation_errors=validation_errors,
                metrics=None,
            )
            outcomes[model_id] = outcome
            failure_reasons.append(f"{model_id}: {notes}")
            if fail_fast:
                break

    for model_id in MODEL_ORDER:
        if model_id in outcomes:
            continue
        checkpoint = checkpoints[model_id]
        run_id = f"{selected_session_id}-{model_id}-eval"
        run_dir = runs_root / run_id
        outcomes[model_id] = ModelOutcome(
            model_id=model_id,
            checkpoint=checkpoint,
            run_dir=run_dir,
            status="skipped",
            notes="not_executed_due_to_fail_fast",
            wallclock_seconds=0.0,
            validation_errors=(),
            metrics=None,
        )
        failure_reasons.append(f"{model_id}: not_executed_due_to_fail_fast")

    base_out = outcomes["base"]
    sft_out = outcomes["sft"]
    grpo_out = outcomes["grpo"]
    comparisons = {
        "sft_vs_base": _comparison(sft_out, base_out),
        "grpo_vs_sft": _comparison(grpo_out, sft_out),
        "grpo_vs_base": _comparison(grpo_out, base_out),
    }

    quality_gates: dict[str, Any] = {}
    for model_id in MODEL_ORDER:
        item = outcomes[model_id]
        metrics = item.metrics
        if metrics is None:
            quality_gates[model_id] = {
                "has_metrics": False,
                "schema_threshold_passed": False,
                "parser_threshold_passed": False,
            }
            continue
        quality_gates[model_id] = {
            "has_metrics": True,
            "schema_threshold_passed": metrics.schema_compliance_rate
            >= resolved.reward.thresholds.schema_compliance_rate_min,
            "parser_threshold_passed": metrics.parser_failure_rate
            <= resolved.reward.thresholds.parser_failure_rate_max,
        }

    all_success = all(outcomes[item].status == "success" for item in MODEL_ORDER)
    required_comparisons = comparisons["sft_vs_base"] is not None and comparisons["grpo_vs_sft"] is not None
    strict_claims_ok = all_success and required_comparisons

    status = "success" if strict_claims_ok else "partial"
    if all(outcomes[item].status != "success" for item in MODEL_ORDER):
        status = "failed"

    if strict_claims and not strict_claims_ok:
        failure_reasons.append("strict_claims_failed")

    by_model_analysis: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        outcome = outcomes[model_id]
        if outcome.status != "success":
            continue
        predictions_path = outcome.run_dir / "predictions.jsonl"
        if predictions_path.exists():
            by_model_analysis[model_id] = analyze_predictions(_read_jsonl(predictions_path))

    payload = {
        "schema_version": resolved.schema_version,
        "session_id": selected_session_id,
        "inputs": {
            "config_path": str(config_path),
            "dataset_dir": str(dataset_dir),
            "base_checkpoint": str(base_checkpoint),
            "sft_checkpoint": str(sft_checkpoint),
            "grpo_checkpoint": str(grpo_checkpoint),
            "mode": mode_lower,
            "seed": seed,
            "max_items": max_items,
        },
        "models": {model_id: _as_payload(outcomes[model_id]) for model_id in MODEL_ORDER},
        "comparisons": comparisons,
        "quality_gates": quality_gates,
        "strict_claims_ok": strict_claims_ok,
        "status": status,
        "failure_reasons": failure_reasons,
    }

    metrics_path = report_dir / "final_metrics.json"
    report_path = report_dir / "final_report.md"
    error_analysis_path = report_dir / "error_analysis.md"
    write_final_metrics_json(payload, metrics_path)
    write_final_report_markdown(payload, report_path)
    write_error_analysis_markdown(
        build_error_analysis_markdown(by_model_analysis),
        error_analysis_path,
    )

    return FinalEvalResult(
        session_id=selected_session_id,
        status=status,
        strict_claims_ok=strict_claims_ok,
        failure_reasons=tuple(failure_reasons),
        metrics_path=metrics_path,
        report_path=report_path,
        error_analysis_path=error_analysis_path,
        outcomes=tuple(outcomes[item] for item in MODEL_ORDER),
    )

