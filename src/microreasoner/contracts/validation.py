from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema


PARSER_FAILURE_RATE_MAX = 0.02
SCHEMA_COMPLIANCE_RATE_MIN = 0.98


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _schema_path() -> Path:
    return _repo_root() / "schemas" / "run_manifest.schema.json"


def _extract_eval_sampling(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    evaluation = config.get("evaluation", {})
    sampled = evaluation.get("sampled", {}) if isinstance(evaluation, dict) else {}
    return (
        sampled.get("temperature"),
        sampled.get("top_p"),
        sampled.get("num_samples"),
    )


def _validate_metrics(metrics: dict[str, Any], errors: list[str]) -> None:
    required_families = ("accuracy", "schema", "parser", "length")
    for family in required_families:
        if family not in metrics:
            errors.append(f"Missing required metrics family: {family}")

    accuracy = metrics.get("accuracy")
    if isinstance(accuracy, dict):
        for benchmark_name, benchmark_values in accuracy.items():
            if not isinstance(benchmark_values, dict):
                errors.append(f"Accuracy benchmark is not an object: {benchmark_name}")
                continue
            for key in ("greedy_pass_at_1", "sampled_pass_at_1"):
                if key not in benchmark_values:
                    errors.append(
                        f"Missing required accuracy metric: accuracy.{benchmark_name}.{key}"
                    )

    schema = metrics.get("schema")
    if isinstance(schema, dict):
        compliance_rate = schema.get("compliance_rate")
        if compliance_rate is None:
            errors.append("Missing schema.compliance_rate")
        elif not isinstance(compliance_rate, (int, float)):
            errors.append("schema.compliance_rate must be numeric")
        elif float(compliance_rate) < SCHEMA_COMPLIANCE_RATE_MIN:
            errors.append(
                f"schema.compliance_rate {compliance_rate} is below minimum "
                f"{SCHEMA_COMPLIANCE_RATE_MIN}"
            )

    parser = metrics.get("parser")
    if isinstance(parser, dict):
        extraction_failure_rate = parser.get("extraction_failure_rate")
        if extraction_failure_rate is None:
            errors.append("Missing parser.extraction_failure_rate")
        elif not isinstance(extraction_failure_rate, (int, float)):
            errors.append("parser.extraction_failure_rate must be numeric")
        elif float(extraction_failure_rate) > PARSER_FAILURE_RATE_MAX:
            errors.append(
                f"parser.extraction_failure_rate {extraction_failure_rate} exceeds maximum "
                f"{PARSER_FAILURE_RATE_MAX}"
            )

    length = metrics.get("length")
    if isinstance(length, dict):
        think_tokens = length.get("think_tokens")
        if not isinstance(think_tokens, dict):
            errors.append("Missing length.think_tokens object")
        else:
            for key in ("mean", "p95"):
                if key not in think_tokens:
                    errors.append(f"Missing length.think_tokens.{key}")


def _validate_checkpoint_state(checkpoints: dict[str, Any], run_dir: Path, errors: list[str]) -> None:
    resume_test = checkpoints.get("resume_test")
    if not isinstance(resume_test, dict):
        errors.append("Missing checkpoints.resume_test object")
    else:
        if resume_test.get("passed") is not True:
            errors.append("Checkpoint resume test failed or is not true")

    latest = checkpoints.get("latest")
    if not isinstance(latest, str) or not latest:
        errors.append("Missing checkpoints.latest pointer")
    else:
        latest_path = Path(latest)
        if latest_path.is_absolute():
            exists = latest_path.exists()
        else:
            # Accept both run-relative pointers (contract default) and
            # repository/CWD-relative pointers emitted by some runners.
            exists = (run_dir / latest_path).exists() or latest_path.exists()
        if not exists:
            errors.append(
                "checkpoints.latest path does not exist "
                f"(checked run-relative and cwd-relative forms): {latest}"
            )

    best = checkpoints.get("best")
    if not isinstance(best, str) or not best:
        errors.append("Missing checkpoints.best pointer")


def _validate_dataset_hashes(
    run_manifest: dict[str, Any], dataset_manifest: dict[str, Any], errors: list[str]
) -> None:
    run_data = run_manifest.get("data")
    datasets = dataset_manifest.get("datasets")
    if not isinstance(run_data, dict):
        errors.append("run_manifest.data must be an object")
        return
    if not isinstance(datasets, dict):
        errors.append("dataset_manifest.datasets must be an object")
        return

    for key in ("sft", "rl", "eval"):
        run_entry = run_data.get(key)
        dataset_entry = datasets.get(key)
        if not isinstance(run_entry, dict):
            errors.append(f"run_manifest.data.{key} missing or invalid")
            continue
        if not isinstance(dataset_entry, dict):
            errors.append(f"dataset_manifest.datasets.{key} missing or invalid")
            continue

        run_hash = run_entry.get("hash")
        ds_hash = dataset_entry.get("hash")
        if run_hash != ds_hash:
            errors.append(
                f"Dataset hash mismatch for {key}: run_manifest has {run_hash}, "
                f"dataset_manifest has {ds_hash}"
            )


def validate_run_dir(run_dir: Path, compare_run_dir: Path | None = None) -> ValidationResult:
    errors: list[str] = []
    run_dir = run_dir.resolve()

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return ValidationResult(ok=False, errors=["Missing required file: run_manifest.json"])

    try:
        run_manifest = _load_json(manifest_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return ValidationResult(ok=False, errors=[f"Invalid run_manifest.json: {exc}"])

    try:
        schema = _load_json(_schema_path())
        jsonschema.validate(instance=run_manifest, schema=schema)
    except jsonschema.ValidationError as exc:
        errors.append(f"run_manifest schema validation failed: {exc.message}")
    except FileNotFoundError:
        errors.append("run manifest schema file is missing")

    artifacts = run_manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return ValidationResult(ok=False, errors=["run_manifest.artifacts must be an object"])

    required_artifacts = {
        "config_path": "config.json",
        "dataset_manifest_path": "dataset_manifest.json",
        "metrics_path": "metrics.json",
        "checkpoints_path": "checkpoints.json",
    }
    loaded: dict[str, dict[str, Any]] = {}
    for key, fallback_name in required_artifacts.items():
        rel = artifacts.get(key, fallback_name)
        if not isinstance(rel, str) or not rel:
            errors.append(f"Invalid artifact pointer for {key}")
            continue
        target = (run_dir / rel).resolve()
        if not target.exists():
            errors.append(f"Missing required artifact file: {rel}")
            continue
        try:
            loaded[key] = _load_json(target)
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid JSON in artifact {rel}: {exc}")

    config = loaded.get("config_path")
    dataset_manifest = loaded.get("dataset_manifest_path")
    metrics = loaded.get("metrics_path")
    checkpoints = loaded.get("checkpoints_path")

    if isinstance(metrics, dict):
        _validate_metrics(metrics, errors)
    if isinstance(checkpoints, dict):
        _validate_checkpoint_state(checkpoints, run_dir, errors)
    if isinstance(dataset_manifest, dict):
        _validate_dataset_hashes(run_manifest, dataset_manifest, errors)

    if compare_run_dir is not None:
        compare_manifest_path = compare_run_dir.resolve() / "run_manifest.json"
        if not compare_manifest_path.exists():
            errors.append(f"Compare run missing run_manifest.json: {compare_run_dir}")
        else:
            try:
                compare_manifest = _load_json(compare_manifest_path)
                compare_artifacts = compare_manifest.get("artifacts", {})
                compare_config_rel = (
                    compare_artifacts.get("config_path")
                    if isinstance(compare_artifacts, dict)
                    else None
                )
                if not isinstance(compare_config_rel, str):
                    errors.append("Compare run has invalid artifacts.config_path")
                elif config is not None:
                    compare_config = _load_json(compare_run_dir.resolve() / compare_config_rel)
                    if _extract_eval_sampling(config) != _extract_eval_sampling(compare_config):
                        errors.append("Evaluation sampling config drift detected across runs")
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid compare run manifest/config: {exc}")

    return ValidationResult(ok=not errors, errors=errors)

