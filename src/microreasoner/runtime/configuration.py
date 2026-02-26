from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from microreasoner.runtime.errors import RuntimeConfigError
from microreasoner.runtime.models import (
    DataConfig,
    EvaluationConfig,
    EvaluationGreedyConfig,
    EvaluationSampledConfig,
    GateConfig,
    ModelConfig,
    ProjectConfig,
    ResolvedConfig,
    RLDataConfig,
    RewardConfig,
    RewardFormatConfig,
    RewardThresholdConfig,
    RewardWeightsConfig,
    SFTDataConfig,
)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeConfigError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_scalar(text: str) -> Any:
    raw = text.strip()
    if raw == "":
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        return raw


def parse_override(assignment: str) -> tuple[list[str], Any]:
    if "=" not in assignment:
        raise RuntimeConfigError(f"Invalid override '{assignment}': expected key=value")
    path_text, value_text = assignment.split("=", 1)
    keys = [part.strip() for part in path_text.split(".") if part.strip()]
    if not keys:
        raise RuntimeConfigError(f"Invalid override '{assignment}': missing key path")
    return keys, _parse_scalar(value_text)


def apply_override(config: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in cursor:
            raise RuntimeConfigError(
                f"Unknown override path '{'.'.join(keys)}': missing segment '{key}'"
            )
        next_value = cursor[key]
        if not isinstance(next_value, dict):
            raise RuntimeConfigError(
                f"Override path '{'.'.join(keys)}' traverses non-object segment '{key}'"
            )
        cursor = next_value

    leaf = keys[-1]
    if leaf not in cursor:
        raise RuntimeConfigError(
            f"Unknown override key '{'.'.join(keys)}': leaf '{leaf}' does not exist"
        )
    cursor[leaf] = value


def apply_overrides(config: dict[str, Any], assignments: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for assignment in assignments:
        keys, value = parse_override(assignment)
        apply_override(resolved, keys, value)
    return resolved


def _as_dict(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"Expected object at '{key}'")
    return value


def _require_str(source: dict[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or value == "":
        raise RuntimeConfigError(f"Expected non-empty string at '{key}'")
    return value


def _optional_str(source: dict[str, Any], key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeConfigError(f"Expected string or null at '{key}'")
    return value


def _require_bool(source: dict[str, Any], key: str) -> bool:
    value = source.get(key)
    if not isinstance(value, bool):
        raise RuntimeConfigError(f"Expected bool at '{key}'")
    return value


def _require_float(source: dict[str, Any], key: str) -> float:
    value = source.get(key)
    if not isinstance(value, (int, float)):
        raise RuntimeConfigError(f"Expected numeric value at '{key}'")
    return float(value)


def _require_int(source: dict[str, Any], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int):
        raise RuntimeConfigError(f"Expected integer at '{key}'")
    return value


def _require_str_list(source: dict[str, Any], key: str) -> tuple[str, ...]:
    value = source.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeConfigError(f"Expected list[str] at '{key}'")
    return tuple(value)


def to_resolved_config(raw: dict[str, Any]) -> ResolvedConfig:
    project_raw = _as_dict(raw, "project")
    model_raw = _as_dict(raw, "model")
    data_raw = _as_dict(raw, "data")
    reward_raw = _as_dict(raw, "reward")
    evaluation_raw = _as_dict(raw, "evaluation")
    gates_raw = _as_dict(raw, "gates")

    sft_raw = _as_dict(data_raw, "sft")
    rl_raw = _as_dict(data_raw, "rl")

    reward_format_raw = _as_dict(reward_raw, "format")
    reward_weights_raw = _as_dict(reward_raw, "weights")
    reward_threshold_raw = _as_dict(reward_raw, "thresholds")

    eval_greedy_raw = _as_dict(evaluation_raw, "greedy")
    eval_sampled_raw = _as_dict(evaluation_raw, "sampled")

    return ResolvedConfig(
        schema_version=_require_str(raw, "schema_version"),
        project=ProjectConfig(
            name=_require_str(project_raw, "name"),
            mode=_require_str(project_raw, "mode"),
        ),
        model=ModelConfig(
            default_base_model=_require_str(model_raw, "default_base_model"),
            fallback_base_model=_optional_str(model_raw, "fallback_base_model"),
        ),
        data=DataConfig(
            strategy=_require_str(data_raw, "strategy"),
            sft=SFTDataConfig(
                primary_dataset=_require_str(sft_raw, "primary_dataset"),
                secondary_datasets=_require_str_list(sft_raw, "secondary_datasets"),
            ),
            rl=RLDataConfig(curriculum=_require_str_list(rl_raw, "curriculum")),
        ),
        reward=RewardConfig(
            format=RewardFormatConfig(
                response_schema=_require_str(reward_format_raw, "response_schema")
            ),
            weights=RewardWeightsConfig(
                correctness=_require_float(reward_weights_raw, "correctness"),
                schema=_require_float(reward_weights_raw, "schema"),
                length=_require_float(reward_weights_raw, "length"),
            ),
            thresholds=RewardThresholdConfig(
                parser_failure_rate_max=_require_float(
                    reward_threshold_raw, "parser_failure_rate_max"
                ),
                schema_compliance_rate_min=_require_float(
                    reward_threshold_raw, "schema_compliance_rate_min"
                ),
            ),
        ),
        evaluation=EvaluationConfig(
            publish=_require_str_list(evaluation_raw, "publish"),
            greedy=EvaluationGreedyConfig(
                temperature=_require_float(eval_greedy_raw, "temperature")
            ),
            sampled=EvaluationSampledConfig(
                temperature=_require_float(eval_sampled_raw, "temperature"),
                top_p=_require_float(eval_sampled_raw, "top_p"),
                num_samples=_require_int(eval_sampled_raw, "num_samples"),
            ),
        ),
        gates=GateConfig(
            strict_qa=_require_bool(gates_raw, "strict_qa"),
            fail_on_missing_metrics=_require_bool(gates_raw, "fail_on_missing_metrics"),
            fail_on_resume_test_failure=_require_bool(
                gates_raw, "fail_on_resume_test_failure"
            ),
            fail_on_dataset_hash_mismatch=_require_bool(
                gates_raw, "fail_on_dataset_hash_mismatch"
            ),
            fail_on_eval_config_drift=_require_bool(
                gates_raw, "fail_on_eval_config_drift"
            ),
        ),
        raw=copy.deepcopy(raw),
    )


def resolve_config(
    defaults_path: Path,
    user_config_path: Path,
    cli_overrides: list[str] | None = None,
) -> ResolvedConfig:
    defaults = load_yaml_mapping(defaults_path)
    user_config = load_yaml_mapping(user_config_path)
    merged = deep_merge(defaults, user_config)
    if cli_overrides:
        merged = apply_overrides(merged, cli_overrides)
    return to_resolved_config(merged)

