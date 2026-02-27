from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from microreasoner.runtime.errors import RuntimeConfigError
from microreasoner.runtime.models import (
    BenchmarkDatasetConfig,
    DataConfig,
    DataFilterConfig,
    DataOutputConfig,
    DataPipelineConfig,
    DataSourceConfig,
    DataSplitConfig,
    EvaluationConfig,
    EvaluationDatasetsConfig,
    EvaluationGreedyConfig,
    EvaluationInferenceConfig,
    EvaluationParserConfig,
    EvaluationSampledConfig,
    GateConfig,
    ModelConfig,
    ProjectConfig,
    ResolvedConfig,
    RLDataConfig,
    RLDataPipelineConfig,
    RLCurriculumRuleConfig,
    RewardConfig,
    RewardFormatConfig,
    RewardThresholdConfig,
    RewardWeightsConfig,
    SFTDataConfig,
    TrainSFTBackendConfig,
    TrainSFTBatchConfig,
    TrainSFTCheckpointConfig,
    TrainSFTConfig,
    TrainSFTGateConfig,
    TrainSFTLoRAConfig,
    TrainSFTOptimConfig,
    TrainSFTQuantizationConfig,
    TrainSFTRunConfig,
    TrainSFTSelectionConfig,
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


def _require_mapping_list(source: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = source.get(key)
    if not isinstance(value, list):
        raise RuntimeConfigError(f"Expected list[object] at '{key}'")
    if not all(isinstance(item, dict) for item in value):
        raise RuntimeConfigError(f"Expected list[object] at '{key}'")
    return list(value)


def _require_float_mapping(source: dict[str, Any], key: str) -> dict[str, float]:
    value = source.get(key)
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"Expected object at '{key}'")
    out: dict[str, float] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str):
            raise RuntimeConfigError(f"Expected string keys in '{key}'")
        if not isinstance(item_value, (int, float)):
            raise RuntimeConfigError(f"Expected numeric values in '{key}'")
        out[item_key] = float(item_value)
    return out


def to_resolved_config(raw: dict[str, Any]) -> ResolvedConfig:
    project_raw = _as_dict(raw, "project")
    model_raw = _as_dict(raw, "model")
    data_raw = _as_dict(raw, "data")
    data_pipeline_raw = _as_dict(raw, "data_pipeline")
    train_sft_raw = _as_dict(raw, "train_sft")
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
    eval_datasets_raw = _as_dict(evaluation_raw, "datasets")
    eval_gsm8k_raw = _as_dict(eval_datasets_raw, "gsm8k")
    eval_math_raw = _as_dict(eval_datasets_raw, "math")
    eval_parser_raw = _as_dict(evaluation_raw, "parser")
    eval_inference_raw = _as_dict(evaluation_raw, "inference")

    pipeline_split_raw = _as_dict(data_pipeline_raw, "split")
    pipeline_filters_raw = _as_dict(data_pipeline_raw, "filters")
    pipeline_outputs_raw = _as_dict(data_pipeline_raw, "outputs")
    pipeline_rl_raw = _as_dict(data_pipeline_raw, "rl")
    pipeline_source_rows = _require_mapping_list(data_pipeline_raw, "input_sources")
    pipeline_rule_rows = _require_mapping_list(pipeline_rl_raw, "curriculum_rules")

    train_sft_lora_raw = _as_dict(train_sft_raw, "lora")
    train_sft_quant_raw = _as_dict(train_sft_raw, "quantization")
    train_sft_optim_raw = _as_dict(train_sft_raw, "optim")
    train_sft_batch_raw = _as_dict(train_sft_raw, "batch")
    train_sft_run_raw = _as_dict(train_sft_raw, "run")
    train_sft_checkpoint_raw = _as_dict(train_sft_raw, "checkpoint")
    train_sft_selection_raw = _as_dict(train_sft_raw, "selection")
    train_sft_gate_raw = _as_dict(train_sft_raw, "gates")
    train_sft_backend_raw = _as_dict(train_sft_raw, "backend")

    if len(pipeline_source_rows) == 0:
        raise RuntimeConfigError("data_pipeline.input_sources must include at least one source")

    source_configs: list[DataSourceConfig] = []
    for source_row in pipeline_source_rows:
        source_configs.append(
            DataSourceConfig(
                name=_require_str(source_row, "name"),
                adapter=_require_str(source_row, "adapter"),
                path=_require_str(source_row, "path"),
            )
        )

    curriculum_rules: list[RLCurriculumRuleConfig] = []
    for rule_row in pipeline_rule_rows:
        curriculum_rules.append(
            RLCurriculumRuleConfig(
                name=_require_str(rule_row, "name"),
                benchmarks=_require_str_list(rule_row, "benchmarks"),
            )
        )

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
        data_pipeline=DataPipelineConfig(
            schema_version=_require_str(data_pipeline_raw, "schema_version"),
            input_sources=tuple(source_configs),
            split=DataSplitConfig(
                strategy=_require_str(pipeline_split_raw, "strategy"),
                train_ratio=_require_float(pipeline_split_raw, "train_ratio"),
                val_ratio=_require_float(pipeline_split_raw, "val_ratio"),
                seed=_require_int(pipeline_split_raw, "seed"),
            ),
            filters=DataFilterConfig(
                min_think_tokens=_require_int(pipeline_filters_raw, "min_think_tokens"),
                max_think_tokens=_require_int(pipeline_filters_raw, "max_think_tokens"),
                require_single_boxed_answer=_require_bool(
                    pipeline_filters_raw, "require_single_boxed_answer"
                ),
                drop_duplicates=_require_bool(pipeline_filters_raw, "drop_duplicates"),
            ),
            outputs=DataOutputConfig(
                root_dir=_require_str(pipeline_outputs_raw, "root_dir"),
                write_rejects=_require_bool(pipeline_outputs_raw, "write_rejects"),
                compression=_require_str(pipeline_outputs_raw, "compression"),
            ),
            rl=RLDataPipelineConfig(
                curriculum_rules=tuple(curriculum_rules),
                benchmark_mix_targets=_require_float_mapping(
                    pipeline_rl_raw, "benchmark_mix_targets"
                ),
            ),
        ),
        train_sft=TrainSFTConfig(
            mode=_require_str(train_sft_raw, "mode"),
            lora=TrainSFTLoRAConfig(
                r=_require_int(train_sft_lora_raw, "r"),
                alpha=_require_int(train_sft_lora_raw, "alpha"),
                dropout=_require_float(train_sft_lora_raw, "dropout"),
                target_modules=_require_str_list(train_sft_lora_raw, "target_modules"),
            ),
            quantization=TrainSFTQuantizationConfig(
                enabled=_require_bool(train_sft_quant_raw, "enabled"),
                bnb_4bit_compute_dtype=_require_str(
                    train_sft_quant_raw, "bnb_4bit_compute_dtype"
                ),
                double_quant=_require_bool(train_sft_quant_raw, "double_quant"),
                quant_type=_require_str(train_sft_quant_raw, "quant_type"),
            ),
            optim=TrainSFTOptimConfig(
                lr=_require_float(train_sft_optim_raw, "lr"),
                weight_decay=_require_float(train_sft_optim_raw, "weight_decay"),
                warmup_ratio=_require_float(train_sft_optim_raw, "warmup_ratio"),
                scheduler=_require_str(train_sft_optim_raw, "scheduler"),
            ),
            batch=TrainSFTBatchConfig(
                per_device=_require_int(train_sft_batch_raw, "per_device"),
                grad_accum=_require_int(train_sft_batch_raw, "grad_accum"),
                max_seq_len=_require_int(train_sft_batch_raw, "max_seq_len"),
            ),
            run=TrainSFTRunConfig(
                epochs=_require_int(train_sft_run_raw, "epochs"),
                max_steps=_require_int(train_sft_run_raw, "max_steps"),
                eval_every_steps=_require_int(train_sft_run_raw, "eval_every_steps"),
                save_every_steps=_require_int(train_sft_run_raw, "save_every_steps"),
                save_every_minutes=_require_int(train_sft_run_raw, "save_every_minutes"),
                logging_steps=_require_int(train_sft_run_raw, "logging_steps"),
                max_eval_samples=_require_int(train_sft_run_raw, "max_eval_samples"),
            ),
            checkpoint=TrainSFTCheckpointConfig(
                save_total_limit=_require_int(train_sft_checkpoint_raw, "save_total_limit"),
                resume_strict=_require_bool(train_sft_checkpoint_raw, "resume_strict"),
            ),
            selection=TrainSFTSelectionConfig(
                primary_metric=_require_str(train_sft_selection_raw, "primary_metric"),
                secondary_metric=_require_str(train_sft_selection_raw, "secondary_metric"),
            ),
            gates=TrainSFTGateConfig(
                schema_min=_require_float(train_sft_gate_raw, "schema_min"),
                baseline_greedy_pass_at_1=_require_float(
                    train_sft_gate_raw, "baseline_greedy_pass_at_1"
                ),
            ),
            backend=TrainSFTBackendConfig(
                trainer=_require_str(train_sft_backend_raw, "trainer")
            ),
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
            datasets=EvaluationDatasetsConfig(
                gsm8k=BenchmarkDatasetConfig(path=_require_str(eval_gsm8k_raw, "path")),
                math=BenchmarkDatasetConfig(path=_require_str(eval_math_raw, "path")),
            ),
            parser=EvaluationParserConfig(
                strict_boxed_only=_require_bool(eval_parser_raw, "strict_boxed_only")
            ),
            inference=EvaluationInferenceConfig(
                backend=_require_str(eval_inference_raw, "backend"),
                max_new_tokens=_require_int(eval_inference_raw, "max_new_tokens"),
                device=_require_str(eval_inference_raw, "device"),
                dtype=_require_str(eval_inference_raw, "dtype"),
            ),
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
