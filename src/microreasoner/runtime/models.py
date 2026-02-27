from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    mode: str


@dataclass(frozen=True)
class ModelConfig:
    default_base_model: str
    fallback_base_model: str | None


@dataclass(frozen=True)
class SFTDataConfig:
    primary_dataset: str
    secondary_datasets: tuple[str, ...]


@dataclass(frozen=True)
class RLDataConfig:
    curriculum: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    strategy: str
    sft: SFTDataConfig
    rl: RLDataConfig


@dataclass(frozen=True)
class RewardFormatConfig:
    response_schema: str


@dataclass(frozen=True)
class RewardWeightsConfig:
    correctness: float
    schema: float
    length: float


@dataclass(frozen=True)
class RewardThresholdConfig:
    parser_failure_rate_max: float
    schema_compliance_rate_min: float


@dataclass(frozen=True)
class RewardConfig:
    format: RewardFormatConfig
    weights: RewardWeightsConfig
    thresholds: RewardThresholdConfig


@dataclass(frozen=True)
class EvaluationGreedyConfig:
    temperature: float


@dataclass(frozen=True)
class EvaluationSampledConfig:
    temperature: float
    top_p: float
    num_samples: int


@dataclass(frozen=True)
class BenchmarkDatasetConfig:
    path: str


@dataclass(frozen=True)
class EvaluationDatasetsConfig:
    gsm8k: BenchmarkDatasetConfig
    math: BenchmarkDatasetConfig


@dataclass(frozen=True)
class EvaluationParserConfig:
    strict_boxed_only: bool


@dataclass(frozen=True)
class EvaluationInferenceConfig:
    backend: str
    max_new_tokens: int
    device: str
    dtype: str


@dataclass(frozen=True)
class EvaluationConfig:
    publish: tuple[str, ...]
    datasets: EvaluationDatasetsConfig
    parser: EvaluationParserConfig
    inference: EvaluationInferenceConfig
    greedy: EvaluationGreedyConfig
    sampled: EvaluationSampledConfig


@dataclass(frozen=True)
class DataSourceConfig:
    name: str
    adapter: str
    path: str


@dataclass(frozen=True)
class DataSplitConfig:
    strategy: str
    train_ratio: float
    val_ratio: float
    seed: int


@dataclass(frozen=True)
class DataFilterConfig:
    min_think_tokens: int
    max_think_tokens: int
    require_single_boxed_answer: bool
    drop_duplicates: bool


@dataclass(frozen=True)
class DataOutputConfig:
    root_dir: str
    write_rejects: bool
    compression: str


@dataclass(frozen=True)
class RLCurriculumRuleConfig:
    name: str
    benchmarks: tuple[str, ...]


@dataclass(frozen=True)
class RLDataPipelineConfig:
    curriculum_rules: tuple[RLCurriculumRuleConfig, ...]
    benchmark_mix_targets: dict[str, float]


@dataclass(frozen=True)
class DataPipelineConfig:
    schema_version: str
    input_sources: tuple[DataSourceConfig, ...]
    split: DataSplitConfig
    filters: DataFilterConfig
    outputs: DataOutputConfig
    rl: RLDataPipelineConfig


@dataclass(frozen=True)
class GateConfig:
    strict_qa: bool
    fail_on_missing_metrics: bool
    fail_on_resume_test_failure: bool
    fail_on_dataset_hash_mismatch: bool
    fail_on_eval_config_drift: bool


@dataclass(frozen=True)
class ResolvedConfig:
    schema_version: str
    project: ProjectConfig
    model: ModelConfig
    data: DataConfig
    data_pipeline: DataPipelineConfig
    reward: RewardConfig
    evaluation: EvaluationConfig
    gates: GateConfig
    raw: dict[str, Any]


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    logs_dir: Path
    events_log_path: Path
    summary_path: Path
    config_path: Path
    command_meta_path: Path
    errors_path: Path


@dataclass(frozen=True)
class RunContext:
    run_id: str
    command_name: str
    seed: int
    started_at: str
    paths: RunPaths
