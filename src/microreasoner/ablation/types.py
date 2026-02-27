from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    family: str
    description: str
    kind: str
    train_overrides: tuple[str, ...]
    alias_of: str | None = None


@dataclass(frozen=True)
class RunArtifacts:
    train_run_dir: Path | None
    eval_run_dir: Path | None


@dataclass(frozen=True)
class MetricSnapshot:
    greedy_pass_at_1: float
    sampled_pass_at_1: float
    schema_compliance_rate: float
    parser_failure_rate: float
    eval_examples: int


@dataclass(frozen=True)
class CostSnapshot:
    wallclock_seconds: float
    train_steps: int


@dataclass(frozen=True)
class ExperimentOutcome:
    experiment_id: str
    family: str
    description: str
    status: str
    notes: str
    artifacts: RunArtifacts
    metrics: MetricSnapshot | None
    cost: CostSnapshot


@dataclass(frozen=True)
class AblationRow:
    experiment_id: str
    family: str
    backend_mode: str
    status: str
    sft_run_dir: str
    train_run_dir: str
    eval_run_dir: str
    greedy_pass_at_1: float
    sampled_pass_at_1: float
    schema_compliance_rate: float
    parser_failure_rate: float
    delta_greedy_vs_sft: float
    delta_sampled_vs_sft: float
    delta_schema_vs_sft: float
    delta_parser_vs_sft: float
    wallclock_seconds: float
    train_steps: int
    eval_examples: int
    notes: str

