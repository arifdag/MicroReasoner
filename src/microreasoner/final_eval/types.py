from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMetrics:
    benchmarks: dict[str, dict[str, float]]
    macro_greedy_pass_at_1: float
    macro_sampled_pass_at_1: float
    schema_compliance_rate: float
    parser_failure_rate: float
    think_tokens_mean: float
    think_tokens_p95: float
    eval_examples: int
    greedy_solved: int
    sampled_solved: int
    cost_per_solved_greedy: float | None
    cost_per_solved_sampled: float | None


@dataclass(frozen=True)
class ModelOutcome:
    model_id: str
    checkpoint: Path
    run_dir: Path
    status: str
    notes: str
    wallclock_seconds: float
    validation_errors: tuple[str, ...]
    metrics: ModelMetrics | None


@dataclass(frozen=True)
class FinalEvalResult:
    session_id: str
    status: str
    strict_claims_ok: bool
    failure_reasons: tuple[str, ...]
    metrics_path: Path
    report_path: Path
    error_analysis_path: Path
    outcomes: tuple[ModelOutcome, ...]

