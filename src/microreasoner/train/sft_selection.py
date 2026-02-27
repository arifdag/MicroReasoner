from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from microreasoner.runtime.models import ResolvedConfig


@dataclass(frozen=True)
class SnapshotMetrics:
    checkpoint_path: str
    step: int
    schema_compliance: float
    greedy_pass_at_1: float
    sampled_pass_at_1: float
    parser_failure_rate: float


def pick_best_snapshot(
    snapshots: list[SnapshotMetrics],
    *,
    primary_metric: str,
    secondary_metric: str,
) -> SnapshotMetrics:
    if len(snapshots) == 0:
        raise ValueError("Cannot select best checkpoint from empty snapshots")

    def metric_value(snapshot: SnapshotMetrics, name: str) -> float:
        if name == "schema_compliance":
            return snapshot.schema_compliance
        if name == "greedy_pass_at_1":
            return snapshot.greedy_pass_at_1
        if name == "sampled_pass_at_1":
            return snapshot.sampled_pass_at_1
        if name == "parser_failure_rate":
            return -snapshot.parser_failure_rate
        raise ValueError(f"Unsupported checkpoint metric: {name}")

    ranked = sorted(
        snapshots,
        key=lambda item: (
            metric_value(item, primary_metric),
            metric_value(item, secondary_metric),
            item.step,
        ),
        reverse=True,
    )
    return ranked[0]


def gate_sft_ready(
    *,
    config: ResolvedConfig,
    final_schema_compliance: float,
    final_greedy_pass_at_1: float,
) -> tuple[bool, str]:
    schema_ok = final_schema_compliance >= config.train_sft.gates.schema_min
    pass_delta = final_greedy_pass_at_1 - config.train_sft.gates.baseline_greedy_pass_at_1
    pass_ok = pass_delta >= 0.0

    if schema_ok and pass_ok:
        return True, "gate_passed"

    reasons: list[str] = []
    if not schema_ok:
        reasons.append(
            f"schema_compliance {final_schema_compliance:.4f} < {config.train_sft.gates.schema_min:.4f}"
        )
    if not pass_ok:
        reasons.append(
            "greedy_pass_at_1 delta "
            f"{pass_delta:.4f} < 0.0000 vs baseline {config.train_sft.gates.baseline_greedy_pass_at_1:.4f}"
        )
    return False, "; ".join(reasons)


def snapshot_from_dict(payload: dict[str, Any]) -> SnapshotMetrics:
    return SnapshotMetrics(
        checkpoint_path=str(payload["checkpoint_path"]),
        step=int(payload["step"]),
        schema_compliance=float(payload["schema_compliance"]),
        greedy_pass_at_1=float(payload["greedy_pass_at_1"]),
        sampled_pass_at_1=float(payload["sampled_pass_at_1"]),
        parser_failure_rate=float(payload["parser_failure_rate"]),
    )
