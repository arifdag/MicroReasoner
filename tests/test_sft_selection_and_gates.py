from __future__ import annotations

from pathlib import Path

from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root
from microreasoner.train.sft_selection import SnapshotMetrics, gate_sft_ready, pick_best_snapshot


def test_pick_best_snapshot_schema_then_accuracy() -> None:
    snapshots = [
        SnapshotMetrics(
            checkpoint_path="c1",
            step=100,
            schema_compliance=0.99,
            greedy_pass_at_1=0.40,
            sampled_pass_at_1=0.42,
            parser_failure_rate=0.01,
        ),
        SnapshotMetrics(
            checkpoint_path="c2",
            step=120,
            schema_compliance=0.98,
            greedy_pass_at_1=0.90,
            sampled_pass_at_1=0.91,
            parser_failure_rate=0.00,
        ),
        SnapshotMetrics(
            checkpoint_path="c3",
            step=130,
            schema_compliance=0.99,
            greedy_pass_at_1=0.70,
            sampled_pass_at_1=0.75,
            parser_failure_rate=0.01,
        ),
    ]
    best = pick_best_snapshot(
        snapshots,
        primary_metric="schema_compliance",
        secondary_metric="greedy_pass_at_1",
    )
    assert best.checkpoint_path == "c3"


def test_gate_sft_ready_enforces_schema_and_delta(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("train_sft:\n  gates:\n    baseline_greedy_pass_at_1: 0.5\n", encoding="utf-8")
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)

    passed, _ = gate_sft_ready(
        config=config,
        final_schema_compliance=0.99,
        final_greedy_pass_at_1=0.51,
    )
    assert passed

    failed_schema, reason_schema = gate_sft_ready(
        config=config,
        final_schema_compliance=0.97,
        final_greedy_pass_at_1=0.51,
    )
    assert not failed_schema
    assert "schema_compliance" in reason_schema

    failed_delta, reason_delta = gate_sft_ready(
        config=config,
        final_schema_compliance=0.99,
        final_greedy_pass_at_1=0.49,
    )
    assert not failed_delta
    assert "delta" in reason_delta

