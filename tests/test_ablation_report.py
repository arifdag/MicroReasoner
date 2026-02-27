from __future__ import annotations

from pathlib import Path

from microreasoner.ablation.report import build_rows, write_csv, write_markdown
from microreasoner.ablation.types import CostSnapshot, ExperimentOutcome, MetricSnapshot, RunArtifacts


def test_ablation_report_writers_emit_csv_and_markdown(tmp_path: Path) -> None:
    outcomes = [
        ExperimentOutcome(
            experiment_id="exp_sft_only",
            family="baseline",
            description="baseline",
            status="success",
            notes="ok",
            artifacts=RunArtifacts(train_run_dir=tmp_path / "sft", eval_run_dir=tmp_path / "sft_eval"),
            metrics=MetricSnapshot(
                greedy_pass_at_1=0.5,
                sampled_pass_at_1=0.6,
                schema_compliance_rate=0.99,
                parser_failure_rate=0.01,
                eval_examples=10,
            ),
            cost=CostSnapshot(wallclock_seconds=1.5, train_steps=0),
        ),
        ExperimentOutcome(
            experiment_id="exp_loss_dr",
            family="loss",
            description="dr",
            status="success",
            notes="ok",
            artifacts=RunArtifacts(train_run_dir=tmp_path / "grpo", eval_run_dir=tmp_path / "grpo_eval"),
            metrics=MetricSnapshot(
                greedy_pass_at_1=0.6,
                sampled_pass_at_1=0.7,
                schema_compliance_rate=0.995,
                parser_failure_rate=0.008,
                eval_examples=10,
            ),
            cost=CostSnapshot(wallclock_seconds=2.0, train_steps=6),
        ),
    ]
    rows = build_rows(
        outcomes,
        backend_mode="fixture",
        sft_baseline_id="exp_sft_only",
        sft_run_dir=tmp_path / "sft",
    )
    assert len(rows) == 2
    assert rows[1].delta_greedy_vs_sft > 0.0

    csv_path = tmp_path / "ablation_results.csv"
    md_path = tmp_path / "ablation_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, metadata={"mode": "fixture"})

    assert csv_path.exists()
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Ablation Summary" in content
    assert "exp_loss_dr" in content

