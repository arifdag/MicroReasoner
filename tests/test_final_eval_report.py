from __future__ import annotations

import json
from pathlib import Path

from microreasoner.final_eval.report import (
    build_final_report_markdown,
    write_final_metrics_json,
    write_final_report_markdown,
)


def _sample_payload(tmp_path: Path) -> dict:
    return {
        "schema_version": "1.0.0",
        "session_id": "final-test",
        "inputs": {
            "config_path": str(tmp_path / "config.yaml"),
            "dataset_dir": str(tmp_path / "eval"),
            "base_checkpoint": str(tmp_path / "base"),
            "sft_checkpoint": str(tmp_path / "sft"),
            "grpo_checkpoint": str(tmp_path / "grpo"),
            "seed": 42,
            "max_items": 8,
        },
        "models": {
            "base": {
                "status": "success",
                "wallclock_seconds": 1.0,
                "metrics": {
                    "macro_greedy_pass_at_1": 0.4,
                    "macro_sampled_pass_at_1": 0.5,
                    "schema_compliance_rate": 0.99,
                    "parser_failure_rate": 0.01,
                    "greedy_solved": 4,
                    "sampled_solved": 5,
                    "cost_per_solved_greedy": 0.25,
                    "cost_per_solved_sampled": 0.2,
                },
            },
            "sft": {
                "status": "success",
                "wallclock_seconds": 1.2,
                "metrics": {
                    "macro_greedy_pass_at_1": 0.5,
                    "macro_sampled_pass_at_1": 0.6,
                    "schema_compliance_rate": 0.995,
                    "parser_failure_rate": 0.008,
                    "greedy_solved": 5,
                    "sampled_solved": 6,
                    "cost_per_solved_greedy": 0.24,
                    "cost_per_solved_sampled": 0.2,
                },
            },
            "grpo": {
                "status": "success",
                "wallclock_seconds": 1.3,
                "metrics": {
                    "macro_greedy_pass_at_1": 0.55,
                    "macro_sampled_pass_at_1": 0.65,
                    "schema_compliance_rate": 0.996,
                    "parser_failure_rate": 0.007,
                    "greedy_solved": 6,
                    "sampled_solved": 7,
                    "cost_per_solved_greedy": 0.216,
                    "cost_per_solved_sampled": 0.186,
                },
            },
        },
        "comparisons": {
            "sft_vs_base": {
                "delta_macro_greedy_pass_at_1": 0.1,
                "delta_macro_sampled_pass_at_1": 0.1,
                "delta_schema_compliance_rate": 0.005,
                "delta_parser_failure_rate": -0.002,
            },
            "grpo_vs_sft": {
                "delta_macro_greedy_pass_at_1": 0.05,
                "delta_macro_sampled_pass_at_1": 0.05,
                "delta_schema_compliance_rate": 0.001,
                "delta_parser_failure_rate": -0.001,
            },
            "grpo_vs_base": {
                "delta_macro_greedy_pass_at_1": 0.15,
                "delta_macro_sampled_pass_at_1": 0.15,
                "delta_schema_compliance_rate": 0.006,
                "delta_parser_failure_rate": -0.003,
            },
        },
        "strict_claims_ok": True,
        "status": "success",
        "failure_reasons": [],
    }


def test_final_eval_report_writers(tmp_path: Path) -> None:
    payload = _sample_payload(tmp_path)
    metrics_path = tmp_path / "final_metrics.json"
    report_path = tmp_path / "final_report.md"

    write_final_metrics_json(payload, metrics_path)
    write_final_report_markdown(payload, report_path)
    content = build_final_report_markdown(payload)

    assert metrics_path.exists()
    parsed = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert parsed["status"] == "success"
    assert report_path.exists()
    assert "Final Evaluation Report" in content
    assert "grpo_vs_sft" in content

