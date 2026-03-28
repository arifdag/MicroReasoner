from __future__ import annotations

from microreasoner.final_eval.report import build_final_report_markdown


def test_build_final_report_markdown_handles_missing_metrics_entries() -> None:
    payload = {
        "session_id": "partial-session",
        "status": "partial",
        "strict_claims_ok": False,
        "inputs": {
            "config_path": "configs/defaults.yaml",
            "dataset_dir": "artifacts/eval",
            "base_checkpoint": "base",
            "sft_checkpoint": "sft",
            "grpo_checkpoint": "grpo",
            "seed": 42,
            "max_items": 8,
        },
        "models": {
            "base": {
                "status": "failed",
                "wallclock_seconds": 1.0,
                "metrics": None,
            },
            "sft": {
                "status": "success",
                "wallclock_seconds": 2.0,
                "metrics": {
                    "macro_greedy_pass_at_1": 0.5,
                    "macro_sampled_pass_at_1": 0.6,
                    "schema_compliance_rate": 1.0,
                    "parser_failure_rate": 0.0,
                    "greedy_solved": 4,
                    "sampled_solved": 5,
                    "cost_per_solved_greedy": 0.5,
                    "cost_per_solved_sampled": 0.4,
                },
            },
            "grpo": {
                "status": "skipped",
                "wallclock_seconds": 0.0,
                "metrics": None,
            },
        },
        "comparisons": {
            "sft_vs_base": None,
            "grpo_vs_sft": None,
            "grpo_vs_base": None,
        },
        "failure_reasons": ["base: validation failed"],
    }

    report = build_final_report_markdown(payload)

    assert "# Final Evaluation Report" in report
    assert "| base | failed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0 |" in report
    assert "| sft | success | 0.5000 | 0.6000 | 1.0000 | 0.0000 | 2.0 |" in report
    assert "| grpo | skipped | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |" in report
