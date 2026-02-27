from __future__ import annotations

import json
from pathlib import Path

from microreasoner.ablation.runner import run_ablation_suite
from microreasoner.ablation.types import ExperimentSpec


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _setup_fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_dir = tmp_path / "raw"
    _write_jsonl(
        source_dir / "source.jsonl",
        [
            {"id": "r1", "benchmark": "gsm8k", "question": "1+1?", "think": "add", "answer_boxed": "2"},
            {"id": "r2", "benchmark": "gsm8k", "question": "2+2?", "think": "add", "answer_boxed": "4"},
            {"id": "r3", "benchmark": "math", "question": "3+3?", "think": "add", "answer_boxed": "6"},
            {"id": "r13", "benchmark": "math", "question": "4+4?", "think": "add", "answer_boxed": "8"},
        ],
    )

    eval_dir = tmp_path / "eval"
    _write_jsonl(
        eval_dir / "gsm8k_eval.jsonl",
        [
            {
                "id": "g1",
                "question": "1+1?",
                "answer": "2",
                "mock_greedy_response": "<think>ok</think><answer>\\boxed{2}</answer>",
                "mock_sampled_responses": [
                    "<think>x</think><answer>\\boxed{0}</answer>",
                    "<think>ok</think><answer>\\boxed{2}</answer>",
                ],
            }
        ],
    )
    _write_jsonl(
        eval_dir / "math_eval.jsonl",
        [
            {
                "id": "m1",
                "question": "2+2?",
                "answer": "4",
                "mock_greedy_response": "<think>ok</think><answer>\\boxed{4}</answer>",
                "mock_sampled_responses": [
                    "<think>x</think><answer>\\boxed{0}</answer>",
                    "<think>ok</think><answer>\\boxed{4}</answer>",
                ],
            }
        ],
    )

    config_path = tmp_path / "config.yaml"
    _write_text(
        config_path,
        "\n".join(
            [
                "data_pipeline:",
                "  input_sources:",
                "    - name: source",
                "      adapter: canonical_jsonl",
                "      path: source.jsonl",
                "evaluation:",
                "  datasets:",
                "    gsm8k:",
                "      path: gsm8k_eval.jsonl",
                "    math:",
                "      path: math_eval.jsonl",
                "",
            ]
        ),
    )
    return config_path, source_dir, eval_dir


def test_run_ablation_suite_fixture_generates_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path, source_dir, eval_dir = _setup_fixture_inputs(tmp_path)

    monkeypatch.setattr(
        "microreasoner.ablation.runner.build_experiment_specs",
        lambda: [
            ExperimentSpec("exp_sft_only", "baseline", "baseline", "sft_only", ()),
            ExperimentSpec(
                "exp_loss_dr",
                "loss",
                "dr",
                "grpo",
                ("train_grpo.algo.loss_type=dr_grpo", "train_grpo.algo.scale_rewards=batch"),
            ),
            ExperimentSpec("exp_schema_020", "schema_weight", "alias", "grpo", (), alias_of="exp_loss_dr"),
            ExperimentSpec("exp_rs_sft", "alternative", "rs", "rs_sft", ()),
        ],
    )

    result = run_ablation_suite(
        config_path=config_path,
        source_dir=source_dir,
        dataset_dir=eval_dir,
        output_root=tmp_path / "ablation_out",
        report_dir=tmp_path / "reports",
        mode="fixture",
        seed=42,
        max_items=8,
        skip_existing=False,
        fail_fast=True,
        session_id="ablation-fixture-test",
    )
    assert result.csv_path.exists()
    assert result.markdown_path.exists()
    assert len(result.rows) == 4
    assert any(row.experiment_id == "exp_rs_sft" for row in result.rows)


def test_run_ablation_suite_skip_existing_reuses_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path, source_dir, eval_dir = _setup_fixture_inputs(tmp_path)
    monkeypatch.setattr(
        "microreasoner.ablation.runner.build_experiment_specs",
        lambda: [
            ExperimentSpec("exp_sft_only", "baseline", "baseline", "sft_only", ()),
            ExperimentSpec(
                "exp_loss_dr",
                "loss",
                "dr",
                "grpo",
                ("train_grpo.algo.loss_type=dr_grpo", "train_grpo.algo.scale_rewards=batch"),
            ),
        ],
    )
    session_id = "ablation-skip-test"

    first = run_ablation_suite(
        config_path=config_path,
        source_dir=source_dir,
        dataset_dir=eval_dir,
        output_root=tmp_path / "ablation_out",
        report_dir=tmp_path / "reports",
        mode="fixture",
        seed=42,
        max_items=8,
        skip_existing=False,
        fail_fast=True,
        session_id=session_id,
    )
    assert first.csv_path.exists()

    train_summary = (
        tmp_path
        / "ablation_out"
        / "runs"
        / f"{session_id}-exp_loss_dr-train"
        / "summary.json"
    )
    first_mtime = train_summary.stat().st_mtime

    second = run_ablation_suite(
        config_path=config_path,
        source_dir=source_dir,
        dataset_dir=eval_dir,
        output_root=tmp_path / "ablation_out",
        report_dir=tmp_path / "reports",
        mode="fixture",
        seed=42,
        max_items=8,
        skip_existing=True,
        fail_fast=True,
        session_id=session_id,
    )
    assert second.csv_path.exists()
    second_mtime = train_summary.stat().st_mtime
    assert second_mtime == first_mtime

