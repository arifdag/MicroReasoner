from __future__ import annotations

import json
from pathlib import Path

from microreasoner.final_eval.runner import run_final_evaluation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")

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

    base_ckpt = tmp_path / "base_ckpt"
    sft_ckpt = tmp_path / "sft_ckpt"
    grpo_ckpt = tmp_path / "grpo_ckpt"
    base_ckpt.mkdir(parents=True, exist_ok=True)
    sft_ckpt.mkdir(parents=True, exist_ok=True)
    grpo_ckpt.mkdir(parents=True, exist_ok=True)
    return config_path, eval_dir, base_ckpt, sft_ckpt, grpo_ckpt


def test_run_final_evaluation_fixture_generates_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path, eval_dir, base_ckpt, sft_ckpt, grpo_ckpt = _prepare_inputs(tmp_path)
    result = run_final_evaluation(
        config_path=config_path,
        dataset_dir=eval_dir,
        base_checkpoint=base_ckpt,
        sft_checkpoint=sft_ckpt,
        grpo_checkpoint=grpo_ckpt,
        output_root=tmp_path / "final_out",
        report_dir=tmp_path / "reports",
        seed=42,
        max_items=8,
        mode="fixture",
        skip_existing=False,
        fail_fast=True,
        strict_claims=False,
        session_id="final-fixture-test",
    )
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    assert result.error_analysis_path.exists()
    assert result.status == "success"
    assert result.strict_claims_ok


def test_run_final_evaluation_skip_existing_reuses_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path, eval_dir, base_ckpt, sft_ckpt, grpo_ckpt = _prepare_inputs(tmp_path)
    session_id = "final-skip-test"

    first = run_final_evaluation(
        config_path=config_path,
        dataset_dir=eval_dir,
        base_checkpoint=base_ckpt,
        sft_checkpoint=sft_ckpt,
        grpo_checkpoint=grpo_ckpt,
        output_root=tmp_path / "final_out",
        report_dir=tmp_path / "reports",
        seed=42,
        max_items=8,
        mode="fixture",
        skip_existing=False,
        fail_fast=True,
        strict_claims=False,
        session_id=session_id,
    )
    assert first.metrics_path.exists()
    summary = (
        tmp_path / "final_out" / "runs" / f"{session_id}-grpo-eval" / "summary.json"
    )
    first_mtime = summary.stat().st_mtime

    second = run_final_evaluation(
        config_path=config_path,
        dataset_dir=eval_dir,
        base_checkpoint=base_ckpt,
        sft_checkpoint=sft_ckpt,
        grpo_checkpoint=grpo_ckpt,
        output_root=tmp_path / "final_out",
        report_dir=tmp_path / "reports",
        seed=42,
        max_items=8,
        mode="fixture",
        skip_existing=True,
        fail_fast=True,
        strict_claims=False,
        session_id=session_id,
    )
    assert second.metrics_path.exists()
    second_mtime = summary.stat().st_mtime
    assert second_mtime == first_mtime

