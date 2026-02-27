from __future__ import annotations

from pathlib import Path

import pytest

from microreasoner.final_eval.runner import run_final_evaluation


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_final_eval_requires_eval_dataset_files(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        run_final_evaluation(
            config_path=config_path,
            dataset_dir=tmp_path / "missing-eval",
            base_checkpoint=checkpoint,
            sft_checkpoint=checkpoint,
            grpo_checkpoint=checkpoint,
            output_root=tmp_path / "out",
            report_dir=tmp_path / "reports",
            seed=42,
            max_items=8,
            mode="fixture",
            skip_existing=False,
            fail_fast=True,
            strict_claims=False,
            session_id="inputs-test",
        )


def test_final_eval_requires_checkpoint_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")
    eval_dir = tmp_path / "eval"
    _write_text(eval_dir / "gsm8k_eval.jsonl", "")
    _write_text(eval_dir / "math_eval.jsonl", "")

    with pytest.raises(ValueError):
        run_final_evaluation(
            config_path=config_path,
            dataset_dir=eval_dir,
            base_checkpoint=tmp_path / "missing_base",
            sft_checkpoint=tmp_path / "missing_sft",
            grpo_checkpoint=tmp_path / "missing_grpo",
            output_root=tmp_path / "out",
            report_dir=tmp_path / "reports",
            seed=42,
            max_items=8,
            mode="fixture",
            skip_existing=False,
            fail_fast=True,
            strict_claims=False,
            session_id="inputs-test-2",
        )

