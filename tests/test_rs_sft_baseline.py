from __future__ import annotations

from pathlib import Path

from microreasoner.ablation.baseline_rs_sft import build_rs_sft_manifest
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root
from microreasoner.train.grpo_data import GRPOTrainInput, RLRecordItem


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_rs_sft_manifest_outputs_valid_files(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")
    resolved = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)

    records = (
        RLRecordItem(
            record_id="r1",
            prompt="Problem: 1+1",
            gold_answer="2",
            benchmark="gsm8k",
            source_name="src",
            difficulty_tag="easy",
            curriculum_stage="gsm8k_heavy",
        ),
        RLRecordItem(
            record_id="r2",
            prompt="Problem: 2+2",
            gold_answer="4",
            benchmark="math",
            source_name="src",
            difficulty_tag="medium",
            curriculum_stage="gsm8k_math_mixed",
        ),
    )
    train_input = GRPOTrainInput(
        dataset_id="rl_dataset",
        train_records=records,
        val_records=records,
        manifest_path=tmp_path / "rl_manifest.json",
        manifest={},
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
    )

    manifest_path, result = build_rs_sft_manifest(
        config=resolved,
        train_input=train_input,
        output_root=tmp_path / "rs",
        candidates_per_prompt=4,
        strict_boxed_only=True,
    )
    assert manifest_path.exists()
    assert result.accepted_count >= 2

