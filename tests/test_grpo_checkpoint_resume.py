from __future__ import annotations

import json
from pathlib import Path

from microreasoner.data.build_rl_dataset import build_rl_dataset
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root
from microreasoner.train.grpo_data import load_grpo_train_input
from microreasoner.train.grpo_trainer import run_grpo_training


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_config_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "raw"
    _write_jsonl(
        source_dir / "source.jsonl",
        [
            {
                "id": "r1",
                "benchmark": "gsm8k",
                "question": "1+2?",
                "think": "add",
                "answer_boxed": "3",
                "metadata": {"difficulty": "easy"},
            },
            {
                "id": "r2",
                "benchmark": "gsm8k",
                "question": "2+2?",
                "think": "add",
                "answer_boxed": "4",
                "metadata": {"difficulty": "easy"},
            },
            {
                "id": "r3",
                "benchmark": "math",
                "question": "3+3?",
                "think": "add",
                "answer_boxed": "6",
                "metadata": {"difficulty": "medium"},
            },
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
                "  split:",
                "    strategy: hash",
                "    train_ratio: 0.67",
                "    val_ratio: 0.33",
                "    seed: 7",
                "train_grpo:",
                "  backend:",
                "    trainer: fixture",
                "  run:",
                "    max_steps: 4",
                "    eval_every_steps: 2",
                "  algo:",
                "    group_size: 4",
                "",
            ]
        ),
    )
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    build_result = build_rl_dataset(config=config, output_root=tmp_path / "datasets", source_dir=source_dir)
    return config_path, Path(build_result.manifest_path)


def test_grpo_training_resume_fixture_backend(tmp_path: Path) -> None:
    config_path, dataset_manifest = _prepare_config_and_manifest(tmp_path)
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    train_input = load_grpo_train_input(
        dataset_manifest,
        max_eval_samples=config.train_grpo.run.max_eval_samples,
    )

    init_checkpoint = tmp_path / "init-checkpoint"
    init_checkpoint.mkdir(parents=True, exist_ok=True)
    (init_checkpoint / "trainer_state.json").write_text(
        json.dumps({"step": 0, "backend": "fixture"}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    first = run_grpo_training(
        config=config,
        train_input=train_input,
        run_dir=run_dir,
        init_checkpoint=init_checkpoint,
        max_steps_override=4,
        eval_every_steps_override=2,
    )
    assert first.latest_checkpoint.exists()
    assert first.resume_test_passed

    second = run_grpo_training(
        config=config,
        train_input=train_input,
        run_dir=run_dir,
        init_checkpoint=init_checkpoint,
        resume_from=first.latest_checkpoint,
        max_steps_override=8,
        eval_every_steps_override=2,
    )
    assert second.resume_test_passed
    assert second.global_step == 8
    assert second.latest_checkpoint.exists()
