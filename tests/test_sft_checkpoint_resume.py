from __future__ import annotations

import json
from pathlib import Path

from microreasoner.data.build_sft_dataset import build_sft_dataset
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root
from microreasoner.train.sft_data import load_sft_train_input
from microreasoner.train.sft_trainer import run_sft_training


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
                "id": "s1",
                "benchmark": "gsm8k",
                "question": "1+1?",
                "think": "add",
                "answer_boxed": "2",
            },
            {
                "id": "s2",
                "benchmark": "math",
                "question": "2+2?",
                "think": "add",
                "answer_boxed": "4",
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
                "    train_ratio: 0.5",
                "    val_ratio: 0.5",
                "    seed: 7",
                "train_sft:",
                "  backend:",
                "    trainer: fixture",
                "  run:",
                "    max_steps: 4",
                "    eval_every_steps: 2",
                "",
            ]
        ),
    )
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    build_result = build_sft_dataset(config=config, output_root=tmp_path / "datasets", source_dir=source_dir)
    return config_path, Path(build_result.manifest_path)


def test_sft_training_resume_fixture_backend(tmp_path: Path) -> None:
    config_path, dataset_manifest = _prepare_config_and_manifest(tmp_path)
    config = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)
    train_input = load_sft_train_input(dataset_manifest, max_eval_samples=32)

    run_dir = tmp_path / "run"
    first = run_sft_training(
        config=config,
        train_input=train_input,
        run_dir=run_dir,
        max_steps_override=4,
        eval_every_steps_override=2,
    )
    assert first.latest_checkpoint.exists()
    assert first.resume_test_passed

    second = run_sft_training(
        config=config,
        train_input=train_input,
        run_dir=run_dir,
        resume_from=first.latest_checkpoint,
        max_steps_override=8,
        eval_every_steps_override=2,
    )
    assert second.resume_test_passed
    assert second.global_step == 8
    assert second.latest_checkpoint.exists()

