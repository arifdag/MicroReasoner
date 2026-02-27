from __future__ import annotations

import json
from pathlib import Path

from microreasoner.cli.main import main
from microreasoner.contracts.validation import validate_run_dir


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_train_grpo_cli_end_to_end_fixture_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "raw"
    _write_jsonl(
        source_dir / "source.jsonl",
        [
            {
                "id": "r1",
                "benchmark": "gsm8k",
                "question": "1+1?",
                "think": "add",
                "answer_boxed": "2",
                "metadata": {"difficulty": "easy"},
            },
            {
                "id": "r2",
                "benchmark": "gsm8k",
                "question": "2+3?",
                "think": "add",
                "answer_boxed": "5",
                "metadata": {"difficulty": "easy"},
            },
            {
                "id": "r3",
                "benchmark": "math",
                "question": "6-1?",
                "think": "subtract",
                "answer_boxed": "5",
                "metadata": {"difficulty": "medium"},
            },
            {
                "id": "r6",
                "benchmark": "math",
                "question": "3*3?",
                "think": "multiply",
                "answer_boxed": "9",
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
                "    train_ratio: 0.75",
                "    val_ratio: 0.25",
                "    seed: 11",
                "train_grpo:",
                "  backend:",
                "    trainer: fixture",
                "  run:",
                "    max_steps: 6",
                "    eval_every_steps: 2",
                "  algo:",
                "    group_size: 4",
                "",
            ]
        ),
    )

    dataset_root = tmp_path / "datasets"
    build_code = main(
        [
            "data",
            "build-rl",
            "--config",
            str(config_path),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(dataset_root),
            "--run-id",
            "build-rl-run",
        ]
    )
    assert build_code == 0

    build_summary = _read_json(tmp_path / "artifacts" / "runs" / "build-rl-run" / "summary.json")
    dataset_manifest = Path(build_summary["artifacts"]["dataset_manifest_path"])
    assert dataset_manifest.exists()

    init_checkpoint = tmp_path / "init-checkpoint"
    init_checkpoint.mkdir(parents=True, exist_ok=True)
    (init_checkpoint / "trainer_state.json").write_text(
        json.dumps({"step": 0, "backend": "fixture"}),
        encoding="utf-8",
    )

    run_root = tmp_path / "runs"
    run_id = "grpo-run"
    train_code = main(
        [
            "train",
            "grpo",
            "--config",
            str(config_path),
            "--dataset-manifest",
            str(dataset_manifest),
            "--init-checkpoint",
            str(init_checkpoint),
            "--run-id",
            run_id,
            "--output-dir",
            str(run_root),
            "--set",
            "train_grpo.backend.trainer=fixture",
        ]
    )
    assert train_code == 0

    run_dir = run_root / run_id
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "checkpoints.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "dataset_manifest.json").exists()
    assert (run_dir / "metrics_history.jsonl").exists()
    assert (run_dir / "reward_history.jsonl").exists()
    assert (run_dir / "curriculum_trace.jsonl").exists()

    validation = validate_run_dir(run_dir)
    assert validation.ok, validation.errors
