from __future__ import annotations

import json
from pathlib import Path

from microreasoner.cli.main import main


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


def test_train_grpo_fails_when_reward_variance_gate_is_too_strict(tmp_path: Path, monkeypatch) -> None:
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
                "benchmark": "math",
                "question": "2+2?",
                "think": "add",
                "answer_boxed": "4",
                "metadata": {"difficulty": "medium"},
            },
            {
                "id": "r13",
                "benchmark": "gsm8k",
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

    build_code = main(
        [
            "data",
            "build-rl",
            "--config",
            str(config_path),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(tmp_path / "datasets"),
            "--run-id",
            "build-rl-gate",
        ]
    )
    assert build_code == 0

    build_summary = _read_json(tmp_path / "artifacts" / "runs" / "build-rl-gate" / "summary.json")
    dataset_manifest = Path(build_summary["artifacts"]["dataset_manifest_path"])

    init_checkpoint = tmp_path / "init-checkpoint"
    init_checkpoint.mkdir(parents=True, exist_ok=True)
    (init_checkpoint / "trainer_state.json").write_text(
        json.dumps({"step": 0, "backend": "fixture"}),
        encoding="utf-8",
    )

    run_root = tmp_path / "runs"
    run_id = "grpo-gate-fail"
    code = main(
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
            "--set",
            "train_grpo.gates.min_reward_std=10.0",
        ]
    )
    assert code == 1

    run_dir = run_root / run_id
    assert run_dir.exists()
    summary = _read_json(run_dir / "summary.json")
    assert summary["status"] == "failed"
    run_manifest = _read_json(run_dir / "run_manifest.json")
    assert run_manifest["status"] == "failed"
    assert "rolling_reward_std" in str(run_manifest["failure_reason"])
