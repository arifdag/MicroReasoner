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


def test_train_sft_cli_end_to_end_fixture_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "raw"
    _write_jsonl(
        source_dir / "source.jsonl",
        [
            {
                "id": "x1",
                "benchmark": "gsm8k",
                "question": "1+2?",
                "think": "add numbers",
                "answer_boxed": "3",
            },
            {
                "id": "x2",
                "benchmark": "math",
                "question": "2+3?",
                "think": "add numbers",
                "answer_boxed": "5",
            },
            {
                "id": "x3",
                "benchmark": "gsm8k",
                "question": "3+3?",
                "think": "add numbers",
                "answer_boxed": "6",
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
                "    seed: 9",
                "train_sft:",
                "  backend:",
                "    trainer: fixture",
                "  run:",
                "    max_steps: 6",
                "    eval_every_steps: 2",
                "  mode: qlora",
                "",
            ]
        ),
    )

    dataset_root = tmp_path / "datasets"
    build_code = main(
        [
            "data",
            "build-sft",
            "--config",
            str(config_path),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(dataset_root),
            "--run-id",
            "build-sft-run",
        ]
    )
    assert build_code == 0

    build_summary = _read_json(tmp_path / "artifacts" / "runs" / "build-sft-run" / "summary.json")
    dataset_manifest = Path(build_summary["artifacts"]["dataset_manifest_path"])
    assert dataset_manifest.exists()

    run_root = tmp_path / "runs"
    run_id = "sft-run"
    train_code = main(
        [
            "train",
            "sft",
            "--config",
            str(config_path),
            "--dataset-manifest",
            str(dataset_manifest),
            "--run-id",
            run_id,
            "--output-dir",
            str(run_root),
            "--set",
            "train_sft.backend.trainer=fixture",
        ]
    )
    assert train_code == 0

    run_dir = run_root / run_id
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "checkpoints.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "dataset_manifest.json").exists()
    assert (run_dir / "metrics_history.jsonl").exists()

    validation = validate_run_dir(run_dir)
    assert validation.ok, validation.errors
