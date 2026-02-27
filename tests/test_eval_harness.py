from __future__ import annotations

import json
from pathlib import Path

from microreasoner.cli.main import main
from microreasoner.contracts.validation import validate_run_dir
from microreasoner.eval.parser import parse_response


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


def test_parse_response_strict_boxed() -> None:
    text = "<think>step by step</think><answer>\\boxed{42}</answer>"
    parsed = parse_response(text, strict_boxed_only=True)
    assert parsed.parse_ok
    assert parsed.schema_ok
    assert parsed.boxed_answer == "42"


def test_parse_response_multiple_boxed_fails() -> None:
    text = "<think>reason</think><answer>\\boxed{1} and \\boxed{2}</answer>"
    parsed = parse_response(text, strict_boxed_only=True)
    assert not parsed.parse_ok
    assert parsed.reason == "multiple_boxed"


def test_eval_command_with_fixture_backend_produces_metrics_and_passes_validation(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    _write_jsonl(
        dataset_dir / "gsm8k_eval.jsonl",
        [
            {
                "id": "g1",
                "question": "What is 1+1?",
                "answer": "2",
                "mock_greedy_response": "<think>add numbers</think><answer>\\boxed{2}</answer>",
                "mock_sampled_responses": [
                    "<think>bad</think><answer>\\boxed{0}</answer>"
                ]
                * 31
                + ["<think>good</think><answer>\\boxed{2}</answer>"],
            }
        ],
    )
    _write_jsonl(
        dataset_dir / "math_eval.jsonl",
        [
            {
                "id": "m1",
                "question": "Compute 3*3",
                "answer": "9",
                "mock_greedy_response": "<think>multiply</think><answer>\\boxed{9}</answer>",
                "mock_sampled_responses": [
                    "<think>multiply</think><answer>\\boxed{9}</answer>"
                ]
                * 32,
            }
        ],
    )

    config_path = tmp_path / "eval_config.yaml"
    _write_text(
        config_path,
        "\n".join(
            [
                "evaluation:",
                "  inference:",
                "    backend: fixture",
                "  datasets:",
                "    gsm8k:",
                "      path: gsm8k_eval.jsonl",
                "    math:",
                "      path: math_eval.jsonl",
                "",
            ]
        ),
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    run_root = tmp_path / "runs"
    run_id = "eval-success"
    code = main(
        [
            "eval",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--dataset-dir",
            str(dataset_dir),
            "--run-id",
            run_id,
            "--output-dir",
            str(run_root),
            "--seed",
            "123",
        ]
    )
    assert code == 0

    run_dir = run_root / run_id
    metrics = _read_json(run_dir / "metrics.json")
    assert metrics["accuracy"]["gsm8k"]["greedy_pass_at_1"] == 1.0
    assert metrics["accuracy"]["gsm8k"]["sampled_pass_at_1"] == 1.0
    assert metrics["accuracy"]["math"]["greedy_pass_at_1"] == 1.0
    assert metrics["schema"]["compliance_rate"] == 1.0
    assert metrics["parser"]["extraction_failure_rate"] == 0.0
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "dataset_manifest.json").exists()
    assert (run_dir / "checkpoints.json").exists()

    validation = validate_run_dir(run_dir)
    assert validation.ok, validation.errors

