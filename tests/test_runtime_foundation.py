from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from microreasoner.cli.main import main
from microreasoner.runtime.configuration import (
    RuntimeConfigError,
    apply_overrides,
    deep_merge,
    parse_override,
)
from microreasoner.runtime.context import generate_run_id


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_config_merge_and_override_precedence() -> None:
    defaults = {"evaluation": {"sampled": {"num_samples": 32, "temperature": 0.6}}}
    user = {"evaluation": {"sampled": {"num_samples": 16}}}
    merged = deep_merge(defaults, user)
    final = apply_overrides(merged, ["evaluation.sampled.num_samples=8"])
    assert final["evaluation"]["sampled"]["temperature"] == 0.6
    assert final["evaluation"]["sampled"]["num_samples"] == 8


def test_override_parser_handles_scalar_types() -> None:
    keys, value = parse_override("evaluation.sampled.top_p=0.9")
    assert keys == ["evaluation", "sampled", "top_p"]
    assert value == pytest.approx(0.9)

    _, value_bool = parse_override("gates.strict_qa=false")
    assert value_bool is False

    _, value_int = parse_override("evaluation.sampled.num_samples=64")
    assert value_int == 64


def test_unknown_override_path_raises() -> None:
    with pytest.raises(RuntimeConfigError):
        apply_overrides({"evaluation": {}}, ["evaluation.unknown.key=1"])


def test_run_id_format() -> None:
    run_id = generate_run_id("train-sft")
    assert re.match(r"^train-sft-\d{8}T\d{6}Z-[0-9a-f]{8}$", run_id)


def test_train_grpo_missing_init_checkpoint_returns_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")
    dataset_manifest = tmp_path / "dataset_manifest.json"
    _write_text(
        dataset_manifest,
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_type": "rl",
                "dataset_id": "abc12345def67890",
                "build_timestamp": "2026-02-27T00:00:00Z",
                "seed": 42,
                "inputs": [
                    {
                        "name": "source",
                        "adapter": "canonical_jsonl",
                        "path": "source.jsonl",
                        "resolved_path": "source.jsonl",
                        "hash": "a" * 64,
                    }
                ],
                "filters": {
                    "min_think_tokens": 1,
                    "max_think_tokens": 1200,
                    "require_single_boxed_answer": True,
                    "drop_duplicates": True,
                },
                "split_counts": {"train": 1, "val": 1},
                "reject_stats": {},
                "artifact_paths": {"train": "train.jsonl", "val": "val.jsonl", "manifest": "manifest.json"},
                "artifact_hashes": {"train": "b" * 64, "val": "c" * 64},
            },
        ),
    )
    output_root = tmp_path / "runs"
    run_id = "unit-train-grpo"
    missing_init = tmp_path / "missing-checkpoint"

    code = main(
        [
            "train",
            "grpo",
            "--config",
            str(config_path),
            "--dataset-manifest",
            str(dataset_manifest),
            "--init-checkpoint",
            str(missing_init),
            "--run-id",
            run_id,
            "--output-dir",
            str(output_root),
            "--set",
            "evaluation.sampled.num_samples=16",
        ]
    )
    assert code == 1
    assert not (output_root / run_id).exists()


def test_eval_scaffold_missing_checkpoint_is_structured_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")
    output_root = tmp_path / "runs"
    run_id = "unit-eval-fail"
    missing_checkpoint = tmp_path / "missing.ckpt"

    code = main(
        [
            "eval",
            "--config",
            str(config_path),
            "--checkpoint",
            str(missing_checkpoint),
            "--run-id",
            run_id,
            "--output-dir",
            str(output_root),
        ]
    )
    assert code == 1

    run_dir = output_root / run_id
    errors = _read_json(run_dir / "errors.json")
    assert errors["code"] == "CHECKPOINT_NOT_FOUND"
    summary = _read_json(run_dir / "summary.json")
    assert summary["status"] == "failed"


def test_override_failure_does_not_create_run_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")
    output_root = tmp_path / "runs"

    code = main(
        [
            "train",
            "grpo",
            "--config",
            str(config_path),
            "--dataset-manifest",
            str(tmp_path / "manifest.json"),
            "--init-checkpoint",
            str(tmp_path / "checkpoint"),
            "--run-id",
            "bad-override",
            "--output-dir",
            str(output_root),
            "--set",
            "nonexistent.section=1",
        ]
    )
    assert code == 1
    assert not (output_root / "bad-override").exists()
