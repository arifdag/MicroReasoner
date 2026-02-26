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


def test_train_scaffold_creates_run_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_text(config_path, "{}\n")
    output_root = tmp_path / "runs"
    run_id = "unit-train-sft"

    code = main(
        [
            "train",
            "sft",
            "--config",
            str(config_path),
            "--run-id",
            run_id,
            "--output-dir",
            str(output_root),
            "--set",
            "evaluation.sampled.num_samples=16",
        ]
    )
    assert code == 2

    run_dir = output_root / run_id
    assert (run_dir / "config.json").exists()
    assert (run_dir / "command_meta.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "errors.json").exists()
    assert (run_dir / "logs" / "events.jsonl").exists()

    config_json = _read_json(run_dir / "config.json")
    assert config_json["evaluation"]["sampled"]["num_samples"] == 16

    summary = _read_json(run_dir / "summary.json")
    assert summary["status"] == "not_implemented"


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

