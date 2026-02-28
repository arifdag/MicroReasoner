from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microreasoner.contracts.validation import validate_run_dir


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_valid_run(run_dir: Path, sampled_temperature: float = 0.6) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    (checkpoints_dir / "latest.bin").write_text("ok", encoding="utf-8")
    (checkpoints_dir / "best.bin").write_text("ok", encoding="utf-8")

    config = {
        "evaluation": {
            "sampled": {
                "temperature": sampled_temperature,
                "top_p": 0.95,
                "num_samples": 32,
            }
        }
    }
    _write_json(run_dir / "config.json", config)

    dataset_manifest = {
        "datasets": {
            "sft": {"hash": "hash_sft_1", "count": 100},
            "rl": {"hash": "hash_rl_1", "count": 200},
            "eval": {"hash": "hash_eval_1", "count": 50},
        }
    }
    _write_json(run_dir / "dataset_manifest.json", dataset_manifest)

    metrics = {
        "accuracy": {
            "gsm8k": {"greedy_pass_at_1": 0.72, "sampled_pass_at_1": 0.8},
            "math": {"greedy_pass_at_1": 0.5, "sampled_pass_at_1": 0.57},
        },
        "schema": {"compliance_rate": 0.985},
        "parser": {"extraction_failure_rate": 0.01},
        "length": {"think_tokens": {"mean": 180, "p95": 420}},
    }
    _write_json(run_dir / "metrics.json", metrics)

    checkpoints = {
        "latest": "checkpoints/latest.bin",
        "best": "checkpoints/best.bin",
        "resume_test": {"passed": True, "tested_at": "2026-02-26T00:00:00Z"},
    }
    _write_json(run_dir / "checkpoints.json", checkpoints)

    run_manifest = {
        "schema_version": "1.0.0",
        "run_id": "run-001",
        "git_commit": "abcdef1",
        "seed": 42,
        "started_at": "2026-02-26T00:00:00Z",
        "finished_at": "2026-02-26T00:15:00Z",
        "model": {"base": "Qwen/Qwen2.5-Math-1.5B-Instruct", "adapter": "lora"},
        "data": {
            "sft": {"name": "openr1_subset", "hash": "hash_sft_1"},
            "rl": {"name": "gsm8k_rl", "hash": "hash_rl_1"},
            "eval": {"name": "gsm8k_math_eval", "hash": "hash_eval_1"},
        },
        "artifacts": {
            "config_path": "config.json",
            "dataset_manifest_path": "dataset_manifest.json",
            "metrics_path": "metrics.json",
            "checkpoints_path": "checkpoints.json",
        },
        "status": "success",
        "failure_reason": None,
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)

    return run_dir


def test_validate_run_passes_for_valid_run(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path / "run_ok")
    result = validate_run_dir(run_dir)
    assert result.ok
    assert result.errors == []


def test_validate_run_fails_on_missing_metrics_file(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path / "run_missing_metrics")
    (run_dir / "metrics.json").unlink()
    result = validate_run_dir(run_dir)
    assert not result.ok
    assert any("Missing required artifact file" in err for err in result.errors)


def test_validate_run_fails_on_missing_metrics_family(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path / "run_missing_family")
    metrics = _load_json(run_dir / "metrics.json")
    metrics.pop("parser")
    _write_json(run_dir / "metrics.json", metrics)
    result = validate_run_dir(run_dir)
    assert not result.ok
    assert any("Missing required metrics family: parser" == err for err in result.errors)


def test_validate_run_fails_on_resume_test_failure(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path / "run_resume_fail")
    checkpoints = _load_json(run_dir / "checkpoints.json")
    checkpoints["resume_test"]["passed"] = False
    _write_json(run_dir / "checkpoints.json", checkpoints)
    result = validate_run_dir(run_dir)
    assert not result.ok
    assert any("Checkpoint resume test failed" in err for err in result.errors)


def test_validate_run_fails_when_parser_failure_rate_exceeds_threshold(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path / "run_parser_fail")
    metrics = _load_json(run_dir / "metrics.json")
    metrics["parser"]["extraction_failure_rate"] = 0.021
    _write_json(run_dir / "metrics.json", metrics)
    result = validate_run_dir(run_dir)
    assert not result.ok
    assert any("parser.extraction_failure_rate" in err for err in result.errors)


def test_validate_run_fails_when_schema_compliance_below_threshold(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path / "run_schema_fail")
    metrics = _load_json(run_dir / "metrics.json")
    metrics["schema"]["compliance_rate"] = 0.97
    _write_json(run_dir / "metrics.json", metrics)
    result = validate_run_dir(run_dir)
    assert not result.ok
    assert any("schema.compliance_rate" in err for err in result.errors)


def test_validate_run_fails_on_dataset_hash_mismatch(tmp_path: Path) -> None:
    run_dir = _make_valid_run(tmp_path / "run_hash_fail")
    dataset_manifest = _load_json(run_dir / "dataset_manifest.json")
    dataset_manifest["datasets"]["eval"]["hash"] = "different_hash"
    _write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    result = validate_run_dir(run_dir)
    assert not result.ok
    assert any("Dataset hash mismatch for eval" in err for err in result.errors)


def test_validate_run_fails_on_eval_sampling_config_drift(tmp_path: Path) -> None:
    run_a = _make_valid_run(tmp_path / "run_a", sampled_temperature=0.6)
    run_b = _make_valid_run(tmp_path / "run_b", sampled_temperature=0.8)
    result = validate_run_dir(run_a, compare_run_dir=run_b)
    assert not result.ok
    assert any("Evaluation sampling config drift detected across runs" == err for err in result.errors)


def test_validate_run_accepts_cwd_relative_checkpoint_pointer_with_run_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = _make_valid_run(tmp_path / "artifacts" / "runs" / "smoke-run")
    checkpoints = _load_json(run_dir / "checkpoints.json")
    checkpoints["latest"] = "artifacts/runs/smoke-run/checkpoints/latest.bin"
    _write_json(run_dir / "checkpoints.json", checkpoints)

    monkeypatch.chdir(tmp_path)
    result = validate_run_dir(run_dir)
    assert result.ok

