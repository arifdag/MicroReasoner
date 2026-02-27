from __future__ import annotations

import json
from pathlib import Path

import pytest

from microreasoner.train.grpo_data import GRPODataError, load_grpo_train_input


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _build_manifest(dataset_dir: Path) -> Path:
    manifest = {
        "schema_version": "1.0.0",
        "dataset_type": "rl",
        "dataset_id": "deadbeefcafebabe",
        "build_timestamp": "2026-02-27T00:00:00Z",
        "seed": 42,
        "inputs": [
            {
                "name": "source",
                "adapter": "canonical_jsonl",
                "path": "source.jsonl",
                "resolved_path": str(dataset_dir / "source.jsonl"),
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
        "artifact_paths": {
            "train": str(dataset_dir / "train_prompts.jsonl"),
            "val": str(dataset_dir / "val_prompts.jsonl"),
            "manifest": str(dataset_dir / "manifest.json"),
        },
        "artifact_hashes": {
            "train": "b" * 64,
            "val": "c" * 64,
        },
    }
    manifest_path = dataset_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_load_grpo_train_input_reads_records(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "rl"
    _write_jsonl(
        dataset_dir / "train_prompts.jsonl",
        [
            {
                "record_id": "t1",
                "prompt": "Q1",
                "gold_answer": "2",
                "split": "train",
                "source_name": "source",
                "benchmark": "gsm8k",
                "difficulty_tag": "easy",
                "curriculum_stage": "gsm8k_heavy",
            }
        ],
    )
    _write_jsonl(
        dataset_dir / "val_prompts.jsonl",
        [
            {
                "record_id": "v1",
                "prompt": "Q2",
                "gold_answer": "5",
                "split": "val",
                "source_name": "source",
                "benchmark": "math",
                "difficulty_tag": "medium",
                "curriculum_stage": "gsm8k_math_mixed",
            }
        ],
    )
    manifest_path = _build_manifest(dataset_dir)

    loaded = load_grpo_train_input(manifest_path, max_eval_samples=8)
    assert loaded.dataset_id == "deadbeefcafebabe"
    assert len(loaded.train_records) == 1
    assert len(loaded.val_records) == 1


def test_load_grpo_train_input_rejects_wrong_manifest_type(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "rl"
    _write_jsonl(dataset_dir / "train_prompts.jsonl", [])
    _write_jsonl(dataset_dir / "val_prompts.jsonl", [])
    manifest_path = _build_manifest(dataset_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_type"] = "sft"
    _write_json(manifest_path, manifest)

    with pytest.raises(GRPODataError):
        load_grpo_train_input(manifest_path)
