from __future__ import annotations

import json
from pathlib import Path

from microreasoner.train.sft_data import load_sft_train_input


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_load_sft_train_input_from_manifest(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(
        train_path,
        [
            {
                "record_id": "a1",
                "prompt": "q1",
                "target_response": "<think>t</think>\n<answer>\\boxed{2}</answer>",
                "split": "train",
                "source_name": "s",
                "benchmark": "gsm8k",
                "quality_flags": ["normalized"],
            }
        ],
    )
    _write_jsonl(
        val_path,
        [
            {
                "record_id": "b1",
                "prompt": "q2",
                "target_response": "<think>t</think>\n<answer>\\boxed{3}</answer>",
                "split": "val",
                "source_name": "s",
                "benchmark": "math",
                "quality_flags": ["normalized"],
            }
        ],
    )

    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "1.0.0",
            "dataset_type": "sft",
            "dataset_id": "abcdef1234567890",
            "build_timestamp": "2026-02-27T00:00:00Z",
            "seed": 42,
            "inputs": [
                {
                    "name": "raw",
                    "adapter": "canonical_jsonl",
                    "path": "x.jsonl",
                    "resolved_path": str(tmp_path / "x.jsonl"),
                    "hash": "0" * 64,
                }
            ],
            "filters": {
                "min_think_tokens": 1,
                "max_think_tokens": 1000,
                "require_single_boxed_answer": True,
                "drop_duplicates": True,
            },
            "split_counts": {"train": 1, "val": 1},
            "reject_stats": {"accepted_total": 2},
            "artifact_paths": {
                "train": str(train_path),
                "val": str(val_path),
                "manifest": str(manifest_path),
            },
            "artifact_hashes": {
                "train": "1" * 64,
                "val": "2" * 64,
            },
        },
    )

    loaded = load_sft_train_input(manifest_path)
    assert loaded.dataset_id == "abcdef1234567890"
    assert len(loaded.train_records) == 1
    assert len(loaded.val_records) == 1
    assert loaded.train_records[0].gold_answer == "2"

