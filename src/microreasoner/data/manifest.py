from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_dataset_id(
    *,
    dataset_type: str,
    config_snapshot: dict[str, Any],
    train_lines: list[str],
    val_lines: list[str],
    input_hashes: list[str],
) -> str:
    payload = {
        "dataset_type": dataset_type,
        "config_snapshot": config_snapshot,
        "train_lines": train_lines,
        "val_lines": val_lines,
        "input_hashes": input_hashes,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(blob)[:16]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def schema_path(repo_root: Path) -> Path:
    return repo_root / "schemas" / "dataset_manifest.schema.json"


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> None:
    path = schema_path(repo_root)
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    jsonschema.validate(instance=manifest, schema=schema)

