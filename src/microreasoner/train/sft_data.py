from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microreasoner.data.manifest import validate_manifest
from microreasoner.eval.parser import parse_response
from microreasoner.runtime.context import repo_root


class SFTDataError(ValueError):
    """Raised when SFT dataset loading/validation fails."""


@dataclass(frozen=True)
class SFTRecordItem:
    record_id: str
    prompt: str
    target_response: str
    benchmark: str
    source_name: str
    gold_answer: str | None


@dataclass(frozen=True)
class SFTTrainInput:
    dataset_id: str
    train_records: tuple[SFTRecordItem, ...]
    val_records: tuple[SFTRecordItem, ...]
    manifest_path: Path
    manifest: dict[str, Any]
    train_path: Path
    val_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SFTDataError(f"Expected object JSON in {path}")
    return data


def _resolve_artifact_path(base: Path, artifact: str) -> Path:
    path = Path(artifact)
    if path.is_absolute():
        return path
    candidates = [
        path,
        repo_root() / path,
        base / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base / path).resolve()


def _require_str(row: dict[str, Any], key: str, path: Path, line_no: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise SFTDataError(f"{path}:{line_no} missing non-empty string field '{key}'")
    return value


def _parse_record(row: dict[str, Any], path: Path, line_no: int) -> SFTRecordItem:
    record_id = _require_str(row, "record_id", path, line_no)
    prompt = _require_str(row, "prompt", path, line_no)
    target = _require_str(row, "target_response", path, line_no)
    benchmark = _require_str(row, "benchmark", path, line_no)
    source_name = _require_str(row, "source_name", path, line_no)

    parsed_target = parse_response(target, strict_boxed_only=True)
    gold = parsed_target.boxed_answer if parsed_target.parse_ok else None
    return SFTRecordItem(
        record_id=record_id,
        prompt=prompt,
        target_response=target,
        benchmark=benchmark,
        source_name=source_name,
        gold_answer=gold,
    )


def _load_jsonl_records(path: Path) -> list[SFTRecordItem]:
    records: list[SFTRecordItem] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SFTDataError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SFTDataError(f"{path}:{line_no} expected JSON object")
            record = _parse_record(row, path, line_no)
            if record.record_id in seen_ids:
                raise SFTDataError(f"{path}:{line_no} duplicate record_id '{record.record_id}'")
            seen_ids.add(record.record_id)
            records.append(record)
    records.sort(key=lambda item: item.record_id)
    return records


def load_sft_train_input(
    dataset_manifest_path: Path,
    *,
    max_eval_samples: int | None = None,
) -> SFTTrainInput:
    if not dataset_manifest_path.exists():
        raise SFTDataError(f"Dataset manifest not found: {dataset_manifest_path}")

    manifest = _load_json(dataset_manifest_path)
    validate_manifest(manifest, repo_root())

    if manifest.get("dataset_type") != "sft":
        raise SFTDataError(
            f"Dataset manifest must be dataset_type='sft', got {manifest.get('dataset_type')}"
        )

    artifact_paths = manifest.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise SFTDataError("dataset manifest missing artifact_paths")
    train_artifact = artifact_paths.get("train")
    val_artifact = artifact_paths.get("val")
    if not isinstance(train_artifact, str) or not isinstance(val_artifact, str):
        raise SFTDataError("dataset manifest missing train/val artifact paths")

    base = dataset_manifest_path.parent
    train_path = _resolve_artifact_path(base, train_artifact)
    val_path = _resolve_artifact_path(base, val_artifact)
    if not train_path.exists():
        raise SFTDataError(f"SFT train file not found: {train_path}")
    if not val_path.exists():
        raise SFTDataError(f"SFT val file not found: {val_path}")

    train_records = _load_jsonl_records(train_path)
    val_records = _load_jsonl_records(val_path)

    if max_eval_samples is not None:
        val_records = val_records[:max_eval_samples]
    if len(train_records) == 0:
        raise SFTDataError("SFT train set is empty")
    if len(val_records) == 0:
        raise SFTDataError("SFT val set is empty")

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or dataset_id.strip() == "":
        raise SFTDataError("dataset manifest missing dataset_id")

    return SFTTrainInput(
        dataset_id=dataset_id,
        train_records=tuple(train_records),
        val_records=tuple(val_records),
        manifest_path=dataset_manifest_path,
        manifest=manifest,
        train_path=train_path,
        val_path=val_path,
    )

