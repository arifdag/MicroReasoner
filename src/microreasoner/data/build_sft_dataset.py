from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from microreasoner.data.adapters import load_sources
from microreasoner.data.filters import apply_filters
from microreasoner.data.manifest import (
    make_dataset_id,
    sha256_file,
    validate_manifest,
    write_json,
    write_jsonl,
)
from microreasoner.data.normalize import normalize_examples
from microreasoner.data.split import split_examples
from microreasoner.data.types import BuildResult, CanonicalExample, SFTRecord
from microreasoner.prompting import build_reasoning_prompt
from microreasoner.runtime.context import repo_root, utc_now_iso
from microreasoner.runtime.models import ResolvedConfig


def _serialize_jsonl(records: list[dict]) -> list[str]:
    return [json.dumps(item, sort_keys=True) for item in records]


def _resolve_source_paths(
    config: ResolvedConfig, source_dir: Path | None
) -> list[tuple[str, str, str, Path]]:
    entries: list[tuple[str, str, str, Path]] = []
    for source in config.data_pipeline.input_sources:
        original = Path(source.path)
        resolved = source_dir / original.name if source_dir is not None else original
        entries.append((source.name, source.adapter, source.path, resolved))
    return entries


def _make_sft_prompt(example: CanonicalExample) -> str:
    return build_reasoning_prompt(example.question)


def _to_sft_record(example: CanonicalExample, split_name: str) -> SFTRecord:
    think = example.think or ""
    boxed = example.answer_boxed or ""
    target = f"<think>{think}</think>\n<answer>\\boxed{{{boxed}}}</answer>"
    return SFTRecord(
        record_id=example.example_id,
        prompt=_make_sft_prompt(example),
        target_response=target,
        split=split_name,
        source_name=example.source_name,
        benchmark=example.benchmark,
        quality_flags=("normalized",),
    )


def build_sft_dataset(
    *,
    config: ResolvedConfig,
    output_root: Path | None = None,
    source_dir: Path | None = None,
) -> BuildResult:
    raw_examples = load_sources(config.data_pipeline.input_sources, source_dir=source_dir)
    normalized = normalize_examples(raw_examples)
    accepted, rejected, reject_stats = apply_filters(normalized, config.data_pipeline.filters)
    if not accepted:
        raise ValueError("No SFT examples left after filtering")

    train_examples, val_examples = split_examples(accepted, config.data_pipeline.split)
    train_records = [_to_sft_record(item, "train") for item in train_examples]
    val_records = [_to_sft_record(item, "val") for item in val_examples]

    train_payload = [asdict(item) for item in train_records]
    val_payload = [asdict(item) for item in val_records]
    reject_payload = [asdict(item) for item in rejected]
    train_lines = _serialize_jsonl(train_payload)
    val_lines = _serialize_jsonl(val_payload)
    reject_lines = _serialize_jsonl(reject_payload)

    source_entries = _resolve_source_paths(config, source_dir)
    input_hashes = [sha256_file(entry[3]) for entry in source_entries]
    snapshot = {
        "split": asdict(config.data_pipeline.split),
        "filters": asdict(config.data_pipeline.filters),
        "schema_version": config.data_pipeline.schema_version,
    }
    dataset_id = make_dataset_id(
        dataset_type="sft",
        config_snapshot=snapshot,
        train_lines=train_lines,
        val_lines=val_lines,
        input_hashes=input_hashes,
    )

    root = output_root if output_root is not None else Path(config.data_pipeline.outputs.root_dir)
    dataset_dir = root / "sft" / dataset_id
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    rejects_path = dataset_dir / "rejects.jsonl"
    manifest_path = dataset_dir / "manifest.json"

    write_jsonl(train_path, train_lines)
    write_jsonl(val_path, val_lines)
    if config.data_pipeline.outputs.write_rejects:
        write_jsonl(rejects_path, reject_lines)

    artifacts = {
        "train": str(train_path),
        "val": str(val_path),
        "manifest": str(manifest_path),
    }
    artifact_hashes = {
        "train": sha256_file(train_path),
        "val": sha256_file(val_path),
    }
    if config.data_pipeline.outputs.write_rejects:
        artifacts["rejects"] = str(rejects_path)
        artifact_hashes["rejects"] = sha256_file(rejects_path)

    manifest = {
        "schema_version": config.data_pipeline.schema_version,
        "dataset_type": "sft",
        "dataset_id": dataset_id,
        "build_timestamp": utc_now_iso(),
        "seed": config.data_pipeline.split.seed,
        "inputs": [
            {
                "name": name,
                "adapter": adapter,
                "path": original_path,
                "resolved_path": str(resolved_path),
                "hash": file_hash,
            }
            for (name, adapter, original_path, resolved_path), file_hash in zip(
                source_entries, input_hashes
            )
        ],
        "filters": asdict(config.data_pipeline.filters),
        "split_counts": {"train": len(train_records), "val": len(val_records)},
        "reject_stats": reject_stats,
        "artifact_paths": artifacts,
        "artifact_hashes": artifact_hashes,
    }
    validate_manifest(manifest, repo_root())
    write_json(manifest_path, manifest)

    return BuildResult(
        dataset_type="sft",
        dataset_id=dataset_id,
        output_dir=str(dataset_dir),
        manifest_path=str(manifest_path),
        train_count=len(train_records),
        val_count=len(val_records),
        reject_count=len(reject_payload),
    )

