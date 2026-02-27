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
from microreasoner.data.types import BuildResult, CanonicalExample, RLPromptRecord
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


def _difficulty_tag(example: CanonicalExample) -> str:
    difficulty = example.metadata.get("difficulty")
    if isinstance(difficulty, str) and difficulty.strip() != "":
        return difficulty.strip().lower()
    token_count = len(example.question.split())
    if token_count < 20:
        return "easy"
    if token_count < 50:
        return "medium"
    return "hard"


def _curriculum_stage(example: CanonicalExample, config: ResolvedConfig) -> str:
    benchmark = example.benchmark.lower()
    for rule in config.data_pipeline.rl.curriculum_rules:
        if benchmark in {item.lower() for item in rule.benchmarks}:
            return rule.name
    return "default"


def _to_rl_record(example: CanonicalExample, split_name: str, config: ResolvedConfig) -> RLPromptRecord:
    answer = example.answer_boxed or ""
    prompt = (
        "Solve the following problem.\n"
        "You may reason internally and return only the final formatted answer.\n\n"
        f"Problem:\n{example.question}"
    )
    return RLPromptRecord(
        record_id=example.example_id,
        prompt=prompt,
        gold_answer=answer,
        split=split_name,
        source_name=example.source_name,
        benchmark=example.benchmark,
        difficulty_tag=_difficulty_tag(example),
        curriculum_stage=_curriculum_stage(example, config),
    )


def build_rl_dataset(
    *,
    config: ResolvedConfig,
    output_root: Path | None = None,
    source_dir: Path | None = None,
) -> BuildResult:
    raw_examples = load_sources(config.data_pipeline.input_sources, source_dir=source_dir)
    normalized = normalize_examples(raw_examples)
    accepted, rejected, reject_stats = apply_filters(normalized, config.data_pipeline.filters)
    if not accepted:
        raise ValueError("No RL examples left after filtering")

    train_examples, val_examples = split_examples(accepted, config.data_pipeline.split)
    train_records = [_to_rl_record(item, "train", config) for item in train_examples]
    val_records = [_to_rl_record(item, "val", config) for item in val_examples]

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
        "rl": {
            "curriculum_rules": [asdict(item) for item in config.data_pipeline.rl.curriculum_rules],
            "benchmark_mix_targets": dict(config.data_pipeline.rl.benchmark_mix_targets),
        },
    }
    dataset_id = make_dataset_id(
        dataset_type="rl",
        config_snapshot=snapshot,
        train_lines=train_lines,
        val_lines=val_lines,
        input_hashes=input_hashes,
    )

    root = output_root if output_root is not None else Path(config.data_pipeline.outputs.root_dir)
    dataset_dir = root / "rl" / dataset_id
    train_path = dataset_dir / "train_prompts.jsonl"
    val_path = dataset_dir / "val_prompts.jsonl"
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
        "dataset_type": "rl",
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
        dataset_type="rl",
        dataset_id=dataset_id,
        output_dir=str(dataset_dir),
        manifest_path=str(manifest_path),
        train_count=len(train_records),
        val_count=len(val_records),
        reject_count=len(reject_payload),
    )

