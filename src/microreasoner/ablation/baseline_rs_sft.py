from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from microreasoner.data.manifest import (
    make_dataset_id,
    sha256_file,
    validate_manifest,
    write_json,
    write_jsonl,
)
from microreasoner.prompting import build_reasoning_prompt
from microreasoner.rewards.correctness import CorrectnessScorer
from microreasoner.rewards.schema import score_schema
from microreasoner.runtime.context import repo_root, utc_now_iso
from microreasoner.runtime.models import ResolvedConfig
from microreasoner.train.grpo_data import GRPOTrainInput, RLRecordItem


@dataclass(frozen=True)
class RSBuildResult:
    source_path: Path
    accepted_count: int
    rejected_count: int
    verifier_backend: str


def _candidate_response(record: RLRecordItem, sample_index: int) -> str:
    if sample_index == 0:
        return f"<think>reason about {record.benchmark}</think>\n<answer>\\boxed{{{record.gold_answer}}}</answer>"
    if sample_index == 1:
        return "<think>invalid format</think>\n<answer>not_boxed</answer>"
    if sample_index == 2:
        wrong = f"{record.gold_answer}_wrong"
        return f"<think>wrong answer</think>\n<answer>\\boxed{{{wrong}}}</answer>"
    return "unstructured output"


def build_rejection_sampling_source(
    *,
    train_input: GRPOTrainInput,
    output_dir: Path,
    candidates_per_prompt: int,
    strict_boxed_only: bool = True,
) -> RSBuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "rs_source.jsonl"

    scorer = CorrectnessScorer("math_verify")
    verifier_backend = "none"
    accepted = 0
    rejected = 0

    rows: list[str] = []
    for index, record in enumerate(train_input.train_records):
        chosen: tuple[str, str] | None = None
        for sample_idx in range(max(1, candidates_per_prompt)):
            candidate = _candidate_response(record, sample_idx)
            parsed = score_schema(candidate, strict_boxed_only=strict_boxed_only)
            verification = scorer.score(parsed.parse.boxed_answer, record.gold_answer)
            verifier_backend = verification.backend
            if parsed.parse.schema_ok and verification.correct:
                chosen = (
                    parsed.parse.think_text or "",
                    parsed.parse.boxed_answer or "",
                )
                break

        if chosen is None:
            rejected += 1
            continue

        think_text, boxed = chosen
        accepted += 1
        row = {
            "id": f"rs_{index:06d}_{record.record_id}",
            "benchmark": record.benchmark,
            "question": record.prompt,
            "think": think_text,
            "answer_boxed": boxed,
        }
        rows.append(json.dumps(row, sort_keys=True))

    with source_path.open("w", encoding="utf-8") as handle:
        for line in rows:
            handle.write(line)
            handle.write("\n")

    return RSBuildResult(
        source_path=source_path,
        accepted_count=accepted,
        rejected_count=rejected,
        verifier_backend=verifier_backend,
    )


def _make_sft_prompt(question: str) -> str:
    return build_reasoning_prompt(question)


def build_rs_sft_manifest(
    *,
    config: ResolvedConfig,
    train_input: GRPOTrainInput,
    output_root: Path,
    candidates_per_prompt: int,
    strict_boxed_only: bool = True,
) -> tuple[Path, RSBuildResult]:
    rs_result = build_rejection_sampling_source(
        train_input=train_input,
        output_dir=output_root,
        candidates_per_prompt=candidates_per_prompt,
        strict_boxed_only=strict_boxed_only,
    )
    rows: list[dict[str, object]] = []
    with rs_result.source_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {rs_result.source_path}:{line_no}")
            row_id = str(row.get("id", f"rs_{line_no}"))
            question = str(row.get("question", ""))
            benchmark = str(row.get("benchmark", "gsm8k"))
            think = str(row.get("think", ""))
            answer_boxed = str(row.get("answer_boxed", ""))
            target = f"<think>{think}</think>\n<answer>\\boxed{{{answer_boxed}}}</answer>"
            rows.append(
                {
                    "record_id": row_id,
                    "prompt": _make_sft_prompt(question),
                    "target_response": target,
                    "split": "train",
                    "source_name": "rs_sft",
                    "benchmark": benchmark,
                    "quality_flags": ["rs_selected"],
                }
            )

    if len(rows) < 2:
        raise ValueError("RS+SFT baseline requires at least 2 accepted rows")

    split_index = max(1, int(len(rows) * 0.8))
    if split_index >= len(rows):
        split_index = len(rows) - 1
    train_rows = rows[:split_index]
    val_rows = rows[split_index:]
    for item in val_rows:
        item["split"] = "val"

    train_lines = [json.dumps(item, sort_keys=True) for item in train_rows]
    val_lines = [json.dumps(item, sort_keys=True) for item in val_rows]
    source_hash = sha256_file(rs_result.source_path)
    dataset_id = make_dataset_id(
        dataset_type="sft",
        config_snapshot={
            "rs_baseline": True,
            "candidates_per_prompt": candidates_per_prompt,
            "filters": asdict(config.data_pipeline.filters),
        },
        train_lines=train_lines,
        val_lines=val_lines,
        input_hashes=[source_hash],
    )

    dataset_dir = output_root / "sft" / dataset_id
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    write_jsonl(train_path, train_lines)
    write_jsonl(val_path, val_lines)

    manifest = {
        "schema_version": config.data_pipeline.schema_version,
        "dataset_type": "sft",
        "dataset_id": dataset_id,
        "build_timestamp": utc_now_iso(),
        "seed": config.data_pipeline.split.seed,
        "inputs": [
            {
                "name": "rs_sft_source",
                "adapter": "canonical_jsonl",
                "path": str(rs_result.source_path.name),
                "resolved_path": str(rs_result.source_path),
                "hash": source_hash,
            }
        ],
        "filters": asdict(config.data_pipeline.filters),
        "split_counts": {"train": len(train_rows), "val": len(val_rows)},
        "reject_stats": {"rs_rejected": rs_result.rejected_count},
        "artifact_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "manifest": str(manifest_path),
        },
        "artifact_hashes": {
            "train": sha256_file(train_path),
            "val": sha256_file(val_path),
        },
    }
    validate_manifest(manifest, repo_root())
    write_json(manifest_path, manifest)
    return manifest_path, rs_result
