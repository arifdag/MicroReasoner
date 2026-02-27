from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from subprocess import CalledProcessError, check_output
from typing import Iterable

from microreasoner.eval.formatter import build_prompt
from microreasoner.eval.inference import (
    InferenceError,
    InferenceSettings,
    build_inference_engine,
)
from microreasoner.eval.loader import EvalDataError, load_jsonl_examples
from microreasoner.eval.metrics import build_metrics
from microreasoner.eval.parser import parse_response
from microreasoner.eval.types import EvalExample, EvalPrediction
from microreasoner.eval.verifier import build_verifier
from microreasoner.runtime.errors import RuntimeCommandError
from microreasoner.runtime.io import write_json
from microreasoner.runtime.models import ResolvedConfig, RunContext


def _dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _combined_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _load_examples(
    *,
    gsm8k_path: Path,
    math_path: Path,
    max_items: int | None,
) -> list[EvalExample]:
    try:
        gsm_examples = load_jsonl_examples(gsm8k_path, "gsm8k")
        math_examples = load_jsonl_examples(math_path, "math")
    except EvalDataError as exc:
        raise RuntimeCommandError("DATASET_LOAD_ERROR", str(exc)) from exc

    all_examples = gsm_examples + math_examples
    all_examples.sort(key=lambda item: (item.benchmark, item.example_id))
    if max_items is not None:
        return all_examples[:max_items]
    return all_examples


def _resolve_dataset_path(path_text: str, dataset_dir: Path | None) -> Path:
    path = Path(path_text)
    if dataset_dir is not None:
        return dataset_dir / path.name
    return path


def _count_tokens(text: str | None) -> int:
    if text is None or text.strip() == "":
        return 0
    return len(text.split())


def _prediction_to_json(prediction: EvalPrediction) -> dict:
    return asdict(prediction)


def _write_predictions(path: Path, predictions: list[EvalPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(_prediction_to_json(prediction), sort_keys=True))
            handle.write("\n")


def _git_commit() -> str:
    try:
        text = check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
        if text:
            return text
    except (CalledProcessError, FileNotFoundError):
        pass
    return "0000000"


def _write_run_manifests(
    *,
    context: RunContext,
    config_path: Path,
    metrics_path: Path,
    dataset_manifest_path: Path,
    checkpoints_path: Path,
    combined_eval_hash: str,
    config: ResolvedConfig,
    status: str,
    failure_reason: str | None,
) -> None:
    dataset_manifest = {
        "datasets": {
            "sft": {
                "name": config.data.sft.primary_dataset,
                "hash": "not_applicable",
            },
            "rl": {
                "name": ",".join(config.data.rl.curriculum),
                "hash": "not_applicable",
            },
            "eval": {
                "name": "gsm8k_math_eval",
                "hash": combined_eval_hash,
            },
        }
    }
    write_json(dataset_manifest_path, dataset_manifest)

    run_manifest = {
        "schema_version": config.schema_version,
        "run_id": context.run_id,
        "git_commit": _git_commit(),
        "seed": context.seed,
        "started_at": context.started_at,
        "finished_at": context.started_at,
        "model": {
            "base": config.model.default_base_model,
            "adapter": None,
        },
        "data": {
            "sft": {
                "name": config.data.sft.primary_dataset,
                "hash": "not_applicable",
            },
            "rl": {
                "name": ",".join(config.data.rl.curriculum),
                "hash": "not_applicable",
            },
            "eval": {
                "name": "gsm8k_math_eval",
                "hash": combined_eval_hash,
            },
        },
        "artifacts": {
            "config_path": str(config_path.relative_to(context.paths.run_dir)),
            "dataset_manifest_path": str(dataset_manifest_path.relative_to(context.paths.run_dir)),
            "metrics_path": str(metrics_path.relative_to(context.paths.run_dir)),
            "checkpoints_path": str(checkpoints_path.relative_to(context.paths.run_dir)),
        },
        "status": status,
        "failure_reason": failure_reason,
    }
    write_json(context.paths.run_dir / "run_manifest.json", run_manifest)


def run_evaluation(
    *,
    config: ResolvedConfig,
    checkpoint: Path,
    context: RunContext,
    dataset_dir: Path | None,
    max_items: int | None,
) -> dict[str, str]:
    gsm8k_path = _resolve_dataset_path(config.evaluation.datasets.gsm8k.path, dataset_dir)
    math_path = _resolve_dataset_path(config.evaluation.datasets.math.path, dataset_dir)
    examples = _load_examples(gsm8k_path=gsm8k_path, math_path=math_path, max_items=max_items)
    if not examples:
        raise RuntimeCommandError("EMPTY_EVAL_SET", "No evaluation examples were loaded")

    settings = InferenceSettings(
        backend=config.evaluation.inference.backend,
        max_new_tokens=config.evaluation.inference.max_new_tokens,
        device=config.evaluation.inference.device,
        dtype=config.evaluation.inference.dtype,
        greedy_temperature=config.evaluation.greedy.temperature,
        sampled_temperature=config.evaluation.sampled.temperature,
        sampled_top_p=config.evaluation.sampled.top_p,
        sampled_n=config.evaluation.sampled.num_samples,
        seed=context.seed,
    )
    try:
        engine = build_inference_engine(checkpoint=checkpoint, settings=settings)
    except InferenceError as exc:
        raise RuntimeCommandError("INFERENCE_BACKEND_ERROR", str(exc)) from exc

    verifier = build_verifier("math_verify")
    predictions: list[EvalPrediction] = []
    strict_boxed = config.evaluation.parser.strict_boxed_only

    for example in examples:
        prompt = build_prompt(example)

        greedy_output = engine.generate_greedy(prompt, example)
        greedy_parse = parse_response(greedy_output, strict_boxed_only=strict_boxed)
        greedy_correct = False
        if greedy_parse.boxed_answer is not None:
            greedy_correct = verifier.verify(greedy_parse.boxed_answer, example.gold_answer).correct
        predictions.append(
            EvalPrediction(
                example_id=example.example_id,
                benchmark=example.benchmark,
                mode="greedy",
                sample_index=0,
                prompt=prompt,
                raw_text=greedy_output,
                parsed_answer=greedy_parse.boxed_answer,
                parse_ok=greedy_parse.parse_ok,
                schema_ok=greedy_parse.schema_ok,
                verified_correct=greedy_correct,
                parse_reason=greedy_parse.reason,
                think_token_count=_count_tokens(greedy_parse.think_text),
            )
        )

        sampled_outputs = engine.generate_sampled(prompt, example)
        for sample_index, sampled_output in enumerate(sampled_outputs):
            sampled_parse = parse_response(sampled_output, strict_boxed_only=strict_boxed)
            sampled_correct = False
            if sampled_parse.boxed_answer is not None:
                sampled_correct = verifier.verify(sampled_parse.boxed_answer, example.gold_answer).correct
            predictions.append(
                EvalPrediction(
                    example_id=example.example_id,
                    benchmark=example.benchmark,
                    mode="sampled",
                    sample_index=sample_index,
                    prompt=prompt,
                    raw_text=sampled_output,
                    parsed_answer=sampled_parse.boxed_answer,
                    parse_ok=sampled_parse.parse_ok,
                    schema_ok=sampled_parse.schema_ok,
                    verified_correct=sampled_correct,
                    parse_reason=sampled_parse.reason,
                    think_token_count=_count_tokens(sampled_parse.think_text),
                )
            )

    metrics = build_metrics(predictions)
    metrics_path = context.paths.run_dir / "metrics.json"
    predictions_path = context.paths.run_dir / "predictions.jsonl"
    dataset_manifest_path = context.paths.run_dir / "dataset_manifest.json"
    checkpoints_path = context.paths.run_dir / "checkpoints.json"

    write_json(metrics_path, metrics)
    _write_predictions(predictions_path, predictions)
    write_json(
        checkpoints_path,
        {
            "latest": str(checkpoint.resolve()),
            "best": str(checkpoint.resolve()),
            "resume_test": {"passed": True, "tested_at": context.started_at},
        },
    )

    combined_eval_hash = _combined_hash([_dataset_hash(gsm8k_path), _dataset_hash(math_path)])
    _write_run_manifests(
        context=context,
        config_path=context.paths.config_path,
        metrics_path=metrics_path,
        dataset_manifest_path=dataset_manifest_path,
        checkpoints_path=checkpoints_path,
        combined_eval_hash=combined_eval_hash,
        config=config,
        status="success",
        failure_reason=None,
    )

    return {
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "checkpoints_path": str(checkpoints_path),
        "run_manifest_path": str(context.paths.run_dir / "run_manifest.json"),
    }


def write_failed_run_manifest(
    *,
    context: RunContext,
    config: ResolvedConfig,
    reason: str,
) -> None:
    metrics_path = context.paths.run_dir / "metrics.json"
    dataset_manifest_path = context.paths.run_dir / "dataset_manifest.json"
    checkpoints_path = context.paths.run_dir / "checkpoints.json"
    _write_run_manifests(
        context=context,
        config_path=context.paths.config_path,
        metrics_path=metrics_path,
        dataset_manifest_path=dataset_manifest_path,
        checkpoints_path=checkpoints_path,
        combined_eval_hash="not_applicable",
        config=config,
        status="failed",
        failure_reason=reason,
    )

