from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import CalledProcessError, check_output

from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import context_as_dict, init_run_context, repo_root
from microreasoner.runtime.errors import RuntimeCommandError, RuntimeConfigError
from microreasoner.runtime.io import write_error, write_json, write_summary
from microreasoner.runtime.logging import JsonlEventLogger
from microreasoner.runtime.scaffold import _seed_from_config
from microreasoner.train.grpo_data import GRPODataError, load_grpo_train_input
from microreasoner.train.grpo_trainer import GRPOTrainingError, run_grpo_training


def _git_commit() -> str:
    try:
        text = check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
        if text:
            return text
    except (CalledProcessError, FileNotFoundError):
        pass
    return "0000000"


def _write_metrics_history(path: Path, snapshots) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in snapshots:
            payload = {
                "checkpoint_path": item.checkpoint_path,
                "step": item.step,
                "schema_compliance": item.schema_compliance,
                "greedy_pass_at_1": item.greedy_pass_at_1,
                "sampled_pass_at_1": item.sampled_pass_at_1,
                "parser_failure_rate": item.parser_failure_rate,
                "reward_std": item.reward_std,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def _write_reward_history(path: Path, steps) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in steps:
            payload = {
                "step": item.step,
                "curriculum_stage": item.curriculum_stage,
                "prompt_count": item.prompt_count,
                "sample_count": item.sample_count,
                "reward_mean": item.reward_mean,
                "reward_std": item.reward_std,
                "correctness_mean": item.correctness_mean,
                "schema_mean": item.schema_mean,
                "length_mean": item.length_mean,
                "parser_failure_rate": item.parser_failure_rate,
                "schema_compliance_rate": item.schema_compliance_rate,
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def _write_curriculum_trace(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, sort_keys=True))
            handle.write("\n")


def _write_run_contract_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    started_at: str,
    config,
    dataset_id: str,
    sft_checkpoint: Path,
    metrics_path: Path,
    checkpoints_path: Path,
    status: str,
    failure_reason: str | None,
    train_count: int,
    val_count: int,
) -> None:
    sft_hash = f"checkpoint:{sft_checkpoint.resolve()}"
    dataset_manifest_payload = {
        "datasets": {
            "sft": {"name": sft_checkpoint.name, "hash": sft_hash, "count": 0},
            "rl": {"name": dataset_id, "hash": dataset_id, "count": train_count},
            "eval": {"name": f"{dataset_id}_val", "hash": dataset_id, "count": val_count},
        }
    }
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    write_json(dataset_manifest_path, dataset_manifest_payload)

    run_manifest = {
        "schema_version": config.schema_version,
        "run_id": run_id,
        "git_commit": _git_commit(),
        "seed": _seed_from_config(config.raw),
        "started_at": started_at,
        "finished_at": started_at,
        "model": {
            "base": str(sft_checkpoint.resolve()),
            "adapter": "lora",
        },
        "data": {
            "sft": {"name": sft_checkpoint.name, "hash": sft_hash},
            "rl": {"name": dataset_id, "hash": dataset_id},
            "eval": {"name": f"{dataset_id}_val", "hash": dataset_id},
        },
        "artifacts": {
            "config_path": "config.json",
            "dataset_manifest_path": "dataset_manifest.json",
            "metrics_path": str(metrics_path.relative_to(run_dir)),
            "checkpoints_path": str(checkpoints_path.relative_to(run_dir)),
        },
        "status": status,
        "failure_reason": failure_reason,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)


def execute_grpo_command(
    *,
    config_path: Path,
    dataset_manifest: Path,
    init_checkpoint: Path,
    cli_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    resume_from: Path | None = None,
    max_steps: int | None = None,
    eval_every_steps: int | None = None,
) -> int:
    defaults_path = repo_root() / "configs" / "defaults.yaml"
    try:
        resolved = resolve_config(
            defaults_path=defaults_path,
            user_config_path=config_path,
            cli_overrides=cli_overrides or [],
        )
    except RuntimeConfigError as exc:
        print(f"Config resolution failed: {exc}", file=sys.stderr)
        return 1

    if not init_checkpoint.exists():
        print(f"Initial checkpoint not found: {init_checkpoint}", file=sys.stderr)
        return 1

    seed = _seed_from_config(resolved.raw)
    context = init_run_context(
        command_name="train-grpo",
        output_dir=output_dir,
        run_id=run_id,
        seed=seed,
    )
    logger = JsonlEventLogger(context.paths.events_log_path)
    logger.log("info", "command_start", {"command": "train grpo"})

    try:
        write_json(context.paths.config_path, resolved.raw)
        write_json(
            context.paths.command_meta_path,
            {
                "command": "train grpo",
                "input": {
                    "config_path": str(config_path),
                    "dataset_manifest": str(dataset_manifest),
                    "init_checkpoint": str(init_checkpoint),
                    "resume_from": str(resume_from) if resume_from else None,
                    "cli_overrides": cli_overrides or [],
                    "max_steps": max_steps,
                    "eval_every_steps": eval_every_steps,
                },
                "context": context_as_dict(context),
            },
        )

        train_input = load_grpo_train_input(
            dataset_manifest,
            max_eval_samples=resolved.train_grpo.run.max_eval_samples,
        )
        logger.log(
            "info",
            "grpo_data_loaded",
            {
                "dataset_id": train_input.dataset_id,
                "train_size": len(train_input.train_records),
                "val_size": len(train_input.val_records),
            },
        )

        result = run_grpo_training(
            config=resolved,
            train_input=train_input,
            run_dir=context.paths.run_dir,
            init_checkpoint=init_checkpoint,
            resume_from=resume_from,
            max_steps_override=max_steps,
            eval_every_steps_override=eval_every_steps,
        )

        metrics = result.final_metrics.to_metrics_json()
        metrics_path = context.paths.run_dir / "metrics.json"
        checkpoints_path = context.paths.run_dir / "checkpoints.json"
        metrics_history_path = context.paths.run_dir / "metrics_history.jsonl"
        reward_history_path = context.paths.run_dir / "reward_history.jsonl"
        curriculum_trace_path = context.paths.run_dir / "curriculum_trace.jsonl"
        best_checkpoint_path = context.paths.run_dir / "best_checkpoint.json"

        write_json(metrics_path, metrics)
        _write_metrics_history(metrics_history_path, result.snapshots)
        _write_reward_history(reward_history_path, result.reward_history)
        _write_curriculum_trace(curriculum_trace_path, result.curriculum_trace)
        write_json(
            best_checkpoint_path,
            {
                "path": str(result.best_checkpoint),
                "backend": result.backend,
                "gate_passed": result.gate_passed,
                "gate_reason": result.gate_reason,
                "verifier_backend": result.verifier_backend,
            },
        )
        write_json(
            checkpoints_path,
            {
                "latest": str(result.latest_checkpoint),
                "best": str(result.best_checkpoint),
                "resume_test": {"passed": result.resume_test_passed, "tested_at": context.started_at},
            },
        )

        status = "success" if result.gate_passed else "failed"
        failure_reason = None if result.gate_passed else result.gate_reason
        _write_run_contract_artifacts(
            run_dir=context.paths.run_dir,
            run_id=context.run_id,
            started_at=context.started_at,
            config=resolved,
            dataset_id=train_input.dataset_id,
            sft_checkpoint=init_checkpoint,
            metrics_path=metrics_path,
            checkpoints_path=checkpoints_path,
            status=status,
            failure_reason=failure_reason,
            train_count=len(train_input.train_records),
            val_count=len(train_input.val_records),
        )

        if not result.gate_passed:
            raise RuntimeCommandError("GRPO_GATE_FAILED", result.gate_reason)

        logger.log(
            "info",
            "grpo_training_complete",
            {
                "latest_checkpoint": str(result.latest_checkpoint),
                "best_checkpoint": str(result.best_checkpoint),
                "schema_compliance": result.final_metrics.schema_compliance,
                "greedy_pass_at_1": result.final_metrics.greedy_pass_at_1,
                "reward_std": result.final_metrics.reward_std,
            },
        )
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command="train grpo",
            status="success",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "metrics_path": str(metrics_path),
                "checkpoints_path": str(checkpoints_path),
                "metrics_history_path": str(metrics_history_path),
                "reward_history_path": str(reward_history_path),
                "curriculum_trace_path": str(curriculum_trace_path),
                "run_manifest_path": str(context.paths.run_dir / "run_manifest.json"),
            },
            message="GRPO training completed successfully",
        )
        print("GRPO training completed successfully")
        print(f"Run directory: {context.paths.run_dir}")
        return 0

    except (GRPODataError, GRPOTrainingError, RuntimeCommandError) as exc:
        logger.log("error", "command_failed", {"message": str(exc)})
        write_error(context.paths.errors_path, "GRPO_TRAIN_ERROR", str(exc), run_id=context.run_id)
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command="train grpo",
            status="failed",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "errors_path": str(context.paths.errors_path),
            },
            message=str(exc),
        )
        print(str(exc), file=sys.stderr)
        print(f"Run directory: {context.paths.run_dir}", file=sys.stderr)
        return 1
