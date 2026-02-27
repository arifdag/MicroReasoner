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
from microreasoner.train.sft_data import SFTDataError, load_sft_train_input
from microreasoner.train.sft_trainer import SFTTrainingError, run_sft_training


def _git_commit() -> str:
    try:
        text = check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
        if text:
            return text
    except (CalledProcessError, FileNotFoundError):
        pass
    return "0000000"


def _write_run_contract_artifacts(
    *,
    run_dir: Path,
    run_id: str,
    started_at: str,
    config,
    dataset_id: str,
    metrics_path: Path,
    checkpoints_path: Path,
    status: str,
    failure_reason: str | None,
) -> None:
    dataset_manifest_payload = {
        "datasets": {
            "sft": {"name": dataset_id, "hash": dataset_id, "count": 0},
            "rl": {"name": "not_applicable", "hash": "not_applicable", "count": 0},
            "eval": {"name": "not_applicable", "hash": "not_applicable", "count": 0},
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
            "base": config.model.default_base_model,
            "adapter": "lora",
        },
        "data": {
            "sft": {"name": dataset_id, "hash": dataset_id},
            "rl": {"name": "not_applicable", "hash": "not_applicable"},
            "eval": {"name": "not_applicable", "hash": "not_applicable"},
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
            }
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def execute_sft_command(
    *,
    config_path: Path,
    dataset_manifest: Path,
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

    seed = _seed_from_config(resolved.raw)
    context = init_run_context(
        command_name="train-sft",
        output_dir=output_dir,
        run_id=run_id,
        seed=seed,
    )
    logger = JsonlEventLogger(context.paths.events_log_path)
    logger.log("info", "command_start", {"command": "train sft"})

    try:
        write_json(context.paths.config_path, resolved.raw)
        write_json(
            context.paths.command_meta_path,
            {
                "command": "train sft",
                "input": {
                    "config_path": str(config_path),
                    "dataset_manifest": str(dataset_manifest),
                    "cli_overrides": cli_overrides or [],
                    "resume_from": str(resume_from) if resume_from else None,
                    "max_steps": max_steps,
                    "eval_every_steps": eval_every_steps,
                },
                "context": context_as_dict(context),
            },
        )

        train_input = load_sft_train_input(
            dataset_manifest,
            max_eval_samples=resolved.train_sft.run.max_eval_samples,
        )
        logger.log(
            "info",
            "sft_data_loaded",
            {
                "dataset_id": train_input.dataset_id,
                "train_size": len(train_input.train_records),
                "val_size": len(train_input.val_records),
            },
        )

        result = run_sft_training(
            config=resolved,
            train_input=train_input,
            run_dir=context.paths.run_dir,
            resume_from=resume_from,
            max_steps_override=max_steps,
            eval_every_steps_override=eval_every_steps,
        )

        metrics = result.final_metrics.to_metrics_json(benchmark_name="sft_val")
        metrics_path = context.paths.run_dir / "metrics.json"
        checkpoints_path = context.paths.run_dir / "checkpoints.json"
        metrics_history_path = context.paths.run_dir / "metrics_history.jsonl"
        best_checkpoint_path = context.paths.run_dir / "best_checkpoint.json"

        write_json(metrics_path, metrics)
        _write_metrics_history(metrics_history_path, result.snapshots)
        write_json(
            best_checkpoint_path,
            {
                "path": str(result.best_checkpoint),
                "selected_mode": result.selected_mode,
                "backend": result.backend,
                "gate_passed": result.gate_passed,
                "gate_reason": result.gate_reason,
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
            metrics_path=metrics_path,
            checkpoints_path=checkpoints_path,
            status=status,
            failure_reason=failure_reason,
        )

        if not result.gate_passed:
            raise RuntimeCommandError("SFT_GATE_FAILED", result.gate_reason)

        logger.log(
            "info",
            "sft_training_complete",
            {
                "latest_checkpoint": str(result.latest_checkpoint),
                "best_checkpoint": str(result.best_checkpoint),
                "schema_compliance": result.final_metrics.schema_compliance,
                "greedy_pass_at_1": result.final_metrics.greedy_pass_at_1,
            },
        )
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command="train sft",
            status="success",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "metrics_path": str(metrics_path),
                "checkpoints_path": str(checkpoints_path),
                "metrics_history_path": str(metrics_history_path),
                "run_manifest_path": str(context.paths.run_dir / "run_manifest.json"),
            },
            message="SFT training completed successfully",
        )
        print("SFT training completed successfully")
        print(f"Run directory: {context.paths.run_dir}")
        return 0

    except (SFTDataError, SFTTrainingError, RuntimeCommandError) as exc:
        logger.log("error", "command_failed", {"message": str(exc)})
        write_error(context.paths.errors_path, "SFT_TRAIN_ERROR", str(exc), run_id=context.run_id)
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command="train sft",
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

