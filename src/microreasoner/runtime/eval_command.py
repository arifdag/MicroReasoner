from __future__ import annotations

import sys
from pathlib import Path

from microreasoner.eval.harness import run_evaluation, write_failed_run_manifest
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import context_as_dict, init_run_context, repo_root
from microreasoner.runtime.errors import RuntimeCommandError, RuntimeConfigError
from microreasoner.runtime.io import write_error, write_json, write_summary
from microreasoner.runtime.logging import JsonlEventLogger
from microreasoner.runtime.scaffold import _seed_from_config


def execute_eval_command(
    *,
    config_path: Path,
    checkpoint: Path,
    cli_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    dataset_dir: Path | None = None,
    max_items: int | None = None,
    seed_override: int | None = None,
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

    seed = seed_override if seed_override is not None else _seed_from_config(resolved.raw)
    context = init_run_context(
        command_name="eval",
        output_dir=output_dir,
        run_id=run_id,
        seed=seed,
    )
    logger = JsonlEventLogger(context.paths.events_log_path)

    try:
        logger.log("info", "command_start", {"command": "eval"})
        write_json(context.paths.config_path, resolved.raw)
        write_json(
            context.paths.command_meta_path,
            {
                "command": "eval",
                "input": {
                    "config_path": str(config_path),
                    "cli_overrides": cli_overrides or [],
                    "checkpoint": str(checkpoint),
                    "dataset_dir": str(dataset_dir) if dataset_dir else None,
                    "max_items": max_items,
                    "seed_override": seed_override,
                },
                "context": context_as_dict(context),
            },
        )
        logger.log(
            "info",
            "config_resolved",
            {
                "schema_version": resolved.schema_version,
                "config_path": str(context.paths.config_path),
            },
        )

        if not checkpoint.exists():
            raise RuntimeCommandError("CHECKPOINT_NOT_FOUND", f"Checkpoint not found: {checkpoint}")

        artifacts = run_evaluation(
            config=resolved,
            checkpoint=checkpoint,
            context=context,
            dataset_dir=dataset_dir,
            max_items=max_items,
        )
        logger.log("info", "evaluation_complete", {"artifacts": artifacts})
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command="eval",
            status="success",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                **artifacts,
            },
            message="Evaluation completed successfully",
        )
        print("Evaluation completed successfully")
        print(f"Run directory: {context.paths.run_dir}")
        return 0

    except RuntimeCommandError as exc:
        logger.log("error", "command_failed", {"code": exc.code, "message": exc.message})
        write_error(context.paths.errors_path, exc.code, exc.message, run_id=context.run_id)
        write_failed_run_manifest(context=context, config=resolved, reason=exc.message)
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command="eval",
            status="failed",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "errors_path": str(context.paths.errors_path),
                "run_manifest_path": str(context.paths.run_dir / "run_manifest.json"),
            },
            message=exc.message,
        )
        print(exc.message, file=sys.stderr)
        print(f"Run directory: {context.paths.run_dir}", file=sys.stderr)
        return 1

