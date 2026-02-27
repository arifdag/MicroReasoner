from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import context_as_dict, init_run_context, repo_root
from microreasoner.runtime.errors import RuntimeCommandError, RuntimeConfigError
from microreasoner.runtime.io import write_error, write_json, write_summary
from microreasoner.runtime.logging import JsonlEventLogger


DEFAULT_SEED = 42


def _seed_from_config(raw: dict[str, Any]) -> int:
    seed = raw.get("seed", DEFAULT_SEED)
    if isinstance(seed, int):
        return seed
    return DEFAULT_SEED


def execute_scaffold_command(
    *,
    command_name: str,
    config_path: Path,
    cli_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    checkpoint: Path | None = None,
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
        command_name=command_name,
        output_dir=output_dir,
        run_id=run_id,
        seed=seed,
    )
    logger = JsonlEventLogger(context.paths.events_log_path)

    try:
        logger.log("info", "command_start", {"command": command_name})

        write_json(context.paths.config_path, resolved.raw)
        write_json(
            context.paths.command_meta_path,
            {
                "command": command_name,
                "input": {
                    "config_path": str(config_path),
                    "cli_overrides": cli_overrides or [],
                    "checkpoint": str(checkpoint) if checkpoint else None,
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

        if checkpoint is not None and not checkpoint.exists():
            raise RuntimeCommandError(
                "CHECKPOINT_NOT_FOUND",
                f"Checkpoint not found: {checkpoint}",
            )

        message = (
            f"{command_name} scaffold initialized successfully; "
            "model execution is not implemented yet."
        )
        logger.log(
            "warning",
            "command_not_implemented",
            {"reason": "execution_pending"},
        )
        write_error(
            context.paths.errors_path,
            "NOT_IMPLEMENTED",
            message,
            run_id=context.run_id,
        )
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command=command_name,
            status="not_implemented",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "errors_path": str(context.paths.errors_path),
            },
            message=message,
        )
        print(message)
        print(f"Run directory: {context.paths.run_dir}")
        return 2

    except RuntimeCommandError as exc:
        logger.log("error", "command_failed", {"code": exc.code, "message": exc.message})
        write_error(context.paths.errors_path, exc.code, exc.message, run_id=context.run_id)
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command=command_name,
            status="failed",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "errors_path": str(context.paths.errors_path),
            },
            message=exc.message,
        )
        print(exc.message, file=sys.stderr)
        print(f"Run directory: {context.paths.run_dir}", file=sys.stderr)
        return 1

