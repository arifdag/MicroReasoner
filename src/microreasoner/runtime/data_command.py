from __future__ import annotations

import json
import sys
from pathlib import Path

from microreasoner.data.build_rl_dataset import build_rl_dataset
from microreasoner.data.build_sft_dataset import build_sft_dataset
from microreasoner.data.manifest import validate_manifest
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import context_as_dict, init_run_context, repo_root
from microreasoner.runtime.errors import RuntimeCommandError, RuntimeConfigError
from microreasoner.runtime.io import write_error, write_json, write_summary
from microreasoner.runtime.logging import JsonlEventLogger


def execute_data_build_command(
    *,
    dataset_type: str,
    config_path: Path,
    cli_overrides: list[str] | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    source_dir: Path | None = None,
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

    seed = seed_override if seed_override is not None else resolved.data_pipeline.split.seed
    context = init_run_context(
        command_name=f"data-build-{dataset_type}",
        output_dir=None,
        run_id=run_id,
        seed=seed,
    )
    logger = JsonlEventLogger(context.paths.events_log_path)
    logger.log("info", "command_start", {"command": f"data build-{dataset_type}"})

    try:
        write_json(context.paths.config_path, resolved.raw)
        write_json(
            context.paths.command_meta_path,
            {
                "command": f"data build-{dataset_type}",
                "input": {
                    "config_path": str(config_path),
                    "cli_overrides": cli_overrides or [],
                    "source_dir": str(source_dir) if source_dir else None,
                    "output_dir": str(output_dir) if output_dir else None,
                    "seed_override": seed_override,
                },
                "context": context_as_dict(context),
            },
        )

        if dataset_type == "sft":
            result = build_sft_dataset(
                config=resolved,
                output_root=output_dir,
                source_dir=source_dir,
            )
        elif dataset_type == "rl":
            result = build_rl_dataset(
                config=resolved,
                output_root=output_dir,
                source_dir=source_dir,
            )
        else:
            raise RuntimeCommandError(
                "UNSUPPORTED_DATASET_TYPE",
                f"Unsupported dataset type: {dataset_type}",
            )

        logger.log(
            "info",
            "dataset_build_complete",
            {
                "dataset_type": result.dataset_type,
                "dataset_id": result.dataset_id,
                "manifest_path": result.manifest_path,
            },
        )
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command=f"data build-{dataset_type}",
            status="success",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "dataset_output_dir": result.output_dir,
                "dataset_manifest_path": result.manifest_path,
            },
            message=(
                f"Built {result.dataset_type} dataset {result.dataset_id} "
                f"(train={result.train_count}, val={result.val_count}, rejects={result.reject_count})"
            ),
        )
        print(
            f"Built {result.dataset_type} dataset {result.dataset_id} "
            f"(train={result.train_count}, val={result.val_count}, rejects={result.reject_count})"
        )
        print(f"Dataset output: {result.output_dir}")
        print(f"Run directory: {context.paths.run_dir}")
        return 0

    except Exception as exc:  # broad catch to ensure structured error outputs
        message = str(exc)
        logger.log("error", "command_failed", {"message": message})
        write_error(context.paths.errors_path, "DATA_BUILD_ERROR", message, run_id=context.run_id)
        write_summary(
            path=context.paths.summary_path,
            run_id=context.run_id,
            command=f"data build-{dataset_type}",
            status="failed",
            started_at=context.started_at,
            artifacts={
                "config_path": str(context.paths.config_path),
                "events_log_path": str(context.paths.events_log_path),
                "command_meta_path": str(context.paths.command_meta_path),
                "errors_path": str(context.paths.errors_path),
            },
            message=message,
        )
        print(message, file=sys.stderr)
        print(f"Run directory: {context.paths.run_dir}", file=sys.stderr)
        return 1


def execute_data_inspect_command(dataset_manifest_path: Path) -> int:
    if not dataset_manifest_path.exists():
        print(f"Manifest file not found: {dataset_manifest_path}", file=sys.stderr)
        return 1

    try:
        with dataset_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            print("Manifest must be a JSON object", file=sys.stderr)
            return 1
        validate_manifest(manifest, repo_root())
    except Exception as exc:
        print(f"Manifest validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset type: {manifest.get('dataset_type')}")
    print(f"Dataset id: {manifest.get('dataset_id')}")
    split_counts = manifest.get("split_counts", {})
    print(f"Train count: {split_counts.get('train')}")
    print(f"Val count: {split_counts.get('val')}")
    print(f"Manifest path: {dataset_manifest_path}")
    return 0

