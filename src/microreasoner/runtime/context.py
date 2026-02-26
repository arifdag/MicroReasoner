from __future__ import annotations

import secrets
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microreasoner.runtime.models import RunContext, RunPaths


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_run_id(command_name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    sanitized = command_name.replace(" ", "-")
    return f"{sanitized}-{stamp}-{suffix}"


def init_run_context(
    command_name: str,
    output_dir: Path | None,
    run_id: str | None,
    seed: int,
) -> RunContext:
    selected_run_id = run_id if run_id else generate_run_id(command_name)
    root = output_dir if output_dir else Path("artifacts") / "runs"
    run_dir = root / selected_run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    paths = RunPaths(
        run_dir=run_dir,
        logs_dir=logs_dir,
        events_log_path=logs_dir / "events.jsonl",
        summary_path=run_dir / "summary.json",
        config_path=run_dir / "config.json",
        command_meta_path=run_dir / "command_meta.json",
        errors_path=run_dir / "errors.json",
    )

    return RunContext(
        run_id=selected_run_id,
        command_name=command_name,
        seed=seed,
        started_at=utc_now_iso(),
        paths=paths,
    )


def context_as_dict(context: RunContext) -> dict[str, Any]:
    data = asdict(context)
    data["paths"] = {key: str(value) for key, value in data["paths"].items()}
    return data

