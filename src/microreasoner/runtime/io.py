from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_summary(
    *,
    path: Path,
    run_id: str,
    command: str,
    status: str,
    started_at: str,
    artifacts: dict[str, str],
    message: str | None = None,
) -> dict[str, Any]:
    finished_at = now_iso()
    summary: dict[str, Any] = {
        "run_id": run_id,
        "command": command,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "artifacts": artifacts,
    }
    if message:
        summary["message"] = message
    write_json(path, summary)
    return summary


def write_error(path: Path, code: str, message: str, *, run_id: str | None = None) -> None:
    data: dict[str, Any] = {
        "code": code,
        "message": message,
        "timestamp": now_iso(),
    }
    if run_id is not None:
        data["run_id"] = run_id
    write_json(path, data)

