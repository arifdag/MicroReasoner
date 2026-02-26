from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "ts": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "level": level,
            "event": event,
            "payload": payload or {},
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

