from __future__ import annotations


class RuntimeConfigError(ValueError):
    """Raised when runtime configuration cannot be resolved."""


class RuntimeCommandError(RuntimeError):
    """Raised when a command fails after run context initialization."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

