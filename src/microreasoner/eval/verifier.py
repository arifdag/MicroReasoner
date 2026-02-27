from __future__ import annotations

import re
from dataclasses import dataclass


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _try_float(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    backend: str


class Verifier:
    def verify(self, predicted: str, gold: str) -> VerificationResult:
        raise NotImplementedError


class SimpleVerifier(Verifier):
    def verify(self, predicted: str, gold: str) -> VerificationResult:
        p_norm = _normalize_text(predicted)
        g_norm = _normalize_text(gold)
        if p_norm == g_norm:
            return VerificationResult(correct=True, backend="simple")

        p_num = _try_float(p_norm)
        g_num = _try_float(g_norm)
        if p_num is not None and g_num is not None and abs(p_num - g_num) < 1e-9:
            return VerificationResult(correct=True, backend="simple")

        # Handle trivial rational formatting differences, e.g. 1/2 and 0.5.
        fraction_match = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*", p_norm)
        if fraction_match and g_num is not None:
            num = int(fraction_match.group(1))
            den = int(fraction_match.group(2))
            if den != 0 and abs((num / den) - g_num) < 1e-9:
                return VerificationResult(correct=True, backend="simple")

        return VerificationResult(correct=False, backend="simple")


class MathVerifyAdapter(Verifier):
    def __init__(self) -> None:
        # Delayed import so environments without math_verify can still run tests.
        try:
            from math_verify import parse as math_parse  # type: ignore
            from math_verify import verify as math_verify  # type: ignore
        except ImportError as exc:
            raise RuntimeError("math_verify is not installed") from exc

        self._parse = math_parse
        self._verify = math_verify

    def verify(self, predicted: str, gold: str) -> VerificationResult:
        pred_parsed = self._parse(predicted)
        gold_parsed = self._parse(gold)
        is_correct = bool(self._verify(pred_parsed, gold_parsed))
        return VerificationResult(correct=is_correct, backend="math_verify")


def build_verifier(preferred_backend: str = "math_verify") -> Verifier:
    if preferred_backend == "simple":
        return SimpleVerifier()
    if preferred_backend == "math_verify":
        try:
            return MathVerifyAdapter()
        except RuntimeError:
            return SimpleVerifier()
    raise ValueError(f"Unsupported verifier backend: {preferred_backend}")

