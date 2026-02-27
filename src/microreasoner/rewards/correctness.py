from __future__ import annotations

from dataclasses import dataclass

from microreasoner.eval.verifier import build_verifier


@dataclass(frozen=True)
class CorrectnessRewardResult:
    score: float
    correct: bool
    backend: str


class CorrectnessScorer:
    def __init__(self, preferred_backend: str = "math_verify") -> None:
        self._preferred_backend = preferred_backend
        self._verifier = build_verifier(preferred_backend)

    @property
    def preferred_backend(self) -> str:
        return self._preferred_backend

    def score(self, predicted_answer: str | None, gold_answer: str) -> CorrectnessRewardResult:
        if predicted_answer is None:
            return CorrectnessRewardResult(score=0.0, correct=False, backend="none")

        verification = self._verifier.verify(predicted_answer, gold_answer)
        return CorrectnessRewardResult(
            score=1.0 if verification.correct else 0.0,
            correct=verification.correct,
            backend=verification.backend,
        )
