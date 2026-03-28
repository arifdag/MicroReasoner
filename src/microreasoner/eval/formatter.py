from __future__ import annotations

from microreasoner.prompting import build_reasoning_prompt
from microreasoner.eval.types import EvalExample


def build_prompt(example: EvalExample) -> str:
    return build_reasoning_prompt(example.question)

