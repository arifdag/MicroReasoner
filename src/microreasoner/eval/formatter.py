from __future__ import annotations

from microreasoner.eval.types import EvalExample


def build_prompt(example: EvalExample) -> str:
    return (
        "Solve the following problem.\n"
        "Respond using the exact format:\n"
        "<think>...</think>\n<answer>\\boxed{...}</answer>\n\n"
        f"Problem:\n{example.question}"
    )

