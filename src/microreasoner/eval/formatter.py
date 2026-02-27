from __future__ import annotations

from microreasoner.eval.types import EvalExample


def build_prompt(example: EvalExample) -> str:
    return (
        "Solve the following problem and respond with the exact schema.\n"
        "Use <think>...</think> for reasoning and "
        "<answer>\\boxed{...}</answer> for the final answer.\n\n"
        f"Problem:\n{example.question}\n"
    )

