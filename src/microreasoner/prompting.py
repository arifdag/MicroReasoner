from __future__ import annotations

from typing import Any


def build_reasoning_prompt(question: str) -> str:
    return (
        "Solve the following problem.\n"
        "Respond using the exact format:\n"
        "<think>...</think>\n<answer>\\boxed{...}</answer>\n\n"
        f"Problem:\n{question}"
    )


def _render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str | None:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return None
    try:
        rendered = apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None
    return str(rendered)


def render_generation_prompt(tokenizer: Any, prompt: str) -> str:
    rendered = _render_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    if rendered is not None:
        return rendered
    return f"User:\n{prompt}\n\nAssistant:\n"


def render_supervised_text(tokenizer: Any, prompt: str, assistant_response: str) -> str:
    rendered = _render_chat_template(
        tokenizer,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        add_generation_prompt=False,
    )
    if rendered is not None:
        return rendered
    return f"{render_generation_prompt(tokenizer, prompt)}{assistant_response}"
