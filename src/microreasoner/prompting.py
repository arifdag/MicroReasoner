from __future__ import annotations

from typing import Any


QWEN_MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
REASONING_RESPONSE_FORMAT = "<think>...</think>\n<answer>\\boxed{...}</answer>"


def build_reasoning_prompt(question: str) -> str:
    return (
        "Solve the following problem.\n"
        "Respond using the exact format:\n"
        f"{REASONING_RESPONSE_FORMAT}\n\n"
        f"Problem:\n{question}"
    )


def build_reasoning_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QWEN_MATH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


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


def _render_generation_prompt(tokenizer: Any, prompt: str) -> tuple[str, bool]:
    rendered = _render_chat_template(
        tokenizer,
        build_reasoning_messages(prompt),
        add_generation_prompt=True,
    )
    if rendered is not None:
        return rendered, True
    return (
        "System:\n"
        f"{QWEN_MATH_SYSTEM_PROMPT}\n\n"
        f"User:\n{prompt}\n\n"
        "Assistant:\n",
        False,
    )


def _render_supervised_text(
    tokenizer: Any,
    prompt: str,
    assistant_response: str,
) -> tuple[str, bool]:
    rendered = _render_chat_template(
        tokenizer,
        [
            *build_reasoning_messages(prompt),
            {"role": "assistant", "content": assistant_response},
        ],
        add_generation_prompt=False,
    )
    if rendered is not None:
        return rendered, True
    generation_prompt, _ = _render_generation_prompt(tokenizer, prompt)
    return f"{generation_prompt}{assistant_response}", False


def render_generation_prompt(tokenizer: Any, prompt: str) -> str:
    rendered, _ = _render_generation_prompt(tokenizer, prompt)
    return rendered


def render_supervised_text(tokenizer: Any, prompt: str, assistant_response: str) -> str:
    rendered, _ = _render_supervised_text(tokenizer, prompt, assistant_response)
    return rendered


def _tokenize_rendered_text(
    tokenizer: Any,
    text: str,
    *,
    used_chat_template: bool,
    truncation: bool | None = None,
    max_length: int | None = None,
    padding: bool | str | None = None,
    return_tensors: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if truncation is not None:
        kwargs["truncation"] = truncation
    if max_length is not None:
        kwargs["max_length"] = max_length
    if padding is not None:
        kwargs["padding"] = padding
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    if used_chat_template:
        kwargs["add_special_tokens"] = False
    try:
        return tokenizer(text, **kwargs)
    except TypeError:
        kwargs.pop("add_special_tokens", None)
        return tokenizer(text, **kwargs)


def tokenize_generation_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    truncation: bool | None = None,
    max_length: int | None = None,
    padding: bool | str | None = None,
    return_tensors: str | None = None,
) -> Any:
    rendered, used_chat_template = _render_generation_prompt(tokenizer, prompt)
    return _tokenize_rendered_text(
        tokenizer,
        rendered,
        used_chat_template=used_chat_template,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )


def tokenize_supervised_text(
    tokenizer: Any,
    prompt: str,
    assistant_response: str,
    *,
    truncation: bool | None = None,
    max_length: int | None = None,
    padding: bool | str | None = None,
    return_tensors: str | None = None,
) -> Any:
    rendered, used_chat_template = _render_supervised_text(
        tokenizer,
        prompt,
        assistant_response,
    )
    return _tokenize_rendered_text(
        tokenizer,
        rendered,
        used_chat_template=used_chat_template,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )
