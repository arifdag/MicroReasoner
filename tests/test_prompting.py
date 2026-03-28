from __future__ import annotations

from microreasoner.prompting import (
    build_reasoning_prompt,
    render_generation_prompt,
    render_supervised_text,
)


class _ChatTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        rendered = ""
        for message in messages:
            rendered += f"<|{message['role']}|>{message['content']}"
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered


def test_build_reasoning_prompt_keeps_existing_instruction_contract() -> None:
    prompt = build_reasoning_prompt("1+1?")
    assert "Solve the following problem." in prompt
    assert "<think>...</think>" in prompt
    assert "<answer>\\boxed{...}</answer>" in prompt
    assert prompt.endswith("Problem:\n1+1?")


def test_render_generation_prompt_falls_back_to_explicit_assistant_boundary() -> None:
    prompt = render_generation_prompt(object(), "1+1?")
    assert prompt.startswith("User:\n")
    assert prompt.endswith("\n\nAssistant:\n")


def test_render_generation_prompt_uses_chat_template_when_available() -> None:
    tokenizer = _ChatTokenizer()
    prompt = render_generation_prompt(tokenizer, "1+1?")
    assert prompt == "<|user|>1+1?<|assistant|>"


def test_render_supervised_text_uses_chat_template_when_available() -> None:
    tokenizer = _ChatTokenizer()
    rendered = render_supervised_text(tokenizer, "1+1?", "<think>x</think>")
    assert rendered == "<|user|>1+1?<|assistant|><think>x</think>"
