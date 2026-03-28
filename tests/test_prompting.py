from __future__ import annotations

from microreasoner.prompting import (
    build_reasoning_prompt,
    build_reasoning_messages,
    render_generation_prompt,
    render_supervised_text,
    tokenize_generation_prompt,
)


class _ChatTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"text": text, **kwargs})
        return {"input_ids": [1], "attention_mask": [1]}


def test_build_reasoning_prompt_keeps_existing_instruction_contract() -> None:
    prompt = build_reasoning_prompt("1+1?")
    assert "Solve the following problem." in prompt
    assert "<think>...</think>" in prompt
    assert "<answer>\\boxed{...}</answer>" in prompt
    assert prompt.endswith("Problem:\n1+1?")


def test_build_reasoning_messages_includes_qwen_math_system_prompt() -> None:
    messages = build_reasoning_messages("1+1?")
    assert messages == [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": "1+1?"},
    ]


def test_render_generation_prompt_falls_back_to_explicit_assistant_boundary() -> None:
    prompt = render_generation_prompt(object(), "1+1?")
    assert prompt.startswith("System:\n")
    assert "User:\n1+1?" in prompt
    assert prompt.endswith("\n\nAssistant:\n")


def test_render_generation_prompt_uses_chat_template_when_available() -> None:
    tokenizer = _ChatTokenizer()
    prompt = render_generation_prompt(tokenizer, "1+1?")
    assert prompt.startswith("<|system|>")
    assert "<|user|>1+1?" in prompt
    assert prompt.endswith("<|assistant|>")


def test_render_supervised_text_uses_chat_template_when_available() -> None:
    tokenizer = _ChatTokenizer()
    rendered = render_supervised_text(tokenizer, "1+1?", "<think>x</think>")
    assert rendered.startswith("<|system|>")
    assert "<|user|>1+1?" in rendered
    assert rendered.endswith("<|assistant|><think>x</think>")


def test_tokenize_generation_prompt_disables_special_tokens_for_chat_template() -> None:
    tokenizer = _ChatTokenizer()
    tokenize_generation_prompt(tokenizer, "1+1?", return_tensors="pt")
    assert tokenizer.calls
    assert tokenizer.calls[0]["add_special_tokens"] is False
    assert tokenizer.calls[0]["return_tensors"] == "pt"
