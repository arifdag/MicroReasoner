from __future__ import annotations

import pytest

from microreasoner.prompting import render_generation_prompt
from microreasoner.train.sft_data import SFTRecordItem
from microreasoner.train.sft_trainer import SFTTrainingError, _build_torch_datasets


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        padding: bool,
    ) -> dict[str, list[int]]:
        del truncation, padding
        truncated = text[:max_length]
        payload: dict[str, list[int]] = {
            "input_ids": [idx + 1 for idx, _ in enumerate(truncated)],
            "attention_mask": [1] * len(truncated),
        }
        return payload


class _FakeChatTokenizer(_FakeTokenizer):
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


class _FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values: list[int], dtype: str | None = None) -> list[int]:
        del dtype
        return list(values)

    class utils:
        class data:
            class Dataset:
                pass


def test_build_torch_datasets_masks_prompt_tokens_from_labels() -> None:
    row = SFTRecordItem(
        record_id="demo",
        prompt="Q",
        target_response="<think>x</think>",
        benchmark="gsm8k",
        source_name="src",
        gold_answer="x",
    )

    dataset = _build_torch_datasets(
        records=[row],
        tokenizer=_FakeTokenizer(),
        max_seq_len=128,
        torch_module=_FakeTorch(),
    )
    item = dataset[0]

    masked_prefix = len(render_generation_prompt(_FakeTokenizer(), row.prompt))
    assert item["labels"][:masked_prefix] == [-100] * masked_prefix
    assert item["labels"][masked_prefix:] == item["input_ids"][masked_prefix:]


def test_build_torch_datasets_masks_chat_template_prompt_tokens() -> None:
    row = SFTRecordItem(
        record_id="chat-demo",
        prompt="Q",
        target_response="<think>x</think>",
        benchmark="gsm8k",
        source_name="src",
        gold_answer="x",
    )

    tokenizer = _FakeChatTokenizer()
    dataset = _build_torch_datasets(
        records=[row],
        tokenizer=tokenizer,
        max_seq_len=128,
        torch_module=_FakeTorch(),
    )
    item = dataset[0]

    masked_prefix = len(render_generation_prompt(tokenizer, row.prompt))
    assert item["labels"][:masked_prefix] == [-100] * masked_prefix
    assert item["labels"][masked_prefix:] == item["input_ids"][masked_prefix:]


def test_build_torch_datasets_fails_when_target_is_fully_truncated() -> None:
    row = SFTRecordItem(
        record_id="truncated",
        prompt="prompt-too-long",
        target_response="<think>x</think>",
        benchmark="gsm8k",
        source_name="src",
        gold_answer="x",
    )

    with pytest.raises(SFTTrainingError, match="has no supervised target tokens"):
        _build_torch_datasets(
            records=[row],
            tokenizer=_FakeTokenizer(),
            max_seq_len=4,
            torch_module=_FakeTorch(),
        )
