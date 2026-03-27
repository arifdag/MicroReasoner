from __future__ import annotations

import pytest

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
        return_offsets_mapping: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        del truncation, padding
        truncated = text[:max_length]
        payload: dict[str, list[int] | list[tuple[int, int]]] = {
            "input_ids": [idx + 1 for idx, _ in enumerate(truncated)],
            "attention_mask": [1] * len(truncated),
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = [(idx, idx + 1) for idx, _ in enumerate(truncated)]
        return payload


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

    assert item["labels"][:2] == [-100, -100]
    assert item["labels"][2:] == item["input_ids"][2:]


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
