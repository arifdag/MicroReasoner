from __future__ import annotations

from types import SimpleNamespace

from microreasoner.train.sft_eval import evaluate_transformers
from microreasoner.train.sft_model import _cuda_bf16_supported, _select_training_dtype
from microreasoner.train.sft_data import SFTRecordItem


def test_select_training_dtype_prefers_bfloat16_when_supported() -> None:
    fake_torch = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
        ),
    )

    assert _cuda_bf16_supported(fake_torch) is True
    assert _select_training_dtype(fake_torch) == "bf16"


def test_select_training_dtype_falls_back_to_float16_on_cuda_without_bf16() -> None:
    fake_torch = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: False,
        ),
    )

    assert _cuda_bf16_supported(fake_torch) is False
    assert _select_training_dtype(fake_torch) == "fp16"


def test_evaluate_transformers_uses_remove_invalid_values() -> None:
    generate_calls: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, payload):
            self.payload = payload
            self.shape = (1, len(payload[0]))

        def to(self, _device):
            return self

        def __getitem__(self, item):
            if isinstance(item, tuple):
                row, col = item
                row_payload = self.payload[row]
                if isinstance(col, slice):
                    return row_payload[col]
                return row_payload[col]
            return self.payload[item]

        def dim(self) -> int:
            return 2

        def unsqueeze(self, _axis):
            return self

        def __iter__(self):
            return iter(self.payload)

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def decode(self, _ids, skip_special_tokens=True):
            del _ids, skip_special_tokens
            return "<think>ok</think><answer>\\boxed{2}</answer>"

    class FakeModel:
        device = "cuda"

        def generate(self, **kwargs):
            generate_calls.append(dict(kwargs))
            return FakeTensor([[10, 11, 12, 13]])

    class FakeTorch:
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

        def no_grad(self):
            return self._NoGrad()

    model_bundle = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        stack={"torch": FakeTorch()},
    )
    record = SFTRecordItem(
        record_id="r1",
        prompt="2+0?",
        target_response="<think>ok</think><answer>\\boxed{2}</answer>",
        gold_answer="2",
        benchmark="gsm8k",
        source_name="fixture",
    )

    from microreasoner.train import sft_eval as sft_eval_module

    original_tokenize = sft_eval_module.tokenize_generation_prompt
    sft_eval_module.tokenize_generation_prompt = lambda *args, **kwargs: {  # type: ignore[assignment]
        "input_ids": FakeTensor([[1, 2]]),
        "attention_mask": FakeTensor([[1, 1]]),
    }
    try:
        evaluate_transformers(
            records=[record],
            model_bundle=model_bundle,
            max_new_tokens=8,
            sampled_temperature=0.6,
            sampled_top_p=0.95,
            sampled_n=2,
        )
    finally:
        sft_eval_module.tokenize_generation_prompt = original_tokenize

    assert len(generate_calls) == 2
    assert all(call.get("remove_invalid_values") is True for call in generate_calls)
