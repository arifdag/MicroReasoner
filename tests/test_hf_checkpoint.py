from __future__ import annotations

import sys
import types
from pathlib import Path

from microreasoner.hf_checkpoint import (
    load_causal_lm_checkpoint,
    prepare_causal_lm_for_training,
)


def test_load_causal_lm_checkpoint_uses_trainable_autopeft_for_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls: dict[str, object] = {}

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "</s>"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, use_fast: bool = True) -> FakeTokenizer:
            calls["tokenizer"] = (path, use_fast)
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            raise AssertionError(f"unexpected plain model load for adapter checkpoint: {path}, {kwargs}")

    class FakeAutoPeftModelForCausalLM:
        @staticmethod
        def from_pretrained(path: str, is_trainable: bool = False, **kwargs: object) -> object:
            calls["model"] = (path, is_trainable, kwargs)
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(AutoPeftModelForCausalLM=FakeAutoPeftModelForCausalLM),
    )

    tokenizer, _model = load_causal_lm_checkpoint(
        tmp_path,
        use_fast=True,
        trainable_adapter=True,
    )

    assert calls["tokenizer"] == (str(tmp_path), True)
    assert calls["model"] == (str(tmp_path), True, {})
    assert tokenizer.pad_token == "</s>"


def test_load_causal_lm_checkpoint_uses_plain_model_for_non_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token = None
            self.eos_token = "</s>"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, use_fast: bool = True) -> FakeTokenizer:
            calls["tokenizer"] = (path, use_fast)
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            calls["model"] = (path, kwargs)
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
        ),
    )

    tokenizer, _model = load_causal_lm_checkpoint(
        tmp_path,
        use_fast=False,
    )

    assert calls["tokenizer"] == (str(tmp_path), False)
    assert calls["model"] == (str(tmp_path), {})
    assert tokenizer.pad_token == "</s>"


def test_prepare_causal_lm_for_training_enables_train_flags() -> None:
    calls: list[str] = []

    class Config:
        use_cache = True

    class FakeModel:
        def __init__(self) -> None:
            self.config = Config()

        def enable_input_require_grads(self) -> None:
            calls.append("input_grads")

        def gradient_checkpointing_enable(self) -> None:
            calls.append("grad_ckpt")

    model = FakeModel()
    result = prepare_causal_lm_for_training(model)

    assert result is model
    assert calls == ["input_grads", "grad_ckpt"]
    assert model.config.use_cache is False
