from __future__ import annotations

import pytest

import microreasoner.train.grpo_trainer as grpo_trainer
from microreasoner.train.grpo_trainer import GRPOTrainingError, _build_grpo_config


def test_build_grpo_config_maps_prompt_completion_aliases() -> None:
    captured: dict[str, object] = {}

    class FakeGRPOConfig:
        def __init__(
            self,
            *,
            output_dir: str,
            max_prompt_len: int,
            max_completion_len: int,
            num_generations: int,
        ) -> None:
            captured["output_dir"] = output_dir
            captured["max_prompt_len"] = max_prompt_len
            captured["max_completion_len"] = max_completion_len
            captured["num_generations"] = num_generations

    _build_grpo_config(
        FakeGRPOConfig,
        {
            "output_dir": "out",
            "max_prompt_length": 768,
            "max_completion_length": 512,
            "num_generations": 8,
            "unknown_argument": "ignored",
        },
    )

    assert captured["output_dir"] == "out"
    assert captured["max_prompt_len"] == 768
    assert captured["max_completion_len"] == 512
    assert captured["num_generations"] == 8


def test_build_grpo_config_maps_legacy_prompt_completion_aliases_to_modern_names() -> None:
    captured: dict[str, object] = {}

    class FakeGRPOConfig:
        def __init__(
            self,
            *,
            output_dir: str,
            max_prompt_length: int,
            max_completion_length: int,
            num_generations: int,
        ) -> None:
            captured["output_dir"] = output_dir
            captured["max_prompt_length"] = max_prompt_length
            captured["max_completion_length"] = max_completion_length
            captured["num_generations"] = num_generations

    _build_grpo_config(
        FakeGRPOConfig,
        {
            "output_dir": "out",
            "max_prompt_len": 768,
            "max_completion_len": 512,
            "num_generations": 8,
        },
    )

    assert captured["output_dir"] == "out"
    assert captured["max_prompt_length"] == 768
    assert captured["max_completion_length"] == 512
    assert captured["num_generations"] == 8


def test_build_grpo_config_passthroughs_var_kwargs() -> None:
    captured: dict[str, object] = {}

    class FakeGRPOConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    _build_grpo_config(
        FakeGRPOConfig,
        {
            "output_dir": "out",
            "max_prompt_length": 768,
            "max_completion_length": 512,
            "extra": 1,
        },
    )

    assert captured["output_dir"] == "out"
    assert captured["max_prompt_length"] == 768
    assert captured["max_completion_length"] == 512
    assert captured["extra"] == 1


def test_build_grpo_config_retries_aliases_when_signature_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeGRPOConfig:
        def __init__(
            self,
            *,
            output_dir: str,
            max_prompt_len: int,
            max_completion_len: int,
            num_generations: int,
        ) -> None:
            captured["output_dir"] = output_dir
            captured["max_prompt_len"] = max_prompt_len
            captured["max_completion_len"] = max_completion_len
            captured["num_generations"] = num_generations

    def _raise_signature_error(_: object) -> inspect.Signature:
        raise ValueError("signature unavailable")

    import inspect

    monkeypatch.setattr(grpo_trainer.inspect, "signature", _raise_signature_error)

    _build_grpo_config(
        FakeGRPOConfig,
        {
            "output_dir": "out",
            "max_prompt_length": 768,
            "max_completion_length": 512,
            "num_generations": 8,
        },
    )

    assert captured["output_dir"] == "out"
    assert captured["max_prompt_len"] == 768
    assert captured["max_completion_len"] == 512
    assert captured["num_generations"] == 8


def test_build_grpo_config_rejects_conflicting_alias_values() -> None:
    class FakeGRPOConfig:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    with pytest.raises(GRPOTrainingError, match="Conflicting GRPO config values"):
        _build_grpo_config(
            FakeGRPOConfig,
            {
                "output_dir": "out",
                "max_prompt_length": 768,
                "max_prompt_len": 512,
            },
        )
