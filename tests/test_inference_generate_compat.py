from __future__ import annotations

import pytest

from microreasoner.eval.inference import _safe_generate


def test_safe_generate_retries_without_generator_on_known_transformers_error() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, **kwargs: object) -> list[int]:
            self.calls.append(dict(kwargs))
            if "generator" in kwargs:
                raise ValueError(
                    "The following `model_kwargs` are not used by the model: ['generator']"
                )
            return [1, 2, 3]

    model = FakeModel()
    output = _safe_generate(model, {"input_ids": [0], "generator": object()})

    assert output == [1, 2, 3]
    assert len(model.calls) == 2
    assert "generator" in model.calls[0]
    assert "generator" not in model.calls[1]


def test_safe_generate_does_not_swallow_unrelated_value_error() -> None:
    class FakeModel:
        def generate(self, **kwargs: object) -> list[int]:
            del kwargs
            raise ValueError("different failure")

    with pytest.raises(ValueError, match="different failure"):
        _safe_generate(FakeModel(), {"input_ids": [0], "generator": object()})
