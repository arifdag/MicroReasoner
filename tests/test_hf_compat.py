from __future__ import annotations

from types import SimpleNamespace

from microreasoner.train.hf_compat import patch_optimizer_mode_methods, prepare_trainer_optimizer_compat


class _BareOptimizer:
    pass


class _WrappedOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer


def test_patch_optimizer_mode_methods_adds_noops_recursively() -> None:
    inner = _BareOptimizer()
    outer = _WrappedOptimizer(inner)

    patch_optimizer_mode_methods(outer)

    assert hasattr(outer, "train")
    assert hasattr(outer, "eval")
    assert hasattr(inner, "train")
    assert hasattr(inner, "eval")
    assert outer.train() is None
    assert inner.eval() is None


def test_prepare_trainer_optimizer_compat_creates_and_patches_optimizer() -> None:
    created = _BareOptimizer()

    class _Trainer:
        def __init__(self) -> None:
            self.optimizer = None

        def create_optimizer(self) -> None:
            self.optimizer = created

    trainer = _Trainer()
    prepare_trainer_optimizer_compat(trainer)

    assert trainer.optimizer is created
    assert hasattr(created, "train")
    assert hasattr(created, "eval")


def test_patch_optimizer_mode_methods_preserves_existing_methods() -> None:
    calls: list[str] = []

    class _ExistingOptimizer:
        def train(self) -> None:
            calls.append("train")

        def eval(self) -> None:
            calls.append("eval")

    optimizer = _ExistingOptimizer()
    patch_optimizer_mode_methods(optimizer)

    optimizer.train()
    optimizer.eval()
    assert calls == ["train", "eval"]
