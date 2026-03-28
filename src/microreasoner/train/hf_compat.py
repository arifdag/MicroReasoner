from __future__ import annotations

from types import MethodType
from typing import Any


def _noop_train(self: Any) -> None:
    return None


def _noop_eval(self: Any) -> None:
    return None


def _attach_method_if_missing(target: Any, name: str, fn: Any) -> None:
    if target is None or hasattr(target, name):
        return
    try:
        setattr(target, name, MethodType(fn, target))
        return
    except (AttributeError, TypeError):
        pass

    target_type = type(target)
    if not hasattr(target_type, name):
        setattr(target_type, name, fn)


def patch_optimizer_mode_methods(optimizer: Any) -> Any:
    current = optimizer
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        _attach_method_if_missing(current, "train", _noop_train)
        _attach_method_if_missing(current, "eval", _noop_eval)
        current = getattr(current, "optimizer", None)
    return optimizer


def prepare_trainer_optimizer_compat(trainer: Any) -> None:
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None and hasattr(trainer, "create_optimizer"):
        trainer.create_optimizer()
        optimizer = getattr(trainer, "optimizer", None)
    patch_optimizer_mode_methods(optimizer)
