from __future__ import annotations

import hashlib

from microreasoner.data.types import CanonicalExample
from microreasoner.runtime.models import DataSplitConfig


def _assignment_value(example_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    num = int(digest, 16)
    return num / float(2**256 - 1)


def split_examples(
    examples: list[CanonicalExample],
    config: DataSplitConfig,
) -> tuple[list[CanonicalExample], list[CanonicalExample]]:
    if config.strategy.lower() != "hash":
        raise ValueError(f"Unsupported split strategy: {config.strategy}")
    total_ratio = config.train_ratio + config.val_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            f"Invalid split ratios: train_ratio + val_ratio must be 1.0, got {total_ratio}"
        )
    if config.train_ratio <= 0 or config.val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")

    train: list[CanonicalExample] = []
    val: list[CanonicalExample] = []
    for example in sorted(examples, key=lambda item: item.example_id):
        value = _assignment_value(example.example_id, config.seed)
        if value < config.train_ratio:
            train.append(example)
        else:
            val.append(example)
    return train, val

