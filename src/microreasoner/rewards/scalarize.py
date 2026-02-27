from __future__ import annotations

from dataclasses import dataclass

from microreasoner.runtime.models import RewardWeightsConfig


@dataclass(frozen=True)
class RewardComponents:
    correctness: float
    schema: float
    length: float


@dataclass(frozen=True)
class ScalarizedReward:
    total: float
    components: RewardComponents


def scalarize_reward(components: RewardComponents, weights: RewardWeightsConfig) -> ScalarizedReward:
    total = (
        (components.correctness * weights.correctness)
        + (components.schema * weights.schema)
        + (components.length * weights.length)
    )
    return ScalarizedReward(total=total, components=components)
