from __future__ import annotations

from microreasoner.rewards.correctness import CorrectnessScorer
from microreasoner.rewards.length import LengthBand, score_length
from microreasoner.rewards.scalarize import RewardComponents, scalarize_reward
from microreasoner.rewards.schema import score_schema
from microreasoner.runtime.models import RewardWeightsConfig


def test_correctness_reward_uses_fallback_or_math_verify() -> None:
    scorer = CorrectnessScorer("math_verify")
    correct = scorer.score("2", "2")
    wrong = scorer.score("3", "2")
    assert correct.score == 1.0
    assert wrong.score == 0.0
    assert correct.backend in {"math_verify", "simple"}


def test_schema_reward_valid_and_invalid() -> None:
    valid = "<think>work</think>\n<answer>\\boxed{5}</answer>"
    invalid = "<answer>\\boxed{5}</answer>"
    assert score_schema(valid, strict_boxed_only=True).score == 1.0
    assert score_schema(invalid, strict_boxed_only=True).score == 0.0


def test_length_reward_penalty_band() -> None:
    band = LengthBand(min_tokens=2, max_tokens=4)
    in_band = score_length("a b c", band)
    short = score_length("a", band)
    long = score_length("a b c d e f g", band)
    assert in_band.score == 0.0
    assert short.score < 0.0
    assert long.score < 0.0


def test_scalarize_reward_applies_weights() -> None:
    weights = RewardWeightsConfig(correctness=1.0, schema=0.2, length=0.05)
    scalarized = scalarize_reward(
        RewardComponents(correctness=1.0, schema=1.0, length=-0.5),
        weights,
    )
    assert scalarized.total == 1.175
