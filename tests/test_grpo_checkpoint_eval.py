from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import microreasoner.train.grpo_trainer as grpo_trainer
from microreasoner.eval.types import EvalPrediction
from microreasoner.rewards.correctness import CorrectnessScorer
from microreasoner.train.grpo_data import RLRecordItem
from microreasoner.train.grpo_trainer import (
    _build_eval_snapshot,
    _evaluate_transformers_checkpoint,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        evaluation=SimpleNamespace(
            inference=SimpleNamespace(
                max_new_tokens=64,
                device="cpu",
                dtype="float32",
            ),
            greedy=SimpleNamespace(temperature=0.0),
            sampled=SimpleNamespace(temperature=0.7, top_p=0.9, num_samples=2),
            parser=SimpleNamespace(strict_boxed_only=True),
        )
    )


def test_evaluate_transformers_checkpoint_scores_real_engine_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeEngine:
        def generate_greedy(self, prompt: str, example) -> str:
            del prompt, example
            return "<think>reason</think><answer>\\boxed{2}</answer>junk"

        def generate_sampled(self, prompt: str, example) -> list[str]:
            del prompt, example
            return [
                "<think>reason</think><answer>\\boxed{2}</answer>",
                "<think>reason</think><answer>\\boxed{5}</answer>",
            ]

    monkeypatch.setattr(grpo_trainer, "build_inference_engine", lambda checkpoint, settings: FakeEngine())

    predictions = _evaluate_transformers_checkpoint(
        checkpoint=tmp_path / "checkpoint",
        config=_config(),
        records=[
            RLRecordItem(
                record_id="r1",
                prompt="Solve 1+1",
                gold_answer="2",
                benchmark="gsm8k",
                source_name="source",
                difficulty_tag="easy",
                curriculum_stage="gsm8k_heavy",
            )
        ],
        scorer=CorrectnessScorer("simple"),
        seed=7,
    )

    assert len(predictions) == 3
    assert predictions[0].mode == "greedy"
    assert not predictions[0].parse_ok
    assert predictions[0].parse_reason == "extra_text_outside_tags"
    assert predictions[1].verified_correct
    assert not predictions[2].verified_correct


def test_build_eval_snapshot_aggregates_accuracy_across_benchmarks(tmp_path: Path) -> None:
    snapshot = _build_eval_snapshot(
        tmp_path / "checkpoint",
        step=1,
        reward_std=0.1,
        predictions=[
            EvalPrediction(
                example_id="gsm-ok",
                benchmark="gsm8k",
                mode="greedy",
                sample_index=0,
                prompt="p1",
                raw_text="x",
                parsed_answer="2",
                parse_ok=True,
                schema_ok=True,
                verified_correct=True,
                parse_reason=None,
                think_token_count=1,
            ),
            EvalPrediction(
                example_id="math-bad",
                benchmark="math",
                mode="greedy",
                sample_index=0,
                prompt="p2",
                raw_text="x",
                parsed_answer="5",
                parse_ok=True,
                schema_ok=True,
                verified_correct=False,
                parse_reason=None,
                think_token_count=1,
            ),
        ],
    )

    assert snapshot.greedy_pass_at_1 == 0.5
