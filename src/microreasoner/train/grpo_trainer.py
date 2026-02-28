from __future__ import annotations

import inspect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from microreasoner.eval.metrics import build_metrics
from microreasoner.eval.types import EvalPrediction
from microreasoner.rewards.correctness import CorrectnessScorer
from microreasoner.rewards.length import LengthBand, score_length
from microreasoner.rewards.scalarize import RewardComponents, scalarize_reward
from microreasoner.rewards.schema import score_schema
from microreasoner.runtime.models import ResolvedConfig
from microreasoner.train.grpo_data import GRPOTrainInput, RLRecordItem


class GRPOTrainingError(RuntimeError):
    """Raised when GRPO training setup/execution fails."""


@dataclass(frozen=True)
class GRPORewardStep:
    step: int
    curriculum_stage: str
    prompt_count: int
    sample_count: int
    reward_mean: float
    reward_std: float
    correctness_mean: float
    schema_mean: float
    length_mean: float
    parser_failure_rate: float
    schema_compliance_rate: float


@dataclass(frozen=True)
class GRPOEvalSnapshot:
    checkpoint_path: str
    step: int
    schema_compliance: float
    parser_failure_rate: float
    greedy_pass_at_1: float
    sampled_pass_at_1: float
    think_tokens_mean: float
    think_tokens_p95: float
    reward_std: float

    def to_metrics_json(self, benchmark_name: str = "rl_val") -> dict[str, Any]:
        return {
            "accuracy": {
                benchmark_name: {
                    "greedy_pass_at_1": self.greedy_pass_at_1,
                    "sampled_pass_at_1": self.sampled_pass_at_1,
                }
            },
            "schema": {"compliance_rate": self.schema_compliance},
            "parser": {"extraction_failure_rate": self.parser_failure_rate},
            "length": {
                "think_tokens": {
                    "mean": self.think_tokens_mean,
                    "p95": self.think_tokens_p95,
                }
            },
        }


@dataclass(frozen=True)
class GRPOTrainingResult:
    backend: str
    latest_checkpoint: Path
    best_checkpoint: Path
    resume_test_passed: bool
    reward_history: tuple[GRPORewardStep, ...]
    curriculum_trace: tuple[dict[str, Any], ...]
    snapshots: tuple[GRPOEvalSnapshot, ...]
    final_metrics: GRPOEvalSnapshot
    gate_passed: bool
    gate_reason: str
    global_step: int
    verifier_backend: str


def _checkpoint_dir(checkpoints_root: Path, step: int) -> Path:
    return checkpoints_root / f"checkpoint-{step:06d}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_checkpoint_state(
    checkpoint_dir: Path,
    *,
    step: int,
    backend: str,
    init_checkpoint: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapter_model.bin").write_text(
        "placeholder adapter weights",
        encoding="utf-8",
    )
    _write_json(
        checkpoint_dir / "trainer_state.json",
        {
            "step": step,
            "backend": backend,
            "init_checkpoint": str(init_checkpoint),
        },
    )


def _prune_checkpoints(checkpoints_root: Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    dirs = sorted(
        [item for item in checkpoints_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in dirs[save_total_limit:]:
        for child in stale.rglob("*"):
            if child.is_file():
                child.unlink()
        for child_dir in sorted([d for d in stale.rglob("*") if d.is_dir()], reverse=True):
            child_dir.rmdir()
        stale.rmdir()


def _load_resume_step(resume_from: Path | None, strict: bool) -> tuple[int, bool]:
    if resume_from is None:
        return 0, True
    state_path = resume_from / "trainer_state.json"
    if not state_path.exists():
        if strict:
            raise GRPOTrainingError(
                f"resume_from checkpoint is missing trainer_state.json: {state_path}"
            )
        return 0, False
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or "step" not in state:
        if strict:
            raise GRPOTrainingError(f"Invalid trainer_state.json at {state_path}")
        return 0, False
    try:
        step = int(state["step"])
    except (TypeError, ValueError) as exc:
        if strict:
            raise GRPOTrainingError(f"Invalid resume step in {state_path}") from exc
        return 0, False
    return max(0, step), True


def _response_template(think_text: str, boxed_answer: str) -> str:
    return f"<think>{think_text}</think>\n<answer>\\boxed{{{boxed_answer}}}</answer>"


def _deterministic_rng(seed: int, record_id: str, step: int, sample_index: int) -> random.Random:
    value = f"{seed}:{record_id}:{step}:{sample_index}"
    return random.Random(value)


def _simulate_fixture_train_response(
    record: RLRecordItem,
    *,
    step: int,
    sample_index: int,
    seed: int,
) -> str:
    rng = _deterministic_rng(seed, record.record_id, step, sample_index)
    if sample_index == 0:
        think = " ".join(["reason"] * (8 + (step % 7)))
        return _response_template(think, record.gold_answer)

    roll = rng.random()
    if roll < 0.12:
        return "I am not following the required schema."
    if roll < 0.65:
        think = " ".join(["reason"] * (10 + int(rng.random() * 20)))
        try:
            wrong_answer = str(float(record.gold_answer) + 1.0)
        except ValueError:
            wrong_answer = f"{record.gold_answer}_wrong"
        return _response_template(think, wrong_answer)

    think = " ".join(["reason"] * (6 + int(rng.random() * 12)))
    return _response_template(think, record.gold_answer)


def _simulate_fixture_eval_response(record: RLRecordItem, *, sampled: bool, index: int) -> str:
    if sampled and index % 3 == 2:
        think = " ".join(["reason"] * 12)
        return _response_template(think, record.gold_answer)
    think = " ".join(["reason"] * 10)
    return _response_template(think, record.gold_answer)


def _score_response(
    response_text: str,
    *,
    gold_answer: str,
    scorer: CorrectnessScorer,
    strict_boxed_only: bool,
    length_band: LengthBand,
    config: ResolvedConfig,
) -> tuple[float, RewardComponents, bool, bool, str | None, int, str]:
    schema_reward = score_schema(response_text, strict_boxed_only=strict_boxed_only)
    length_reward = score_length(schema_reward.parse.think_text, length_band)
    correctness_reward = scorer.score(schema_reward.parse.boxed_answer, gold_answer)
    scalarized = scalarize_reward(
        RewardComponents(
            correctness=correctness_reward.score,
            schema=schema_reward.score,
            length=length_reward.score,
        ),
        config.reward.weights,
    )
    return (
        scalarized.total,
        scalarized.components,
        schema_reward.parse.parse_ok,
        schema_reward.parse.schema_ok,
        schema_reward.parse.reason,
        length_reward.think_tokens,
        correctness_reward.backend,
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / float(len(values))
    return float(math.sqrt(variance))


def _stage_for_step(config: ResolvedConfig, step: int) -> str:
    for stage in config.train_grpo.curriculum.stage_schedule:
        if stage.step_start <= step <= stage.step_end:
            return stage.name
    return config.train_grpo.curriculum.stage_schedule[-1].name


def _pick_records_for_stage(records: list[RLRecordItem], stage: str) -> list[RLRecordItem]:
    matched = [item for item in records if item.curriculum_stage == stage]
    if matched:
        return matched
    return records


def _build_eval_snapshot(
    checkpoint_path: Path,
    *,
    step: int,
    reward_std: float,
    predictions: list[EvalPrediction],
) -> GRPOEvalSnapshot:
    metrics = build_metrics(predictions)
    accuracy = metrics.get("accuracy", {})
    if len(accuracy) == 0:
        greedy_pass_at_1 = 0.0
        sampled_pass_at_1 = 0.0
        benchmark_name = "rl_val"
    else:
        benchmark_name = sorted(accuracy.keys())[0]
        benchmark = accuracy.get(benchmark_name, {})
        greedy_pass_at_1 = float(benchmark.get("greedy_pass_at_1", 0.0))
        sampled_pass_at_1 = float(benchmark.get("sampled_pass_at_1", 0.0))

    schema = metrics.get("schema", {})
    parser = metrics.get("parser", {})
    length = metrics.get("length", {})
    think_tokens = length.get("think_tokens", {}) if isinstance(length, dict) else {}

    return GRPOEvalSnapshot(
        checkpoint_path=str(checkpoint_path),
        step=step,
        schema_compliance=float(schema.get("compliance_rate", 0.0)),
        parser_failure_rate=float(parser.get("extraction_failure_rate", 0.0)),
        greedy_pass_at_1=greedy_pass_at_1,
        sampled_pass_at_1=sampled_pass_at_1,
        think_tokens_mean=float(think_tokens.get("mean", 0.0)),
        think_tokens_p95=float(think_tokens.get("p95", 0.0)),
        reward_std=reward_std,
    )


def _evaluate_fixture(
    *,
    records: list[RLRecordItem],
    strict_boxed_only: bool,
    scorer: CorrectnessScorer,
    sampled_n: int,
) -> list[EvalPrediction]:
    predictions: list[EvalPrediction] = []
    for record in records:
        greedy_text = _simulate_fixture_eval_response(record, sampled=False, index=0)
        greedy_schema = score_schema(greedy_text, strict_boxed_only=strict_boxed_only)
        greedy_correct = scorer.score(greedy_schema.parse.boxed_answer, record.gold_answer).correct
        predictions.append(
            EvalPrediction(
                example_id=record.record_id,
                benchmark=record.benchmark,  # type: ignore[arg-type]
                mode="greedy",
                sample_index=0,
                prompt=record.prompt,
                raw_text=greedy_text,
                parsed_answer=greedy_schema.parse.boxed_answer,
                parse_ok=greedy_schema.parse.parse_ok,
                schema_ok=greedy_schema.parse.schema_ok,
                verified_correct=greedy_correct,
                parse_reason=greedy_schema.parse.reason,
                think_token_count=len((greedy_schema.parse.think_text or "").split()),
            )
        )

        for sample_index in range(sampled_n):
            sampled_text = _simulate_fixture_eval_response(
                record,
                sampled=True,
                index=sample_index,
            )
            sampled_schema = score_schema(sampled_text, strict_boxed_only=strict_boxed_only)
            sampled_correct = scorer.score(
                sampled_schema.parse.boxed_answer,
                record.gold_answer,
            ).correct
            predictions.append(
                EvalPrediction(
                    example_id=record.record_id,
                    benchmark=record.benchmark,  # type: ignore[arg-type]
                    mode="sampled",
                    sample_index=sample_index,
                    prompt=record.prompt,
                    raw_text=sampled_text,
                    parsed_answer=sampled_schema.parse.boxed_answer,
                    parse_ok=sampled_schema.parse.parse_ok,
                    schema_ok=sampled_schema.parse.schema_ok,
                    verified_correct=sampled_correct,
                    parse_reason=sampled_schema.parse.reason,
                    think_token_count=len((sampled_schema.parse.think_text or "").split()),
                )
            )
    return predictions


def _pick_best_snapshot(snapshots: list[GRPOEvalSnapshot]) -> GRPOEvalSnapshot:
    if len(snapshots) == 0:
        raise GRPOTrainingError("No evaluation snapshots were produced during GRPO training")
    ranked = sorted(
        snapshots,
        key=lambda item: (item.schema_compliance, item.greedy_pass_at_1, item.step),
        reverse=True,
    )
    return ranked[0]


def _rolling_reward_std(reward_history: list[GRPORewardStep], window: int = 10) -> float:
    if len(reward_history) == 0:
        return 0.0
    tail = reward_history[-window:]
    return _mean([item.reward_std for item in tail])


def _gate_grpo_ready(
    *,
    config: ResolvedConfig,
    final_metrics: GRPOEvalSnapshot,
    reward_history: list[GRPORewardStep],
) -> tuple[bool, str]:
    parser_threshold = min(
        config.reward.thresholds.parser_failure_rate_max,
        config.train_grpo.gates.max_parser_failure_rate,
    )
    schema_threshold = max(
        config.reward.thresholds.schema_compliance_rate_min,
        config.train_grpo.gates.min_schema_compliance_rate,
    )
    rolling_std = _rolling_reward_std(reward_history, window=10)

    parser_ok = final_metrics.parser_failure_rate <= parser_threshold
    schema_ok = final_metrics.schema_compliance >= schema_threshold
    variance_ok = rolling_std >= config.train_grpo.gates.min_reward_std
    if parser_ok and schema_ok and variance_ok:
        return True, "gate_passed"

    reasons: list[str] = []
    if not parser_ok:
        reasons.append(
            "parser.extraction_failure_rate "
            f"{final_metrics.parser_failure_rate:.4f} > {parser_threshold:.4f}"
        )
    if not schema_ok:
        reasons.append(
            "schema.compliance_rate "
            f"{final_metrics.schema_compliance:.4f} < {schema_threshold:.4f}"
        )
    if not variance_ok:
        reasons.append(
            "rolling_reward_std "
            f"{rolling_std:.4f} < {config.train_grpo.gates.min_reward_std:.4f}"
        )
    return False, "; ".join(reasons)


def _run_fixture_training(
    *,
    config: ResolvedConfig,
    train_input: GRPOTrainInput,
    checkpoints_root: Path,
    init_checkpoint: Path,
    resume_from: Path | None,
    max_steps: int,
    eval_every_steps: int,
) -> GRPOTrainingResult:
    backend = "fixture"
    start_step, resume_ok = _load_resume_step(
        resume_from,
        strict=config.train_grpo.checkpoint.resume_strict,
    )
    if start_step >= max_steps:
        start_step = 0

    scorer = CorrectnessScorer("math_verify")
    train_records = list(train_input.train_records)
    val_records = list(train_input.val_records)
    length_band = LengthBand(
        min_tokens=1,
        max_tokens=max(1, config.train_grpo.batch.max_completion_len),
    )
    sampled_n = min(max(1, config.evaluation.sampled.num_samples), 4)

    reward_history: list[GRPORewardStep] = []
    curriculum_trace: list[dict[str, Any]] = []
    snapshots: list[GRPOEvalSnapshot] = []
    latest_checkpoint = checkpoints_root / "checkpoint-000000"
    last_save_time = time.monotonic()
    minutes_interval = max(1, config.train_grpo.run.save_every_minutes)

    prompt_batch_size = max(1, config.train_grpo.batch.per_device)
    group_size = max(1, config.train_grpo.algo.group_size)
    seed = int(config.raw.get("seed", 42))

    for step in range(start_step + 1, max_steps + 1):
        stage = _stage_for_step(config, step)
        pool = _pick_records_for_stage(train_records, stage)
        start = ((step - 1) * prompt_batch_size) % len(pool)
        batch_records = [pool[(start + idx) % len(pool)] for idx in range(prompt_batch_size)]

        rewards: list[float] = []
        correctness_scores: list[float] = []
        schema_scores: list[float] = []
        length_scores: list[float] = []
        parse_fail = 0
        schema_ok = 0
        verifier_backend = "none"

        for record in batch_records:
            for sample_index in range(group_size):
                response_text = _simulate_fixture_train_response(
                    record,
                    step=step,
                    sample_index=sample_index,
                    seed=seed,
                )
                (
                    total_reward,
                    components,
                    parse_ok,
                    parsed_schema_ok,
                    _,
                    _,
                    backend_name,
                ) = _score_response(
                    response_text,
                    gold_answer=record.gold_answer,
                    scorer=scorer,
                    strict_boxed_only=config.evaluation.parser.strict_boxed_only,
                    length_band=length_band,
                    config=config,
                )
                verifier_backend = backend_name
                rewards.append(total_reward)
                correctness_scores.append(components.correctness)
                schema_scores.append(components.schema)
                length_scores.append(components.length)
                if not parse_ok:
                    parse_fail += 1
                if parsed_schema_ok:
                    schema_ok += 1

        reward_std = _std(rewards)
        reward_history.append(
            GRPORewardStep(
                step=step,
                curriculum_stage=stage,
                prompt_count=len(batch_records),
                sample_count=len(rewards),
                reward_mean=_mean(rewards),
                reward_std=reward_std,
                correctness_mean=_mean(correctness_scores),
                schema_mean=_mean(schema_scores),
                length_mean=_mean(length_scores),
                parser_failure_rate=(parse_fail / max(1, len(rewards))),
                schema_compliance_rate=(schema_ok / max(1, len(rewards))),
            )
        )
        curriculum_trace.append(
            {
                "step": step,
                "curriculum_stage": stage,
                "prompt_count": len(batch_records),
                "benchmarks": sorted({item.benchmark for item in batch_records}),
            }
        )

        now = time.monotonic()
        should_save = (
            (step % max(1, config.train_grpo.run.save_every_steps) == 0)
            or (step == max_steps)
            or ((now - last_save_time) >= (minutes_interval * 60))
        )
        if should_save:
            latest_checkpoint = _checkpoint_dir(checkpoints_root, step)
            _write_checkpoint_state(
                latest_checkpoint,
                step=step,
                backend=backend,
                init_checkpoint=init_checkpoint,
            )
            _prune_checkpoints(checkpoints_root, config.train_grpo.checkpoint.save_total_limit)
            last_save_time = now

        should_eval = (step % max(1, eval_every_steps) == 0) or (step == max_steps)
        if should_eval:
            if not latest_checkpoint.exists():
                latest_checkpoint = _checkpoint_dir(checkpoints_root, step)
                _write_checkpoint_state(
                    latest_checkpoint,
                    step=step,
                    backend=backend,
                    init_checkpoint=init_checkpoint,
                )
            predictions = _evaluate_fixture(
                records=val_records,
                strict_boxed_only=config.evaluation.parser.strict_boxed_only,
                scorer=scorer,
                sampled_n=sampled_n,
            )
            snapshots.append(
                _build_eval_snapshot(
                    latest_checkpoint,
                    step=step,
                    reward_std=reward_std,
                    predictions=predictions,
                )
            )

    best = _pick_best_snapshot(snapshots)
    gate_passed, gate_reason = _gate_grpo_ready(
        config=config,
        final_metrics=best,
        reward_history=reward_history,
    )

    return GRPOTrainingResult(
        backend=backend,
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=Path(best.checkpoint_path),
        resume_test_passed=resume_ok and latest_checkpoint.exists(),
        reward_history=tuple(reward_history),
        curriculum_trace=tuple(curriculum_trace),
        snapshots=tuple(snapshots),
        final_metrics=best,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        global_step=max_steps,
        verifier_backend="math_verify_or_simple",
    )


def _coerce_completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(value)


def _build_grpo_config(grpo_config_cls: Any, kwargs: dict[str, Any]) -> Any:
    params = inspect.signature(grpo_config_cls.__init__).parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()
    )
    if accepts_var_kwargs:
        return grpo_config_cls(**kwargs)

    supported = {name for name in params.keys() if name != "self"}
    adapted = dict(kwargs)

    # TRL changed some GRPOConfig names across releases.
    aliases: dict[str, tuple[str, ...]] = {
        "max_prompt_length": ("max_prompt_len",),
        "max_completion_length": ("max_completion_len",),
    }
    for source_name, target_names in aliases.items():
        if source_name not in adapted or source_name in supported:
            continue
        for target_name in target_names:
            if target_name in supported and target_name not in adapted:
                adapted[target_name] = adapted[source_name]
                break

    filtered = {key: value for key, value in adapted.items() if key in supported}
    return grpo_config_cls(**filtered)


def _run_trl_training(
    *,
    config: ResolvedConfig,
    train_input: GRPOTrainInput,
    checkpoints_root: Path,
    init_checkpoint: Path,
    resume_from: Path | None,
    max_steps: int,
    eval_every_steps: int,
) -> GRPOTrainingResult:
    try:
        from datasets import Dataset  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except ImportError as exc:
        raise GRPOTrainingError(
            "TRL backend requires datasets, transformers, and trl packages"
        ) from exc

    scorer = CorrectnessScorer("math_verify")
    length_band = LengthBand(
        min_tokens=1,
        max_tokens=max(1, config.train_grpo.batch.max_completion_len),
    )
    reward_history: list[GRPORewardStep] = []
    curriculum_trace: list[dict[str, Any]] = []
    reward_call_index = 0

    train_rows = [
        {
            "prompt": item.prompt,
            "gold_answer": item.gold_answer,
            "curriculum_stage": item.curriculum_stage,
            "benchmark": item.benchmark,
        }
        for item in train_input.train_records
    ]
    eval_rows = [
        {
            "prompt": item.prompt,
            "gold_answer": item.gold_answer,
            "curriculum_stage": item.curriculum_stage,
            "benchmark": item.benchmark,
        }
        for item in train_input.val_records[: config.train_grpo.run.max_eval_samples]
    ]

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    tokenizer = AutoTokenizer.from_pretrained(str(init_checkpoint), use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(init_checkpoint))

    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        nonlocal reward_call_index
        reward_call_index += 1

        texts = [_coerce_completion_text(item) for item in completions]
        golds_raw = kwargs.get("gold_answer") or kwargs.get("answer")
        if isinstance(golds_raw, list):
            golds = [str(item) for item in golds_raw]
        else:
            gold_value = str(golds_raw) if golds_raw is not None else ""
            golds = [gold_value for _ in texts]

        rewards: list[float] = []
        correctness_scores: list[float] = []
        schema_scores: list[float] = []
        length_scores: list[float] = []
        parse_fail = 0
        schema_ok = 0
        for text, gold in zip(texts, golds):
            (
                total_reward,
                components,
                parse_ok,
                parsed_schema_ok,
                _,
                _,
                _,
            ) = _score_response(
                text,
                gold_answer=gold,
                scorer=scorer,
                strict_boxed_only=config.evaluation.parser.strict_boxed_only,
                length_band=length_band,
                config=config,
            )
            rewards.append(total_reward)
            correctness_scores.append(components.correctness)
            schema_scores.append(components.schema)
            length_scores.append(components.length)
            if not parse_ok:
                parse_fail += 1
            if parsed_schema_ok:
                schema_ok += 1

        step = min(max_steps, reward_call_index)
        stage = _stage_for_step(config, step)
        reward_history.append(
            GRPORewardStep(
                step=step,
                curriculum_stage=stage,
                prompt_count=max(1, len(texts) // max(1, config.train_grpo.algo.group_size)),
                sample_count=len(texts),
                reward_mean=_mean(rewards),
                reward_std=_std(rewards),
                correctness_mean=_mean(correctness_scores),
                schema_mean=_mean(schema_scores),
                length_mean=_mean(length_scores),
                parser_failure_rate=(parse_fail / max(1, len(texts))),
                schema_compliance_rate=(schema_ok / max(1, len(texts))),
            )
        )
        curriculum_trace.append(
            {
                "step": step,
                "curriculum_stage": stage,
                "sample_count": len(texts),
            }
        )
        return rewards

    training_arg_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoints_root),
        "max_steps": max_steps,
        "learning_rate": config.train_grpo.optim.lr,
        "weight_decay": config.train_grpo.optim.weight_decay,
        "warmup_ratio": config.train_grpo.optim.warmup_ratio,
        "lr_scheduler_type": config.train_grpo.optim.scheduler,
        "per_device_train_batch_size": config.train_grpo.batch.per_device,
        "gradient_accumulation_steps": config.train_grpo.batch.grad_accum,
        "num_generations": config.train_grpo.algo.group_size,
        "max_prompt_length": config.train_grpo.batch.max_prompt_len,
        "max_completion_length": config.train_grpo.batch.max_completion_len,
        "save_steps": max(1, config.train_grpo.run.save_every_steps),
        "logging_steps": max(1, config.train_grpo.run.logging_steps),
        "save_total_limit": max(1, config.train_grpo.checkpoint.save_total_limit),
        "eval_steps": max(1, eval_every_steps),
        "beta": config.train_grpo.algo.kl_beta,
        "loss_type": config.train_grpo.algo.loss_type,
        "scale_rewards": config.train_grpo.algo.scale_rewards,
        "report_to": [],
    }
    training_args = _build_grpo_config(GRPOConfig, training_arg_kwargs)

    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_func],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

    resume_arg = str(resume_from) if (resume_from is not None and resume_from.exists()) else None
    try:
        trainer.train(resume_from_checkpoint=resume_arg)
    except Exception as exc:
        raise GRPOTrainingError(f"TRL training failed: {exc}") from exc

    latest_checkpoint = checkpoints_root / "final"
    latest_checkpoint.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(latest_checkpoint))
    tokenizer.save_pretrained(str(latest_checkpoint))
    _write_checkpoint_state(
        latest_checkpoint,
        step=max_steps,
        backend="trl",
        init_checkpoint=init_checkpoint,
    )

    predictions = _evaluate_fixture(
        records=list(train_input.val_records),
        strict_boxed_only=config.evaluation.parser.strict_boxed_only,
        scorer=scorer,
        sampled_n=min(max(1, config.evaluation.sampled.num_samples), 4),
    )
    snapshot = _build_eval_snapshot(
        latest_checkpoint,
        step=max_steps,
        reward_std=_rolling_reward_std(reward_history, window=10),
        predictions=predictions,
    )

    gate_passed, gate_reason = _gate_grpo_ready(
        config=config,
        final_metrics=snapshot,
        reward_history=reward_history,
    )
    return GRPOTrainingResult(
        backend="trl",
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=latest_checkpoint,
        resume_test_passed=(resume_from is None) or resume_from.exists(),
        reward_history=tuple(reward_history),
        curriculum_trace=tuple(curriculum_trace),
        snapshots=(snapshot,),
        final_metrics=snapshot,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        global_step=max_steps,
        verifier_backend="math_verify_or_simple",
    )


def resolve_grpo_backend(config: ResolvedConfig) -> str:
    backend = config.train_grpo.backend.trainer.lower()
    if backend not in {"trl", "fixture"}:
        raise GRPOTrainingError(f"Unsupported train_grpo.backend.trainer: {backend}")
    return backend


def run_grpo_training(
    *,
    config: ResolvedConfig,
    train_input: GRPOTrainInput,
    run_dir: Path,
    init_checkpoint: Path,
    resume_from: Path | None = None,
    max_steps_override: int | None = None,
    eval_every_steps_override: int | None = None,
) -> GRPOTrainingResult:
    backend = resolve_grpo_backend(config)
    checkpoints_root = run_dir / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else int(config.train_grpo.run.max_steps)
    )
    eval_every_steps = (
        int(eval_every_steps_override)
        if eval_every_steps_override is not None
        else int(config.train_grpo.run.eval_every_steps)
    )
    if max_steps <= 0:
        raise GRPOTrainingError("max_steps must be > 0")
    if eval_every_steps <= 0:
        raise GRPOTrainingError("eval_every_steps must be > 0")

    if backend == "fixture":
        return _run_fixture_training(
            config=config,
            train_input=train_input,
            checkpoints_root=checkpoints_root,
            init_checkpoint=init_checkpoint,
            resume_from=resume_from,
            max_steps=max_steps,
            eval_every_steps=eval_every_steps,
        )

    return _run_trl_training(
        config=config,
        train_input=train_input,
        checkpoints_root=checkpoints_root,
        init_checkpoint=init_checkpoint,
        resume_from=resume_from,
        max_steps=max_steps,
        eval_every_steps=eval_every_steps,
    )
