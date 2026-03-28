from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from microreasoner.prompting import tokenize_generation_prompt, tokenize_supervised_text
from microreasoner.runtime.models import ResolvedConfig
from microreasoner.train.hf_compat import prepare_trainer_optimizer_compat
from microreasoner.train.sft_data import SFTRecordItem, SFTTrainInput
from microreasoner.train.sft_eval import SFTMetrics, evaluate_fixture, evaluate_transformers
from microreasoner.train.sft_model import (
    SFTModelSetupError,
    _cuda_bf16_supported,
    build_transformers_model,
    resolve_sft_backend,
    select_sft_mode,
)
from microreasoner.train.sft_selection import SnapshotMetrics, gate_sft_ready, pick_best_snapshot


class SFTTrainingError(RuntimeError):
    """Raised when SFT training setup/execution fails."""


@dataclass(frozen=True)
class SFTTrainingResult:
    selected_mode: str
    backend: str
    latest_checkpoint: Path
    best_checkpoint: Path
    resume_test_passed: bool
    snapshots: tuple[SnapshotMetrics, ...]
    final_metrics: SFTMetrics
    gate_passed: bool
    gate_reason: str
    global_step: int


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _checkpoint_dir(checkpoints_root: Path, step: int) -> Path:
    return checkpoints_root / f"checkpoint-{step:06d}"


def _write_checkpoint_state(
    checkpoint_dir: Path,
    *,
    step: int,
    selected_mode: str,
    backend: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "adapter_model.bin").write_text(
        "placeholder adapter weights",
        encoding="utf-8",
    )
    _write_json(
        checkpoint_dir / "trainer_state.json",
        {"step": step, "selected_mode": selected_mode, "backend": backend},
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
            raise SFTTrainingError(
                f"resume_from checkpoint is missing trainer_state.json: {state_path}"
            )
        return 0, False
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict) or "step" not in state:
        if strict:
            raise SFTTrainingError(f"Invalid trainer_state.json at {state_path}")
        return 0, False
    try:
        step = int(state["step"])
    except (TypeError, ValueError) as exc:
        if strict:
            raise SFTTrainingError(f"Invalid resume step in {state_path}") from exc
        return 0, False
    return max(0, step), True


def _record_snapshot(
    snapshots: list[SnapshotMetrics],
    *,
    checkpoint_path: Path,
    step: int,
    metrics: SFTMetrics,
) -> None:
    snapshots.append(
        SnapshotMetrics(
            checkpoint_path=str(checkpoint_path),
            step=step,
            schema_compliance=metrics.schema_compliance,
            greedy_pass_at_1=metrics.greedy_pass_at_1,
            sampled_pass_at_1=metrics.sampled_pass_at_1,
            parser_failure_rate=metrics.parser_failure_rate,
        )
    )


def _run_fixture_training(
    *,
    config: ResolvedConfig,
    train_input: SFTTrainInput,
    checkpoints_root: Path,
    resume_from: Path | None,
    max_steps: int,
    eval_every_steps: int,
) -> SFTTrainingResult:
    selected_mode = select_sft_mode(config).selected_mode
    backend = "fixture"

    start_step, resume_ok = _load_resume_step(
        resume_from,
        strict=config.train_sft.checkpoint.resume_strict,
    )
    if start_step >= max_steps:
        start_step = 0

    snapshots: list[SnapshotMetrics] = []
    latest_checkpoint = checkpoints_root / "checkpoint-000000"
    last_save_time = time.monotonic()
    minutes_interval = max(1, config.train_sft.run.save_every_minutes)

    for step in range(start_step + 1, max_steps + 1):
        now = time.monotonic()
        should_save = (
            (step % max(1, config.train_sft.run.save_every_steps) == 0)
            or (step == max_steps)
            or ((now - last_save_time) >= (minutes_interval * 60))
        )
        if should_save:
            latest_checkpoint = _checkpoint_dir(checkpoints_root, step)
            _write_checkpoint_state(
                latest_checkpoint,
                step=step,
                selected_mode=selected_mode,
                backend=backend,
            )
            _prune_checkpoints(checkpoints_root, config.train_sft.checkpoint.save_total_limit)
            last_save_time = now

        should_eval = (step % max(1, eval_every_steps) == 0) or (step == max_steps)
        if should_eval:
            metrics = evaluate_fixture(list(train_input.val_records))
            if not latest_checkpoint.exists():
                latest_checkpoint = _checkpoint_dir(checkpoints_root, step)
                _write_checkpoint_state(
                    latest_checkpoint,
                    step=step,
                    selected_mode=selected_mode,
                    backend=backend,
                )
            _record_snapshot(
                snapshots,
                checkpoint_path=latest_checkpoint,
                step=step,
                metrics=metrics,
            )

    if len(snapshots) == 0:
        raise SFTTrainingError("No evaluation snapshots were produced during fixture training")

    best = pick_best_snapshot(
        snapshots,
        primary_metric=config.train_sft.selection.primary_metric,
        secondary_metric=config.train_sft.selection.secondary_metric,
    )
    best_checkpoint = Path(best.checkpoint_path)
    final_metrics = SFTMetrics(
        schema_compliance=best.schema_compliance,
        parser_failure_rate=best.parser_failure_rate,
        greedy_pass_at_1=best.greedy_pass_at_1,
        sampled_pass_at_1=best.sampled_pass_at_1,
        think_tokens_mean=evaluate_fixture(list(train_input.val_records)).think_tokens_mean,
        think_tokens_p95=evaluate_fixture(list(train_input.val_records)).think_tokens_p95,
        eval_size=len(train_input.val_records),
    )
    gate_passed, gate_reason = gate_sft_ready(
        config=config,
        final_schema_compliance=final_metrics.schema_compliance,
        final_greedy_pass_at_1=final_metrics.greedy_pass_at_1,
    )
    return SFTTrainingResult(
        selected_mode=selected_mode,
        backend=backend,
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=best_checkpoint,
        resume_test_passed=resume_ok and latest_checkpoint.exists(),
        snapshots=tuple(snapshots),
        final_metrics=final_metrics,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        global_step=max_steps,
    )


def _build_torch_datasets(
    *,
    records: list[SFTRecordItem],
    tokenizer: Any,
    max_seq_len: int,
    torch_module: Any,
) -> Any:
    def _build_supervised_example(row: SFTRecordItem) -> dict[str, list[int]]:
        prompt_encoded = tokenize_generation_prompt(
            tokenizer,
            row.prompt,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        encoded = tokenize_supervised_text(
            tokenizer,
            row.prompt,
            row.target_response,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        prompt_ids = [int(item) for item in prompt_encoded["input_ids"]]
        input_ids = [int(item) for item in encoded["input_ids"]]
        attention_mask = [int(item) for item in encoded["attention_mask"]]

        prompt_token_count = 0
        for prompt_token, full_token in zip(prompt_ids, input_ids):
            if prompt_token != full_token:
                break
            prompt_token_count += 1

        labels = ([-100] * prompt_token_count) + input_ids[prompt_token_count:]

        if prompt_token_count >= len(input_ids):
            raise SFTTrainingError(
                "SFT example "
                f"{row.record_id!r} has no supervised target tokens after tokenization/truncation; "
                "increase train_sft.batch.max_seq_len"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    class PromptTargetDataset(torch_module.utils.data.Dataset):
        def __init__(self, items: list[SFTRecordItem]) -> None:
            self._rows: list[dict[str, Any]] = []
            for row in items:
                encoded = _build_supervised_example(row)
                input_ids = torch_module.tensor(encoded["input_ids"], dtype=torch_module.long)
                attention_mask = torch_module.tensor(
                    encoded["attention_mask"],
                    dtype=torch_module.long,
                )
                labels = torch_module.tensor(encoded["labels"], dtype=torch_module.long)
                self._rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
                )

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self._rows[idx]

    return PromptTargetDataset(records)


def _build_collator(torch_module: Any, pad_token_id: int) -> Any:
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [item["input_ids"] for item in features]
        attention_mask = [item["attention_mask"] for item in features]
        labels = [item["labels"] for item in features]
        return {
            "input_ids": torch_module.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=pad_token_id,
            ),
            "attention_mask": torch_module.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch_module.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            ),
        }

    return collate


def _build_training_arguments(training_arguments_cls: Any, kwargs: dict[str, Any]) -> Any:
    params = inspect.signature(training_arguments_cls.__init__).parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()
    )
    if accepts_var_kwargs:
        return training_arguments_cls(**kwargs)

    supported = {name for name in params.keys() if name != "self"}
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return training_arguments_cls(**filtered)


def _run_transformers_training(
    *,
    config: ResolvedConfig,
    train_input: SFTTrainInput,
    checkpoints_root: Path,
    resume_from: Path | None,
    max_steps: int,
    eval_every_steps: int,
) -> SFTTrainingResult:
    mode_selection = select_sft_mode(config)
    selected_mode = mode_selection.selected_mode

    model_bundle = build_transformers_model(
        config=config,
        selected_mode=selected_mode,
        checkpoint_or_model=config.model.default_base_model,
    )
    stack = model_bundle.stack
    torch = stack["torch"]
    Trainer = stack["Trainer"]
    TrainingArguments = stack["TrainingArguments"]

    train_dataset = _build_torch_datasets(
        records=list(train_input.train_records),
        tokenizer=model_bundle.tokenizer,
        max_seq_len=config.train_sft.batch.max_seq_len,
        torch_module=torch,
    )
    eval_records = list(train_input.val_records)[: config.train_sft.run.max_eval_samples]
    eval_dataset = _build_torch_datasets(
        records=eval_records,
        tokenizer=model_bundle.tokenizer,
        max_seq_len=config.train_sft.batch.max_seq_len,
        torch_module=torch,
    )
    collator = _build_collator(torch, model_bundle.tokenizer.pad_token_id)
    bf16_enabled = _cuda_bf16_supported(torch)

    training_arg_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoints_root),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": config.train_sft.batch.per_device,
        "per_device_eval_batch_size": config.train_sft.batch.per_device,
        "gradient_accumulation_steps": config.train_sft.batch.grad_accum,
        "learning_rate": config.train_sft.optim.lr,
        "warmup_ratio": config.train_sft.optim.warmup_ratio,
        "lr_scheduler_type": config.train_sft.optim.scheduler,
        "weight_decay": config.train_sft.optim.weight_decay,
        "num_train_epochs": float(config.train_sft.run.epochs),
        "max_steps": max_steps,
        "logging_steps": max(1, config.train_sft.run.logging_steps),
        "save_steps": max(1, config.train_sft.run.save_every_steps),
        "save_total_limit": max(1, config.train_sft.checkpoint.save_total_limit),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": True,
        "bf16": bf16_enabled,
        "fp16": bool(torch.cuda.is_available() and not bf16_enabled),
        "max_grad_norm": 1.0,
    }
    training_args = _build_training_arguments(TrainingArguments, training_arg_kwargs)

    trainer_kwargs = {
        "model": model_bundle.model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    try:
        trainer = Trainer(**trainer_kwargs, tokenizer=model_bundle.tokenizer)
    except TypeError:
        try:
            trainer = Trainer(**trainer_kwargs, processing_class=model_bundle.tokenizer)
        except TypeError:
            trainer = Trainer(**trainer_kwargs)
    prepare_trainer_optimizer_compat(trainer)

    resume_arg: str | None = None
    if resume_from is not None and resume_from.exists():
        resume_arg = str(resume_from)
    trainer.train(resume_from_checkpoint=resume_arg)

    latest_checkpoint = checkpoints_root / "final"
    latest_checkpoint.mkdir(parents=True, exist_ok=True)
    model_bundle.model.save_pretrained(latest_checkpoint)
    model_bundle.tokenizer.save_pretrained(latest_checkpoint)
    _write_json(
        latest_checkpoint / "trainer_state.json",
        {
            "step": max_steps,
            "selected_mode": selected_mode,
            "backend": "transformers",
        },
    )

    sampled_n = min(max(1, config.evaluation.sampled.num_samples), 4)
    eval_metrics = evaluate_transformers(
        records=eval_records,
        model_bundle=model_bundle,
        max_new_tokens=config.evaluation.inference.max_new_tokens,
        sampled_temperature=config.evaluation.sampled.temperature,
        sampled_top_p=config.evaluation.sampled.top_p,
        sampled_n=sampled_n,
    )
    snapshot = SnapshotMetrics(
        checkpoint_path=str(latest_checkpoint),
        step=max_steps,
        schema_compliance=eval_metrics.schema_compliance,
        greedy_pass_at_1=eval_metrics.greedy_pass_at_1,
        sampled_pass_at_1=eval_metrics.sampled_pass_at_1,
        parser_failure_rate=eval_metrics.parser_failure_rate,
    )
    gate_passed, gate_reason = gate_sft_ready(
        config=config,
        final_schema_compliance=eval_metrics.schema_compliance,
        final_greedy_pass_at_1=eval_metrics.greedy_pass_at_1,
    )
    return SFTTrainingResult(
        selected_mode=selected_mode,
        backend="transformers",
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=latest_checkpoint,
        resume_test_passed=(resume_from is None) or resume_from.exists(),
        snapshots=(snapshot,),
        final_metrics=eval_metrics,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        global_step=max_steps,
    )


def run_sft_training(
    *,
    config: ResolvedConfig,
    train_input: SFTTrainInput,
    run_dir: Path,
    resume_from: Path | None = None,
    max_steps_override: int | None = None,
    eval_every_steps_override: int | None = None,
) -> SFTTrainingResult:
    backend = resolve_sft_backend(config)
    checkpoints_root = run_dir / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    max_steps = (
        int(max_steps_override)
        if max_steps_override is not None
        else int(config.train_sft.run.max_steps)
    )
    eval_every_steps = (
        int(eval_every_steps_override)
        if eval_every_steps_override is not None
        else int(config.train_sft.run.eval_every_steps)
    )
    if max_steps <= 0:
        raise SFTTrainingError("max_steps must be > 0")
    if eval_every_steps <= 0:
        raise SFTTrainingError("eval_every_steps must be > 0")

    if backend == "fixture":
        return _run_fixture_training(
            config=config,
            train_input=train_input,
            checkpoints_root=checkpoints_root,
            resume_from=resume_from,
            max_steps=max_steps,
            eval_every_steps=eval_every_steps,
        )

    if backend == "transformers":
        try:
            return _run_transformers_training(
                config=config,
                train_input=train_input,
                checkpoints_root=checkpoints_root,
                resume_from=resume_from,
                max_steps=max_steps,
                eval_every_steps=eval_every_steps,
            )
        except SFTModelSetupError as exc:
            raise SFTTrainingError(str(exc)) from exc

    raise SFTTrainingError(f"Unsupported backend: {backend}")

