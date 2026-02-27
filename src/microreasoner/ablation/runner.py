from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microreasoner.ablation.baseline_rs_sft import build_rs_sft_manifest
from microreasoner.ablation.report import build_rows, write_csv, write_markdown
from microreasoner.ablation.types import (
    AblationRow,
    CostSnapshot,
    ExperimentOutcome,
    ExperimentSpec,
    MetricSnapshot,
    RunArtifacts,
)
from microreasoner.cli.main import main as cli_main
from microreasoner.runtime.configuration import resolve_config
from microreasoner.runtime.context import repo_root
from microreasoner.train.grpo_data import load_grpo_train_input


@dataclass(frozen=True)
class AblationRunResult:
    session_id: str
    rows: tuple[AblationRow, ...]
    csv_path: Path
    markdown_path: Path


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line == "":
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSON at {path}:{line_no}")
            rows.append(payload)
    return rows


def _parse_iso(text: str) -> datetime:
    value = text.replace("Z", "+00:00")
    return datetime.fromisoformat(value)


def _duration_seconds(summary_path: Path) -> float:
    if not summary_path.exists():
        return 0.0
    summary = _read_json(summary_path)
    started = summary.get("started_at")
    finished = summary.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return 0.0
    delta = _parse_iso(finished) - _parse_iso(started)
    return max(0.0, float(delta.total_seconds()))


def _failure_note(run_dir: Path) -> str:
    errors_path = run_dir / "errors.json"
    if not errors_path.exists():
        return "command_failed_without_errors_json"
    payload = _read_json(errors_path)
    message = payload.get("message")
    if isinstance(message, str) and message.strip() != "":
        return message
    return "command_failed"


def _checkpoints_best_path(train_run_dir: Path) -> Path:
    checkpoints = _read_json(train_run_dir / "checkpoints.json")
    value = checkpoints.get("best")
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"Missing checkpoints.best in {train_run_dir}")
    return Path(value)


def _train_steps(train_run_dir: Path) -> int:
    history = train_run_dir / "metrics_history.jsonl"
    if history.exists():
        rows = _read_jsonl(history)
        if rows:
            step = rows[-1].get("step")
            if isinstance(step, int):
                return step
    checkpoints = train_run_dir / "checkpoints.json"
    if checkpoints.exists():
        payload = _read_json(checkpoints)
        latest = payload.get("latest")
        if isinstance(latest, str) and latest.strip() != "":
            state = Path(latest) / "trainer_state.json"
            if state.exists():
                trainer_state = _read_json(state)
                step = trainer_state.get("step")
                if isinstance(step, int):
                    return step
    return 0


def _eval_metrics(eval_run_dir: Path) -> MetricSnapshot:
    metrics = _read_json(eval_run_dir / "metrics.json")
    accuracy = metrics.get("accuracy")
    if not isinstance(accuracy, dict) or len(accuracy) == 0:
        raise ValueError(f"Missing metrics.accuracy in {eval_run_dir}")
    greedy_values: list[float] = []
    sampled_values: list[float] = []
    for _, bench in sorted(accuracy.items()):
        if not isinstance(bench, dict):
            continue
        greedy = bench.get("greedy_pass_at_1")
        sampled = bench.get("sampled_pass_at_1")
        if isinstance(greedy, (int, float)):
            greedy_values.append(float(greedy))
        if isinstance(sampled, (int, float)):
            sampled_values.append(float(sampled))
    if not greedy_values or not sampled_values:
        raise ValueError(f"Missing pass@1 values in {eval_run_dir}")

    schema = metrics.get("schema", {})
    parser = metrics.get("parser", {})
    schema_rate = float(schema.get("compliance_rate", 0.0)) if isinstance(schema, dict) else 0.0
    parser_rate = (
        float(parser.get("extraction_failure_rate", 0.0)) if isinstance(parser, dict) else 0.0
    )

    predictions_path = eval_run_dir / "predictions.jsonl"
    eval_examples = 0
    if predictions_path.exists():
        rows = _read_jsonl(predictions_path)
        ids = {row.get("example_id") for row in rows if row.get("mode") == "greedy"}
        eval_examples = len([item for item in ids if isinstance(item, str)])

    return MetricSnapshot(
        greedy_pass_at_1=sum(greedy_values) / len(greedy_values),
        sampled_pass_at_1=sum(sampled_values) / len(sampled_values),
        schema_compliance_rate=schema_rate,
        parser_failure_rate=parser_rate,
        eval_examples=eval_examples,
    )


def _run_command(
    *,
    argv: list[str],
    run_dir: Path,
    skip_existing: bool,
) -> tuple[int, bool]:
    summary_path = run_dir / "summary.json"
    if skip_existing and summary_path.exists():
        payload = _read_json(summary_path)
        if payload.get("status") == "success":
            return 0, True
    code = cli_main(argv)
    return int(code), False


def _fixture_train_overrides_sft() -> list[str]:
    return [
        "train_sft.backend.trainer=fixture",
        "train_sft.run.max_steps=6",
        "train_sft.run.eval_every_steps=2",
    ]


def _fixture_train_overrides_grpo() -> list[str]:
    return [
        "train_grpo.backend.trainer=fixture",
        "train_grpo.run.max_steps=6",
        "train_grpo.run.eval_every_steps=2",
        "train_grpo.algo.group_size=4",
    ]


def _fixture_eval_overrides() -> list[str]:
    return [
        "evaluation.inference.backend=fixture",
        "evaluation.sampled.num_samples=4",
    ]


def build_experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            experiment_id="exp_sft_only",
            family="baseline",
            description="SFT checkpoint evaluation baseline",
            kind="sft_only",
            train_overrides=(),
        ),
        ExperimentSpec(
            experiment_id="exp_loss_dr",
            family="loss",
            description="GRPO loss=dr_grpo, scale=batch, schema_weight=0.2",
            kind="grpo",
            train_overrides=(
                "train_grpo.algo.loss_type=dr_grpo",
                "train_grpo.algo.scale_rewards=batch",
                "reward.weights.schema=0.2",
            ),
        ),
        ExperimentSpec(
            experiment_id="exp_loss_grpo",
            family="loss",
            description="GRPO loss=grpo (vs dr_grpo)",
            kind="grpo",
            train_overrides=(
                "train_grpo.algo.loss_type=grpo",
                "train_grpo.algo.scale_rewards=batch",
                "reward.weights.schema=0.2",
            ),
        ),
        ExperimentSpec(
            experiment_id="exp_scale_batch",
            family="scale_rewards",
            description="Reward scaling=batch",
            kind="grpo",
            train_overrides=(),
            alias_of="exp_loss_dr",
        ),
        ExperimentSpec(
            experiment_id="exp_scale_group",
            family="scale_rewards",
            description="Reward scaling=group",
            kind="grpo",
            train_overrides=(
                "train_grpo.algo.loss_type=dr_grpo",
                "train_grpo.algo.scale_rewards=group",
                "reward.weights.schema=0.2",
            ),
        ),
        ExperimentSpec(
            experiment_id="exp_scale_none",
            family="scale_rewards",
            description="Reward scaling=none",
            kind="grpo",
            train_overrides=(
                "train_grpo.algo.loss_type=dr_grpo",
                "train_grpo.algo.scale_rewards=none",
                "reward.weights.schema=0.2",
            ),
        ),
        ExperimentSpec(
            experiment_id="exp_schema_005",
            family="schema_weight",
            description="Schema reward weight=0.05",
            kind="grpo",
            train_overrides=(
                "train_grpo.algo.loss_type=dr_grpo",
                "train_grpo.algo.scale_rewards=batch",
                "reward.weights.schema=0.05",
            ),
        ),
        ExperimentSpec(
            experiment_id="exp_schema_020",
            family="schema_weight",
            description="Schema reward weight=0.2",
            kind="grpo",
            train_overrides=(),
            alias_of="exp_loss_dr",
        ),
        ExperimentSpec(
            experiment_id="exp_schema_050",
            family="schema_weight",
            description="Schema reward weight=0.5",
            kind="grpo",
            train_overrides=(
                "train_grpo.algo.loss_type=dr_grpo",
                "train_grpo.algo.scale_rewards=batch",
                "reward.weights.schema=0.5",
            ),
        ),
        ExperimentSpec(
            experiment_id="exp_rs_sft",
            family="alternative",
            description="RS+SFT alternative baseline",
            kind="rs_sft",
            train_overrides=(),
        ),
    ]


def _load_dataset_manifest_from_data_run(run_id: str) -> Path:
    summary_path = Path("artifacts") / "runs" / run_id / "summary.json"
    summary = _read_json(summary_path)
    artifacts = summary.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError(f"Missing summary.artifacts in {summary_path}")
    manifest_path = artifacts.get("dataset_manifest_path")
    if not isinstance(manifest_path, str):
        raise ValueError(f"Missing dataset_manifest_path in {summary_path}")
    return Path(manifest_path)


def _ensure_fixture_eval_dir(
    *,
    rl_manifest: Path,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    gsm_path = out_dir / "gsm8k_eval.jsonl"
    math_path = out_dir / "math_eval.jsonl"
    if gsm_path.exists() and math_path.exists():
        return out_dir

    train_input = load_grpo_train_input(rl_manifest, max_eval_samples=64)
    by_benchmark: dict[str, list[Any]] = {"gsm8k": [], "math": []}
    all_records = list(train_input.val_records) + list(train_input.train_records)
    for item in all_records:
        key = item.benchmark.lower()
        if key not in by_benchmark:
            continue
        by_benchmark[key].append(item)

    for benchmark, path in (("gsm8k", gsm_path), ("math", math_path)):
        rows = by_benchmark.get(benchmark, [])
        if not rows and all_records:
            rows = [all_records[0]]
        lines: list[str] = []
        for index, row in enumerate(rows[:8]):
            payload = {
                "id": f"{benchmark}_{index}_{row.record_id}",
                "question": row.prompt,
                "answer": row.gold_answer,
                "mock_greedy_response": (
                    f"<think>fixture reasoning</think><answer>\\boxed{{{row.gold_answer}}}</answer>"
                ),
                "mock_sampled_responses": [
                    "<think>fixture wrong</think><answer>\\boxed{0}</answer>",
                    f"<think>fixture reasoning</think><answer>\\boxed{{{row.gold_answer}}}</answer>",
                ],
            }
            lines.append(json.dumps(payload, sort_keys=True))
        with path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
    return out_dir


def run_ablation_suite(
    *,
    config_path: Path,
    source_dir: Path | None,
    dataset_dir: Path | None,
    output_root: Path,
    report_dir: Path,
    mode: str,
    seed: int | None,
    max_items: int | None,
    skip_existing: bool,
    fail_fast: bool,
    session_id: str | None = None,
) -> AblationRunResult:
    mode_lower = mode.lower()
    if mode_lower not in {"fixture", "real"}:
        raise ValueError(f"Unsupported mode: {mode}")

    selected_session_id = session_id or f"ablate-{_now_stamp()}-{secrets.token_hex(3)}"
    runs_root = (output_root / "runs").resolve()
    datasets_root = (output_root / "datasets").resolve()
    rs_root = (output_root / "rs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)
    rs_root.mkdir(parents=True, exist_ok=True)

    data_sft_run_id = f"{selected_session_id}-data-sft"
    data_rl_run_id = f"{selected_session_id}-data-rl"

    data_sft_args = [
        "data",
        "build-sft",
        "--config",
        str(config_path),
        "--run-id",
        data_sft_run_id,
        "--output-dir",
        str(datasets_root),
    ]
    if source_dir is not None:
        data_sft_args.extend(["--source-dir", str(source_dir)])
    if seed is not None:
        data_sft_args.extend(["--seed", str(seed)])
    data_sft_run_dir = Path("artifacts") / "runs" / data_sft_run_id
    code, _ = _run_command(argv=data_sft_args, run_dir=data_sft_run_dir, skip_existing=skip_existing)
    if code != 0:
        raise RuntimeError(f"data build-sft failed: {_failure_note(data_sft_run_dir)}")
    sft_manifest = _load_dataset_manifest_from_data_run(data_sft_run_id)

    data_rl_args = [
        "data",
        "build-rl",
        "--config",
        str(config_path),
        "--run-id",
        data_rl_run_id,
        "--output-dir",
        str(datasets_root),
    ]
    if source_dir is not None:
        data_rl_args.extend(["--source-dir", str(source_dir)])
    if seed is not None:
        data_rl_args.extend(["--seed", str(seed)])
    data_rl_run_dir = Path("artifacts") / "runs" / data_rl_run_id
    code, _ = _run_command(argv=data_rl_args, run_dir=data_rl_run_dir, skip_existing=skip_existing)
    if code != 0:
        raise RuntimeError(f"data build-rl failed: {_failure_note(data_rl_run_dir)}")
    rl_manifest = _load_dataset_manifest_from_data_run(data_rl_run_id)

    eval_dataset_dir = dataset_dir
    if eval_dataset_dir is None and mode_lower == "fixture":
        eval_dataset_dir = _ensure_fixture_eval_dir(
            rl_manifest=rl_manifest,
            out_dir=(output_root / "eval_fixture").resolve(),
        )

    sft_train_run_id = f"{selected_session_id}-anchor-sft"
    sft_train_run_dir = runs_root / sft_train_run_id
    sft_train_args = [
        "train",
        "sft",
        "--config",
        str(config_path),
        "--dataset-manifest",
        str(sft_manifest),
        "--run-id",
        sft_train_run_id,
        "--output-dir",
        str(runs_root),
    ]
    sft_overrides: list[str] = []
    if mode_lower == "fixture":
        sft_overrides.extend(_fixture_train_overrides_sft())
    for item in sft_overrides:
        sft_train_args.extend(["--set", item])
    code, _ = _run_command(argv=sft_train_args, run_dir=sft_train_run_dir, skip_existing=skip_existing)
    if code != 0:
        raise RuntimeError(f"anchor train-sft failed: {_failure_note(sft_train_run_dir)}")

    anchor_checkpoint = _checkpoints_best_path(sft_train_run_dir)
    baseline_eval_run_id = f"{selected_session_id}-anchor-eval"
    baseline_eval_run_dir = runs_root / baseline_eval_run_id
    eval_args = [
        "eval",
        "--config",
        str(config_path),
        "--checkpoint",
        str(anchor_checkpoint),
        "--run-id",
        baseline_eval_run_id,
        "--output-dir",
        str(runs_root),
    ]
    if eval_dataset_dir is not None:
        eval_args.extend(["--dataset-dir", str(eval_dataset_dir)])
    if max_items is not None:
        eval_args.extend(["--max-items", str(max_items)])
    if seed is not None:
        eval_args.extend(["--seed", str(seed)])
    eval_overrides: list[str] = []
    if mode_lower == "fixture":
        eval_overrides.extend(_fixture_eval_overrides())
    for item in eval_overrides:
        eval_args.extend(["--set", item])
    code, _ = _run_command(argv=eval_args, run_dir=baseline_eval_run_dir, skip_existing=skip_existing)
    if code != 0:
        raise RuntimeError(f"anchor eval failed: {_failure_note(baseline_eval_run_dir)}")

    baseline_outcome = ExperimentOutcome(
        experiment_id="exp_sft_only",
        family="baseline",
        description="SFT checkpoint evaluation baseline",
        status="success",
        notes="baseline_sft_eval",
        artifacts=RunArtifacts(train_run_dir=sft_train_run_dir, eval_run_dir=baseline_eval_run_dir),
        metrics=_eval_metrics(baseline_eval_run_dir),
        cost=CostSnapshot(
            wallclock_seconds=_duration_seconds(baseline_eval_run_dir / "summary.json"),
            train_steps=0,
        ),
    )

    specs = build_experiment_specs()
    by_id: dict[str, ExperimentOutcome] = {"exp_sft_only": baseline_outcome}
    outcomes: list[ExperimentOutcome] = [baseline_outcome]
    resolved_cfg = resolve_config(repo_root() / "configs" / "defaults.yaml", config_path)

    for spec in specs:
        if spec.experiment_id == "exp_sft_only":
            continue

        if spec.alias_of is not None:
            source = by_id.get(spec.alias_of)
            if source is None:
                raise RuntimeError(f"Alias source missing: {spec.alias_of}")
            alias = ExperimentOutcome(
                experiment_id=spec.experiment_id,
                family=spec.family,
                description=spec.description,
                status=source.status,
                notes=f"alias_of={spec.alias_of}",
                artifacts=source.artifacts,
                metrics=source.metrics,
                cost=source.cost,
            )
            by_id[alias.experiment_id] = alias
            outcomes.append(alias)
            continue

        try:
            if spec.kind == "grpo":
                train_run_id = f"{selected_session_id}-{spec.experiment_id}-train"
                train_run_dir = runs_root / train_run_id
                train_args = [
                    "train",
                    "grpo",
                    "--config",
                    str(config_path),
                    "--dataset-manifest",
                    str(rl_manifest),
                    "--init-checkpoint",
                    str(anchor_checkpoint),
                    "--run-id",
                    train_run_id,
                    "--output-dir",
                    str(runs_root),
                ]
                train_overrides: list[str] = []
                if mode_lower == "fixture":
                    train_overrides.extend(_fixture_train_overrides_grpo())
                train_overrides.extend(spec.train_overrides)
                for item in train_overrides:
                    train_args.extend(["--set", item])
                code, _ = _run_command(
                    argv=train_args,
                    run_dir=train_run_dir,
                    skip_existing=skip_existing,
                )
                if code != 0:
                    raise RuntimeError(_failure_note(train_run_dir))

                checkpoint = _checkpoints_best_path(train_run_dir)
                eval_run_id = f"{selected_session_id}-{spec.experiment_id}-eval"
                eval_run_dir = runs_root / eval_run_id
                eval_exp_args = [
                    "eval",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--run-id",
                    eval_run_id,
                    "--output-dir",
                    str(runs_root),
                ]
                if eval_dataset_dir is not None:
                    eval_exp_args.extend(["--dataset-dir", str(eval_dataset_dir)])
                if max_items is not None:
                    eval_exp_args.extend(["--max-items", str(max_items)])
                if seed is not None:
                    eval_exp_args.extend(["--seed", str(seed)])
                exp_eval_overrides: list[str] = []
                if mode_lower == "fixture":
                    exp_eval_overrides.extend(_fixture_eval_overrides())
                for item in exp_eval_overrides:
                    eval_exp_args.extend(["--set", item])
                code, _ = _run_command(
                    argv=eval_exp_args,
                    run_dir=eval_run_dir,
                    skip_existing=skip_existing,
                )
                if code != 0:
                    raise RuntimeError(_failure_note(eval_run_dir))

                outcome = ExperimentOutcome(
                    experiment_id=spec.experiment_id,
                    family=spec.family,
                    description=spec.description,
                    status="success",
                    notes=spec.description,
                    artifacts=RunArtifacts(train_run_dir=train_run_dir, eval_run_dir=eval_run_dir),
                    metrics=_eval_metrics(eval_run_dir),
                    cost=CostSnapshot(
                        wallclock_seconds=_duration_seconds(train_run_dir / "summary.json")
                        + _duration_seconds(eval_run_dir / "summary.json"),
                        train_steps=_train_steps(train_run_dir),
                    ),
                )
            elif spec.kind == "rs_sft":
                rs_dir = rs_root / spec.experiment_id
                train_input = load_grpo_train_input(
                    rl_manifest,
                    max_eval_samples=resolved_cfg.train_grpo.run.max_eval_samples,
                )
                rs_manifest, rs_result = build_rs_sft_manifest(
                    config=resolved_cfg,
                    train_input=train_input,
                    output_root=rs_dir,
                    candidates_per_prompt=4 if mode_lower == "fixture" else 8,
                    strict_boxed_only=resolved_cfg.evaluation.parser.strict_boxed_only,
                )
                rs_train_run_id = f"{selected_session_id}-{spec.experiment_id}-train"
                rs_train_run_dir = runs_root / rs_train_run_id
                rs_train_args = [
                    "train",
                    "sft",
                    "--config",
                    str(config_path),
                    "--dataset-manifest",
                    str(rs_manifest),
                    "--run-id",
                    rs_train_run_id,
                    "--output-dir",
                    str(runs_root),
                ]
                rs_overrides: list[str] = [f"model.default_base_model={anchor_checkpoint}"]
                if mode_lower == "fixture":
                    rs_overrides.extend(_fixture_train_overrides_sft())
                for item in rs_overrides:
                    rs_train_args.extend(["--set", item])
                code, _ = _run_command(
                    argv=rs_train_args,
                    run_dir=rs_train_run_dir,
                    skip_existing=skip_existing,
                )
                if code != 0:
                    raise RuntimeError(_failure_note(rs_train_run_dir))

                rs_checkpoint = _checkpoints_best_path(rs_train_run_dir)
                rs_eval_run_id = f"{selected_session_id}-{spec.experiment_id}-eval"
                rs_eval_run_dir = runs_root / rs_eval_run_id
                rs_eval_args = [
                    "eval",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(rs_checkpoint),
                    "--run-id",
                    rs_eval_run_id,
                    "--output-dir",
                    str(runs_root),
                ]
                if eval_dataset_dir is not None:
                    rs_eval_args.extend(["--dataset-dir", str(eval_dataset_dir)])
                if max_items is not None:
                    rs_eval_args.extend(["--max-items", str(max_items)])
                if seed is not None:
                    rs_eval_args.extend(["--seed", str(seed)])
                rs_eval_overrides: list[str] = []
                if mode_lower == "fixture":
                    rs_eval_overrides.extend(_fixture_eval_overrides())
                for item in rs_eval_overrides:
                    rs_eval_args.extend(["--set", item])
                code, _ = _run_command(
                    argv=rs_eval_args,
                    run_dir=rs_eval_run_dir,
                    skip_existing=skip_existing,
                )
                if code != 0:
                    raise RuntimeError(_failure_note(rs_eval_run_dir))

                notes = (
                    f"accepted={rs_result.accepted_count}; rejected={rs_result.rejected_count}; "
                    f"verifier={rs_result.verifier_backend}"
                )
                outcome = ExperimentOutcome(
                    experiment_id=spec.experiment_id,
                    family=spec.family,
                    description=spec.description,
                    status="success",
                    notes=notes,
                    artifacts=RunArtifacts(train_run_dir=rs_train_run_dir, eval_run_dir=rs_eval_run_dir),
                    metrics=_eval_metrics(rs_eval_run_dir),
                    cost=CostSnapshot(
                        wallclock_seconds=_duration_seconds(rs_train_run_dir / "summary.json")
                        + _duration_seconds(rs_eval_run_dir / "summary.json"),
                        train_steps=_train_steps(rs_train_run_dir),
                    ),
                )
            else:
                raise RuntimeError(f"Unsupported experiment kind: {spec.kind}")

        except Exception as exc:
            failed_train = runs_root / f"{selected_session_id}-{spec.experiment_id}-train"
            failed_eval = runs_root / f"{selected_session_id}-{spec.experiment_id}-eval"
            wallclock = _duration_seconds(failed_train / "summary.json") + _duration_seconds(
                failed_eval / "summary.json"
            )
            outcome = ExperimentOutcome(
                experiment_id=spec.experiment_id,
                family=spec.family,
                description=spec.description,
                status="failed",
                notes=str(exc),
                artifacts=RunArtifacts(train_run_dir=failed_train, eval_run_dir=failed_eval),
                metrics=None,
                cost=CostSnapshot(wallclock_seconds=wallclock, train_steps=_train_steps(failed_train)),
            )
            if fail_fast:
                by_id[outcome.experiment_id] = outcome
                outcomes.append(outcome)
                break

        by_id[outcome.experiment_id] = outcome
        outcomes.append(outcome)

    ordered_outcomes: list[ExperimentOutcome] = []
    for spec in specs:
        found = by_id.get(spec.experiment_id)
        if found is not None:
            ordered_outcomes.append(found)

    rows = build_rows(
        ordered_outcomes,
        backend_mode=mode_lower,
        sft_baseline_id="exp_sft_only",
        sft_run_dir=sft_train_run_dir,
    )
    csv_path = report_dir / "ablation_results.csv"
    markdown_path = report_dir / "ablation_summary.md"
    write_csv(rows, csv_path)
    write_markdown(
        rows,
        markdown_path,
        metadata={
            "session_id": selected_session_id,
            "mode": mode_lower,
            "config": str(config_path),
            "source_dir": str(source_dir) if source_dir else "",
            "dataset_dir": str(eval_dataset_dir) if eval_dataset_dir else "",
        },
    )
    return AblationRunResult(
        session_id=selected_session_id,
        rows=tuple(rows),
        csv_path=csv_path,
        markdown_path=markdown_path,
    )
