from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microreasoner.contracts.validation import validate_run_dir
from microreasoner.runtime.eval_command import execute_eval_command
from microreasoner.runtime.scaffold import execute_scaffold_command


def _add_train_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train_parser = subparsers.add_parser("train", help="Training commands")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)

    train_sft = train_subparsers.add_parser("sft", help="Train SFT model")
    train_sft.add_argument("--config", required=True, help="Path to training config")
    train_sft.add_argument("--run-id", required=False, help="Optional explicit run id")
    train_sft.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Config override in dotted form key=value (can be provided multiple times)",
    )
    train_sft.add_argument(
        "--output-dir",
        required=False,
        help="Optional output root (default: artifacts/runs)",
    )

    train_grpo = train_subparsers.add_parser("grpo", help="Train GRPO model")
    train_grpo.add_argument("--config", required=True, help="Path to training config")
    train_grpo.add_argument("--run-id", required=False, help="Optional explicit run id")
    train_grpo.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Config override in dotted form key=value (can be provided multiple times)",
    )
    train_grpo.add_argument(
        "--output-dir",
        required=False,
        help="Optional output root (default: artifacts/runs)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microreasoner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_subcommands(subparsers)

    eval_parser = subparsers.add_parser("eval", help="Evaluation command")
    eval_parser.add_argument("--config", required=True, help="Path to eval config")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    eval_parser.add_argument("--run-id", required=False, help="Optional explicit run id")
    eval_parser.add_argument(
        "--dataset-dir",
        required=False,
        help="Optional directory overriding eval dataset paths by filename",
    )
    eval_parser.add_argument(
        "--max-items",
        required=False,
        type=int,
        help="Optional max number of examples after deterministic ordering",
    )
    eval_parser.add_argument(
        "--seed",
        required=False,
        type=int,
        help="Optional seed override for deterministic sampled generation",
    )
    eval_parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Config override in dotted form key=value (can be provided multiple times)",
    )
    eval_parser.add_argument(
        "--output-dir",
        required=False,
        help="Optional output root (default: artifacts/runs)",
    )

    validate_parser = subparsers.add_parser("validate-run", help="Validate run artifacts")
    validate_parser.add_argument("--run-dir", required=True, help="Run directory path")
    validate_parser.add_argument(
        "--compare-run-dir",
        required=False,
        help="Optional run directory for evaluation-config drift checks",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-run":
        compare = Path(args.compare_run_dir) if args.compare_run_dir else None
        result = validate_run_dir(Path(args.run_dir), compare)
        if result.ok:
            print("Validation passed")
            return 0
        print("Validation failed:")
        for err in result.errors:
            print(f"- {err}")
        return 1

    if args.command == "train":
        command_name = f"train-{args.train_command}"
        return execute_scaffold_command(
            command_name=command_name,
            config_path=Path(args.config),
            cli_overrides=args.set_overrides,
            run_id=args.run_id,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )

    if args.command == "eval":
        return execute_eval_command(
            config_path=Path(args.config),
            checkpoint=Path(args.checkpoint),
            cli_overrides=args.set_overrides,
            run_id=args.run_id,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            max_items=args.max_items,
            seed_override=args.seed,
        )

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
