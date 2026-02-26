from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microreasoner.contracts.validation import validate_run_dir


def _add_train_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    train_parser = subparsers.add_parser("train", help="Training commands (interface reserved)")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)

    train_sft = train_subparsers.add_parser("sft", help="Train SFT model")
    train_sft.add_argument("--config", required=True, help="Path to training config")

    train_grpo = train_subparsers.add_parser("grpo", help="Train GRPO model")
    train_grpo.add_argument("--config", required=True, help="Path to training config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microreasoner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_train_subcommands(subparsers)

    eval_parser = subparsers.add_parser("eval", help="Evaluation command (interface reserved)")
    eval_parser.add_argument("--config", required=True, help="Path to eval config")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")

    validate_parser = subparsers.add_parser("validate-run", help="Validate run artifacts")
    validate_parser.add_argument("--run-dir", required=True, help="Run directory path")
    validate_parser.add_argument(
        "--compare-run-dir",
        required=False,
        help="Optional run directory for evaluation-config drift checks",
    )

    return parser


def _reserved_command_message(command: str) -> int:
    print(
        f"Command '{command}' interface is locked by Phase 0 but implementation is pending.",
        file=sys.stderr,
    )
    return 2


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
        return _reserved_command_message(f"train {args.train_command}")
    if args.command == "eval":
        return _reserved_command_message("eval")

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

