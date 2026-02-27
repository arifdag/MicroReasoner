from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microreasoner.final_eval.runner import run_final_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run final locked evaluation for Base/SFT/GRPO and generate publishable reports."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing evaluation files named as in config",
    )
    parser.add_argument("--base-checkpoint", required=True, help="Path to base checkpoint")
    parser.add_argument("--sft-checkpoint", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--grpo-checkpoint", required=True, help="Path to GRPO checkpoint")
    parser.add_argument(
        "--output-root",
        required=False,
        default="artifacts/final_eval",
        help="Output root for final eval run directories",
    )
    parser.add_argument(
        "--report-dir",
        required=False,
        default="reports",
        help="Directory for final metrics/report outputs",
    )
    parser.add_argument(
        "--mode",
        required=False,
        choices=["fixture", "real"],
        default="fixture",
        help="Execution mode for evaluation backend assumptions",
    )
    parser.add_argument("--seed", required=False, type=int, help="Optional evaluation seed override")
    parser.add_argument("--max-items", required=False, type=int, help="Optional eval item cap")
    parser.add_argument("--session-id", required=False, help="Optional fixed session id")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse successful eval run directories when present",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first eval failure")
    parser.add_argument(
        "--strict-claims",
        action="store_true",
        help="Return non-zero when required comparison claims are missing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_final_evaluation(
            config_path=Path(args.config),
            dataset_dir=Path(args.dataset_dir),
            base_checkpoint=Path(args.base_checkpoint),
            sft_checkpoint=Path(args.sft_checkpoint),
            grpo_checkpoint=Path(args.grpo_checkpoint),
            output_root=Path(args.output_root),
            report_dir=Path(args.report_dir),
            seed=args.seed,
            max_items=args.max_items,
            mode=args.mode,
            skip_existing=bool(args.skip_existing),
            fail_fast=bool(args.fail_fast),
            strict_claims=bool(args.strict_claims),
            session_id=args.session_id,
        )
    except Exception as exc:
        print(f"Final evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Final evaluation session: {result.session_id}")
    print(f"Status: {result.status}")
    print(f"final_metrics.json: {result.metrics_path}")
    print(f"final_report.md: {result.report_path}")
    print(f"error_analysis.md: {result.error_analysis_path}")

    if args.strict_claims and not result.strict_claims_ok:
        print("Strict claims check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

