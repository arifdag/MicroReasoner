from __future__ import annotations

import argparse
import sys
from pathlib import Path

from microreasoner.ablation.runner import run_ablation_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MicroReasoner ablations and generate reports.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--source-dir", required=False, help="Optional source directory for data builds")
    parser.add_argument(
        "--dataset-dir",
        required=False,
        help="Optional evaluation dataset directory (expects gsm8k_eval.jsonl and math_eval.jsonl)",
    )
    parser.add_argument(
        "--output-root",
        required=False,
        default="artifacts/ablations",
        help="Output root for run directories",
    )
    parser.add_argument(
        "--report-dir",
        required=False,
        default="reports",
        help="Directory to write ablation reports",
    )
    parser.add_argument(
        "--mode",
        required=False,
        choices=["fixture", "real"],
        default="fixture",
        help="Ablation execution mode",
    )
    parser.add_argument("--seed", required=False, type=int, help="Optional seed override")
    parser.add_argument(
        "--session-id",
        required=False,
        help="Optional stable ablation session id (useful with --skip-existing)",
    )
    parser.add_argument(
        "--max-items",
        required=False,
        type=int,
        help="Optional eval max items override",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing successful run directories when possible",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop when the first experiment fails",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_ablation_suite(
            config_path=Path(args.config),
            source_dir=Path(args.source_dir) if args.source_dir else None,
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            output_root=Path(args.output_root),
            report_dir=Path(args.report_dir),
            mode=args.mode,
            seed=args.seed,
            max_items=args.max_items,
            skip_existing=bool(args.skip_existing),
            fail_fast=bool(args.fail_fast),
            session_id=args.session_id,
        )
    except Exception as exc:
        print(f"Ablation run failed: {exc}", file=sys.stderr)
        return 1

    print(f"Ablation session: {result.session_id}")
    print(f"CSV report: {result.csv_path}")
    print(f"Markdown report: {result.markdown_path}")
    print(f"Rows: {len(result.rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
