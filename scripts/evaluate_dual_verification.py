from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genfine.data.loader import load_dataset
from genfine.evaluation.dual_verification import (
    DualVerificationError,
    DualVerificationEvaluator,
    DualVerificationSummary,
)
from genfine.evaluation.run_loader import (
    RunLoadError,
    load_run_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify each GenFINE output against both its predicted "
            "EditPlan and the dataset Gold EditPlan."
        )
    )

    parser.add_argument(
        "run_file",
        help="RunRecord JSONL file.",
    )
    parser.add_argument(
        "--dataset",
        default="data/seed/seed_v0.2.jsonl",
        help="Gold GenFINE dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON output. Defaults to "
            "<run_file>.dual-verification.json."
        ),
    )
    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=0.5,
        help=(
            "Minimum character-overlap coefficient used when "
            "comparing predicted and gold span actions."
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        instances = load_dataset(args.dataset)
        run_records = load_run_records(args.run_file)

        summary = DualVerificationEvaluator(
            minimum_overlap=args.minimum_overlap
        ).evaluate(
            instances=instances,
            run_records=run_records,
            require_complete=args.require_complete,
        )
    except (
        FileNotFoundError,
        RunLoadError,
        DualVerificationError,
        ValueError,
    ) as exc:
        print(
            f"Dual verification failed: {exc}",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else Path(
            f"{args.run_file}.dual-verification.json"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(summary)
    print(f"Output: {output_path}")

    return 0


def print_summary(
    summary: DualVerificationSummary,
) -> None:
    print(
        "Dataset instances: "
        f"{summary.dataset_instance_count}"
    )
    print(
        "Evaluated instances: "
        f"{summary.evaluated_instance_count}"
    )

    print_metric(
        "Predicted-plan verification coverage",
        summary.predicted_plan_verification_coverage,
    )
    print_metric(
        "Predicted-plan verification pass",
        summary.predicted_plan_verification_pass_rate,
    )
    print_metric(
        "Gold-plan verification pass",
        summary.gold_plan_verification_pass_rate,
    )
    print_metric(
        "Upstream plan error",
        summary.upstream_plan_error_rate,
    )
    print_metric(
        "Rewrite execution error",
        summary.rewrite_execution_error_rate,
    )
    print_metric(
        "Gold-compliant despite plan mismatch",
        summary.gold_compliant_despite_plan_mismatch_rate,
    )

    print("Per-instance categories:")

    for record in summary.records:
        print(
            f"  {record.instance_id}: "
            f"{record.category.value}"
        )


def print_metric(label: str, metric) -> None:
    print(
        f"{label}: {metric.value:.4f} "
        f"({metric.numerator}/{metric.denominator})"
    )


if __name__ == "__main__":
    raise SystemExit(main())
