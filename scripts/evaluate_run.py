from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genfine.data.loader import load_dataset
from genfine.evaluation import (
    EvaluationError,
    RunEvaluator,
    RunLoadError,
    load_run_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a GenFINE RunRecord JSONL "
            "file against a gold dataset."
        )
    )

    parser.add_argument(
        "run_file",
        help="RunRecord JSONL file.",
    )

    parser.add_argument(
        "--dataset",
        default="data/seed/seed_v0.1.jsonl",
        help="Gold GenFINE dataset.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional evaluation JSON output. "
            "Defaults to <run_file>.metrics.json."
        ),
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail when any dataset instance "
            "has no run record."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        instances = load_dataset(
            args.dataset
        )

        run_records = load_run_records(
            args.run_file
        )

        summary = RunEvaluator().evaluate(
            instances=instances,
            run_records=run_records,
            require_complete=(
                args.require_complete
            ),
        )

    except (
        FileNotFoundError,
        RunLoadError,
        EvaluationError,
    ) as exc:
        print(
            f"Evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else Path(
            f"{args.run_file}.metrics.json"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(summary)
    print(f"Metrics output: {output_path}")

    return 0


def print_summary(summary) -> None:
    print(
        "Dataset instances: "
        f"{summary.dataset_instance_count}"
    )

    print(
        "Evaluated instances: "
        f"{summary.evaluated_instance_count}"
    )

    print(
        "Failed instances: "
        f"{summary.failed_instances}"
    )

    print_metric(
        "Analysis exact match",
        summary.analysis_exact_match,
    )

    print_metric(
        "Span action accuracy",
        summary.span_action_accuracy,
    )

    print_metric(
        "Instance action accuracy",
        summary.instance_action_accuracy,
    )

    print_metric(
        "Edit scope accuracy",
        summary.edit_scope_accuracy,
    )

    print_metric(
        "Output exact match",
        summary.output_exact_match,
    )

    print_metric(
        "Verification coverage",
        summary.verification_coverage,
    )

    print_metric(
        "Verification pass rate",
        summary.verification_pass_rate,
    )

    print_metric(
        "Protected-fact preservation",
        summary.protected_fact_preservation_rate,
    )

    print_metric(
        "Action compliance",
        summary.action_compliance_rate,
    )

    print_metric(
        "Unsupported gender insertion",
        summary.unsupported_gender_insertion_rate,
    )

    print_metric(
        "Over-neutralization",
        summary.over_neutralization_rate,
    )

    print_metric(
        "Under-correction",
        summary.under_correction_rate,
    )


def print_metric(
    label: str,
    metric,
) -> None:
    print(
        f"{label}: "
        f"{metric.value:.4f} "
        f"({metric.numerator}/"
        f"{metric.denominator})"
    )


if __name__ == "__main__":
    raise SystemExit(main())