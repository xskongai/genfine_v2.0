from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from genfine.data.loader import load_dataset
from genfine.evaluation.analysis_evaluator import (
    AnalysisEvaluationSummary,
    evaluate_analysis_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GenFINE structured analysis predictions."
        )
    )

    parser.add_argument(
        "predictions",
        nargs="?",
        default="runs/llm_analysis_v0.1.jsonl",
        help="LLMAnalyzer prediction JSONL file.",
    )

    parser.add_argument(
        "--dataset",
        default="data/seed/seed_v0.2.jsonl",
        help="Gold GenFINE dataset JSONL file.",
    )

    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=0.5,
        help=(
            "Minimum overlap coefficient used for "
            "one-to-one span alignment."
        ),
    )

    parser.add_argument(
        "--json-output",
        default=None,
        help=(
            "Optional path for the machine-readable "
            "evaluation summary."
        ),
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Exit with status 1 unless every dataset instance "
            "has a schema-valid prediction."
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: "
                    f"{exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Record at {path}:{line_number} "
                    "must be a JSON object"
                )

            records.append(payload)

    return records


def _print_prf(
    name: str,
    metric,
) -> None:
    print(
        f"{name}: "
        f"P={metric.precision:.4f} "
        f"R={metric.recall:.4f} "
        f"F1={metric.f1:.4f} "
        f"(TP={metric.true_positive}, "
        f"FP={metric.false_positive}, "
        f"FN={metric.false_negative})"
    )


def _print_accuracy(
    name: str,
    metric,
) -> None:
    print(
        f"{name}: "
        f"{metric.accuracy:.4f} "
        f"({metric.correct}/{metric.total})"
    )


def print_summary(
    summary: AnalysisEvaluationSummary,
) -> None:
    print(
        f"Dataset instances: "
        f"{summary.dataset_instances}"
    )
    print(
        f"Prediction records: "
        f"{summary.prediction_records}"
    )
    print(
        f"Missing predictions: "
        f"{summary.missing_predictions}"
    )
    print(
        f"Extra predictions: "
        f"{summary.extra_predictions}"
    )

    print()

    _print_accuracy(
        "Schema validity",
        summary.schema_validity,
    )
    _print_accuracy(
        "Analysis exact match",
        summary.analysis_exact_match,
    )

    print()

    _print_prf(
        "Exact span detection",
        summary.exact_span,
    )
    _print_prf(
        "Overlap span detection",
        summary.overlap_span,
    )
    _print_accuracy(
        "Matched gold-span coverage",
        summary.matched_span_coverage,
    )

    print()

    _print_prf(
        "Function labels (micro)",
        summary.function_labels,
    )
    _print_accuracy(
        "Necessity status",
        summary.necessity_status,
    )
    _print_accuracy(
        "Bias status",
        summary.bias_status,
    )
    _print_prf(
        "Bias mechanisms (micro)",
        summary.bias_mechanisms,
    )
    _print_accuracy(
        "Span stance",
        summary.span_stance,
    )

    print()

    _print_accuracy(
        "Instance speaker stance",
        summary.speaker_stance,
    )
    _print_accuracy(
        "Context sufficient",
        summary.context_sufficient,
    )

    if summary.invalid_instance_ids:
        print()
        print(
            "Invalid instances: "
            + ", ".join(
                summary.invalid_instance_ids
            )
        )

    if summary.extra_instance_ids:
        print(
            "Extra instance IDs: "
            + ", ".join(
                summary.extra_instance_ids
            )
        )


def main() -> int:
    args = parse_args()

    prediction_path = (
        PROJECT_ROOT / args.predictions
    )
    dataset_path = PROJECT_ROOT / args.dataset

    if not 0.0 < args.minimum_overlap <= 1.0:
        print(
            "--minimum-overlap must be in (0, 1].",
            file=sys.stderr,
        )
        return 2

    try:
        instances = load_dataset(dataset_path)
        prediction_records = load_jsonl(
            prediction_path
        )

        summary = evaluate_analysis_predictions(
            instances=instances,
            prediction_records=prediction_records,
            minimum_overlap=args.minimum_overlap,
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(
            f"Evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print_summary(summary)

    if args.json_output:
        json_output_path = (
            PROJECT_ROOT / args.json_output
        )
        json_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_output_path.write_text(
            json.dumps(
                summary.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            "JSON summary: "
            f"{json_output_path}"
        )

    if (
        args.require_complete
        and summary.schema_validity.correct
        != summary.schema_validity.total
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
