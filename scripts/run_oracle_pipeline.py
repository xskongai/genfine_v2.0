from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import (
    DatasetLoadError,
    load_dataset_records,
)
from genfine.data.validator import (
    DatasetValidator,
)
from genfine.generation import GoldRewriter
from genfine.pipeline import (
    EditPlanBuilder,
    PipelineRunner,
)
from genfine.policy import DecisionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete GenFINE Oracle pipeline."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/seed/seed_v0.1.jsonl",
        help="Input GenFINE JSONL dataset.",
    )

    parser.add_argument(
        "--rules",
        default="configs/decision_rules.yaml",
        help="Decision-rule YAML file.",
    )

    parser.add_argument(
        "--output",
        default="runs/oracle_pipeline_v0.1.jsonl",
        help="Output RunRecord JSONL file.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Write failed instances as error records "
            "instead of stopping immediately."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        loaded_records = load_dataset_records(
            args.dataset
        )
    except (
        DatasetLoadError,
        FileNotFoundError,
    ) as exc:
        print(
            f"Failed to load dataset: {exc}",
            file=sys.stderr,
        )
        return 1

    validation_report = (
        DatasetValidator().validate(
            loaded_records
        )
    )

    if not validation_report.passed:
        print(
            "Dataset validation failed.",
            file=sys.stderr,
        )

        for issue in validation_report.errors:
            print(
                f"[{issue.code}] "
                f"{issue.instance_id}: "
                f"{issue.message}",
                file=sys.stderr,
            )

        return 1

    instances = [
        record.instance
        for record in loaded_records
    ]

    runner = PipelineRunner(
        analyzer=OracleAnalyzer(),
        decision_engine=(
            DecisionEngine.from_yaml(
                args.rules
            )
        ),
        edit_plan_builder=EditPlanBuilder(),
        rewriter=GoldRewriter(
            strict_plan_match=True
        ),
    )

    run_records = runner.run_many(
        instances,
        continue_on_error=(
            args.continue_on_error
        ),
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in run_records:
            payload = record.model_dump(
                mode="json",
                exclude_none=True,
            )

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            file.write("\n")

    metrics = calculate_oracle_metrics(
        instances=instances,
        run_records=run_records,
    )

    print(f"Instances: {metrics['instance_count']}")
    print(f"Failed: {metrics['failed_count']}")

    print(
        "Analysis exact match: "
        f"{metrics['analysis_exact_match']:.4f}"
    )

    print(
        "Span action accuracy: "
        f"{metrics['span_action_accuracy']:.4f}"
    )

    print(
        "Instance action accuracy: "
        f"{metrics['instance_action_accuracy']:.4f}"
    )

    print(
        "Edit scope accuracy: "
        f"{metrics['edit_scope_accuracy']:.4f}"
    )

    print(
        "Output exact match: "
        f"{metrics['output_exact_match']:.4f}"
    )

    print(f"Output: {output_path}")

    return 0 if metrics["failed_count"] == 0 else 1


def calculate_oracle_metrics(
    *,
    instances,
    run_records,
) -> dict[str, int | float]:
    instance_count = len(instances)

    failed_count = sum(
        bool(record.errors)
        for record in run_records
    )

    analysis_correct = 0
    instance_action_correct = 0
    edit_scope_correct = 0
    output_correct = 0

    total_span_actions = 0
    correct_span_actions = 0

    for instance, record in zip(
        instances,
        run_records,
        strict=True,
    ):
        if (
            record.predicted_analysis
            == instance.gold_analysis
        ):
            analysis_correct += 1

        if record.edit_plan is not None:
            if (
                record.edit_plan.instance_action
                == instance.gold_decision.instance_action
            ):
                instance_action_correct += 1

            if (
                record.edit_plan.edit_scope
                == instance.gold_decision.edit_scope
            ):
                edit_scope_correct += 1

            predicted_actions = {
                item.span_id: item.action
                for item
                in record.edit_plan.span_decisions
            }

            gold_actions = {
                item.span_id: item.action
                for item
                in instance.gold_decision.span_actions
            }

            for span_id, gold_action in (
                gold_actions.items()
            ):
                total_span_actions += 1

                if (
                    predicted_actions.get(span_id)
                    == gold_action
                ):
                    correct_span_actions += 1

        if record.output_text == instance.gold_output:
            output_correct += 1

    return {
        "instance_count": instance_count,
        "failed_count": failed_count,
        "analysis_exact_match": _safe_divide(
            analysis_correct,
            instance_count,
        ),
        "span_action_accuracy": _safe_divide(
            correct_span_actions,
            total_span_actions,
        ),
        "instance_action_accuracy": _safe_divide(
            instance_action_correct,
            instance_count,
        ),
        "edit_scope_accuracy": _safe_divide(
            edit_scope_correct,
            instance_count,
        ),
        "output_exact_match": _safe_divide(
            output_correct,
            instance_count,
        ),
    }


def _safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())