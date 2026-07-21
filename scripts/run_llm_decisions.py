from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from genfine.data.loader import load_dataset
from genfine.domain.models import AnalysisResult
from genfine.evaluation.analysis_evaluator import match_spans
from genfine.pipeline import EditPlanBuilder
from genfine.policy import DecisionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DecisionEngine and EditPlanBuilder over saved "
            "LLMAnalyzer predictions."
        )
    )

    parser.add_argument(
        "predictions",
        nargs="?",
        default="runs/llm_analysis_v0.1.jsonl",
    )
    parser.add_argument(
        "--dataset",
        default="data/seed/seed_v0.2.jsonl",
    )
    parser.add_argument(
        "--rules",
        default="configs/decision_rules.yaml",
    )
    parser.add_argument(
        "--output",
        default="runs/llm_decisions_v0.1.jsonl",
    )
    parser.add_argument(
        "--json-output",
        default="runs/llm_decision_metrics_v0.1.json",
    )
    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Record at {path}:{line_number} must be an object"
                )

            records.append(payload)

    return records


def index_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for record in records:
        instance_id = record.get("instance_id")

        if not isinstance(instance_id, str):
            raise ValueError(
                "Every analysis record must contain "
                "a string instance_id"
            )

        if instance_id in indexed:
            raise ValueError(
                f"Duplicate analysis record for {instance_id!r}"
            )

        indexed[instance_id] = record

    return indexed


def make_error_record(
    *,
    instance,
    source: dict[str, Any] | None,
    stage: str,
    error: str,
    rule_version: str,
) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "original_text": instance.context.target_text,
        "analyzer": (
            source.get("analyzer", "unknown")
            if source
            else "unknown"
        ),
        "prompt_version": (
            source.get("prompt_version")
            if source
            else None
        ),
        "rule_version": rule_version,
        "failure_stage": stage,
        "error": error,
    }


def run_decisions(
    *,
    instances,
    analysis_records: list[dict[str, Any]],
    engine: DecisionEngine,
    builder: EditPlanBuilder,
) -> list[dict[str, Any]]:
    analysis_by_id = index_records(analysis_records)
    output_records: list[dict[str, Any]] = []

    for instance in instances:
        source = analysis_by_id.get(instance.instance_id)

        if source is None:
            output_records.append(
                make_error_record(
                    instance=instance,
                    source=None,
                    stage="analysis",
                    error="Missing analysis prediction record",
                    rule_version=engine.version,
                )
            )
            continue

        if "predicted_analysis" not in source:
            output_records.append(
                make_error_record(
                    instance=instance,
                    source=source,
                    stage="analysis",
                    error=str(
                        source.get(
                            "error",
                            "Missing predicted_analysis",
                        )
                    ),
                    rule_version=engine.version,
                )
            )
            continue

        try:
            analysis = AnalysisResult.model_validate(
                source["predicted_analysis"]
            )
        except ValidationError as exc:
            output_records.append(
                make_error_record(
                    instance=instance,
                    source=source,
                    stage="analysis",
                    error=f"ValidationError: {exc}",
                    rule_version=engine.version,
                )
            )
            continue

        try:
            decisions = engine.decide_analysis(
                analysis=analysis,
                task_mode=instance.task_mode,
            )
        except Exception as exc:
            output_records.append(
                make_error_record(
                    instance=instance,
                    source=source,
                    stage="decision",
                    error=f"{type(exc).__name__}: {exc}",
                    rule_version=engine.version,
                )
            )
            continue

        try:
            edit_plan = builder.build(
                instance=instance,
                analysis=analysis,
                decisions=decisions,
            )
        except Exception as exc:
            output_records.append(
                make_error_record(
                    instance=instance,
                    source=source,
                    stage="edit_plan",
                    error=f"{type(exc).__name__}: {exc}",
                    rule_version=engine.version,
                )
            )
            continue

        output_records.append(
            {
                "instance_id": instance.instance_id,
                "original_text": instance.context.target_text,
                "analyzer": source.get("analyzer", "unknown"),
                "prompt_version": source.get("prompt_version"),
                "rule_version": engine.version,
                "predicted_analysis": analysis.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "span_decisions": [
                    decision.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                    for decision in decisions
                ],
                "edit_plan": edit_plan.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            }
        )

    return output_records


def evaluate_records(
    *,
    instances,
    records: list[dict[str, Any]],
    minimum_overlap: float,
) -> dict[str, Any]:
    records_by_id = index_records(records)

    successful_instances = 0
    analysis_failures = 0
    decision_failures = 0

    span_correct = 0
    span_total = 0

    matched_span_correct = 0
    matched_span_total = 0

    instance_action_correct = 0
    edit_scope_correct = 0
    plan_exact_correct = 0

    unmatched_gold_spans = 0
    extra_predicted_decisions = 0
    failed_instance_ids: list[str] = []

    for instance in instances:
        record = records_by_id.get(instance.instance_id)
        gold_spans = instance.gold_analysis.spans
        span_total += len(gold_spans)

        if record is None or record.get("error"):
            failed_instance_ids.append(instance.instance_id)
            unmatched_gold_spans += len(gold_spans)

            stage = (
                record.get("failure_stage")
                if record
                else "analysis"
            )

            if stage == "analysis":
                analysis_failures += 1
            else:
                decision_failures += 1

            continue

        successful_instances += 1

        analysis = AnalysisResult.model_validate(
            record["predicted_analysis"]
        )

        predicted_decisions = {
            decision["span_id"]: decision["action"]
            for decision in record["span_decisions"]
        }

        gold_actions = {
            decision.span_id: decision.action.value
            for decision
            in instance.gold_decision.span_actions
        }

        matches = match_spans(
            gold_spans=gold_spans,
            predicted_spans=analysis.spans,
            minimum_overlap=minimum_overlap,
        )

        matched_prediction_ids: set[str] = set()
        all_gold_actions_correct = True

        for gold_span, predicted_span in matches:
            matched_prediction_ids.add(
                predicted_span.span_id
            )

            predicted_action = predicted_decisions.get(
                predicted_span.span_id
            )
            gold_action = gold_actions.get(
                gold_span.span_id
            )

            is_correct = predicted_action == gold_action

            matched_span_total += 1

            if is_correct:
                span_correct += 1
                matched_span_correct += 1
            else:
                all_gold_actions_correct = False

        missing_gold = len(gold_spans) - len(matches)
        unmatched_gold_spans += missing_gold

        if missing_gold:
            all_gold_actions_correct = False

        extra_decisions = (
            set(predicted_decisions)
            - matched_prediction_ids
        )

        extra_predicted_decisions += len(extra_decisions)

        edit_plan = record["edit_plan"]

        predicted_instance_action = edit_plan[
            "instance_action"
        ]
        predicted_edit_scope = edit_plan["edit_scope"]

        gold_instance_action = (
            instance.gold_decision.instance_action.value
        )
        gold_edit_scope = (
            instance.gold_decision.edit_scope.value
        )

        instance_is_correct = (
            predicted_instance_action
            == gold_instance_action
        )
        scope_is_correct = (
            predicted_edit_scope
            == gold_edit_scope
        )

        if instance_is_correct:
            instance_action_correct += 1

        if scope_is_correct:
            edit_scope_correct += 1

        if (
            instance_is_correct
            and scope_is_correct
            and all_gold_actions_correct
            and not extra_decisions
        ):
            plan_exact_correct += 1

    instance_total = len(instances)

    def ratio(
        numerator: int,
        denominator: int,
    ) -> float:
        return (
            numerator / denominator
            if denominator
            else 0.0
        )

    return {
        "dataset_instances": instance_total,
        "successful_instances": successful_instances,
        "pipeline_coverage": {
            "correct": successful_instances,
            "total": instance_total,
            "accuracy": ratio(
                successful_instances,
                instance_total,
            ),
        },
        "analysis_failures": analysis_failures,
        "analysis_failure_rate": ratio(
            analysis_failures,
            instance_total,
        ),
        "decision_or_plan_failures": decision_failures,
        "span_action": {
            "correct": span_correct,
            "total": span_total,
            "accuracy": ratio(
                span_correct,
                span_total,
            ),
        },
        "matched_span_action": {
            "correct": matched_span_correct,
            "total": matched_span_total,
            "accuracy": ratio(
                matched_span_correct,
                matched_span_total,
            ),
        },
        "instance_action": {
            "correct": instance_action_correct,
            "total": instance_total,
            "accuracy": ratio(
                instance_action_correct,
                instance_total,
            ),
        },
        "edit_scope": {
            "correct": edit_scope_correct,
            "total": instance_total,
            "accuracy": ratio(
                edit_scope_correct,
                instance_total,
            ),
        },
        "plan_exact_match": {
            "correct": plan_exact_correct,
            "total": instance_total,
            "accuracy": ratio(
                plan_exact_correct,
                instance_total,
            ),
        },
        "unmatched_gold_spans": unmatched_gold_spans,
        "extra_predicted_decisions": (
            extra_predicted_decisions
        ),
        "failed_instance_ids": failed_instance_ids,
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def print_metric(
    name: str,
    metric: dict[str, Any],
) -> None:
    print(
        f"{name}: "
        f"{metric['accuracy']:.4f} "
        f"({metric['correct']}/{metric['total']})"
    )


def main() -> int:
    args = parse_args()

    if not 0.0 < args.minimum_overlap <= 1.0:
        print(
            "--minimum-overlap must be in (0, 1].",
            file=sys.stderr,
        )
        return 2

    try:
        instances = load_dataset(
            PROJECT_ROOT / args.dataset
        )
        analysis_records = load_jsonl(
            PROJECT_ROOT / args.predictions
        )

        engine = DecisionEngine.from_yaml(
            PROJECT_ROOT / args.rules
        )

        records = run_decisions(
            instances=instances,
            analysis_records=analysis_records,
            engine=engine,
            builder=EditPlanBuilder(),
        )

        metrics = evaluate_records(
            instances=instances,
            records=records,
            minimum_overlap=args.minimum_overlap,
        )

        output_path = PROJECT_ROOT / args.output
        metrics_path = PROJECT_ROOT / args.json_output

        write_jsonl(
            output_path,
            records,
        )

        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        metrics_path.write_text(
            json.dumps(
                metrics,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(
            f"Run failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Dataset instances: "
        f"{metrics['dataset_instances']}"
    )
    print(
        f"Successful instances: "
        f"{metrics['successful_instances']}"
    )
    print()

    print_metric(
        "Pipeline coverage",
        metrics["pipeline_coverage"],
    )
    print(
        "Analysis failure rate: "
        f"{metrics['analysis_failure_rate']:.4f} "
        f"({metrics['analysis_failures']}/"
        f"{metrics['dataset_instances']})"
    )
    print(
        "Decision/edit-plan failures: "
        f"{metrics['decision_or_plan_failures']}"
    )
    print()

    print_metric(
        "Span action accuracy",
        metrics["span_action"],
    )
    print_metric(
        "Matched-span action accuracy",
        metrics["matched_span_action"],
    )
    print_metric(
        "Instance action accuracy",
        metrics["instance_action"],
    )
    print_metric(
        "Edit scope accuracy",
        metrics["edit_scope"],
    )
    print_metric(
        "Plan exact match",
        metrics["plan_exact_match"],
    )
    print()

    print(
        "Unmatched gold spans: "
        f"{metrics['unmatched_gold_spans']}"
    )
    print(
        "Extra predicted decisions: "
        f"{metrics['extra_predicted_decisions']}"
    )

    if metrics["failed_instance_ids"]:
        print(
            "Failed instances: "
            + ", ".join(
                metrics["failed_instance_ids"]
            )
        )

    print()
    print(f"Output: {output_path}")
    print(f"JSON summary: {metrics_path}")

    if (
        args.require_complete
        and metrics["successful_instances"]
        != metrics["dataset_instances"]
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())