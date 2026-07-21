from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from genfine.data.loader import load_dataset
from genfine.domain.enums import InstanceAction
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    RunRecord,
)
from genfine.factual_support import (
    FactualSupportJudge,
    FactualSupportJudgeError,
    FactualSupportLabel,
    FactualSupportPromptBuilder,
    FactualSupportPromptError,
    FactualSupportStatus,
    LLMFactualSupportJudge,
)
from genfine.providers import OpenAITextClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved GenFINE rewritten outputs for "
            "unsupported factual insertion."
        )
    )

    parser.add_argument(
        "run_file",
        nargs="?",
        default="runs/llm_gpt_rewriter_v0.2.jsonl",
        help=(
            "RunRecord JSONL containing predicted_analysis, "
            "edit_plan and output_text."
        ),
    )

    parser.add_argument(
        "--dataset",
        default="data/seed/seed_v0.2.jsonl",
        help="Gold dataset used to identify source context.",
    )

    parser.add_argument(
        "--prompt",
        default=os.getenv(
            "GENFINE_FACTUAL_SUPPORT_PROMPT",
            "configs/prompts/factual_support_v0.1.yaml",
        ),
        help="Factual-support prompt configuration.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "OpenAI model used as the independent judge. "
            "Defaults to OPENAI_MODEL."
        ),
    )

    parser.add_argument(
        "--output",
        default="runs/factual_support_v0.1.jsonl",
        help="Per-instance factual-support results.",
    )

    parser.add_argument(
        "--metrics-output",
        default="runs/factual_support_metrics_v0.1.json",
        help="Aggregated factual-support metrics.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N dataset instances.",
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail before evaluation when a selected dataset "
            "instance has no saved RunRecord."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Write an error record and continue when one "
            "instance cannot be evaluated."
        ),
    )

    return parser.parse_args()


def load_run_records(
    path: Path,
) -> list[RunRecord]:
    """
    Load and validate RunRecord JSONL.
    """

    records: list[RunRecord] = []

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
                    "must be a JSON object."
                )

            try:
                record = RunRecord.model_validate(
                    payload
                )
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid RunRecord at "
                    f"{path}:{line_number}: {exc}"
                ) from exc

            records.append(record)

    return records


def index_run_records(
    records: Sequence[RunRecord],
) -> dict[str, RunRecord]:
    """
    Index RunRecords and reject duplicated instance IDs.
    """

    indexed: dict[str, RunRecord] = {}

    for record in records:
        if record.instance_id in indexed:
            raise ValueError(
                "Duplicate RunRecord for "
                f"{record.instance_id!r}."
            )

        indexed[record.instance_id] = record

    return indexed


def validate_record_coverage(
    *,
    selected_instances: Sequence[DatasetInstance],
    all_dataset_instances: Sequence[DatasetInstance],
    indexed_records: dict[str, RunRecord],
    require_complete: bool,
) -> None:
    """
    Validate unknown and missing instance identifiers.

    Unknown records are always rejected. Missing selected records
    are rejected during preflight only when require_complete=True.
    """

    selected_ids = {
        instance.instance_id
        for instance in selected_instances
    }

    all_dataset_ids = {
        instance.instance_id
        for instance in all_dataset_instances
    }

    record_ids = set(indexed_records)

    unknown_ids = (
        record_ids
        - all_dataset_ids
    )

    if unknown_ids:
        raise ValueError(
            "RunRecords refer to unknown dataset instances: "
            f"{sorted(unknown_ids)}"
        )

    missing_ids = (
        selected_ids
        - record_ids
    )

    if require_complete and missing_ids:
        raise ValueError(
            "Selected dataset instances have no RunRecord: "
            f"{sorted(missing_ids)}"
        )


def validate_saved_inputs(
    *,
    instance: DatasetInstance,
    record: RunRecord,
) -> tuple[
    AnalysisResult,
    EditPlan,
    str | None,
]:
    """
    Validate the saved semantic inputs used by the judge.

    The factual-support evaluator must use exactly the analysis,
    plan and output that were present during generation.
    """

    if record.errors:
        raise ValueError(
            "Source RunRecord contains errors: "
            f"{record.errors}"
        )

    if (
        record.instance_id
        != instance.instance_id
    ):
        raise ValueError(
            "RunRecord instance_id does not match "
            f"{instance.instance_id!r}."
        )

    if (
        record.original_text
        != instance.context.target_text
    ):
        raise ValueError(
            "RunRecord original_text does not match "
            f"the dataset text for "
            f"{instance.instance_id!r}."
        )

    analysis = record.predicted_analysis
    edit_plan = record.edit_plan
    output_text = record.output_text

    if analysis is None:
        raise ValueError(
            "RunRecord has no predicted_analysis."
        )

    if edit_plan is None:
        raise ValueError(
            "RunRecord has no edit_plan."
        )

    if (
        edit_plan.instance_id
        != instance.instance_id
    ):
        raise ValueError(
            "EditPlan instance_id does not match "
            f"{instance.instance_id!r}."
        )

    if (
        edit_plan.original_text
        != instance.context.target_text
    ):
        raise ValueError(
            "EditPlan original_text does not match "
            f"the dataset text for "
            f"{instance.instance_id!r}."
        )

    if (
        output_text is None
        and edit_plan.instance_action
        != InstanceAction.ABSTAIN
    ):
        raise ValueError(
            "A non-ABSTAIN EditPlan has no output_text."
        )

    if (
        output_text is not None
        and edit_plan.instance_action
        == InstanceAction.ABSTAIN
    ):
        raise ValueError(
            "An ABSTAIN EditPlan produced output_text."
        )

    return (
        analysis,
        edit_plan,
        output_text,
    )


def extract_plan_actions(
    edit_plan: EditPlan,
) -> list[str]:
    """
    Return deterministic unique action names.
    """

    return sorted(
        {
            decision.action.value
            for decision
            in edit_plan.span_decisions
        }
    )


def evaluate_saved_record(
    *,
    instance: DatasetInstance,
    record: RunRecord,
    judge: FactualSupportJudge,
) -> dict[str, Any]:
    """
    Evaluate one saved generated output.
    """

    (
        analysis,
        edit_plan,
        output_text,
    ) = validate_saved_inputs(
        instance=instance,
        record=record,
    )

    result = judge.evaluate(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
        output_text=output_text,
    )

    if (
        result.instance_id
        != instance.instance_id
    ):
        raise ValueError(
            "Factual-support result instance_id "
            "does not match the dataset instance."
        )

    return {
        "instance_id": (
            instance.instance_id
        ),
        "original_text": (
            instance.context.target_text
        ),
        "output_text": output_text,
        "changed": (
            output_text is not None
            and output_text
            != instance.context.target_text
        ),
        "instance_action": (
            edit_plan.instance_action.value
        ),
        "edit_scope": (
            edit_plan.edit_scope.value
        ),
        "plan_actions": (
            extract_plan_actions(
                edit_plan
            )
        ),
        "factual_support": (
            result.model_dump(
                mode="json",
                exclude_none=True,
            )
        ),
        "source_metadata": dict(
            record.metadata
        ),
    }


def make_error_record(
    *,
    instance: DatasetInstance,
    source: RunRecord | None,
    error: Exception,
) -> dict[str, Any]:
    """
    Construct one serializable evaluation failure record.
    """

    return {
        "instance_id": (
            instance.instance_id
        ),
        "original_text": (
            instance.context.target_text
        ),
        "output_text": (
            source.output_text
            if source is not None
            else None
        ),
        "changed": (
            source is not None
            and source.output_text is not None
            and source.output_text
            != instance.context.target_text
        ),
        "error": (
            f"{type(error).__name__}: "
            f"{error}"
        ),
        "source_metadata": (
            dict(source.metadata)
            if source is not None
            else {}
        ),
    }


def run_evaluation(
    *,
    instances: Sequence[DatasetInstance],
    indexed_records: dict[str, RunRecord],
    judge: FactualSupportJudge,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    """
    Evaluate selected instances in dataset order.
    """

    results: list[dict[str, Any]] = []

    for instance in instances:
        source = indexed_records.get(
            instance.instance_id
        )

        try:
            if source is None:
                raise ValueError(
                    "Missing saved RunRecord."
                )

            result = evaluate_saved_record(
                instance=instance,
                record=source,
                judge=judge,
            )

        except Exception as exc:
            if not continue_on_error:
                raise

            result = make_error_record(
                instance=instance,
                source=source,
                error=exc,
            )

        results.append(result)

    return results


def build_metric(
    *,
    numerator: int,
    denominator: int,
) -> dict[str, int | float]:
    """
    Build the same numerator/denominator/value shape used by
    the rest of the evaluation system.
    """

    value = (
        numerator / denominator
        if denominator
        else 0.0
    )

    return {
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
    }


def build_metrics(
    *,
    all_dataset_instance_count: int,
    selected_instance_count: int,
    records: Sequence[dict[str, Any]],
    judge_name: str,
    prompt_version: str,
    run_file: str,
    dataset_file: str,
) -> dict[str, Any]:
    """
    Aggregate instance-level and claim-level factual-support metrics.
    """

    successful_records = [
        record
        for record in records
        if "factual_support" in record
    ]

    failed_records = [
        record
        for record in records
        if "error" in record
    ]

    evaluated_records = [
        record
        for record in successful_records
        if (
            record["factual_support"]["status"]
            == FactualSupportStatus.EVALUATED.value
        )
    ]

    no_output_records = [
        record
        for record in successful_records
        if (
            record["factual_support"]["status"]
            == (
                FactualSupportStatus
                .NOT_APPLICABLE_NO_OUTPUT
                .value
            )
        )
    ]

    changed_records = [
        record
        for record in successful_records
        if record["changed"]
    ]

    unsupported_records = [
        record
        for record in evaluated_records
        if (
            record["factual_support"]
            ["unsupported_factual_insertion"]
        )
    ]

    all_claims = [
        claim
        for record in evaluated_records
        for claim in (
            record["factual_support"]
            ["claims"]
        )
    ]

    label_counts = Counter(
        claim["label"]
        for claim in all_claims
    )

    unsupported_claim_count = (
        label_counts[
            FactualSupportLabel
            .UNSUPPORTED_FACTUAL_INSERTION
            .value
        ]
    )

    uncertain_claim_count = (
        label_counts[
            FactualSupportLabel
            .UNCERTAIN
            .value
        ]
    )

    uncertain_instance_ids = sorted(
        {
            record["instance_id"]
            for record in evaluated_records
            if any(
                claim["label"]
                == (
                    FactualSupportLabel
                    .UNCERTAIN
                    .value
                )
                for claim in (
                    record["factual_support"]
                    ["claims"]
                )
            )
        }
    )

    action_counts: Counter[str] = Counter()

    for record in successful_records:
        action_counts.update(
            record.get(
                "plan_actions",
                [],
            )
        )

    return {
        "dataset_instance_count": (
            all_dataset_instance_count
        ),
        "selected_instance_count": (
            selected_instance_count
        ),
        "successful_instance_count": (
            len(successful_records)
        ),
        "failed_instance_count": (
            len(failed_records)
        ),
        "evaluated_output_count": (
            len(evaluated_records)
        ),
        "no_output_count": (
            len(no_output_records)
        ),
        "changed_output_count": (
            len(changed_records)
        ),
        "total_claim_count": (
            len(all_claims)
        ),
        "factual_support_coverage": (
            build_metric(
                numerator=len(
                    successful_records
                ),
                denominator=(
                    selected_instance_count
                ),
            )
        ),
        "semantic_evaluation_coverage": (
            build_metric(
                numerator=len(
                    evaluated_records
                ),
                denominator=(
                    selected_instance_count
                ),
            )
        ),
        "unsupported_factual_insertion_rate": (
            build_metric(
                numerator=len(
                    unsupported_records
                ),
                denominator=len(
                    evaluated_records
                ),
            )
        ),
        "unsupported_factual_claim_rate": (
            build_metric(
                numerator=(
                    unsupported_claim_count
                ),
                denominator=len(
                    all_claims
                ),
            )
        ),
        "uncertain_claim_rate": (
            build_metric(
                numerator=(
                    uncertain_claim_count
                ),
                denominator=len(
                    all_claims
                ),
            )
        ),
        "judge_failure_rate": (
            build_metric(
                numerator=len(
                    failed_records
                ),
                denominator=(
                    selected_instance_count
                ),
            )
        ),
        "claim_label_counts": dict(
            sorted(
                label_counts.items()
            )
        ),
        "plan_action_counts": dict(
            sorted(
                action_counts.items()
            )
        ),
        "unsupported_instance_ids": [
            record["instance_id"]
            for record in unsupported_records
        ],
        "uncertain_instance_ids": (
            uncertain_instance_ids
        ),
        "failed_instance_ids": [
            record["instance_id"]
            for record in failed_records
        ],
        "metadata": {
            "judge": judge_name,
            "prompt_version": prompt_version,
            "source_run_file": run_file,
            "dataset_file": dataset_file,
        },
    }


def write_jsonl(
    *,
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def write_json(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_metric(
    label: str,
    metric: dict[str, Any],
) -> None:
    print(
        f"{label}: "
        f"{metric['value']:.4f} "
        f"({metric['numerator']}/"
        f"{metric['denominator']})"
    )


def print_summary(
    *,
    metrics: dict[str, Any],
    output_path: Path,
    metrics_path: Path,
    judge_name: str,
) -> None:
    print(f"Judge: {judge_name}")

    print(
        "Dataset instances: "
        f"{metrics['dataset_instance_count']}"
    )

    print(
        "Selected instances: "
        f"{metrics['selected_instance_count']}"
    )

    print(
        "Successful instances: "
        f"{metrics['successful_instance_count']}"
    )

    print(
        "Failed instances: "
        f"{metrics['failed_instance_count']}"
    )

    print(
        "Evaluated outputs: "
        f"{metrics['evaluated_output_count']}"
    )

    print(
        "No-output instances: "
        f"{metrics['no_output_count']}"
    )

    print(
        "Total claims: "
        f"{metrics['total_claim_count']}"
    )

    print_metric(
        "Factual-support coverage",
        metrics[
            "factual_support_coverage"
        ],
    )

    print_metric(
        "Unsupported factual insertion",
        metrics[
            "unsupported_factual_insertion_rate"
        ],
    )

    print_metric(
        "Unsupported factual claims",
        metrics[
            "unsupported_factual_claim_rate"
        ],
    )

    print_metric(
        "Uncertain claims",
        metrics[
            "uncertain_claim_rate"
        ],
    )

    print(
        "Unsupported instances: "
        f"{metrics['unsupported_instance_ids']}"
    )

    print(f"Results output: {output_path}")
    print(f"Metrics output: {metrics_path}")


def main() -> int:
    env_path = PROJECT_ROOT / ".env"

    load_dotenv(
        dotenv_path=env_path,
        override=False,
    )

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not configured. "
            f"Expected it in {env_path}",
            file=sys.stderr,
        )
        return 1

    args = parse_args()

    try:
        all_instances = load_dataset(
            args.dataset
        )

        instances = all_instances

        if args.limit is not None:
            if args.limit < 1:
                raise ValueError(
                    "--limit must be at least 1."
                )

            instances = all_instances[
                :args.limit
            ]

        run_records = load_run_records(
            Path(args.run_file)
        )

        indexed_records = index_run_records(
            run_records
        )

        validate_record_coverage(
            selected_instances=instances,
            all_dataset_instances=(
                all_instances
            ),
            indexed_records=(
                indexed_records
            ),
            require_complete=(
                args.require_complete
            ),
        )

        prompt_builder = (
            FactualSupportPromptBuilder
            .from_yaml(
                args.prompt
            )
        )

        client = OpenAITextClient(
            model=args.model
        )

        judge = LLMFactualSupportJudge(
            client=client,
            prompt_builder=prompt_builder,
        )

        results = run_evaluation(
            instances=instances,
            indexed_records=(
                indexed_records
            ),
            judge=judge,
            continue_on_error=(
                args.continue_on_error
            ),
        )

        metrics = build_metrics(
            all_dataset_instance_count=(
                len(all_instances)
            ),
            selected_instance_count=(
                len(instances)
            ),
            records=results,
            judge_name=judge.name,
            prompt_version=(
                prompt_builder.version
            ),
            run_file=args.run_file,
            dataset_file=args.dataset,
        )

        output_path = Path(
            args.output
        )

        metrics_path = Path(
            args.metrics_output
        )

        write_jsonl(
            path=output_path,
            records=results,
        )

        write_json(
            path=metrics_path,
            payload=metrics,
        )

        print_summary(
            metrics=metrics,
            output_path=output_path,
            metrics_path=metrics_path,
            judge_name=judge.name,
        )

    except (
        FileNotFoundError,
        ValueError,
        FactualSupportPromptError,
        FactualSupportJudgeError,
    ) as exc:
        print(
            f"Factual-support evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    return (
        0
        if metrics["failed_instance_count"] == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())