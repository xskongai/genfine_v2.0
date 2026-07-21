from __future__ import annotations

import argparse
import json
import os
import sys
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
from genfine.generation import (
    GPTRewriter,
    RewritePromptBuilder,
)
from genfine.providers import OpenAITextClient
from genfine.verification import OutputVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute saved LLM analysis/edit plans with GPTRewriter, "
            "then run deterministic output verification."
        )
    )

    parser.add_argument(
        "decisions",
        nargs="?",
        default="runs/llm_decisions_v0.2.jsonl",
        help=(
            "JSONL produced by scripts/run_llm_decisions.py."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="data/seed/seed_v0.2.jsonl",
    )
    parser.add_argument(
        "--prompt",
        default=os.getenv(
            "GENFINE_REWRITE_PROMPT",
            "configs/prompts/rewrite_v0.1.yaml",
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model name. Defaults to OPENAI_MODEL.",
    )
    parser.add_argument(
        "--output",
        default="runs/llm_gpt_rewriter_v0.2.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail before rewriting when a dataset instance has no "
            "saved decision record."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
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
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for record in records:
        instance_id = record.get("instance_id")

        if not isinstance(instance_id, str):
            raise ValueError(
                "Every decision record must contain a string instance_id"
            )

        if instance_id in indexed:
            raise ValueError(
                f"Duplicate decision record for {instance_id!r}"
            )

        indexed[instance_id] = record

    return indexed


def validate_record_coverage(
    *,
    selected_instances: Sequence[DatasetInstance],
    all_dataset_instances: Sequence[DatasetInstance],
    indexed_records: dict[str, dict[str, Any]],
    require_complete: bool,
) -> None:
    selected_ids = {
        instance.instance_id
        for instance in selected_instances
    }
    all_dataset_ids = {
        instance.instance_id
        for instance in all_dataset_instances
    }
    record_ids = set(indexed_records)

    unknown_ids = record_ids - all_dataset_ids

    if unknown_ids:
        raise ValueError(
            "Decision records refer to unknown dataset instances: "
            f"{sorted(unknown_ids)}"
        )

    missing_ids = selected_ids - record_ids

    if require_complete and missing_ids:
        raise ValueError(
            "Selected dataset instances have no saved decision record: "
            f"{sorted(missing_ids)}"
        )


def load_saved_inputs(
    *,
    instance: DatasetInstance,
    source: dict[str, Any],
) -> tuple[AnalysisResult, EditPlan]:
    source_error = source.get("error")

    if source_error:
        stage = source.get("failure_stage", "unknown")
        raise ValueError(
            f"Saved decision record failed at {stage}: {source_error}"
        )

    try:
        analysis_payload = source["predicted_analysis"]
        edit_plan_payload = source["edit_plan"]
    except KeyError as exc:
        raise ValueError(
            f"Saved decision record is missing {exc.args[0]!r}"
        ) from exc

    try:
        analysis = AnalysisResult.model_validate(
            analysis_payload
        )
        edit_plan = EditPlan.model_validate(
            edit_plan_payload
        )
    except ValidationError as exc:
        raise ValueError(
            f"Invalid saved analysis/edit plan: {exc}"
        ) from exc

    if edit_plan.instance_id != instance.instance_id:
        raise ValueError(
            "Saved edit plan instance_id does not match "
            f"{instance.instance_id!r}"
        )

    if (
        edit_plan.original_text
        != instance.context.target_text
    ):
        raise ValueError(
            "Saved edit plan original_text does not match "
            f"the dataset text for {instance.instance_id!r}"
        )

    return analysis, edit_plan


def execute_saved_record(
    *,
    instance: DatasetInstance,
    source: dict[str, Any],
    rewriter: GPTRewriter,
    verifier: OutputVerifier,
    rewrite_prompt_version: str,
) -> RunRecord:
    analysis, edit_plan = load_saved_inputs(
        instance=instance,
        source=source,
    )

    output_text = rewriter.rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
    )

    verification = verifier.verify(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
        output_text=output_text,
    )

    return RunRecord(
        instance_id=instance.instance_id,
        original_text=instance.context.target_text,
        predicted_analysis=analysis,
        edit_plan=edit_plan,
        output_text=output_text,
        verification=verification,
        metadata={
            "analyzer": source.get("analyzer", "unknown"),
            "analysis_prompt_version": source.get(
                "prompt_version"
            ),
            "rewriter": rewriter.name,
            "rewrite_prompt_version": (
                rewrite_prompt_version
            ),
            "decision_rule_version": source.get(
                "rule_version"
            ),
            "task_mode": instance.task_mode.value,
            "saved_decision_source": True,
            "verification_enabled": True,
        },
    )


def make_error_record(
    *,
    instance: DatasetInstance,
    source: dict[str, Any] | None,
    rewriter_name: str,
    rewrite_prompt_version: str,
    error: Exception,
) -> RunRecord:
    return RunRecord(
        instance_id=instance.instance_id,
        original_text=instance.context.target_text,
        errors=[
            f"{type(error).__name__}: {error}"
        ],
        metadata={
            "analyzer": (
                source.get("analyzer", "unknown")
                if source
                else "unknown"
            ),
            "analysis_prompt_version": (
                source.get("prompt_version")
                if source
                else None
            ),
            "rewriter": rewriter_name,
            "rewrite_prompt_version": (
                rewrite_prompt_version
            ),
            "decision_rule_version": (
                source.get("rule_version")
                if source
                else None
            ),
            "task_mode": instance.task_mode.value,
            "saved_decision_source": True,
            "verification_enabled": True,
        },
    )


def run_saved_records(
    *,
    instances: Sequence[DatasetInstance],
    indexed_records: dict[str, dict[str, Any]],
    rewriter: GPTRewriter,
    verifier: OutputVerifier,
    rewrite_prompt_version: str,
    continue_on_error: bool,
) -> list[RunRecord]:
    results: list[RunRecord] = []

    for instance in instances:
        source = indexed_records.get(
            instance.instance_id
        )

        try:
            if source is None:
                raise ValueError(
                    "Missing saved decision record"
                )

            record = execute_saved_record(
                instance=instance,
                source=source,
                rewriter=rewriter,
                verifier=verifier,
                rewrite_prompt_version=(
                    rewrite_prompt_version
                ),
            )
        except Exception as exc:
            if not continue_on_error:
                raise

            record = make_error_record(
                instance=instance,
                source=source,
                rewriter_name=rewriter.name,
                rewrite_prompt_version=(
                    rewrite_prompt_version
                ),
                error=exc,
            )

        results.append(record)

    return results


def write_run_records(
    *,
    path: Path,
    records: Sequence[RunRecord],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record.model_dump(
                        mode="json",
                        exclude_none=True,
                    ),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def print_summary(
    *,
    instances: Sequence[DatasetInstance],
    records: Sequence[RunRecord],
    output_path: Path,
    model_name: str,
) -> None:
    instance_by_id = {
        instance.instance_id: instance
        for instance in instances
    }

    failed = sum(
        bool(record.errors)
        for record in records
    )

    changed = sum(
        record.output_text is not None
        and record.output_text != record.original_text
        for record in records
    )

    exact = sum(
        record.output_text
        == instance_by_id[record.instance_id].gold_output
        for record in records
    )

    verification_covered = sum(
        record.verification is not None
        for record in records
    )

    verification_passed = sum(
        bool(
            record.verification
            and record.verification.passed
        )
        for record in records
    )

    api_eligible = sum(
        bool(
            record.edit_plan
            and record.edit_plan.instance_action
            in {
                InstanceAction.EDIT,
                InstanceAction.SPAN_LEVEL_EDIT,
            }
        )
        for record in records
    )

    print(f"Model: {model_name}")
    print(f"Instances: {len(records)}")
    print(f"Failed: {failed}")
    print(f"API-eligible edit plans: {api_eligible}")
    print(f"Changed outputs: {changed}")
    print(
        "Gold exact match: "
        f"{exact}/{len(records)}"
    )
    print(
        "Verification coverage: "
        f"{verification_covered}/{len(records)}"
    )
    print(
        "Verification passed: "
        f"{verification_passed}/{len(records)}"
    )
    print(f"Output: {output_path}")


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
                    "--limit must be at least 1"
                )

            instances = all_instances[:args.limit]

        decision_records = load_jsonl(
            Path(args.decisions)
        )
        indexed_records = index_records(
            decision_records
        )

        validate_record_coverage(
            selected_instances=instances,
            all_dataset_instances=all_instances,
            indexed_records=indexed_records,
            require_complete=args.require_complete,
        )

        prompt_builder = (
            RewritePromptBuilder.from_yaml(
                args.prompt
            )
        )
        client = OpenAITextClient(
            model=args.model
        )
        rewriter = GPTRewriter(
            client=client,
            prompt_builder=prompt_builder,
        )
        verifier = OutputVerifier.default()

        run_records = run_saved_records(
            instances=instances,
            indexed_records=indexed_records,
            rewriter=rewriter,
            verifier=verifier,
            rewrite_prompt_version=(
                prompt_builder.version
            ),
            continue_on_error=(
                args.continue_on_error
            ),
        )

        output_path = Path(args.output)

        write_run_records(
            path=output_path,
            records=run_records,
        )

        print_summary(
            instances=instances,
            records=run_records,
            output_path=output_path,
            model_name=client.model,
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(
            f"Execution failed: {exc}",
            file=sys.stderr,
        )
        return 1

    failed_count = sum(
        bool(record.errors)
        for record in run_records
    )

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
