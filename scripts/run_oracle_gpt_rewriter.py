from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import (
    DatasetLoadError,
    load_dataset_records,
)
from genfine.data.validator import DatasetValidator
from genfine.generation import (
    GPTRewriter,
    RewritePromptBuilder,
)
from genfine.pipeline import (
    EditPlanBuilder,
    PipelineRunner,
)
from genfine.policy import DecisionEngine
from genfine.providers import OpenAITextClient

from genfine.verification import OutputVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Oracle analysis and policy, "
            "then execute edits with an OpenAI model."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/seed/seed_v0.1.jsonl",
    )

    parser.add_argument(
        "--rules",
        default="configs/decision_rules.yaml",
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
        help=(
            "OpenAI model name. Defaults to OPENAI_MODEL."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        default=(
            "runs/oracle_gpt_rewriter_v0.1.jsonl"
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    env_path = PROJECT_ROOT / ".env"

    load_dotenv(
        dotenv_path=env_path,
        override=False,
    )

    if not os.getenv("OPENAI_API_KEY"):
        print(
            f"OPENAI_API_KEY is not configured. "
            f"Expected it in {env_path}",
            file=sys.stderr,
        )
        return 1

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

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError(
                "--limit must be at least 1"
            )

        instances = instances[:args.limit]

    client = OpenAITextClient(
        model=args.model
    )

    prompt_builder = (
        RewritePromptBuilder.from_yaml(
            args.prompt
        )
    )

    runner = PipelineRunner(
        analyzer=OracleAnalyzer(),
        decision_engine=(
            DecisionEngine.from_yaml(
                args.rules
            )
        ),
        edit_plan_builder=EditPlanBuilder(),
        rewriter=GPTRewriter(
            client=client,
            prompt_builder=prompt_builder,
        ),
        verifier=OutputVerifier.default(),
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

    failed_count = sum(
        bool(record.errors)
        for record in run_records
    )

    changed_count = sum(
        record.output_text
        != record.original_text
        for record in run_records
        if record.output_text is not None
    )

    exact_match_count = sum(
        record.output_text
        == instance.gold_output
        for instance, record in zip(
            instances,
            run_records,
            strict=True,
        )
    )

    verification_passed_count = sum(
        bool(
            record.verification
            and record.verification.passed
        )
        for record in run_records
    )

    print(f"Model: {client.model}")
    print(f"Instances: {len(run_records)}")
    print(f"Failed: {failed_count}")
    print(f"Changed outputs: {changed_count}")

    print(
        "Gold exact match: "
        f"{exact_match_count}/{len(run_records)}"
    )

    print(
        "Verification passed: "
        f"{verification_passed_count}/"
        f"{len(run_records)}"
    )

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
