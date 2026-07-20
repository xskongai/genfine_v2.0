from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from genfine.analysis import (
    AnalyzerError,
    AnalysisPromptBuilder,
    LLMAnalyzer,
)
from genfine.data.loader import load_dataset
from genfine.providers import OpenAITextClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an LLMAnalyzer over a GenFINE dataset."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/seed/seed_v0.2.jsonl",
    )

    parser.add_argument(
        "--prompt",
        default=(
            "configs/prompts/analyze_v0.1.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        default="runs/llm_analysis_v0.1.jsonl",
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env_path = PROJECT_ROOT / ".env"

    load_dotenv(
        dotenv_path=env_path,
        override=False,
    )

    model = (
        args.model
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o"
    )

    instances = load_dataset(
        PROJECT_ROOT / args.dataset
    )

    if args.limit is not None:
        instances = instances[:args.limit]

    prompt_builder = (
        AnalysisPromptBuilder.from_yaml(
            PROJECT_ROOT / args.prompt
        )
    )

    client = OpenAITextClient(
        model=model,
    )

    analyzer = LLMAnalyzer(
        client=client,
        prompt_builder=prompt_builder,
    )

    output_path = PROJECT_ROOT / args.output

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed = 0
    exact_matches = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for instance in instances:
            try:
                predicted_analysis = (
                    analyzer.analyze(instance)
                )

                exact_match = (
                    predicted_analysis
                    == instance.gold_analysis
                )

                if exact_match:
                    exact_matches += 1

                payload = {
                    "instance_id": (
                        instance.instance_id
                    ),
                    "original_text": (
                        instance.context.target_text
                    ),
                    "analyzer": analyzer.name,
                    "prompt_version": (
                        prompt_builder.version
                    ),
                    "predicted_analysis": (
                        predicted_analysis.model_dump(
                            mode="json",
                            exclude_none=True,
                        )
                    ),
                    "gold_analysis_exact_match": (
                        exact_match
                    ),
                }

            except AnalyzerError as exc:
                failed += 1

                payload = {
                    "instance_id": (
                        instance.instance_id
                    ),
                    "original_text": (
                        instance.context.target_text
                    ),
                    "analyzer": analyzer.name,
                    "prompt_version": (
                        prompt_builder.version
                    ),
                    "error": str(exc),
                }

                if not args.continue_on_error:
                    print(
                        str(exc),
                        file=sys.stderr,
                    )
                    return 1

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    print(f"Model: {model}")
    print(f"Instances: {len(instances)}")
    print(f"Failed: {failed}")
    print(
        "Analysis exact match: "
        f"{exact_matches}/{len(instances)}"
    )
    print(
        "Output: "
        f"{output_path.relative_to(PROJECT_ROOT)}"
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())