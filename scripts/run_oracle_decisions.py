from __future__ import annotations

import argparse
import json
from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.policy import DecisionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the GenFINE decision engine "
            "with Oracle analyses."
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
        "--output",
        default="runs/oracle_decisions_v0.1.jsonl",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    instances = load_dataset(
        args.dataset
    )

    analyzer = OracleAnalyzer()

    engine = DecisionEngine.from_yaml(
        args.rules
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_spans = 0
    correct_spans = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for instance in instances:
            analysis = analyzer.analyze(
                instance
            )

            decisions = engine.decide_analysis(
                analysis=analysis,
                task_mode=instance.task_mode,
            )

            gold_by_span = {
                item.span_id: item.action
                for item
                in instance.gold_decision.span_actions
            }

            predicted_by_span = {
                item.span_id: item.action
                for item in decisions
            }

            for span_id, predicted_action in (
                predicted_by_span.items()
            ):
                total_spans += 1

                if (
                    gold_by_span.get(span_id)
                    == predicted_action
                ):
                    correct_spans += 1

            payload = {
                "instance_id": instance.instance_id,
                "original_text": (
                    instance.context.target_text
                ),
                "rule_version": engine.version,
                "span_decisions": [
                    decision.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                    for decision in decisions
                ],
                "gold_span_actions": {
                    span_id: action.value
                    for span_id, action
                    in gold_by_span.items()
                },
                "all_actions_match_gold": (
                    predicted_by_span
                    == gold_by_span
                ),
            }

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    accuracy = (
        correct_spans / total_spans
        if total_spans
        else 0.0
    )

    print(f"Instances: {len(instances)}")
    print(f"Span decisions: {total_spans}")
    print(f"Correct decisions: {correct_spans}")
    print(f"Span action accuracy: {accuracy:.4f}")
    print(f"Output: {output_path}")

    return 0 if correct_spans == total_spans else 1


if __name__ == "__main__":
    raise SystemExit(main())