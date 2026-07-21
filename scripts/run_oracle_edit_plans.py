from __future__ import annotations

import argparse
import json
from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.pipeline import EditPlanBuilder
from genfine.policy import DecisionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GenFINE edit plans "
            "using Oracle analyses."
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
        default=(
            "runs/oracle_edit_plans_v0.1.jsonl"
        ),
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

    builder = EditPlanBuilder()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    correct_actions = 0
    correct_scopes = 0

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

            edit_plan = builder.build(
                instance=instance,
                analysis=analysis,
                decisions=decisions,
            )

            action_matches = (
                edit_plan.instance_action
                == instance.gold_decision.instance_action
            )

            scope_matches = (
                edit_plan.edit_scope
                == instance.gold_decision.edit_scope
            )

            correct_actions += int(
                action_matches
            )

            correct_scopes += int(
                scope_matches
            )

            payload = {
                "instance_id": instance.instance_id,
                "original_text": (
                    instance.context.target_text
                ),
                "rule_version": engine.version,
                "edit_plan": edit_plan.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
                "gold_instance_action": (
                    instance
                    .gold_decision
                    .instance_action
                    .value
                ),
                "gold_edit_scope": (
                    instance
                    .gold_decision
                    .edit_scope
                    .value
                ),
                "instance_action_matches": (
                    action_matches
                ),
                "edit_scope_matches": (
                    scope_matches
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

    count = len(instances)

    print(f"Instances: {count}")
    print(
        "Correct instance actions: "
        f"{correct_actions}/{count}"
    )
    print(
        "Correct edit scopes: "
        f"{correct_scopes}/{count}"
    )
    print(f"Output: {output_path}")

    success = (
        correct_actions == count
        and correct_scopes == count
    )

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())