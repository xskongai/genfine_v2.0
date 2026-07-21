from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.enums import (
    Action,
    EditScope,
    InstanceAction,
)
from genfine.pipeline import EditPlanBuilder
from genfine.policy import DecisionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.1.jsonl"
)

RULE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "decision_rules.yaml"
)


def build_plan(index: int):
    instance = load_dataset(SEED_PATH)[index]

    analysis = OracleAnalyzer().analyze(
        instance
    )

    decisions = (
        DecisionEngine
        .from_yaml(RULE_PATH)
        .decide_analysis(
            analysis=analysis,
            task_mode=instance.task_mode,
        )
    )

    plan = EditPlanBuilder().build(
        instance=instance,
        analysis=analysis,
        decisions=decisions,
    )

    return instance, analysis, decisions, plan


def test_keep_plan() -> None:
    instance, _, _, plan = build_plan(0)

    assert plan.instance_id == instance.instance_id
    assert (
        plan.instance_action
        == InstanceAction.KEEP
    )
    assert plan.edit_scope == EditScope.NONE
    assert (
        "Return the original text unchanged."
        in plan.global_constraints
    )


def test_generic_male_edit_plan() -> None:
    _, _, _, plan = build_plan(2)

    assert (
        plan.instance_action
        == InstanceAction.EDIT
    )
    assert plan.edit_scope == EditScope.SPAN
    assert (
        plan.span_decisions[0].action
        == Action.REPLACE_GENERIC_FORM
    )

    assert any(
        "generic gender form"
        in constraint
        for constraint
        in plan.global_constraints
    )


def test_preserve_ambiguity_constraint() -> None:
    _, _, _, plan = build_plan(4)

    assert plan.instance_action == InstanceAction.KEEP

    assert any(
        "Do not infer or insert"
        in constraint
        for constraint
        in plan.global_constraints
    )


def test_protected_facts_are_included() -> None:
    _, analysis, _, plan = build_plan(0)

    expected_fact_ids = {
        fact.fact_id
        for fact in analysis.protected_facts
        if fact.must_preserve
    }

    actual_fact_ids = {
        fact.fact_id
        for fact in plan.protected_facts
    }

    assert actual_fact_ids == expected_fact_ids


def test_plan_contains_deep_copies() -> None:
    _, analysis, decisions, plan = build_plan(0)

    assert (
        plan.span_decisions[0]
        is not decisions[0]
    )

    assert (
        plan.protected_facts[0]
        is not analysis.protected_facts[0]
    )


def test_all_seed_plans_match_gold_actions() -> None:
    instances = load_dataset(SEED_PATH)
    analyzer = OracleAnalyzer()
    engine = DecisionEngine.from_yaml(
        RULE_PATH
    )
    builder = EditPlanBuilder()

    for instance in instances:
        analysis = analyzer.analyze(
            instance
        )

        decisions = engine.decide_analysis(
            analysis=analysis,
            task_mode=instance.task_mode,
        )

        plan = builder.build(
            instance=instance,
            analysis=analysis,
            decisions=decisions,
        )

        assert (
            plan.instance_action
            == instance.gold_decision.instance_action
        )

        assert (
            plan.edit_scope
            == instance.gold_decision.edit_scope
        )