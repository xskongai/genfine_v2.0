from pathlib import Path

import pytest

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.enums import InstanceAction
from genfine.generation import (
    GoldRewriter,
    RewriterError,
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


def build_rewrite_inputs(index: int):
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

    edit_plan = EditPlanBuilder().build(
        instance=instance,
        analysis=analysis,
        decisions=decisions,
    )

    return instance, analysis, edit_plan


def test_gold_rewriter_name() -> None:
    rewriter = GoldRewriter()

    assert rewriter.name == "gold"


def test_gold_rewriter_returns_keep_output() -> None:
    instance, analysis, edit_plan = (
        build_rewrite_inputs(0)
    )

    output = GoldRewriter().rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
    )

    assert (
        edit_plan.instance_action
        == InstanceAction.KEEP
    )
    assert output == instance.context.target_text
    assert output == instance.gold_output


def test_gold_rewriter_returns_edited_output() -> None:
    instance, analysis, edit_plan = (
        build_rewrite_inputs(2)
    )

    output = GoldRewriter().rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
    )

    assert (
        edit_plan.instance_action
        == InstanceAction.EDIT
    )

    assert output == "每位学生都应提交自己的作业"
    assert output == instance.gold_output


def test_gold_rewriter_rejects_wrong_instance_id() -> None:
    instance, analysis, edit_plan = (
        build_rewrite_inputs(0)
    )

    invalid_plan = edit_plan.model_copy(
        update={
            "instance_id": "wrong_instance"
        }
    )

    with pytest.raises(
        RewriterError,
        match="instance_id does not match",
    ):
        GoldRewriter().rewrite(
            instance=instance,
            analysis=analysis,
            edit_plan=invalid_plan,
        )


def test_strict_gold_rewriter_rejects_wrong_action() -> None:
    instance, analysis, edit_plan = (
        build_rewrite_inputs(0)
    )

    invalid_plan = edit_plan.model_copy(
        update={
            "instance_action": (
                InstanceAction.EDIT
            )
        }
    )

    with pytest.raises(
        RewriterError,
        match="instance_action does not match gold",
    ):
        GoldRewriter(
            strict_plan_match=True
        ).rewrite(
            instance=instance,
            analysis=analysis,
            edit_plan=invalid_plan,
        )


def test_gold_rewriter_does_not_modify_inputs() -> None:
    instance, analysis, edit_plan = (
        build_rewrite_inputs(2)
    )

    instance_before = instance.model_dump()
    analysis_before = analysis.model_dump()
    plan_before = edit_plan.model_dump()

    GoldRewriter().rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
    )

    assert instance.model_dump() == instance_before
    assert analysis.model_dump() == analysis_before
    assert edit_plan.model_dump() == plan_before