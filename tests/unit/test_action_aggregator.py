from pathlib import Path

import pytest

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.enums import (
    Action,
    EditScope,
    InstanceAction,
)
from genfine.domain.models import SpanDecision
from genfine.policy import DecisionEngine
from genfine.policy.action_aggregator import (
    ActionAggregationError,
    aggregate_span_decisions,
    infer_instance_action,
)


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


def build_decisions(index: int):
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

    return instance, analysis, decisions


def test_keep_like_actions_aggregate_to_keep() -> None:
    instance, analysis, decisions = (
        build_decisions(0)
    )

    result = aggregate_span_decisions(
        decisions=decisions,
        analysis=analysis,
        original_text=(
            instance.context.target_text
        ),
    )

    assert (
        result.instance_action
        == InstanceAction.KEEP
    )
    assert result.edit_scope == EditScope.NONE
    assert result.edited_span_ids == ()


def test_single_edit_aggregates_to_edit() -> None:
    instance, analysis, decisions = (
        build_decisions(2)
    )

    result = aggregate_span_decisions(
        decisions=decisions,
        analysis=analysis,
        original_text=(
            instance.context.target_text
        ),
    )

    assert (
        result.instance_action
        == InstanceAction.EDIT
    )
    assert result.edit_scope == EditScope.SPAN
    assert result.edited_span_ids == ("s1",)


def test_keep_and_edit_become_span_level_edit() -> None:
    instance, analysis, decisions = (
        build_decisions(0)
    )

    modified_decisions = [
        decisions[0],
        decisions[1].model_copy(
            update={
                "action": (
                    Action.REFRAME_PROPOSITION
                )
            }
        ),
    ]

    result = aggregate_span_decisions(
        decisions=modified_decisions,
        analysis=analysis,
        original_text=(
            instance.context.target_text
        ),
    )

    assert (
        result.instance_action
        == InstanceAction.SPAN_LEVEL_EDIT
    )
    assert result.edit_scope == EditScope.SPAN


def test_multiple_edits_use_multi_span_scope() -> None:
    instance, analysis, decisions = (
        build_decisions(0)
    )

    modified_decisions = [
        decision.model_copy(
            update={
                "action": Action.UNMARK
            }
        )
        for decision in decisions
    ]

    result = aggregate_span_decisions(
        decisions=modified_decisions,
        analysis=analysis,
        original_text=(
            instance.context.target_text
        ),
    )

    assert (
        result.instance_action
        == InstanceAction.SPAN_LEVEL_EDIT
    )
    assert (
        result.edit_scope
        == EditScope.MULTI_SPAN
    )


def test_abstain_has_highest_precedence() -> None:
    instance, analysis, decisions = (
        build_decisions(0)
    )

    modified_decisions = [
        decisions[0].model_copy(
            update={"action": Action.ABSTAIN}
        ),
        decisions[1],
    ]

    result = aggregate_span_decisions(
        decisions=modified_decisions,
        analysis=analysis,
        original_text=(
            instance.context.target_text
        ),
    )

    assert (
        result.instance_action
        == InstanceAction.ABSTAIN
    )
    assert result.edit_scope == EditScope.NONE
    assert result.abstained_span_ids == ("s1",)


def test_missing_decision_is_rejected() -> None:
    instance, analysis, decisions = (
        build_decisions(0)
    )

    with pytest.raises(
        ActionAggregationError,
        match="have no decision",
    ):
        aggregate_span_decisions(
            decisions=decisions[:1],
            analysis=analysis,
            original_text=(
                instance.context.target_text
            ),
        )


def test_duplicate_decision_is_rejected() -> None:
    instance, analysis, decisions = (
        build_decisions(0)
    )

    duplicated = [
        decisions[0],
        decisions[0].model_copy(deep=True),
        decisions[1],
    ]

    with pytest.raises(
        ActionAggregationError,
        match="duplicate span decisions",
    ):
        aggregate_span_decisions(
            decisions=duplicated,
            analysis=analysis,
            original_text=(
                instance.context.target_text
            ),
        )


def test_infer_multiple_edits_is_span_level() -> None:
    result = infer_instance_action(
        [
            Action.UNMARK,
            Action.UNMARK,
        ]
    )

    assert (
        result
        == InstanceAction.SPAN_LEVEL_EDIT
    )


def test_unknown_span_decision_is_rejected() -> None:
    instance, analysis, decisions = (
        build_decisions(0)
    )

    unknown = SpanDecision(
        span_id="missing_span",
        action=Action.KEEP,
        rule_id="test_rule",
        reason_code="TEST",
    )

    with pytest.raises(
        ActionAggregationError,
        match="unknown analysis spans",
    ):
        aggregate_span_decisions(
            decisions=[
                *decisions,
                unknown,
            ],
            analysis=analysis,
            original_text=(
                instance.context.target_text
            ),
        )