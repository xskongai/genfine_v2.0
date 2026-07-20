from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from genfine.domain.enums import (
    Action,
    EditScope,
    InstanceAction,
)
from genfine.domain.models import (
    AnalysisResult,
    SpanDecision,
)


class ActionAggregationError(ValueError):
    """Raised when span decisions cannot be aggregated consistently."""


KEEP_LIKE_ACTIONS: frozenset[Action] = frozenset(
    {
        Action.KEEP,
        Action.KEEP_WITH_ATTRIBUTION,
        Action.PRESERVE_AMBIGUITY,
    }
)


@dataclass(frozen=True)
class AggregationResult:
    """Instance-level result aggregated from span-level decisions."""

    instance_action: InstanceAction
    edit_scope: EditScope

    preserved_span_ids: tuple[str, ...]
    edited_span_ids: tuple[str, ...]
    abstained_span_ids: tuple[str, ...]


def infer_instance_action(
    actions: Sequence[Action],
) -> InstanceAction:
    """
    Infer an instance-level action from span-level actions.

    Precedence:
    1. Any ABSTAIN -> ABSTAIN
    2. All actions are keep-like -> KEEP
    3. Exactly one edit and no keep-like action -> EDIT
    4. Mixed or multiple edits -> SPAN_LEVEL_EDIT
    """

    if not actions:
        return InstanceAction.KEEP

    if Action.ABSTAIN in actions:
        return InstanceAction.ABSTAIN

    keep_like_count = sum(
        action in KEEP_LIKE_ACTIONS
        for action in actions
    )

    edit_count = len(actions) - keep_like_count

    if edit_count == 0:
        return InstanceAction.KEEP

    if edit_count == 1 and keep_like_count == 0:
        return InstanceAction.EDIT

    return InstanceAction.SPAN_LEVEL_EDIT


def aggregate_span_decisions(
    *,
    decisions: Sequence[SpanDecision],
    analysis: AnalysisResult,
    original_text: str,
    require_complete: bool = True,
) -> AggregationResult:
    """
    Aggregate span decisions and infer the required editing scope.

    When require_complete=True, every analysis span must have exactly one
    decision.
    """

    decision_span_ids = [
        decision.span_id
        for decision in decisions
    ]

    duplicate_ids = _find_duplicates(
        decision_span_ids
    )

    if duplicate_ids:
        raise ActionAggregationError(
            "duplicate span decisions: "
            f"{sorted(duplicate_ids)}"
        )

    analysis_span_ids = {
        span.span_id
        for span in analysis.spans
    }

    decision_span_id_set = set(
        decision_span_ids
    )

    unknown_ids = (
        decision_span_id_set - analysis_span_ids
    )

    if unknown_ids:
        raise ActionAggregationError(
            "decisions refer to unknown analysis spans: "
            f"{sorted(unknown_ids)}"
        )

    if require_complete:
        missing_ids = (
            analysis_span_ids
            - decision_span_id_set
        )

        if missing_ids:
            raise ActionAggregationError(
                "analysis spans have no decision: "
                f"{sorted(missing_ids)}"
            )

    preserved_span_ids = tuple(
        decision.span_id
        for decision in decisions
        if decision.action in KEEP_LIKE_ACTIONS
    )

    abstained_span_ids = tuple(
        decision.span_id
        for decision in decisions
        if decision.action == Action.ABSTAIN
    )

    edited_span_ids = tuple(
        decision.span_id
        for decision in decisions
        if (
            decision.action
            not in KEEP_LIKE_ACTIONS
            and decision.action != Action.ABSTAIN
        )
    )

    instance_action = infer_instance_action(
        [
            decision.action
            for decision in decisions
        ]
    )

    if instance_action in {
        InstanceAction.KEEP,
        InstanceAction.ABSTAIN,
    }:
        edit_scope = EditScope.NONE
    else:
        edit_scope = infer_edit_scope(
            edited_span_ids=edited_span_ids,
            analysis=analysis,
            original_text=original_text,
        )

    return AggregationResult(
        instance_action=instance_action,
        edit_scope=edit_scope,
        preserved_span_ids=preserved_span_ids,
        edited_span_ids=edited_span_ids,
        abstained_span_ids=abstained_span_ids,
    )


def infer_edit_scope(
    *,
    edited_span_ids: Sequence[str],
    analysis: AnalysisResult,
    original_text: str,
) -> EditScope:
    """
    Infer the smallest practical edit scope.

    - no edited spans -> NONE
    - one partial span -> SPAN
    - one span covering the entire input -> SENTENCE
    - multiple spans -> MULTI_SPAN
    """

    if not edited_span_ids:
        return EditScope.NONE

    if len(edited_span_ids) > 1:
        return EditScope.MULTI_SPAN

    span_by_id = {
        span.span_id: span
        for span in analysis.spans
    }

    span_id = edited_span_ids[0]

    if span_id not in span_by_id:
        raise ActionAggregationError(
            f"unknown edited span: {span_id!r}"
        )

    span = span_by_id[span_id]

    if (
        span.start == 0
        and span.end == len(original_text)
    ):
        return EditScope.SENTENCE

    return EditScope.SPAN


def _find_duplicates(
    values: Sequence[str],
) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    return duplicates