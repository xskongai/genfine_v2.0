from __future__ import annotations

from collections.abc import Sequence

from genfine.domain.enums import (
    Action,
    InstanceAction,
)
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    SpanDecision,
)
from genfine.policy.action_aggregator import (
    AggregationResult,
    aggregate_span_decisions,
)


class EditPlanError(ValueError):
    """Raised when an executable edit plan cannot be built."""


class EditPlanBuilder:
    """
    Convert structured analysis and span decisions into an executable plan.

    The builder does not rewrite text. It only describes what must be
    preserved, edited or avoided.
    """

    def build(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        decisions: Sequence[SpanDecision],
    ) -> EditPlan:
        original_text = (
            instance.context.target_text
        )

        aggregation = aggregate_span_decisions(
            decisions=decisions,
            analysis=analysis,
            original_text=original_text,
            require_complete=True,
        )

        self._validate_abstention(
            aggregation=aggregation,
            decisions=decisions,
        )

        global_constraints = (
            self._build_global_constraints(
                aggregation=aggregation,
                analysis=analysis,
                decisions=decisions,
            )
        )

        return EditPlan(
            instance_id=instance.instance_id,
            original_text=original_text,
            instance_action=(
                aggregation.instance_action
            ),
            edit_scope=aggregation.edit_scope,
            span_decisions=[
                decision.model_copy(deep=True)
                for decision in decisions
            ],
            protected_facts=[
                fact.model_copy(deep=True)
                for fact in analysis.protected_facts
                if fact.must_preserve
            ],
            global_constraints=global_constraints,
        )

    def build_gold(
        self,
        *,
        instance: DatasetInstance,
    ) -> EditPlan:
        """Build an executable plan from the instance gold annotations.

        Gold annotations use ``SpanActionAnnotation`` while the executable
        pipeline uses ``SpanDecision``. This method performs only that
        structural conversion, then reuses the normal plan builder so gold
        and predicted plans are checked by the same aggregation logic.
        """
        decisions = [
            SpanDecision(
                span_id=annotation.span_id,
                action=annotation.action,
                rule_id="gold_annotation",
                reason_code=(
                    annotation.reason_code
                    or "GOLD_ANNOTATION"
                ),
                priority=0,
                confidence=1.0,
                constraints=(
                    [annotation.rationale]
                    if annotation.rationale
                    else []
                ),
            )
            for annotation
            in instance.gold_decision.span_actions
        ]

        plan = self.build(
            instance=instance,
            analysis=instance.gold_analysis,
            decisions=decisions,
        )

        gold_decision = instance.gold_decision

        if (
            plan.instance_action
            != gold_decision.instance_action
        ):
            raise EditPlanError(
                "Gold span actions aggregate to "
                f"{plan.instance_action.value}, but the dataset "
                "declares "
                f"{gold_decision.instance_action.value} for "
                f"{instance.instance_id!r}."
            )

        if plan.edit_scope != gold_decision.edit_scope:
            raise EditPlanError(
                "Gold span actions imply edit_scope="
                f"{plan.edit_scope.value}, but the dataset declares "
                f"{gold_decision.edit_scope.value} for "
                f"{instance.instance_id!r}."
            )

        return plan

    @staticmethod
    def _validate_abstention(
        *,
        aggregation: AggregationResult,
        decisions: Sequence[SpanDecision],
    ) -> None:
        if (
            aggregation.instance_action
            != InstanceAction.ABSTAIN
        ):
            return

        non_abstain_edits = [
            decision
            for decision in decisions
            if (
                decision.action
                not in {
                    Action.ABSTAIN,
                    Action.KEEP,
                    Action.KEEP_WITH_ATTRIBUTION,
                    Action.PRESERVE_AMBIGUITY,
                }
            )
        ]

        if non_abstain_edits:
            # This is not forbidden, but must be treated as a conservative
            # whole-instance abstention rather than a partial rewrite.
            return

    def _build_global_constraints(
        self,
        *,
        aggregation: AggregationResult,
        analysis: AnalysisResult,
        decisions: Sequence[SpanDecision],
    ) -> list[str]:
        constraints = [
            (
                "Do not introduce gender information that is not grounded "
                "in the source text or provided context."
            ),
            (
                "Preserve named entities, factual relations, negation, "
                "modality, attribution and coreference unless an explicit "
                "span action requires a change."
            ),
        ]

        protected_facts = [
            fact
            for fact in analysis.protected_facts
            if fact.must_preserve
        ]

        if protected_facts:
            constraints.append(
                "Preserve every protected fact included in this edit plan."
            )

        if (
            aggregation.instance_action
            == InstanceAction.KEEP
        ):
            constraints.append(
                "Return the original text unchanged."
            )

        elif (
            aggregation.instance_action
            == InstanceAction.ABSTAIN
        ):
            constraints.extend(
                [
                    (
                        "Do not produce a definitive rewritten sentence."
                    ),
                    (
                        "Preserve the original text until sufficient "
                        "context is available."
                    ),
                ]
            )

        else:
            constraints.extend(
                [
                    (
                        "Apply changes only to spans whose actions require "
                        "editing."
                    ),
                    (
                        "Preserve spans assigned KEEP, "
                        "KEEP_WITH_ATTRIBUTION or PRESERVE_AMBIGUITY."
                    ),
                ]
            )

        actions = {
            decision.action
            for decision in decisions
        }

        if Action.PRESERVE_AMBIGUITY in actions:
            constraints.append(
                "Do not infer or insert a gendered pronoun, identity or role."
            )

        if Action.KEEP_WITH_ATTRIBUTION in actions:
            constraints.append(
                "Preserve quoted content, attribution and speaker stance."
            )

        if Action.REFRAME_PROPOSITION in actions:
            constraints.append(
                "Remove the biased generalization without deleting "
                "surrounding non-biased facts."
            )

        if (
            Action.ADD_SCOPE_OR_QUALIFICATION
            in actions
        ):
            constraints.append(
                "Preserve the reported observation while adding only "
                "supported scope or uncertainty."
            )

        if Action.REPLACE_GENERIC_FORM in actions:
            constraints.append(
                "Preserve the original referent and grammatical relation "
                "while replacing the generic gender form."
            )

        return _deduplicate(
            constraints
        )


def _deduplicate(
    values: Sequence[str],
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result