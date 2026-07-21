from __future__ import annotations

from genfine.domain.enums import InstanceAction
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)
from genfine.generation.base import (
    Rewriter,
    RewriterError,
)


class GoldRewriter(Rewriter):
    """
    Return the human-authored gold output.

    This component is used to test pipeline wiring independently of
    generation-model errors.
    """

    def __init__(
        self,
        *,
        strict_plan_match: bool = True,
    ) -> None:
        self.strict_plan_match = strict_plan_match

    @property
    def name(self) -> str:
        return "gold"

    def rewrite(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
    ) -> str | None:
        del analysis  # Reserved for compatibility with other rewriters.

        self._validate_instance_identity(
            instance=instance,
            edit_plan=edit_plan,
        )

        if self.strict_plan_match:
            self._validate_plan_matches_gold(
                instance=instance,
                edit_plan=edit_plan,
            )

        if (
            edit_plan.instance_action
            == InstanceAction.ABSTAIN
        ):
            if instance.gold_output is not None:
                raise RewriterError(
                    "ABSTAIN edit plan requires gold_output=None"
                )

            return None

        if instance.gold_output is None:
            raise RewriterError(
                f"Instance {instance.instance_id!r} has no gold_output"
            )

        if (
            edit_plan.instance_action
            == InstanceAction.KEEP
            and instance.gold_output
            != instance.context.target_text
        ):
            raise RewriterError(
                "KEEP plan requires gold_output to equal target_text"
            )

        return instance.gold_output

    @staticmethod
    def _validate_instance_identity(
        *,
        instance: DatasetInstance,
        edit_plan: EditPlan,
    ) -> None:
        if edit_plan.instance_id != instance.instance_id:
            raise RewriterError(
                "edit plan instance_id does not match dataset instance: "
                f"{edit_plan.instance_id!r} != "
                f"{instance.instance_id!r}"
            )

        if (
            edit_plan.original_text
            != instance.context.target_text
        ):
            raise RewriterError(
                "edit plan original_text does not match target_text"
            )

    @staticmethod
    def _validate_plan_matches_gold(
        *,
        instance: DatasetInstance,
        edit_plan: EditPlan,
    ) -> None:
        gold_decision = instance.gold_decision

        if (
            edit_plan.instance_action
            != gold_decision.instance_action
        ):
            raise RewriterError(
                "edit plan instance_action does not match gold: "
                f"{edit_plan.instance_action.value} != "
                f"{gold_decision.instance_action.value}"
            )

        if (
            edit_plan.edit_scope
            != gold_decision.edit_scope
        ):
            raise RewriterError(
                "edit plan edit_scope does not match gold: "
                f"{edit_plan.edit_scope.value} != "
                f"{gold_decision.edit_scope.value}"
            )

        gold_actions = {
            item.span_id: item.action
            for item in gold_decision.span_actions
        }

        predicted_actions = {
            item.span_id: item.action
            for item in edit_plan.span_decisions
        }

        if predicted_actions != gold_actions:
            raise RewriterError(
                "edit plan span actions do not match gold: "
                f"predicted={predicted_actions}, "
                f"gold={gold_actions}"
            )