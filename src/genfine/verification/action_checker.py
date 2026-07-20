from __future__ import annotations

from collections import defaultdict

from genfine.domain.enums import (
    Action,
    InstanceAction,
)
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    VerificationIssue,
)
from genfine.verification.base import OutputChecker


PRESERVE_ACTIONS = {
    Action.KEEP,
    Action.KEEP_WITH_ATTRIBUTION,
    Action.PRESERVE_AMBIGUITY,
}


REMOVE_OR_REPLACE_ACTIONS = {
    Action.NEUTRALIZE,
    Action.UNMARK,
    Action.GENERALIZE,
    Action.SPECIFY_OR_RESTORE,
    Action.CORRECT_IDENTITY,
    Action.REPLACE_GENERIC_FORM,
    Action.STANDARDIZE_REFERENCE,
    Action.REFRAME_PROPOSITION,
}


class ActionComplianceChecker(OutputChecker):
    """Check whether span-level editing instructions were executed."""

    @property
    def name(self) -> str:
        return "action_compliance"

    def check(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> list[VerificationIssue]:
        if output_text is None:
            return []

        issues: list[VerificationIssue] = []

        original = instance.context.target_text

        if (
            edit_plan.instance_action
            in {
                InstanceAction.EDIT,
                InstanceAction.SPAN_LEVEL_EDIT,
            }
            and output_text == original
        ):
            issues.append(
                self._error(
                    code="EDIT_NOT_APPLIED",
                    message=(
                        "The edit plan requires a change, "
                        "but the output is identical to the input."
                    ),
                )
            )

        span_by_id = {
            span.span_id: span
            for span in analysis.spans
        }

        decisions_by_text: dict[
            str,
            list[Action],
        ] = defaultdict(list)

        for decision in edit_plan.span_decisions:
            span = span_by_id.get(
                decision.span_id
            )

            if span is None:
                issues.append(
                    self._error(
                        code="UNKNOWN_DECISION_SPAN",
                        message=(
                            f"Decision refers to unknown span "
                            f"{decision.span_id!r}."
                        ),
                    )
                )
                continue

            decisions_by_text[
                span.text
            ].append(decision.action)

        for span_text, actions in decisions_by_text.items():
            original_count = original.count(
                span_text
            )

            output_count = output_text.count(
                span_text
            )

            preserve_count = sum(
                action in PRESERVE_ACTIONS
                for action in actions
            )

            replacement_count = sum(
                action in REMOVE_OR_REPLACE_ACTIONS
                for action in actions
            )

            if output_count < preserve_count:
                issues.append(
                    self._error(
                        code="PRESERVED_SPAN_REMOVED",
                        message=(
                            f"Span {span_text!r} was assigned a "
                            "preserve action but is missing from output."
                        ),
                        span_text=span_text,
                    )
                )

            maximum_allowed = max(
                0,
                original_count - replacement_count,
            )

            if (
                replacement_count > 0
                and output_count > maximum_allowed
            ):
                issues.append(
                    self._error(
                        code="EDIT_SPAN_REMAINS",
                        message=(
                            f"Span {span_text!r} should have been "
                            "removed or replaced but remains in output."
                        ),
                        span_text=span_text,
                    )
                )

        return issues

    def _error(
        self,
        *,
        code: str,
        message: str,
        span_text: str | None = None,
    ) -> VerificationIssue:
        return VerificationIssue(
            checker=self.name,
            code=code,
            message=message,
            severity="ERROR",
            span_text=span_text,
        )