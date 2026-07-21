from __future__ import annotations

from genfine.domain.enums import InstanceAction
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    VerificationIssue,
)
from genfine.verification.base import OutputChecker


class KeepIntegrityChecker(OutputChecker):
    """Check instance-level KEEP, EDIT and ABSTAIN behavior."""

    @property
    def name(self) -> str:
        return "keep_integrity"

    def check(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> list[VerificationIssue]:
        del analysis

        issues: list[VerificationIssue] = []

        original = instance.context.target_text
        action = edit_plan.instance_action

        if action == InstanceAction.KEEP:
            if output_text is None:
                issues.append(
                    self._error(
                        code="KEEP_OUTPUT_MISSING",
                        message=(
                            "KEEP plan produced no output."
                        ),
                    )
                )

            elif output_text != original:
                issues.append(
                    self._error(
                        code="KEEP_TEXT_CHANGED",
                        message=(
                            "KEEP plan changed the original text."
                        ),
                    )
                )

        elif action == InstanceAction.ABSTAIN:
            if output_text is not None:
                issues.append(
                    self._error(
                        code="ABSTAIN_PRODUCED_OUTPUT",
                        message=(
                            "ABSTAIN plan produced a definitive output."
                        ),
                    )
                )

        elif action in {
            InstanceAction.EDIT,
            InstanceAction.SPAN_LEVEL_EDIT,
        }:
            if output_text is None:
                issues.append(
                    self._error(
                        code="EDIT_OUTPUT_MISSING",
                        message=(
                            f"{action.value} plan produced no output."
                        ),
                    )
                )

        return issues

    def _error(
        self,
        *,
        code: str,
        message: str,
    ) -> VerificationIssue:
        return VerificationIssue(
            checker=self.name,
            code=code,
            message=message,
            severity="ERROR",
        )