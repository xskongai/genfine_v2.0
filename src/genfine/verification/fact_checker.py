from __future__ import annotations

from genfine.domain.enums import Action
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    ProtectedFact,
    VerificationIssue,
)
from genfine.verification.base import OutputChecker


KEEP_LIKE_ACTIONS = {
    Action.KEEP,
    Action.KEEP_WITH_ATTRIBUTION,
    Action.PRESERVE_AMBIGUITY,
}


class ProtectedFactChecker(OutputChecker):
    """
    Check machine-verifiable protected facts.

    This first version uses explicit required and forbidden phrases.
    Facts without executable phrases receive a warning instead of being
    silently treated as verified.
    """

    @property
    def name(self) -> str:
        return "protected_fact"

    def check(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> list[VerificationIssue]:
        del instance

        if output_text is None:
            return []

        issues: list[VerificationIssue] = []

        span_by_id = {
            span.span_id: span
            for span in analysis.spans
        }

        action_by_span_id = {
            decision.span_id: decision.action
            for decision in edit_plan.span_decisions
        }

        for fact in edit_plan.protected_facts:
            if not fact.must_preserve:
                continue

            required_phrases = list(
                fact.required_output_phrases
            )

            # Fallback: when a protected fact points to a KEEP-like span,
            # the original span itself becomes a required phrase.
            for span_id in fact.source_span_ids:
                span = span_by_id.get(span_id)
                action = action_by_span_id.get(span_id)

                if (
                    span is not None
                    and action in KEEP_LIKE_ACTIONS
                    and span.text not in required_phrases
                ):
                    required_phrases.append(
                        span.text
                    )

            for phrase in required_phrases:
                if phrase not in output_text:
                    issues.append(
                        self._error(
                            code="PROTECTED_PHRASE_MISSING",
                            message=(
                                f"Protected fact {fact.fact_id!r} "
                                f"requires phrase {phrase!r}, "
                                "but it is missing from output."
                            ),
                            span_text=phrase,
                        )
                    )

            for phrase in fact.forbidden_output_phrases:
                if phrase in output_text:
                    issues.append(
                        self._error(
                            code="FORBIDDEN_FACT_PHRASE_PRESENT",
                            message=(
                                f"Protected fact {fact.fact_id!r} "
                                f"forbids phrase {phrase!r}, "
                                "but it appears in output."
                            ),
                            span_text=phrase,
                        )
                    )

            if (
                not required_phrases
                and not fact.forbidden_output_phrases
            ):
                issues.append(
                    self._warning(
                        code="PROTECTED_FACT_NOT_MACHINE_CHECKABLE",
                        message=(
                            f"Protected fact {fact.fact_id!r} "
                            "has no executable verification phrases."
                        ),
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

    def _warning(
        self,
        *,
        code: str,
        message: str,
    ) -> VerificationIssue:
        return VerificationIssue(
            checker=self.name,
            code=code,
            message=message,
            severity="WARNING",
        )