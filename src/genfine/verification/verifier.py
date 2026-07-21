from __future__ import annotations

from collections.abc import Sequence

from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    VerificationIssue,
    VerificationResult,
)
from genfine.verification.action_checker import (
    ActionComplianceChecker,
)
from genfine.verification.base import OutputChecker
from genfine.verification.fact_checker import (
    ProtectedFactChecker,
)
from genfine.verification.gender_insertion_checker import (
    GenderInsertionChecker,
)
from genfine.verification.keep_checker import (
    KeepIntegrityChecker,
)


class OutputVerifier:
    """Run independent deterministic checkers over one output."""

    def __init__(
        self,
        checkers: Sequence[OutputChecker],
    ) -> None:
        self.checkers = list(checkers)

    @classmethod
    def default(cls) -> "OutputVerifier":
        return cls(
            checkers=[
                KeepIntegrityChecker(),
                ActionComplianceChecker(),
                ProtectedFactChecker(),
                GenderInsertionChecker(),
            ]
        )

    def verify(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> VerificationResult:
        issues: list[VerificationIssue] = []

        for checker in self.checkers:
            checker_issues = checker.check(
                instance=instance,
                analysis=analysis,
                edit_plan=edit_plan,
                output_text=output_text,
            )

            issues.extend(checker_issues)

        error_checkers = {
            issue.checker
            for issue in issues
            if issue.severity == "ERROR"
        }

        protected_facts_preserved = (
            "protected_fact"
            not in error_checkers
        )

        action_compliant = not bool(
            error_checkers.intersection(
                {
                    "keep_integrity",
                    "action_compliance",
                }
            )
        )

        unsupported_gender_inserted = (
            "gender_insertion"
            in error_checkers
        )

        passed = (
            protected_facts_preserved
            and action_compliant
            and not unsupported_gender_inserted
            and not any(
                issue.severity == "ERROR"
                for issue in issues
            )
        )

        return VerificationResult(
            passed=passed,
            protected_facts_preserved=(
                protected_facts_preserved
            ),
            action_compliant=action_compliant,
            unsupported_gender_inserted=(
                unsupported_gender_inserted
            ),
            issues=issues,
        )