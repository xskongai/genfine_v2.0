from __future__ import annotations

from abc import ABC, abstractmethod

from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    VerificationIssue,
)


class OutputChecker(ABC):
    """Common interface for independent output validators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable checker identifier."""

    @abstractmethod
    def check(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> list[VerificationIssue]:
        """Return all issues found by this checker."""