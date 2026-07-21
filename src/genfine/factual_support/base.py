from __future__ import annotations

from abc import ABC, abstractmethod

from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)
from genfine.factual_support.models import (
    FactualSupportResult,
)


class FactualSupportJudgeError(RuntimeError):
    """
    Raised when factual-support evaluation cannot be completed.
    """


class FactualSupportJudge(ABC):
    """
    Independent semantic evaluator for unsupported factual insertion.

    The judge does not rewrite text and does not modify the EditPlan.
    It only evaluates an already generated output.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable judge identifier."""

    @property
    @abstractmethod
    def prompt_version(self) -> str:
        """Version of the factual-support judgment policy."""

    @abstractmethod
    def evaluate(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> FactualSupportResult:
        """
        Evaluate whether output claims are source-supported,
        action-licensed, non-factual, unsupported or uncertain.
        """