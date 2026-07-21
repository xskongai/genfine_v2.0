from __future__ import annotations

from abc import ABC, abstractmethod

from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)


class RewriterError(RuntimeError):
    """Base error raised by a rewriting component."""


class Rewriter(ABC):
    """
    Common interface for all GenFINE rewriting components.

    Implementations may use:
    - the human gold rewrite;
    - GPT;
    - a local language model;
    - a deterministic baseline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in pipeline metadata."""

    @abstractmethod
    def rewrite(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
    ) -> str | None:
        """
        Execute an edit plan.

        Returns:
            A rewritten string, or None when the plan abstains.
        """