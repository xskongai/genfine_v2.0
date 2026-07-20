from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
)


class AnalyzerError(RuntimeError):
    """Base error raised by an analysis component."""


class Analyzer(ABC):
    """
    Common interface for all GenFINE analysis components.

    Implementations may obtain the analysis from:
    - human gold annotations;
    - GPT structured output;
    - a local classifier;
    - a heuristic baseline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in run metadata."""

    @abstractmethod
    def analyze(
        self,
        instance: DatasetInstance,
    ) -> AnalysisResult:
        """Produce a structured analysis for one dataset instance."""

    def analyze_many(
        self,
        instances: Iterable[DatasetInstance],
    ) -> list[AnalysisResult]:
        """
        Analyze multiple instances while preserving their input order.

        Implementations can override this method later to support batching.
        """

        return [
            self.analyze(instance)
            for instance in instances
        ]