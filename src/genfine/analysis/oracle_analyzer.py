from __future__ import annotations

from genfine.analysis.base import Analyzer
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
)


class OracleAnalyzer(Analyzer):
    """
    Return human gold analysis as the predicted analysis.

    This analyzer is used to isolate downstream policy, generation and
    verification errors from analysis-model errors.
    """

    @property
    def name(self) -> str:
        return "oracle"

    def analyze(
        self,
        instance: DatasetInstance,
    ) -> AnalysisResult:
        """
        Return a deep copy of the instance's gold analysis.

        A deep copy is required because downstream components may enrich or
        modify the predicted analysis. Those operations must never mutate the
        original dataset annotation.
        """

        return instance.gold_analysis.model_copy(
            deep=True
        )