from __future__ import annotations

from collections.abc import Iterable

from genfine.analysis.base import Analyzer
from genfine.domain.models import (
    DatasetInstance,
    RunRecord,
)
from genfine.generation.base import Rewriter
from genfine.pipeline.edit_plan import (
    EditPlanBuilder,
)
from genfine.policy.decision_engine import (
    DecisionEngine,
)


class PipelineError(RuntimeError):
    """Raised when the GenFINE pipeline cannot complete an instance."""


class PipelineRunner:
    """
    Orchestrate the complete GenFINE processing pipeline.

    The runner contains no language-analysis or rewriting logic itself.
    It only coordinates independently testable components.
    """

    def __init__(
        self,
        *,
        analyzer: Analyzer,
        decision_engine: DecisionEngine,
        edit_plan_builder: EditPlanBuilder,
        rewriter: Rewriter,
    ) -> None:
        self.analyzer = analyzer
        self.decision_engine = decision_engine
        self.edit_plan_builder = edit_plan_builder
        self.rewriter = rewriter

    def run(
        self,
        instance: DatasetInstance,
    ) -> RunRecord:
        """Run one dataset instance through the full pipeline."""

        analysis = self.analyzer.analyze(
            instance
        )

        span_decisions = (
            self.decision_engine.decide_analysis(
                analysis=analysis,
                task_mode=instance.task_mode,
            )
        )

        edit_plan = self.edit_plan_builder.build(
            instance=instance,
            analysis=analysis,
            decisions=span_decisions,
        )

        output_text = self.rewriter.rewrite(
            instance=instance,
            analysis=analysis,
            edit_plan=edit_plan,
        )

        return RunRecord(
            instance_id=instance.instance_id,
            original_text=(
                instance.context.target_text
            ),
            predicted_analysis=analysis,
            edit_plan=edit_plan,
            output_text=output_text,
            metadata={
                "analyzer": self.analyzer.name,
                "rewriter": self.rewriter.name,
                "decision_rule_version": (
                    self.decision_engine.version
                ),
                "task_mode": instance.task_mode.value,
            },
        )

    def run_many(
        self,
        instances: Iterable[DatasetInstance],
        *,
        continue_on_error: bool = False,
    ) -> list[RunRecord]:
        """
        Run multiple instances while preserving input order.

        When continue_on_error=True, failed instances are returned as
        RunRecords containing error messages instead of stopping the run.
        """

        records: list[RunRecord] = []

        for instance in instances:
            try:
                record = self.run(instance)
            except Exception as exc:
                if not continue_on_error:
                    raise

                record = RunRecord(
                    instance_id=instance.instance_id,
                    original_text=(
                        instance.context.target_text
                    ),
                    errors=[
                        f"{type(exc).__name__}: {exc}"
                    ],
                    metadata={
                        "analyzer": self.analyzer.name,
                        "rewriter": self.rewriter.name,
                        "decision_rule_version": (
                            self.decision_engine.version
                        ),
                        "task_mode": (
                            instance.task_mode.value
                        ),
                    },
                )

            records.append(record)

        return records