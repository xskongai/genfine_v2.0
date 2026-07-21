from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from genfine.domain.models import (
    DatasetInstance,
    RunRecord,
    VerificationResult,
)
from genfine.evaluation.analysis_evaluator import match_spans
from genfine.evaluation.models import MetricValue
from genfine.pipeline import EditPlanBuilder
from genfine.verification import OutputVerifier


class DualVerificationError(ValueError):
    """Raised when dual verification inputs are inconsistent."""


class DualVerificationCategory(str, Enum):
    """Primary diagnosis for one evaluated output."""

    PASS = "PASS"
    UPSTREAM_PLAN_ERROR = "UPSTREAM_PLAN_ERROR"
    REWRITE_EXECUTION_ERROR = "REWRITE_EXECUTION_ERROR"
    GOLD_COMPLIANT_DESPITE_PLAN_MISMATCH = (
        "GOLD_COMPLIANT_DESPITE_PLAN_MISMATCH"
    )
    MISSING_PREDICTED_PLAN = "MISSING_PREDICTED_PLAN"


class DualVerificationRecord(BaseModel):
    """Predicted-plan and gold-plan checks for one output."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    predicted_plan_matches_gold: bool
    predicted_plan_verification: VerificationResult | None
    gold_plan_verification: VerificationResult
    category: DualVerificationCategory


class DualVerificationSummary(BaseModel):
    """Aggregate and per-instance dual-verification results."""

    model_config = ConfigDict(extra="forbid")

    dataset_instance_count: int = Field(ge=0)
    evaluated_instance_count: int = Field(ge=0)

    predicted_plan_verification_coverage: MetricValue
    predicted_plan_verification_pass_rate: MetricValue
    gold_plan_verification_pass_rate: MetricValue

    upstream_plan_error_rate: MetricValue
    rewrite_execution_error_rate: MetricValue
    gold_compliant_despite_plan_mismatch_rate: MetricValue

    records: list[DualVerificationRecord] = Field(
        default_factory=list
    )


class DualVerificationEvaluator:
    """Verify each output against predicted and gold edit plans."""

    def __init__(
        self,
        *,
        verifier: OutputVerifier | None = None,
        edit_plan_builder: EditPlanBuilder | None = None,
        minimum_overlap: float = 0.5,
    ) -> None:
        if not 0.0 < minimum_overlap <= 1.0:
            raise ValueError(
                "minimum_overlap must be in (0, 1]"
            )

        self.verifier = verifier or OutputVerifier.default()
        self.edit_plan_builder = (
            edit_plan_builder or EditPlanBuilder()
        )
        self.minimum_overlap = minimum_overlap

    def evaluate(
        self,
        *,
        instances: Sequence[DatasetInstance],
        run_records: Sequence[RunRecord],
        require_complete: bool = False,
    ) -> DualVerificationSummary:
        dataset_by_id = self._index_instances(instances)
        record_by_id = self._index_records(run_records)

        unknown_ids = set(record_by_id) - set(dataset_by_id)

        if unknown_ids:
            raise DualVerificationError(
                "Run records refer to unknown dataset instances: "
                f"{sorted(unknown_ids)}"
            )

        missing_ids = set(dataset_by_id) - set(record_by_id)

        if require_complete and missing_ids:
            raise DualVerificationError(
                "Dataset instances have no run record: "
                f"{sorted(missing_ids)}"
            )

        details: list[DualVerificationRecord] = []

        predicted_covered = 0
        predicted_passed = 0
        gold_passed = 0
        upstream_errors = 0
        rewrite_errors = 0
        gold_compliant_mismatches = 0

        for record in run_records:
            instance = dataset_by_id[record.instance_id]

            predicted_result = (
                self._verify_predicted_plan(
                    instance=instance,
                    record=record,
                )
            )

            if predicted_result is not None:
                predicted_covered += 1

                if predicted_result.passed:
                    predicted_passed += 1

            gold_plan = self.edit_plan_builder.build_gold(
                instance=instance
            )

            gold_result = self.verifier.verify(
                instance=instance,
                analysis=instance.gold_analysis,
                edit_plan=gold_plan,
                output_text=record.output_text,
            )

            if gold_result.passed:
                gold_passed += 1

            plan_matches = self._plan_matches_gold(
                instance=instance,
                record=record,
            )

            if (
                predicted_result is not None
                and predicted_result.passed
                and not gold_result.passed
            ):
                upstream_errors += 1

            if (
                predicted_result is not None
                and not predicted_result.passed
            ):
                rewrite_errors += 1

            if not plan_matches and gold_result.passed:
                gold_compliant_mismatches += 1

            details.append(
                DualVerificationRecord(
                    instance_id=instance.instance_id,
                    predicted_plan_matches_gold=(
                        plan_matches
                    ),
                    predicted_plan_verification=(
                        predicted_result
                    ),
                    gold_plan_verification=gold_result,
                    category=self._classify(
                        predicted_result=predicted_result,
                        gold_result=gold_result,
                        plan_matches_gold=plan_matches,
                    ),
                )
            )

        evaluated_count = len(run_records)

        return DualVerificationSummary(
            dataset_instance_count=len(instances),
            evaluated_instance_count=evaluated_count,
            predicted_plan_verification_coverage=(
                MetricValue(
                    numerator=predicted_covered,
                    denominator=evaluated_count,
                )
            ),
            predicted_plan_verification_pass_rate=(
                MetricValue(
                    numerator=predicted_passed,
                    denominator=predicted_covered,
                )
            ),
            gold_plan_verification_pass_rate=(
                MetricValue(
                    numerator=gold_passed,
                    denominator=evaluated_count,
                )
            ),
            upstream_plan_error_rate=MetricValue(
                numerator=upstream_errors,
                denominator=evaluated_count,
            ),
            rewrite_execution_error_rate=MetricValue(
                numerator=rewrite_errors,
                denominator=predicted_covered,
            ),
            gold_compliant_despite_plan_mismatch_rate=(
                MetricValue(
                    numerator=gold_compliant_mismatches,
                    denominator=evaluated_count,
                )
            ),
            records=details,
        )

    def _verify_predicted_plan(
        self,
        *,
        instance: DatasetInstance,
        record: RunRecord,
    ) -> VerificationResult | None:
        if (
            record.predicted_analysis is None
            or record.edit_plan is None
        ):
            return None

        return self.verifier.verify(
            instance=instance,
            analysis=record.predicted_analysis,
            edit_plan=record.edit_plan,
            output_text=record.output_text,
        )

    def _plan_matches_gold(
        self,
        *,
        instance: DatasetInstance,
        record: RunRecord,
    ) -> bool:
        predicted_analysis = record.predicted_analysis
        predicted_plan = record.edit_plan

        if (
            predicted_analysis is None
            or predicted_plan is None
        ):
            return False

        gold_decision = instance.gold_decision

        if (
            predicted_plan.instance_action
            != gold_decision.instance_action
        ):
            return False

        if predicted_plan.edit_scope != gold_decision.edit_scope:
            return False

        matches = match_spans(
            gold_spans=instance.gold_analysis.spans,
            predicted_spans=predicted_analysis.spans,
            minimum_overlap=self.minimum_overlap,
        )

        if len(matches) != len(instance.gold_analysis.spans):
            return False

        predicted_actions = {
            decision.span_id: decision.action
            for decision in predicted_plan.span_decisions
        }
        gold_actions = {
            decision.span_id: decision.action
            for decision in gold_decision.span_actions
        }

        matched_predicted_ids: set[str] = set()

        for gold_span, predicted_span in matches:
            matched_predicted_ids.add(predicted_span.span_id)

            if (
                predicted_actions.get(predicted_span.span_id)
                != gold_actions.get(gold_span.span_id)
            ):
                return False

        extra_predicted_decisions = (
            set(predicted_actions) - matched_predicted_ids
        )

        return not extra_predicted_decisions

    @staticmethod
    def _classify(
        *,
        predicted_result: VerificationResult | None,
        gold_result: VerificationResult,
        plan_matches_gold: bool,
    ) -> DualVerificationCategory:
        if predicted_result is None:
            return DualVerificationCategory.MISSING_PREDICTED_PLAN

        if not predicted_result.passed:
            return (
                DualVerificationCategory.REWRITE_EXECUTION_ERROR
            )

        if not gold_result.passed:
            return DualVerificationCategory.UPSTREAM_PLAN_ERROR

        if not plan_matches_gold:
            return (
                DualVerificationCategory
                .GOLD_COMPLIANT_DESPITE_PLAN_MISMATCH
            )

        return DualVerificationCategory.PASS

    @staticmethod
    def _index_instances(
        instances: Sequence[DatasetInstance],
    ) -> dict[str, DatasetInstance]:
        indexed: dict[str, DatasetInstance] = {}

        for instance in instances:
            if instance.instance_id in indexed:
                raise DualVerificationError(
                    "Duplicate dataset instance ID: "
                    f"{instance.instance_id!r}"
                )

            indexed[instance.instance_id] = instance

        return indexed

    @staticmethod
    def _index_records(
        records: Sequence[RunRecord],
    ) -> dict[str, RunRecord]:
        indexed: dict[str, RunRecord] = {}

        for record in records:
            if record.instance_id in indexed:
                raise DualVerificationError(
                    "Duplicate run record ID: "
                    f"{record.instance_id!r}"
                )

            indexed[record.instance_id] = record

        return indexed
