from __future__ import annotations

from collections.abc import Sequence

from genfine.domain.enums import InstanceAction
from genfine.domain.models import (
    DatasetInstance,
    RunRecord,
)
from genfine.evaluation.models import (
    EvaluationSummary,
    MetricValue,
)


class EvaluationError(ValueError):
    """Raised when dataset instances and run records are inconsistent."""


EDIT_INSTANCE_ACTIONS = {
    InstanceAction.EDIT,
    InstanceAction.SPAN_LEVEL_EDIT,
}


class RunEvaluator:
    """Evaluate pipeline RunRecords against annotated dataset instances."""

    def evaluate(
        self,
        *,
        instances: Sequence[DatasetInstance],
        run_records: Sequence[RunRecord],
        require_complete: bool = False,
    ) -> EvaluationSummary:
        dataset_by_id = self._index_instances(
            instances
        )

        records_by_id = self._index_run_records(
            run_records
        )

        unknown_run_ids = (
            set(records_by_id)
            - set(dataset_by_id)
        )

        if unknown_run_ids:
            raise EvaluationError(
                "Run records refer to unknown dataset instances: "
                f"{sorted(unknown_run_ids)}"
            )

        missing_run_ids = (
            set(dataset_by_id)
            - set(records_by_id)
        )

        if require_complete and missing_run_ids:
            raise EvaluationError(
                "Dataset instances have no run record: "
                f"{sorted(missing_run_ids)}"
            )

        evaluated_pairs = [
            (
                dataset_by_id[record.instance_id],
                record,
            )
            for record in run_records
        ]

        return self._calculate_summary(
            instances=instances,
            evaluated_pairs=evaluated_pairs,
            missing_run_ids=missing_run_ids,
        )

    def _calculate_summary(
        self,
        *,
        instances: Sequence[DatasetInstance],
        evaluated_pairs: list[
            tuple[DatasetInstance, RunRecord]
        ],
        missing_run_ids: set[str],
    ) -> EvaluationSummary:
        analysis_correct = 0
        analysis_total = 0

        span_action_correct = 0
        span_action_total = 0

        instance_action_correct = 0
        instance_action_total = 0

        edit_scope_correct = 0
        edit_scope_total = 0

        output_exact_correct = 0
        output_exact_total = 0

        verification_covered = 0
        verification_passed = 0

        protected_fact_preserved = 0
        action_compliant = 0
        unsupported_gender_inserted = 0

        keep_total = 0
        over_neutralized = 0

        edit_total = 0
        under_corrected = 0

        failed_instances = 0

        analyzers: set[str] = set()
        rewriters: set[str] = set()
        rule_versions: set[str] = set()

        for instance, record in evaluated_pairs:
            failed = bool(record.errors)

            if failed:
                failed_instances += 1

            analyzer = record.metadata.get(
                "analyzer"
            )

            if isinstance(analyzer, str):
                analyzers.add(analyzer)

            rewriter = record.metadata.get(
                "rewriter"
            )

            if isinstance(rewriter, str):
                rewriters.add(rewriter)

            rule_version = record.metadata.get(
                "decision_rule_version"
            )

            if isinstance(rule_version, str):
                rule_versions.add(
                    rule_version
                )

            # Analysis exact match.
            analysis_total += 1

            if (
                record.predicted_analysis
                == instance.gold_analysis
            ):
                analysis_correct += 1

            # Instance-level decision and edit scope.
            instance_action_total += 1
            edit_scope_total += 1

            if record.edit_plan is not None:
                if (
                    record.edit_plan.instance_action
                    == instance.gold_decision.instance_action
                ):
                    instance_action_correct += 1

                if (
                    record.edit_plan.edit_scope
                    == instance.gold_decision.edit_scope
                ):
                    edit_scope_correct += 1

            # Span-level actions.
            gold_span_actions = {
                item.span_id: item.action
                for item in (
                    instance
                    .gold_decision
                    .span_actions
                )
            }

            predicted_span_actions = {}

            if record.edit_plan is not None:
                predicted_span_actions = {
                    item.span_id: item.action
                    for item in (
                        record
                        .edit_plan
                        .span_decisions
                    )
                }

            for span_id, gold_action in (
                gold_span_actions.items()
            ):
                span_action_total += 1

                if (
                    predicted_span_actions.get(
                        span_id
                    )
                    == gold_action
                ):
                    span_action_correct += 1

            # Surface output exact match.
            output_exact_total += 1

            if (
                record.output_text
                == instance.gold_output
            ):
                output_exact_correct += 1

            # Verification metrics.
            verification = record.verification

            if verification is not None:
                verification_covered += 1

                if verification.passed:
                    verification_passed += 1

                if (
                    verification
                    .protected_facts_preserved
                ):
                    protected_fact_preserved += 1

                if verification.action_compliant:
                    action_compliant += 1

                if (
                    verification
                    .unsupported_gender_inserted
                ):
                    unsupported_gender_inserted += 1

            # Over-neutralization.
            gold_instance_action = (
                instance
                .gold_decision
                .instance_action
            )

            if (
                gold_instance_action
                == InstanceAction.KEEP
            ):
                keep_total += 1

                if (
                    record.output_text
                    != instance.context.target_text
                ):
                    over_neutralized += 1

            # Under-correction.
            if (
                gold_instance_action
                in EDIT_INSTANCE_ACTIONS
            ):
                edit_total += 1

                if self._is_under_corrected(
                    instance=instance,
                    record=record,
                ):
                    under_corrected += 1

        evaluated_count = len(
            evaluated_pairs
        )

        return EvaluationSummary(
            dataset_instance_count=len(instances),
            evaluated_instance_count=(
                evaluated_count
            ),
            successful_instances=(
                evaluated_count
                - failed_instances
            ),
            failed_instances=failed_instances,
            analysis_exact_match=MetricValue(
                numerator=analysis_correct,
                denominator=analysis_total,
            ),
            span_action_accuracy=MetricValue(
                numerator=span_action_correct,
                denominator=span_action_total,
            ),
            instance_action_accuracy=MetricValue(
                numerator=instance_action_correct,
                denominator=instance_action_total,
            ),
            edit_scope_accuracy=MetricValue(
                numerator=edit_scope_correct,
                denominator=edit_scope_total,
            ),
            output_exact_match=MetricValue(
                numerator=output_exact_correct,
                denominator=output_exact_total,
            ),
            verification_coverage=MetricValue(
                numerator=verification_covered,
                denominator=evaluated_count,
            ),
            verification_pass_rate=MetricValue(
                numerator=verification_passed,
                denominator=verification_covered,
            ),
            protected_fact_preservation_rate=MetricValue(
                numerator=protected_fact_preserved,
                denominator=verification_covered,
            ),
            action_compliance_rate=MetricValue(
                numerator=action_compliant,
                denominator=verification_covered,
            ),
            unsupported_gender_insertion_rate=MetricValue(
                numerator=unsupported_gender_inserted,
                denominator=verification_covered,
            ),
            over_neutralization_rate=MetricValue(
                numerator=over_neutralized,
                denominator=keep_total,
            ),
            under_correction_rate=MetricValue(
                numerator=under_corrected,
                denominator=edit_total,
            ),
            metadata={
                "analyzers": sorted(analyzers),
                "rewriters": sorted(rewriters),
                "decision_rule_versions": sorted(
                    rule_versions
                ),
                "missing_run_record_count": len(
                    missing_run_ids
                ),
                "missing_run_record_ids": sorted(
                    missing_run_ids
                ),
            },
        )

    @staticmethod
    def _is_under_corrected(
        *,
        instance: DatasetInstance,
        record: RunRecord,
    ) -> bool:
        """
        Return True when a required edit was not successfully executed.

        A case is under-corrected when:
        - no output was produced;
        - the output is unchanged;
        - or the independent verifier reports action non-compliance.
        """

        if record.output_text is None:
            return True

        if (
            record.output_text
            == instance.context.target_text
        ):
            return True

        if (
            record.verification is not None
            and not record.verification.action_compliant
        ):
            return True

        return False

    @staticmethod
    def _index_instances(
        instances: Sequence[DatasetInstance],
    ) -> dict[str, DatasetInstance]:
        result: dict[
            str,
            DatasetInstance,
        ] = {}

        duplicates: set[str] = set()

        for instance in instances:
            if instance.instance_id in result:
                duplicates.add(
                    instance.instance_id
                )
                continue

            result[
                instance.instance_id
            ] = instance

        if duplicates:
            raise EvaluationError(
                "Duplicate dataset instance IDs: "
                f"{sorted(duplicates)}"
            )

        return result

    @staticmethod
    def _index_run_records(
        records: Sequence[RunRecord],
    ) -> dict[str, RunRecord]:
        result: dict[str, RunRecord] = {}
        duplicates: set[str] = set()

        for record in records:
            if record.instance_id in result:
                duplicates.add(
                    record.instance_id
                )
                continue

            result[
                record.instance_id
            ] = record

        if duplicates:
            raise EvaluationError(
                "Duplicate run record IDs: "
                f"{sorted(duplicates)}"
            )

        return result