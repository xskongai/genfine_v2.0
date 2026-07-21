from pathlib import Path

import pytest

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.evaluation import (
    EvaluationError,
    RunEvaluator,
)
from genfine.generation import GoldRewriter
from genfine.pipeline import (
    EditPlanBuilder,
    PipelineRunner,
)
from genfine.policy import DecisionEngine
from genfine.verification import OutputVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.1.jsonl"
)

RULE_PATH = (
    PROJECT_ROOT
    / "configs"
    / "decision_rules.yaml"
)


def build_records():
    instances = load_dataset(SEED_PATH)

    runner = PipelineRunner(
        analyzer=OracleAnalyzer(),
        decision_engine=(
            DecisionEngine.from_yaml(
                RULE_PATH
            )
        ),
        edit_plan_builder=EditPlanBuilder(),
        rewriter=GoldRewriter(),
        verifier=OutputVerifier.default(),
    )

    records = runner.run_many(
        instances
    )

    return instances, records


def test_gold_pipeline_has_perfect_metrics() -> None:
    instances, records = build_records()

    summary = RunEvaluator().evaluate(
        instances=instances,
        run_records=records,
        require_complete=True,
    )

    assert (
        summary.analysis_exact_match.value
        == 1.0
    )

    assert (
        summary.span_action_accuracy.value
        == 1.0
    )

    assert (
        summary.instance_action_accuracy.value
        == 1.0
    )

    assert (
        summary.edit_scope_accuracy.value
        == 1.0
    )

    assert (
        summary.output_exact_match.value
        == 1.0
    )

    assert (
        summary.verification_pass_rate.value
        == 1.0
    )

    assert (
        summary.protected_fact_preservation_rate.value
        == 1.0
    )

    assert (
        summary.action_compliance_rate.value
        == 1.0
    )

    assert (
        summary.unsupported_gender_insertion_rate.value
        == 0.0
    )

    assert (
        summary.over_neutralization_rate.value
        == 0.0
    )

    assert (
        summary.under_correction_rate.value
        == 0.0
    )


def test_changed_keep_counts_as_over_neutralization() -> None:
    instances, records = build_records()

    changed_record = records[0].model_copy(
        update={
            "output_text": (
                "我的手足打电话说TA会回来"
            )
        }
    )

    modified_records = [
        changed_record,
        *records[1:],
    ]

    summary = RunEvaluator().evaluate(
        instances=instances,
        run_records=modified_records,
    )

    assert (
        summary
        .over_neutralization_rate
        .numerator
        == 1
    )

    assert (
        summary
        .over_neutralization_rate
        .denominator
        == 4
    )


def test_unchanged_edit_counts_as_under_correction() -> None:
    instances, records = build_records()

    edit_instance_index = 2

    unchanged_record = (
        records[
            edit_instance_index
        ].model_copy(
            update={
                "output_text": (
                    instances[
                        edit_instance_index
                    ]
                    .context
                    .target_text
                )
            }
        )
    )

    modified_records = list(records)

    modified_records[
        edit_instance_index
    ] = unchanged_record

    summary = RunEvaluator().evaluate(
        instances=instances,
        run_records=modified_records,
    )

    assert (
        summary
        .under_correction_rate
        .numerator
        == 1
    )

    assert (
        summary
        .under_correction_rate
        .denominator
        == 1
    )


def test_partial_run_is_allowed_by_default() -> None:
    instances, records = build_records()

    summary = RunEvaluator().evaluate(
        instances=instances,
        run_records=records[:3],
    )

    assert (
        summary.evaluated_instance_count
        == 3
    )

    assert (
        summary.metadata[
            "missing_run_record_count"
        ]
        == 2
    )


def test_complete_run_can_be_required() -> None:
    instances, records = build_records()

    with pytest.raises(
        EvaluationError,
        match="have no run record",
    ):
        RunEvaluator().evaluate(
            instances=instances,
            run_records=records[:3],
            require_complete=True,
        )


def test_duplicate_run_ids_are_rejected() -> None:
    instances, records = build_records()

    duplicated = [
        records[0],
        records[0].model_copy(
            deep=True
        ),
    ]

    with pytest.raises(
        EvaluationError,
        match="Duplicate run record IDs",
    ):
        RunEvaluator().evaluate(
            instances=instances,
            run_records=duplicated,
        )