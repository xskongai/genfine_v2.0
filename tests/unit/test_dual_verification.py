from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.models import RunRecord
from genfine.evaluation import (
    DualVerificationCategory,
    DualVerificationEvaluator,
)
from genfine.generation import GoldRewriter
from genfine.pipeline import EditPlanBuilder
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


def build_gold_record(index: int) -> tuple[object, RunRecord]:
    instance = load_dataset(SEED_PATH)[index]
    analysis = OracleAnalyzer().analyze(instance)
    decisions = (
        DecisionEngine
        .from_yaml(RULE_PATH)
        .decide_analysis(
            analysis=analysis,
            task_mode=instance.task_mode,
        )
    )
    plan = EditPlanBuilder().build(
        instance=instance,
        analysis=analysis,
        decisions=decisions,
    )
    output = GoldRewriter().rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
    )
    verification = OutputVerifier.default().verify(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
        output_text=output,
    )

    return instance, RunRecord(
        instance_id=instance.instance_id,
        original_text=instance.context.target_text,
        predicted_analysis=analysis,
        edit_plan=plan,
        output_text=output,
        verification=verification,
    )


def test_gold_plan_builder_matches_dataset_annotations() -> None:
    builder = EditPlanBuilder()

    for instance in load_dataset(SEED_PATH):
        plan = builder.build_gold(instance=instance)

        assert (
            plan.instance_action
            == instance.gold_decision.instance_action
        )
        assert (
            plan.edit_scope
            == instance.gold_decision.edit_scope
        )
        assert [
            decision.action
            for decision in plan.span_decisions
        ] == [
            annotation.action
            for annotation
            in instance.gold_decision.span_actions
        ]


def test_gold_pipeline_passes_both_verification_layers() -> None:
    instances = []
    records = []

    for index in range(len(load_dataset(SEED_PATH))):
        instance, record = build_gold_record(index)
        instances.append(instance)
        records.append(record)

    summary = DualVerificationEvaluator().evaluate(
        instances=instances,
        run_records=records,
        require_complete=True,
    )

    assert (
        summary
        .predicted_plan_verification_pass_rate
        .value
        == 1.0
    )
    assert (
        summary
        .gold_plan_verification_pass_rate
        .value
        == 1.0
    )
    assert summary.upstream_plan_error_rate.numerator == 0
    assert summary.rewrite_execution_error_rate.numerator == 0
    assert all(
        item.category == DualVerificationCategory.PASS
        for item in summary.records
    )


def test_wrong_keep_plan_is_an_upstream_error() -> None:
    instance, correct_record = build_gold_record(2)

    keep_instance, keep_record = build_gold_record(0)
    del keep_instance

    wrong_plan = keep_record.edit_plan.model_copy(
        update={
            "instance_id": instance.instance_id,
            "original_text": instance.context.target_text,
            "protected_facts": [],
            "span_decisions": [],
        }
    )

    wrong_record = correct_record.model_copy(
        update={
            "edit_plan": wrong_plan,
            "output_text": instance.context.target_text,
        }
    )

    summary = DualVerificationEvaluator().evaluate(
        instances=[instance],
        run_records=[wrong_record],
        require_complete=True,
    )

    detail = summary.records[0]

    assert detail.predicted_plan_verification is not None
    assert detail.predicted_plan_verification.passed
    assert not detail.gold_plan_verification.passed
    assert (
        detail.category
        == DualVerificationCategory.UPSTREAM_PLAN_ERROR
    )
    assert summary.upstream_plan_error_rate.numerator == 1


def test_unexecuted_predicted_edit_is_rewrite_error() -> None:
    instance, record = build_gold_record(2)

    unchanged_record = record.model_copy(
        update={
            "output_text": instance.context.target_text,
        }
    )

    summary = DualVerificationEvaluator().evaluate(
        instances=[instance],
        run_records=[unchanged_record],
    )

    assert (
        summary.records[0].category
        == DualVerificationCategory.REWRITE_EXECUTION_ERROR
    )
    assert summary.rewrite_execution_error_rate.numerator == 1
