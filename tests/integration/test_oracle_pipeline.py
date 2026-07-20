from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
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


def build_runner() -> PipelineRunner:
    return PipelineRunner(
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


def test_oracle_pipeline_runs_all_seed_instances() -> None:
    instances = load_dataset(SEED_PATH)

    records = build_runner().run_many(
        instances
    )

    assert len(records) == len(instances)
    assert len(records) == 5


def test_oracle_pipeline_has_no_errors() -> None:
    instances = load_dataset(SEED_PATH)

    records = build_runner().run_many(
        instances
    )

    for record in records:
        assert record.errors == []


def test_oracle_pipeline_outputs_match_gold() -> None:
    instances = load_dataset(SEED_PATH)

    records = build_runner().run_many(
        instances
    )

    for instance, record in zip(
        instances,
        records,
        strict=True,
    ):
        assert (
            record.output_text
            == instance.gold_output
        )


def test_oracle_pipeline_analyses_match_gold() -> None:
    instances = load_dataset(SEED_PATH)

    records = build_runner().run_many(
        instances
    )

    for instance, record in zip(
        instances,
        records,
        strict=True,
    ):
        assert (
            record.predicted_analysis
            == instance.gold_analysis
        )


def test_oracle_pipeline_plans_match_gold() -> None:
    instances = load_dataset(SEED_PATH)

    records = build_runner().run_many(
        instances
    )

    for instance, record in zip(
        instances,
        records,
        strict=True,
    ):
        assert record.edit_plan is not None

        assert (
            record.edit_plan.instance_action
            == instance.gold_decision.instance_action
        )

        assert (
            record.edit_plan.edit_scope
            == instance.gold_decision.edit_scope
        )

        predicted_actions = {
            item.span_id: item.action
            for item
            in record.edit_plan.span_decisions
        }

        gold_actions = {
            item.span_id: item.action
            for item
            in instance.gold_decision.span_actions
        }

        assert predicted_actions == gold_actions


def test_pipeline_metadata_is_recorded() -> None:
    instance = load_dataset(SEED_PATH)[0]

    record = build_runner().run(
        instance
    )

    assert record.metadata["analyzer"] == "oracle"
    assert record.metadata["rewriter"] == "gold"
    assert (
        record.metadata[
            "decision_rule_version"
        ]
        == "0.1"
    )


def test_continue_on_error_returns_error_record() -> None:
    from genfine.domain.enums import (
        InstanceAction,
    )

    instances = load_dataset(SEED_PATH)
    original = instances[0]

    invalid_gold_decision = (
        original.gold_decision.model_copy(
            update={
                "instance_action": (
                    InstanceAction.EDIT
                ),
            }
        )
    )

    broken_instance = original.model_copy(
        update={
            "gold_decision": invalid_gold_decision,
        }
    )

    records = build_runner().run_many(
        [broken_instance],
        continue_on_error=True,
    )

    assert len(records) == 1
    assert records[0].errors
    assert records[0].output_text is None

def test_oracle_pipeline_verification_passes() -> None:
    instances = load_dataset(SEED_PATH)

    records = build_runner().run_many(
        instances
    )

    for record in records:
        assert record.verification is not None
        assert record.verification.passed
        assert record.verification.issues == []
