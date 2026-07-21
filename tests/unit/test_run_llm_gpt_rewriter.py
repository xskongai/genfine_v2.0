from pathlib import Path
from runpy import run_path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.models import RunRecord
from genfine.pipeline import EditPlanBuilder
from genfine.policy import DecisionEngine
from genfine.verification import OutputVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "run_llm_gpt_rewriter.py"
)

SCRIPT_NAMESPACE = run_path(
    str(SCRIPT_PATH)
)

index_records = SCRIPT_NAMESPACE["index_records"]
run_saved_records = SCRIPT_NAMESPACE["run_saved_records"]
validate_record_coverage = (
    SCRIPT_NAMESPACE["validate_record_coverage"]
)

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


class FakeRewriter:
    name = "fake/rewriter"

    def __init__(self) -> None:
        self.call_count = 0

    def rewrite(
        self,
        *,
        instance,
        analysis,
        edit_plan,
    ):
        del analysis, edit_plan
        self.call_count += 1
        return instance.context.target_text


def build_source_record(instance):
    analysis = OracleAnalyzer().analyze(
        instance
    )

    decisions = (
        DecisionEngine
        .from_yaml(RULE_PATH)
        .decide_analysis(
            analysis=analysis,
            task_mode=instance.task_mode,
        )
    )

    edit_plan = EditPlanBuilder().build(
        instance=instance,
        analysis=analysis,
        decisions=decisions,
    )

    return {
        "instance_id": instance.instance_id,
        "analyzer": "oracle",
        "prompt_version": None,
        "rule_version": "0.1",
        "predicted_analysis": analysis.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "edit_plan": edit_plan.model_dump(
            mode="json",
            exclude_none=True,
        ),
    }


def test_saved_record_is_executed_and_verified() -> None:
    instance = load_dataset(SEED_PATH)[0]
    source = build_source_record(instance)
    rewriter = FakeRewriter()

    records = run_saved_records(
        instances=[instance],
        indexed_records={
            instance.instance_id: source
        },
        rewriter=rewriter,  # type: ignore[arg-type]
        verifier=OutputVerifier.default(),
        rewrite_prompt_version="test",
        continue_on_error=False,
    )

    assert len(records) == 1

    record = records[0]

    assert isinstance(record, RunRecord)
    assert record.predicted_analysis is not None
    assert record.edit_plan is not None
    assert record.output_text == instance.context.target_text
    assert record.verification is not None
    assert record.verification.passed
    assert record.metadata["saved_decision_source"] is True
    assert rewriter.call_count == 1


def test_missing_saved_record_becomes_error_record() -> None:
    instance = load_dataset(SEED_PATH)[0]
    rewriter = FakeRewriter()

    records = run_saved_records(
        instances=[instance],
        indexed_records={},
        rewriter=rewriter,  # type: ignore[arg-type]
        verifier=OutputVerifier.default(),
        rewrite_prompt_version="test",
        continue_on_error=True,
    )

    record = records[0]

    assert record.errors
    assert "Missing saved decision record" in record.errors[0]
    assert record.output_text is None
    assert rewriter.call_count == 0


def test_duplicate_decision_records_are_rejected() -> None:
    duplicate = [
        {"instance_id": "same"},
        {"instance_id": "same"},
    ]

    try:
        index_records(duplicate)
    except ValueError as exc:
        assert "Duplicate decision record" in str(exc)
    else:
        raise AssertionError(
            "Expected duplicate decision records to fail"
        )


def test_limit_allows_unselected_saved_records() -> None:
    instances = load_dataset(SEED_PATH)
    indexed = {
        instance.instance_id: {
            "instance_id": instance.instance_id
        }
        for instance in instances
    }

    validate_record_coverage(
        selected_instances=instances[:1],
        all_dataset_instances=instances,
        indexed_records=indexed,
        require_complete=True,
    )
