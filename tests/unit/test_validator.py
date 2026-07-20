from pathlib import Path

from genfine.data.loader import (
    LoadedInstance,
    load_dataset_records,
)
from genfine.data.validator import (
    DatasetValidator,
    infer_instance_action,
)
from genfine.domain.enums import (
    Action,
    InstanceAction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.1.jsonl"
)


def test_valid_seed_dataset_passes() -> None:
    records = load_dataset_records(SEED_PATH)

    report = DatasetValidator().validate(records)

    assert report.passed
    assert report.errors == []


def test_duplicate_instance_id_is_rejected() -> None:
    records = load_dataset_records(SEED_PATH)

    duplicated_instance = records[1].instance.model_copy(
        update={
            "instance_id": (
                records[0].instance.instance_id
            )
        }
    )

    duplicate_record = LoadedInstance(
        source_path=records[1].source_path,
        line_number=records[1].line_number,
        instance=duplicated_instance,
    )

    modified_records = [
        records[0],
        duplicate_record,
        *records[2:],
    ]

    report = DatasetValidator().validate(
        modified_records
    )

    error_codes = {
        issue.code
        for issue in report.errors
    }

    assert "DUPLICATE_INSTANCE_ID" in error_codes


def test_keep_output_must_equal_original() -> None:
    records = load_dataset_records(SEED_PATH)
    original_record = records[0]

    modified_instance = (
        original_record.instance.model_copy(
            update={
                "gold_output": "错误的改写结果"
            }
        )
    )

    modified_record = LoadedInstance(
        source_path=original_record.source_path,
        line_number=original_record.line_number,
        instance=modified_instance,
    )

    report = DatasetValidator().validate(
        [modified_record]
    )

    error_codes = {
        issue.code
        for issue in report.errors
    }

    assert "KEEP_OUTPUT_CHANGED" in error_codes


def test_missing_span_action_is_rejected() -> None:
    records = load_dataset_records(SEED_PATH)
    original_record = records[0]

    modified_decision = (
        original_record
        .instance
        .gold_decision
        .model_copy(
            update={
                "span_actions": []
            }
        )
    )

    modified_instance = (
        original_record.instance.model_copy(
            update={
                "gold_decision": modified_decision
            }
        )
    )

    modified_record = LoadedInstance(
        source_path=original_record.source_path,
        line_number=original_record.line_number,
        instance=modified_instance,
    )

    report = DatasetValidator().validate(
        [modified_record]
    )

    error_codes = {
        issue.code
        for issue in report.errors
    }

    assert "MISSING_SPAN_ACTION" in error_codes


def test_infer_keep_instance_action() -> None:
    result = infer_instance_action(
        [
            Action.KEEP,
            Action.PRESERVE_AMBIGUITY,
        ]
    )

    assert result == InstanceAction.KEEP


def test_infer_edit_instance_action() -> None:
    result = infer_instance_action(
        [
            Action.REPLACE_GENERIC_FORM,
        ]
    )

    assert result == InstanceAction.EDIT


def test_infer_span_level_edit_action() -> None:
    result = infer_instance_action(
        [
            Action.KEEP,
            Action.REFRAME_PROPOSITION,
        ]
    )

    assert result == InstanceAction.SPAN_LEVEL_EDIT


def test_infer_abstain_action() -> None:
    result = infer_instance_action(
        [
            Action.ABSTAIN,
        ]
    )

    assert result == InstanceAction.ABSTAIN