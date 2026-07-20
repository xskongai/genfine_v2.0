from pathlib import Path

import pytest

from genfine.data.loader import (
    DatasetLoadError,
    load_dataset,
    load_dataset_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.1.jsonl"
)


def test_load_seed_dataset() -> None:
    instances = load_dataset(SEED_PATH)

    assert len(instances) == 5
    assert instances[0].instance_id == "zh_seed_0001"


def test_loader_preserves_line_numbers() -> None:
    records = load_dataset_records(SEED_PATH)

    assert records[0].line_number == 1
    assert records[-1].line_number == 5


def test_blank_lines_are_ignored(
    tmp_path: Path,
) -> None:
    first_line = (
        SEED_PATH
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    test_path = tmp_path / "blank_lines.jsonl"
    test_path.write_text(
        f"\n{first_line}\n\n",
        encoding="utf-8",
    )

    records = load_dataset_records(test_path)

    assert len(records) == 1
    assert records[0].line_number == 2


def test_invalid_json_reports_line_number(
    tmp_path: Path,
) -> None:
    test_path = tmp_path / "invalid.jsonl"

    test_path.write_text(
        '{"instance_id": "broken"\n',
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetLoadError,
        match=r"invalid\.jsonl:1",
    ):
        load_dataset_records(test_path)


def test_empty_dataset_is_rejected(
    tmp_path: Path,
) -> None:
    test_path = tmp_path / "empty.jsonl"
    test_path.write_text(
        "\n\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetLoadError,
        match="contains no instances",
    ):
        load_dataset_records(test_path)