#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 7/20/26
Description: validate_dataset
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

from genfine.data.loader import (
    DatasetLoadError,
    load_dataset_records,
)
from genfine.data.validator import (
    DatasetValidationReport,
    DatasetValidator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a GenFINE JSONL dataset."
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/seed/seed_v0.1.jsonl",
        help=(
            "Path to the dataset JSONL file "
            "(default: data/seed/seed_v0.1.jsonl)"
        ),
    )

    return parser.parse_args()


def print_report(
    dataset_path: Path,
    report: DatasetValidationReport,
) -> None:
    print(f"Dataset: {dataset_path}")
    print(f"Instances: {report.instance_count}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")

    for issue in report.issues:
        location = issue.instance_id or "<dataset>"

        if issue.line_number is not None:
            location += f" (line {issue.line_number})"

        print(
            f"[{issue.severity}] "
            f"{issue.code} "
            f"{location}: "
            f"{issue.message}"
        )

    if report.passed:
        print("Validation passed.")
    else:
        print("Validation failed.")


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)

    try:
        records = load_dataset_records(dataset_path)
    except (
        DatasetLoadError,
        FileNotFoundError,
    ) as exc:
        print(
            f"Failed to load dataset: {exc}",
            file=sys.stderr,
        )
        return 1

    validator = DatasetValidator()
    report = validator.validate(records)

    print_report(
        dataset_path=dataset_path,
        report=report,
    )

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())