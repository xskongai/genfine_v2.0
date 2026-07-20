#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 7/20/26
Description: loader
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from genfine.domain.models import DatasetInstance


@dataclass(frozen=True)
class LoadedInstance:
    """
    A validated dataset instance together with its source location.

    Keeping the line number is useful for reporting annotation errors.
    """

    source_path: Path
    line_number: int
    instance: DatasetInstance


class DatasetLoadError(ValueError):
    """Raised when a JSONL dataset cannot be parsed or validated."""

    def __init__(
        self,
        path: Path,
        line_number: int | None,
        message: str,
    ) -> None:
        self.path = path
        self.line_number = line_number
        self.detail = message

        location = str(path)

        if line_number is not None:
            location = f"{location}:{line_number}"

        super().__init__(f"{location}: {message}")


def iter_dataset_records(
    path: str | Path,
) -> Iterator[LoadedInstance]:
    """
    Read and validate a GenFINE JSONL file lazily.

    Blank lines are ignored. Each non-empty line must contain exactly one
    valid DatasetInstance JSON object.
    """

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist: {dataset_path}"
        )

    if not dataset_path.is_file():
        raise DatasetLoadError(
            path=dataset_path,
            line_number=None,
            message="dataset path is not a regular file",
        )

    with dataset_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                instance = DatasetInstance.model_validate_json(line)
            except ValidationError as exc:
                raise DatasetLoadError(
                    path=dataset_path,
                    line_number=line_number,
                    message=_format_validation_error(exc),
                ) from exc
            except ValueError as exc:
                raise DatasetLoadError(
                    path=dataset_path,
                    line_number=line_number,
                    message=f"invalid JSON: {exc}",
                ) from exc

            yield LoadedInstance(
                source_path=dataset_path,
                line_number=line_number,
                instance=instance,
            )


def load_dataset_records(
    path: str | Path,
    *,
    require_non_empty: bool = True,
) -> list[LoadedInstance]:
    """Load all JSONL records while preserving line-number metadata."""

    records = list(iter_dataset_records(path))

    if require_non_empty and not records:
        raise DatasetLoadError(
            path=Path(path),
            line_number=None,
            message="dataset contains no instances",
        )

    return records


def load_dataset(
    path: str | Path,
    *,
    require_non_empty: bool = True,
) -> list[DatasetInstance]:
    """
    Convenience function returning only DatasetInstance objects.

    Use load_dataset_records() when line numbers are needed for diagnostics.
    """

    records = load_dataset_records(
        path,
        require_non_empty=require_non_empty,
    )

    return [record.instance for record in records]


def _format_validation_error(
    error: ValidationError,
) -> str:
    """Convert a Pydantic validation error into a compact readable message."""

    messages: list[str] = []

    for item in error.errors():
        location = ".".join(
            str(part)
            for part in item.get("loc", ())
        )

        message = item.get(
            "msg",
            "validation error",
        )

        if location:
            messages.append(f"{location}: {message}")
        else:
            messages.append(message)

    return "; ".join(messages)