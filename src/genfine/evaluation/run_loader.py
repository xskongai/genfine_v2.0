from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from genfine.domain.models import RunRecord


class RunLoadError(ValueError):
    """Raised when a RunRecord JSONL file cannot be loaded."""


def load_run_records(
    path: str | Path,
) -> list[RunRecord]:
    run_path = Path(path)

    if not run_path.exists():
        raise FileNotFoundError(
            f"Run file does not exist: {run_path}"
        )

    records: list[RunRecord] = []

    with run_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = (
                    RunRecord
                    .model_validate_json(line)
                )
            except ValidationError as exc:
                raise RunLoadError(
                    f"Invalid RunRecord at "
                    f"{run_path}:{line_number}: "
                    f"{exc}"
                ) from exc

            records.append(record)

    if not records:
        raise RunLoadError(
            f"Run file contains no records: "
            f"{run_path}"
        )

    return records