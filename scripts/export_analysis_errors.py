from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from genfine.data.loader import load_dataset
from genfine.domain.models import AnalysisResult, AnalysisSpan
from genfine.evaluation.analysis_evaluator import match_spans


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = [
    "instance_id",
    "target_text",
    "prediction_valid",
    "prediction_error",
    "row_type",
    "span_match_type",
    "auto_flags",
    "gold_span",
    "predicted_span",
    "gold_functions",
    "predicted_functions",
    "gold_necessity",
    "predicted_necessity",
    "gold_bias_status",
    "predicted_bias_status",
    "gold_bias_mechanisms",
    "predicted_bias_mechanisms",
    "gold_span_stance",
    "predicted_span_stance",
    "gold_speaker_stance",
    "predicted_speaker_stance",
    "gold_context_sufficient",
    "predicted_context_sufficient",
    "error_category",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GenFINE analysis errors for manual review."
    )
    parser.add_argument(
        "predictions",
        nargs="?",
        default="runs/llm_analysis_v0.1.jsonl",
    )
    parser.add_argument(
        "--dataset",
        default="data/seed/seed_v0.2.jsonl",
    )
    parser.add_argument(
        "--output",
        default="runs/analysis_error_report_v0.1.csv",
    )
    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at {path}:{line_number} must be an object"
                )

            records.append(record)

    return records


def index_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for record in records:
        instance_id = record.get("instance_id")

        if not isinstance(instance_id, str):
            raise ValueError(
                "Every prediction record needs a string instance_id"
            )

        if instance_id in result:
            raise ValueError(
                f"Duplicate prediction for {instance_id}"
            )

        result[instance_id] = record

    return result


def value(item: Any) -> str:
    if item is None:
        return ""
    return str(getattr(item, "value", item))


def values(items: Iterable[Any]) -> str:
    return json.dumps(
        [value(item) for item in items],
        ensure_ascii=False,
    )


def parse_analysis(
    record: dict[str, Any] | None,
) -> tuple[AnalysisResult | None, str]:
    if record is None:
        return None, "Missing prediction record"

    if "predicted_analysis" not in record:
        return None, str(
            record.get("error", "Missing predicted_analysis")
        )

    try:
        return (
            AnalysisResult.model_validate(
                record["predicted_analysis"]
            ),
            "",
        )
    except ValidationError as exc:
        return None, str(exc)


def exact_span(
    gold: AnalysisSpan,
    predicted: AnalysisSpan,
) -> bool:
    return (
        gold.start == predicted.start
        and gold.end == predicted.end
        and gold.text == predicted.text
    )


def span_flags(
    gold: AnalysisSpan,
    predicted: AnalysisSpan,
) -> list[str]:
    flags: list[str] = []

    if not exact_span(gold, predicted):
        flags.append("SPAN_BOUNDARY_MISMATCH")
    if set(gold.functions) != set(predicted.functions):
        flags.append("FUNCTION_MISMATCH")
    if gold.necessity.status != predicted.necessity.status:
        flags.append("NECESSITY_MISMATCH")
    if gold.bias.status != predicted.bias.status:
        flags.append("BIAS_STATUS_MISMATCH")
    if set(gold.bias.mechanisms) != set(
        predicted.bias.mechanisms
    ):
        flags.append("BIAS_MECHANISM_MISMATCH")
    if gold.stance != predicted.stance:
        flags.append("SPAN_STANCE_MISMATCH")

    return flags or ["MATCH"]


def base_row(
    instance,
    predicted: AnalysisResult | None,
    error: str,
) -> dict[str, Any]:
    gold = instance.gold_analysis

    return {
        "instance_id": instance.instance_id,
        "target_text": instance.context.target_text,
        "prediction_valid": predicted is not None,
        "prediction_error": error,
        "row_type": "",
        "span_match_type": "",
        "auto_flags": "",
        "gold_span": "",
        "predicted_span": "",
        "gold_functions": "",
        "predicted_functions": "",
        "gold_necessity": "",
        "predicted_necessity": "",
        "gold_bias_status": "",
        "predicted_bias_status": "",
        "gold_bias_mechanisms": "",
        "predicted_bias_mechanisms": "",
        "gold_span_stance": "",
        "predicted_span_stance": "",
        "gold_speaker_stance": value(
            gold.speaker_stance
        ),
        "predicted_speaker_stance": (
            value(predicted.speaker_stance)
            if predicted is not None
            else ""
        ),
        "gold_context_sufficient": (
            gold.context_sufficient
        ),
        "predicted_context_sufficient": (
            predicted.context_sufficient
            if predicted is not None
            else ""
        ),
        "error_category": "",
        "review_notes": "",
    }


def add_instance_flags(
    flags: list[str],
    gold: AnalysisResult,
    predicted: AnalysisResult,
) -> None:
    if gold.speaker_stance != predicted.speaker_stance:
        flags.append("SPEAKER_STANCE_MISMATCH")
    if (
        gold.context_sufficient
        != predicted.context_sufficient
    ):
        flags.append("CONTEXT_SUFFICIENCY_MISMATCH")


def fill_gold(
    row: dict[str, Any],
    span: AnalysisSpan,
) -> None:
    row.update(
        {
            "gold_span": span.text,
            "gold_functions": values(span.functions),
            "gold_necessity": value(
                span.necessity.status
            ),
            "gold_bias_status": value(
                span.bias.status
            ),
            "gold_bias_mechanisms": values(
                span.bias.mechanisms
            ),
            "gold_span_stance": value(span.stance),
        }
    )


def fill_predicted(
    row: dict[str, Any],
    span: AnalysisSpan,
) -> None:
    row.update(
        {
            "predicted_span": span.text,
            "predicted_functions": values(
                span.functions
            ),
            "predicted_necessity": value(
                span.necessity.status
            ),
            "predicted_bias_status": value(
                span.bias.status
            ),
            "predicted_bias_mechanisms": values(
                span.bias.mechanisms
            ),
            "predicted_span_stance": value(
                span.stance
            ),
        }
    )


def build_rows(
    instances,
    prediction_records: list[dict[str, Any]],
    minimum_overlap: float,
) -> list[dict[str, Any]]:
    prediction_by_id = index_records(
        prediction_records
    )
    rows: list[dict[str, Any]] = []

    for instance in instances:
        predicted, error = parse_analysis(
            prediction_by_id.get(instance.instance_id)
        )
        gold = instance.gold_analysis

        if predicted is None:
            row = base_row(instance, None, error)
            row["row_type"] = "INVALID_INSTANCE"
            row["auto_flags"] = (
                "SCHEMA_OR_GENERATION_FAILURE"
            )
            rows.append(row)
            continue

        matches = match_spans(
            gold_spans=gold.spans,
            predicted_spans=predicted.spans,
            minimum_overlap=minimum_overlap,
        )
        matched_gold = {id(item[0]) for item in matches}
        matched_predicted = {
            id(item[1]) for item in matches
        }

        for gold_span, predicted_span in matches:
            row = base_row(instance, predicted, error)
            flags = span_flags(
                gold_span,
                predicted_span,
            )
            add_instance_flags(flags, gold, predicted)

            row["row_type"] = "MATCHED_SPAN"
            row["span_match_type"] = (
                "EXACT"
                if exact_span(
                    gold_span,
                    predicted_span,
                )
                else "OVERLAP"
            )
            row["auto_flags"] = "|".join(
                dict.fromkeys(flags)
            )
            fill_gold(row, gold_span)
            fill_predicted(row, predicted_span)
            rows.append(row)

        for gold_span in gold.spans:
            if id(gold_span) in matched_gold:
                continue

            row = base_row(instance, predicted, error)
            flags = ["UNMATCHED_GOLD_SPAN"]
            add_instance_flags(flags, gold, predicted)
            row["row_type"] = "GOLD_ONLY"
            row["span_match_type"] = "UNMATCHED"
            row["auto_flags"] = "|".join(flags)
            fill_gold(row, gold_span)
            rows.append(row)

        for predicted_span in predicted.spans:
            if id(predicted_span) in matched_predicted:
                continue

            row = base_row(instance, predicted, error)
            flags = ["UNMATCHED_PREDICTED_SPAN"]
            add_instance_flags(flags, gold, predicted)
            row["row_type"] = "PREDICTED_ONLY"
            row["span_match_type"] = "UNMATCHED"
            row["auto_flags"] = "|".join(flags)
            fill_predicted(row, predicted_span)
            rows.append(row)

        if not gold.spans and not predicted.spans:
            row = base_row(instance, predicted, error)
            flags: list[str] = []
            add_instance_flags(flags, gold, predicted)
            row["row_type"] = "INSTANCE_ONLY"
            row["auto_flags"] = "|".join(
                flags or ["MATCH"]
            )
            rows.append(row)

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    if not 0.0 < args.minimum_overlap <= 1.0:
        print(
            "--minimum-overlap must be in (0, 1].",
            file=sys.stderr,
        )
        return 2

    try:
        instances = load_dataset(
            PROJECT_ROOT / args.dataset
        )
        predictions = load_jsonl(
            PROJECT_ROOT / args.predictions
        )
        rows = build_rows(
            instances,
            predictions,
            args.minimum_overlap,
        )
        output_path = PROJECT_ROOT / args.output
        write_csv(output_path, rows)
    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset instances: {len(instances)}")
    print(f"Review rows: {len(rows)}")
    print(
        "Invalid instances: "
        f"{sum(row['row_type'] == 'INVALID_INSTANCE' for row in rows)}"
    )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
