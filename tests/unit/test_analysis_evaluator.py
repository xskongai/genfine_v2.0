from __future__ import annotations

from pathlib import Path

from genfine.data.loader import load_dataset
from genfine.evaluation.analysis_evaluator import (
    evaluate_analysis_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.2.jsonl"
)


def _gold_record(instance) -> dict:
    return {
        "instance_id": instance.instance_id,
        "predicted_analysis": (
            instance.gold_analysis.model_dump(
                mode="json",
                exclude_none=True,
            )
        ),
    }


def test_perfect_predictions_score_one() -> None:
    instances = load_dataset(SEED_PATH)[:3]
    records = [
        _gold_record(instance)
        for instance in instances
    ]

    summary = evaluate_analysis_predictions(
        instances=instances,
        prediction_records=records,
    )

    assert summary.schema_validity.accuracy == 1.0
    assert summary.analysis_exact_match.accuracy == 1.0

    assert summary.exact_span.precision == 1.0
    assert summary.exact_span.recall == 1.0
    assert summary.exact_span.f1 == 1.0

    assert summary.overlap_span.f1 == 1.0
    assert summary.matched_span_coverage.accuracy == 1.0

    assert summary.function_labels.f1 == 1.0
    assert summary.necessity_status.accuracy == 1.0
    assert summary.bias_status.accuracy == 1.0
    assert summary.bias_mechanisms.f1 == 1.0
    assert summary.span_stance.accuracy == 1.0

    assert summary.speaker_stance.accuracy == 1.0
    assert summary.context_sufficient.accuracy == 1.0


def test_failed_prediction_counts_as_invalid() -> None:
    instance = load_dataset(SEED_PATH)[0]

    summary = evaluate_analysis_predictions(
        instances=[instance],
        prediction_records=[
            {
                "instance_id": instance.instance_id,
                "error": "schema failure",
            }
        ],
    )

    assert summary.schema_validity.accuracy == 0.0
    assert summary.analysis_exact_match.accuracy == 0.0

    assert summary.exact_span.true_positive == 0
    assert summary.exact_span.false_negative == len(
        instance.gold_analysis.spans
    )

    assert summary.overlap_span.true_positive == 0
    assert summary.overlap_span.false_negative == len(
        instance.gold_analysis.spans
    )

    assert summary.speaker_stance.accuracy == 0.0
    assert summary.context_sufficient.accuracy == 0.0


def test_contained_span_matches_only_in_overlap_metric() -> None:
    instance = load_dataset(SEED_PATH)[0]
    payload = instance.gold_analysis.model_dump(
        mode="json",
        exclude_none=True,
    )

    gold_span = instance.gold_analysis.spans[0]

    payload["spans"][0]["text"] = gold_span.text[0]
    payload["spans"][0]["start"] = gold_span.start
    payload["spans"][0]["end"] = gold_span.start + 1

    summary = evaluate_analysis_predictions(
        instances=[instance],
        prediction_records=[
            {
                "instance_id": instance.instance_id,
                "predicted_analysis": payload,
            }
        ],
    )

    assert summary.exact_span.f1 < 1.0
    assert summary.overlap_span.f1 == 1.0
    assert summary.matched_span_coverage.accuracy == 1.0


def test_missing_multilabel_value_lowers_function_f1() -> None:
    instance = load_dataset(SEED_PATH)[0]
    payload = instance.gold_analysis.model_dump(
        mode="json",
        exclude_none=True,
    )

    assert len(payload["spans"][0]["functions"]) >= 2
    payload["spans"][0]["functions"] = (
        payload["spans"][0]["functions"][:1]
    )

    summary = evaluate_analysis_predictions(
        instances=[instance],
        prediction_records=[
            {
                "instance_id": instance.instance_id,
                "predicted_analysis": payload,
            }
        ],
    )

    assert summary.overlap_span.f1 == 1.0
    assert summary.function_labels.recall < 1.0
    assert summary.function_labels.f1 < 1.0


def test_missing_prediction_is_reported() -> None:
    instance = load_dataset(SEED_PATH)[0]

    summary = evaluate_analysis_predictions(
        instances=[instance],
        prediction_records=[],
    )

    assert summary.missing_predictions == 1
    assert summary.schema_validity.total == 1
    assert summary.schema_validity.correct == 0
