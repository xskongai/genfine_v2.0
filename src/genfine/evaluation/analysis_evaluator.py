from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from pydantic import ValidationError

from genfine.domain.models import (
    AnalysisResult,
    AnalysisSpan,
    DatasetInstance,
)


@dataclass
class PRFMetric:
    """Micro precision, recall and F1 counts."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = (
            self.true_positive
            + self.false_positive
        )
        return (
            self.true_positive / denominator
            if denominator
            else 0.0
        )

    @property
    def recall(self) -> float:
        denominator = (
            self.true_positive
            + self.false_negative
        )
        return (
            self.true_positive / denominator
            if denominator
            else 0.0
        )

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return (
            2.0
            * self.precision
            * self.recall
            / denominator
            if denominator
            else 0.0
        )

    def add(
        self,
        *,
        true_positive: int = 0,
        false_positive: int = 0,
        false_negative: int = 0,
    ) -> None:
        self.true_positive += true_positive
        self.false_positive += false_positive
        self.false_negative += false_negative

    def to_dict(self) -> dict[str, int | float]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class AccuracyMetric:
    """Simple correct/total accuracy."""

    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return (
            self.correct / self.total
            if self.total
            else 0.0
        )

    def add(
        self,
        *,
        correct: bool,
    ) -> None:
        self.total += 1
        if correct:
            self.correct += 1

    def to_dict(self) -> dict[str, int | float]:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
        }


@dataclass
class AnalysisEvaluationSummary:
    """Aggregate metrics for one analysis prediction run."""

    dataset_instances: int
    prediction_records: int
    missing_predictions: int
    extra_predictions: int

    schema_validity: AccuracyMetric
    analysis_exact_match: AccuracyMetric

    exact_span: PRFMetric
    overlap_span: PRFMetric
    matched_span_coverage: AccuracyMetric

    function_labels: PRFMetric
    necessity_status: AccuracyMetric
    bias_status: AccuracyMetric
    bias_mechanisms: PRFMetric
    span_stance: AccuracyMetric

    speaker_stance: AccuracyMetric
    context_sufficient: AccuracyMetric

    invalid_instance_ids: list[str]
    extra_instance_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_instances": self.dataset_instances,
            "prediction_records": self.prediction_records,
            "missing_predictions": self.missing_predictions,
            "extra_predictions": self.extra_predictions,
            "schema_validity": self.schema_validity.to_dict(),
            "analysis_exact_match": (
                self.analysis_exact_match.to_dict()
            ),
            "exact_span": self.exact_span.to_dict(),
            "overlap_span": self.overlap_span.to_dict(),
            "matched_span_coverage": (
                self.matched_span_coverage.to_dict()
            ),
            "function_labels": (
                self.function_labels.to_dict()
            ),
            "necessity_status": (
                self.necessity_status.to_dict()
            ),
            "bias_status": self.bias_status.to_dict(),
            "bias_mechanisms": (
                self.bias_mechanisms.to_dict()
            ),
            "span_stance": self.span_stance.to_dict(),
            "speaker_stance": (
                self.speaker_stance.to_dict()
            ),
            "context_sufficient": (
                self.context_sufficient.to_dict()
            ),
            "invalid_instance_ids": (
                self.invalid_instance_ids
            ),
            "extra_instance_ids": (
                self.extra_instance_ids
            ),
        }


def _enum_values(
    values: Iterable[Any],
) -> set[str]:
    return {
        str(getattr(value, "value", value))
        for value in values
    }


def _update_multilabel_metric(
    metric: PRFMetric,
    *,
    gold_values: Iterable[Any],
    predicted_values: Iterable[Any],
) -> None:
    gold = _enum_values(gold_values)
    predicted = _enum_values(predicted_values)

    metric.add(
        true_positive=len(gold & predicted),
        false_positive=len(predicted - gold),
        false_negative=len(gold - predicted),
    )


def _span_key(
    span: AnalysisSpan,
) -> tuple[int, int, str]:
    return (
        span.start,
        span.end,
        span.text,
    )


def _update_exact_span_metric(
    metric: PRFMetric,
    *,
    gold_spans: list[AnalysisSpan],
    predicted_spans: list[AnalysisSpan],
) -> None:
    gold = Counter(
        _span_key(span)
        for span in gold_spans
    )
    predicted = Counter(
        _span_key(span)
        for span in predicted_spans
    )

    true_positive = sum(
        (gold & predicted).values()
    )

    metric.add(
        true_positive=true_positive,
        false_positive=(
            len(predicted_spans) - true_positive
        ),
        false_negative=(
            len(gold_spans) - true_positive
        ),
    )


def _interval_intersection(
    first: AnalysisSpan,
    second: AnalysisSpan,
) -> int:
    return max(
        0,
        min(first.end, second.end)
        - max(first.start, second.start),
    )


def _overlap_coefficient(
    first: AnalysisSpan,
    second: AnalysisSpan,
) -> float:
    """
    Character overlap divided by the shorter span length.

    This gives full credit to containment during alignment, so a model
    span such as “女性” can be aligned with a longer gold proposition
    containing that cue. Detection quality is still reported separately
    by the strict exact-span metric.
    """

    intersection = _interval_intersection(
        first,
        second,
    )

    if intersection == 0:
        return 0.0

    shorter_length = min(
        first.end - first.start,
        second.end - second.start,
    )

    return (
        intersection / shorter_length
        if shorter_length > 0
        else 0.0
    )


def _intersection_over_union(
    first: AnalysisSpan,
    second: AnalysisSpan,
) -> float:
    intersection = _interval_intersection(
        first,
        second,
    )

    if intersection == 0:
        return 0.0

    union = (
        max(first.end, second.end)
        - min(first.start, second.start)
    )

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def match_spans(
    *,
    gold_spans: list[AnalysisSpan],
    predicted_spans: list[AnalysisSpan],
    minimum_overlap: float = 0.5,
) -> list[tuple[AnalysisSpan, AnalysisSpan]]:
    """
    Greedily create one-to-one span matches.

    Candidates are ranked by:
    1. overlap coefficient;
    2. interval IoU;
    3. exact text equality;
    4. proximity of start offsets.
    """

    candidates: list[
        tuple[
            float,
            float,
            int,
            int,
            int,
            int,
        ]
    ] = []

    for gold_index, gold_span in enumerate(
        gold_spans
    ):
        for predicted_index, predicted_span in enumerate(
            predicted_spans
        ):
            overlap = _overlap_coefficient(
                gold_span,
                predicted_span,
            )

            if overlap < minimum_overlap:
                continue

            iou = _intersection_over_union(
                gold_span,
                predicted_span,
            )
            exact_text = int(
                gold_span.text == predicted_span.text
            )
            start_distance = abs(
                gold_span.start
                - predicted_span.start
            )

            candidates.append(
                (
                    overlap,
                    iou,
                    exact_text,
                    -start_distance,
                    gold_index,
                    predicted_index,
                )
            )

    candidates.sort(reverse=True)

    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    matches: list[
        tuple[AnalysisSpan, AnalysisSpan]
    ] = []

    for (
        _overlap,
        _iou,
        _exact_text,
        _negative_distance,
        gold_index,
        predicted_index,
    ) in candidates:
        if gold_index in used_gold:
            continue

        if predicted_index in used_predicted:
            continue

        used_gold.add(gold_index)
        used_predicted.add(predicted_index)

        matches.append(
            (
                gold_spans[gold_index],
                predicted_spans[predicted_index],
            )
        )

    return matches


def _index_prediction_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for record in records:
        instance_id = record.get("instance_id")

        if not isinstance(instance_id, str):
            raise ValueError(
                "Every prediction record must contain "
                "a string instance_id"
            )

        if instance_id in indexed:
            raise ValueError(
                "Duplicate prediction record for "
                f"{instance_id!r}"
            )

        indexed[instance_id] = record

    return indexed


def evaluate_analysis_predictions(
    *,
    instances: list[DatasetInstance],
    prediction_records: list[dict[str, Any]],
    minimum_overlap: float = 0.5,
) -> AnalysisEvaluationSummary:
    """
    Evaluate LLM analysis output against dataset gold analysis.

    Failed or missing predictions:
    - count as schema-invalid;
    - contribute all gold spans as false negatives;
    - count as incorrect for instance-level labels.
    """

    prediction_by_id = _index_prediction_records(
        prediction_records
    )
    dataset_ids = {
        instance.instance_id
        for instance in instances
    }

    extra_instance_ids = sorted(
        set(prediction_by_id) - dataset_ids
    )

    schema_validity = AccuracyMetric()
    analysis_exact_match = AccuracyMetric()

    exact_span = PRFMetric()
    overlap_span = PRFMetric()
    matched_span_coverage = AccuracyMetric()

    function_labels = PRFMetric()
    necessity_status = AccuracyMetric()
    bias_status = AccuracyMetric()
    bias_mechanisms = PRFMetric()
    span_stance = AccuracyMetric()

    speaker_stance = AccuracyMetric()
    context_sufficient = AccuracyMetric()

    missing_predictions = 0
    invalid_instance_ids: list[str] = []

    for instance in instances:
        record = prediction_by_id.get(
            instance.instance_id
        )
        gold_analysis = instance.gold_analysis
        predicted_analysis: AnalysisResult | None = None

        if record is None:
            missing_predictions += 1
        elif "predicted_analysis" in record:
            try:
                predicted_analysis = (
                    AnalysisResult.model_validate(
                        record["predicted_analysis"]
                    )
                )
            except ValidationError:
                predicted_analysis = None

        is_valid = predicted_analysis is not None
        schema_validity.add(correct=is_valid)

        if not is_valid:
            invalid_instance_ids.append(
                instance.instance_id
            )

        analysis_exact_match.add(
            correct=(
                is_valid
                and predicted_analysis
                == gold_analysis
            )
        )

        predicted_spans = (
            predicted_analysis.spans
            if predicted_analysis is not None
            else []
        )
        gold_spans = gold_analysis.spans

        _update_exact_span_metric(
            exact_span,
            gold_spans=gold_spans,
            predicted_spans=predicted_spans,
        )

        matches = match_spans(
            gold_spans=gold_spans,
            predicted_spans=predicted_spans,
            minimum_overlap=minimum_overlap,
        )

        overlap_span.add(
            true_positive=len(matches),
            false_positive=(
                len(predicted_spans) - len(matches)
            ),
            false_negative=(
                len(gold_spans) - len(matches)
            ),
        )

        for gold_span in gold_spans:
            matched_span_coverage.add(
                correct=any(
                    matched_gold is gold_span
                    for matched_gold, _ in matches
                )
            )

        for gold_span, predicted_span in matches:
            _update_multilabel_metric(
                function_labels,
                gold_values=gold_span.functions,
                predicted_values=(
                    predicted_span.functions
                ),
            )

            necessity_status.add(
                correct=(
                    predicted_span.necessity.status
                    == gold_span.necessity.status
                )
            )

            bias_status.add(
                correct=(
                    predicted_span.bias.status
                    == gold_span.bias.status
                )
            )

            _update_multilabel_metric(
                bias_mechanisms,
                gold_values=(
                    gold_span.bias.mechanisms
                ),
                predicted_values=(
                    predicted_span.bias.mechanisms
                ),
            )

            span_stance.add(
                correct=(
                    predicted_span.stance
                    == gold_span.stance
                )
            )

        speaker_stance.add(
            correct=(
                is_valid
                and predicted_analysis is not None
                and predicted_analysis.speaker_stance
                == gold_analysis.speaker_stance
            )
        )

        context_sufficient.add(
            correct=(
                is_valid
                and predicted_analysis is not None
                and predicted_analysis.context_sufficient
                == gold_analysis.context_sufficient
            )
        )

    return AnalysisEvaluationSummary(
        dataset_instances=len(instances),
        prediction_records=len(prediction_records),
        missing_predictions=missing_predictions,
        extra_predictions=len(extra_instance_ids),
        schema_validity=schema_validity,
        analysis_exact_match=analysis_exact_match,
        exact_span=exact_span,
        overlap_span=overlap_span,
        matched_span_coverage=matched_span_coverage,
        function_labels=function_labels,
        necessity_status=necessity_status,
        bias_status=bias_status,
        bias_mechanisms=bias_mechanisms,
        span_stance=span_stance,
        speaker_stance=speaker_stance,
        context_sufficient=context_sufficient,
        invalid_instance_ids=invalid_instance_ids,
        extra_instance_ids=extra_instance_ids,
    )
