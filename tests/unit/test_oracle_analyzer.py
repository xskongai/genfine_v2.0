from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.enums import FunctionLabel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.1.jsonl"
)


def test_oracle_analyzer_name() -> None:
    analyzer = OracleAnalyzer()

    assert analyzer.name == "oracle"


def test_oracle_returns_gold_analysis() -> None:
    instance = load_dataset(SEED_PATH)[0]
    analyzer = OracleAnalyzer()

    predicted = analyzer.analyze(instance)

    assert predicted == instance.gold_analysis


def test_oracle_returns_a_different_object() -> None:
    instance = load_dataset(SEED_PATH)[0]
    analyzer = OracleAnalyzer()

    predicted = analyzer.analyze(instance)

    assert predicted is not instance.gold_analysis
    assert (
        predicted.spans[0]
        is not instance.gold_analysis.spans[0]
    )


def test_modifying_prediction_does_not_modify_gold() -> None:
    instance = load_dataset(SEED_PATH)[0]
    analyzer = OracleAnalyzer()

    predicted = analyzer.analyze(instance)

    original_confidence = (
        instance.gold_analysis.spans[0].confidence
    )

    predicted.spans[0].confidence = 0.25
    predicted.spans[0].functions.append(
        FunctionLabel.UNKNOWN_FUNCTION
    )

    assert (
        instance.gold_analysis.spans[0].confidence
        == original_confidence
    )

    assert (
        FunctionLabel.UNKNOWN_FUNCTION
        not in instance.gold_analysis.spans[0].functions
    )


def test_analyze_many_preserves_order() -> None:
    instances = load_dataset(SEED_PATH)
    analyzer = OracleAnalyzer()

    predictions = analyzer.analyze_many(
        instances
    )

    assert len(predictions) == len(instances)

    for prediction, instance in zip(
        predictions,
        instances,
        strict=True,
    ):
        assert prediction == instance.gold_analysis


def test_analyze_many_accepts_generator() -> None:
    instances = load_dataset(SEED_PATH)
    analyzer = OracleAnalyzer()

    predictions = analyzer.analyze_many(
        instance
        for instance in instances
    )

    assert len(predictions) == len(instances)


def test_analyze_many_handles_empty_input() -> None:
    analyzer = OracleAnalyzer()

    predictions = analyzer.analyze_many([])

    assert predictions == []