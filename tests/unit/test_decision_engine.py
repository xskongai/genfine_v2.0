from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.enums import Action
from genfine.policy import DecisionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def build_engine() -> DecisionEngine:
    return DecisionEngine.from_yaml(
        RULE_PATH
    )


def test_rule_file_loads() -> None:
    engine = build_engine()

    assert engine.version == "0.1"
    assert len(engine.rules) >= 1


def test_rules_are_sorted_by_priority() -> None:
    engine = build_engine()

    priorities = [
        rule.priority
        for rule in engine.rules
    ]

    assert priorities == sorted(
        priorities,
        reverse=True,
    )


def test_all_seed_span_actions_match_gold() -> None:
    instances = load_dataset(SEED_PATH)

    analyzer = OracleAnalyzer()
    engine = build_engine()

    for instance in instances:
        analysis = analyzer.analyze(instance)

        predicted = engine.decide_analysis(
            analysis=analysis,
            task_mode=instance.task_mode,
        )

        gold_by_span = {
            item.span_id: item.action
            for item
            in instance.gold_decision.span_actions
        }

        predicted_by_span = {
            item.span_id: item.action
            for item in predicted
        }

        assert predicted_by_span == gold_by_span, (
            f"Decision mismatch for "
            f"{instance.instance_id}: "
            f"predicted={predicted_by_span}, "
            f"gold={gold_by_span}"
        )


def test_generic_male_rule() -> None:
    instance = load_dataset(SEED_PATH)[2]
    analysis = OracleAnalyzer().analyze(
        instance
    )

    decision = build_engine().decide_span(
        span=analysis.spans[0],
        analysis=analysis,
        task_mode=instance.task_mode,
    )

    assert (
        decision.action
        == Action.REPLACE_GENERIC_FORM
    )
    assert decision.rule_id == "replace_generic_male"
    assert (
        decision.reason_code
        == "GENERIC_MALE_DEFAULT"
    )


def test_rejected_quote_rule() -> None:
    instance = load_dataset(SEED_PATH)[3]
    analysis = OracleAnalyzer().analyze(
        instance
    )

    decision = build_engine().decide_span(
        span=analysis.spans[0],
        analysis=analysis,
        task_mode=instance.task_mode,
    )

    assert (
        decision.action
        == Action.KEEP_WITH_ATTRIBUTION
    )

    assert (
        decision.rule_id
        == "preserve_rejected_biased_quotation"
    )


def test_preserve_ambiguity_rule() -> None:
    instance = load_dataset(SEED_PATH)[4]
    analysis = OracleAnalyzer().analyze(
        instance
    )

    decision = build_engine().decide_span(
        span=analysis.spans[0],
        analysis=analysis,
        task_mode=instance.task_mode,
    )

    assert (
        decision.action
        == Action.PRESERVE_AMBIGUITY
    )

    assert (
        decision.reason_code
        == "NO_GROUNDED_GENDER_INFORMATION"
    )


def test_protected_fact_becomes_constraint() -> None:
    instance = load_dataset(SEED_PATH)[0]
    analysis = OracleAnalyzer().analyze(
        instance
    )

    decision = build_engine().decide_span(
        span=analysis.spans[0],
        analysis=analysis,
        task_mode=instance.task_mode,
    )

    assert any(
        "person_1 是说话者的姐姐"
        in constraint
        for constraint in decision.constraints
    )


def test_engine_does_not_modify_analysis() -> None:
    instance = load_dataset(SEED_PATH)[0]
    analysis = OracleAnalyzer().analyze(
        instance
    )

    before = analysis.model_dump()

    build_engine().decide_analysis(
        analysis=analysis,
        task_mode=instance.task_mode,
    )

    after = analysis.model_dump()

    assert before == after