from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.pipeline import EditPlanBuilder
from genfine.policy import DecisionEngine
from genfine.verification import OutputVerifier


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


def build_inputs(index: int):
    instance = load_dataset(SEED_PATH)[index]

    analysis = OracleAnalyzer().analyze(
        instance
    )

    decisions = (
        DecisionEngine
        .from_yaml(RULE_PATH)
        .decide_analysis(
            analysis=analysis,
            task_mode=instance.task_mode,
        )
    )

    edit_plan = EditPlanBuilder().build(
        instance=instance,
        analysis=analysis,
        decisions=decisions,
    )

    return instance, analysis, edit_plan


def issue_codes(result) -> set[str]:
    return {
        issue.code
        for issue in result.issues
    }


def test_all_gold_outputs_pass_verification() -> None:
    instances = load_dataset(SEED_PATH)
    verifier = OutputVerifier.default()

    for index, instance in enumerate(instances):
        current, analysis, plan = build_inputs(index)

        result = verifier.verify(
            instance=current,
            analysis=analysis,
            edit_plan=plan,
            output_text=current.gold_output,
        )

        assert result.passed, (
            instance.instance_id,
            result.issues,
        )


def test_keep_change_is_detected() -> None:
    instance, analysis, plan = build_inputs(0)

    result = OutputVerifier.default().verify(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
        output_text="我的手足打电话说TA会回来",
    )

    assert not result.passed
    assert "KEEP_TEXT_CHANGED" in issue_codes(
        result
    )


def test_missing_protected_kinship_is_detected() -> None:
    instance, analysis, plan = build_inputs(0)

    result = OutputVerifier.default().verify(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
        output_text="她打电话说她会回来",
    )

    assert not result.protected_facts_preserved

    assert (
        "PROTECTED_PHRASE_MISSING"
        in issue_codes(result)
    )


def test_unapplied_generic_male_edit_is_detected() -> None:
    instance, analysis, plan = build_inputs(2)

    result = OutputVerifier.default().verify(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
        output_text=instance.context.target_text,
    )

    assert not result.action_compliant

    codes = issue_codes(result)

    assert "EDIT_NOT_APPLIED" in codes
    assert "EDIT_SPAN_REMAINS" in codes


def test_unsupported_gender_insertion_is_detected() -> None:
    instance, analysis, plan = build_inputs(4)

    output = "外科医生完成手术后，他离开了手术室"

    result = OutputVerifier.default().verify(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
        output_text=output,
    )

    assert not result.passed
    assert result.unsupported_gender_inserted

    assert (
        "UNSUPPORTED_GENDER_INSERTION"
        in issue_codes(result)
    )