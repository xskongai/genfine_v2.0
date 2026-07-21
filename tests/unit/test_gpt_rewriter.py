from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset
from genfine.domain.enums import InstanceAction
from genfine.generation import (
    GPTRewriter,
    RewritePromptBuilder,
)
from genfine.pipeline import EditPlanBuilder
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

PROMPT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prompts"
    / "rewrite_v0.1.yaml"
)


class FakeOpenAITextClient:
    def __init__(
        self,
        output: str,
    ) -> None:
        self.output = output
        self.model = "fake-model"
        self.call_count = 0

        self.last_instructions: str | None = None
        self.last_input_text: str | None = None

    def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        self.call_count += 1
        self.last_instructions = instructions
        self.last_input_text = input_text

        return self.output


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

    plan = EditPlanBuilder().build(
        instance=instance,
        analysis=analysis,
        decisions=decisions,
    )

    return instance, analysis, plan


def build_rewriter(
    client: FakeOpenAITextClient,
) -> GPTRewriter:
    return GPTRewriter(
        client=client,  # type: ignore[arg-type]
        prompt_builder=(
            RewritePromptBuilder.from_yaml(
                PROMPT_PATH
            )
        ),
    )


def test_keep_plan_skips_api_call() -> None:
    instance, analysis, plan = (
        build_inputs(0)
    )

    client = FakeOpenAITextClient(
        "不应使用该输出"
    )

    output = build_rewriter(
        client
    ).rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
    )

    assert (
        plan.instance_action
        == InstanceAction.KEEP
    )

    assert output == instance.context.target_text
    assert client.call_count == 0


def test_edit_plan_calls_api_once() -> None:
    instance, analysis, plan = (
        build_inputs(2)
    )

    client = FakeOpenAITextClient(
        "每位学生都应提交自己的作业"
    )

    output = build_rewriter(
        client
    ).rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
    )

    assert (
        plan.instance_action
        == InstanceAction.EDIT
    )

    assert output == "每位学生都应提交自己的作业"
    assert client.call_count == 1


def test_prompt_contains_edit_action() -> None:
    instance, analysis, plan = (
        build_inputs(2)
    )

    client = FakeOpenAITextClient(
        "每位学生都应提交自己的作业"
    )

    build_rewriter(
        client
    ).rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
    )

    assert client.last_input_text is not None

    assert (
        "REPLACE_GENERIC_FORM"
        in client.last_input_text
    )

    assert (
        "GENERIC_MALE_DEFAULT"
        in client.last_input_text
    )

    assert (
        "每位学生都应提交他的作业"
        in client.last_input_text
    )


def test_prompt_contains_protected_facts() -> None:
    instance, analysis, plan = (
        build_inputs(2)
    )

    client = FakeOpenAITextClient(
        "每位学生都应提交自己的作业"
    )

    build_rewriter(
        client
    ).rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
    )

    assert client.last_input_text is not None

    assert (
        "每位学生都应提交属于自己的作业"
        in client.last_input_text
    )


def test_output_prefix_is_removed() -> None:
    instance, analysis, plan = (
        build_inputs(2)
    )

    client = FakeOpenAITextClient(
        "改写结果：每位学生都应提交自己的作业"
    )

    output = build_rewriter(
        client
    ).rewrite(
        instance=instance,
        analysis=analysis,
        edit_plan=plan,
    )

    assert output == "每位学生都应提交自己的作业"