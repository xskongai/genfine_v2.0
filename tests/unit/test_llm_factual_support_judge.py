from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from genfine.domain.enums import Action
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)
from genfine.factual_support import (
    FactualSupportJudgeError,
    FactualSupportLabel,
    FactualSupportPromptBuilder,
    FactualSupportStatus,
    LLMFactualSupportJudge,
)


class FakeTextClient:
    model = "fake-model"

    def __init__(
        self,
        *,
        response: str = "",
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.call_count = 0

    def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        del instructions
        del input_text

        self.call_count += 1

        if self.error is not None:
            raise self.error

        return self.response


class StubPromptBuilder:
    version = "0.1"
    system_instruction = "Evaluate factual support."

    def build_input(
        self,
        *,
        instance,
        analysis,
        edit_plan,
        output_text,
    ) -> str:
        del analysis
        del edit_plan

        return json.dumps(
            {
                "instance_id": instance.instance_id,
                "rewritten_output": output_text,
            },
            ensure_ascii=False,
        )


def build_instance() -> DatasetInstance:
    return cast(
        DatasetInstance,
        SimpleNamespace(
            instance_id="zh_seed_0013",
        ),
    )


def build_analysis() -> AnalysisResult:
    return cast(
        AnalysisResult,
        SimpleNamespace(),
    )


def build_edit_plan(
    *actions: Action,
) -> EditPlan:
    return cast(
        EditPlan,
        SimpleNamespace(
            span_decisions=[
                SimpleNamespace(
                    action=action,
                )
                for action in actions
            ]
        ),
    )


def build_judge(
    response: str,
) -> tuple[
    LLMFactualSupportJudge,
    FakeTextClient,
]:
    client = FakeTextClient(
        response=response
    )

    judge = LLMFactualSupportJudge(
        client=client,
        prompt_builder=cast(
            FactualSupportPromptBuilder,
            StubPromptBuilder(),
        ),
    )

    return judge, client


def valid_supported_payload() -> dict:
    return {
        "instance_id": "zh_seed_0013",
        "status": "EVALUATED",
        "claims": [
            {
                "claim_id": "c1",
                "claim": (
                    "她是本院第一位女性院长"
                ),
                "kind": "SPECIFIC_ENTITY_FACT",
                "label": "SOURCE_SUPPORTED",
                "evidence": [
                    "她是本院第一位女性院长"
                ],
                "relevant_actions": [
                    "KEEP"
                ],
                "rationale": (
                    "该事实在原文中直接出现。"
                ),
                "confidence": 1.0,
            }
        ],
        "unsupported_factual_insertion": False,
    }


def test_valid_result_is_parsed() -> None:
    judge, client = build_judge(
        json.dumps(
            valid_supported_payload(),
            ensure_ascii=False,
        )
    )

    result = judge.evaluate(
        instance=build_instance(),
        analysis=build_analysis(),
        edit_plan=build_edit_plan(
            Action.KEEP,
        ),
        output_text=(
            "她是本院第一位女性院长"
        ),
    )

    assert client.call_count == 1
    assert result.instance_id == "zh_seed_0013"
    assert (
        result.status
        == FactualSupportStatus.EVALUATED
    )
    assert result.judge_name == (
        "llm-factual-support/fake-model"
    )
    assert result.prompt_version == "0.1"
    assert (
        result.claims[0].label
        == FactualSupportLabel.SOURCE_SUPPORTED
    )


def test_markdown_fence_is_removed() -> None:
    payload = json.dumps(
        valid_supported_payload(),
        ensure_ascii=False,
    )

    judge, _ = build_judge(
        f"```json\n{payload}\n```"
    )

    result = judge.evaluate(
        instance=build_instance(),
        analysis=build_analysis(),
        edit_plan=build_edit_plan(
            Action.KEEP,
        ),
        output_text=(
            "她是本院第一位女性院长"
        ),
    )

    assert len(result.claims) == 1


def test_no_output_skips_model_call() -> None:
    judge, client = build_judge("")

    result = judge.evaluate(
        instance=build_instance(),
        analysis=build_analysis(),
        edit_plan=build_edit_plan(
            Action.ABSTAIN,
        ),
        output_text=None,
    )

    assert client.call_count == 0
    assert (
        result.status
        == FactualSupportStatus
        .NOT_APPLICABLE_NO_OUTPUT
    )


def test_invalid_json_raises_error() -> None:
    judge, _ = build_judge(
        "not valid json"
    )

    with pytest.raises(
        FactualSupportJudgeError,
        match="invalid factual-support JSON",
    ):
        judge.evaluate(
            instance=build_instance(),
            analysis=build_analysis(),
            edit_plan=build_edit_plan(
                Action.KEEP,
            ),
            output_text="测试输出",
        )


def test_instance_id_mismatch_is_rejected() -> None:
    payload = valid_supported_payload()
    payload["instance_id"] = "wrong_id"

    judge, _ = build_judge(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    with pytest.raises(
        FactualSupportJudgeError,
        match="instance_id mismatch",
    ):
        judge.evaluate(
            instance=build_instance(),
            analysis=build_analysis(),
            edit_plan=build_edit_plan(
                Action.KEEP,
            ),
            output_text="测试输出",
        )


def test_inconsistent_unsupported_flag_is_rejected() -> None:
    payload = valid_supported_payload()

    payload["claims"][0]["label"] = (
        "UNSUPPORTED_FACTUAL_INSERTION"
    )

    payload[
        "unsupported_factual_insertion"
    ] = False

    judge, _ = build_judge(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    with pytest.raises(
        FactualSupportJudgeError,
        match="invalid factual-support result",
    ):
        judge.evaluate(
            instance=build_instance(),
            analysis=build_analysis(),
            edit_plan=build_edit_plan(
                Action.KEEP,
            ),
            output_text="测试输出",
        )


def test_action_not_present_in_plan_is_rejected() -> None:
    payload = valid_supported_payload()

    payload["claims"][0]["label"] = (
        "LICENSED_REFRAMING"
    )
    payload["claims"][0]["kind"] = (
        "NORMATIVE_REFRAMING"
    )
    payload["claims"][0]["evidence"] = []
    payload["claims"][0][
        "relevant_actions"
    ] = [
        "REFRAME_PROPOSITION"
    ]

    judge, _ = build_judge(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    with pytest.raises(
        FactualSupportJudgeError,
        match="not present in the EditPlan",
    ):
        judge.evaluate(
            instance=build_instance(),
            analysis=build_analysis(),
            edit_plan=build_edit_plan(
                Action.KEEP,
            ),
            output_text=(
                "领导能力不应由性别决定。"
            ),
        )


def test_client_failure_is_wrapped() -> None:
    client = FakeTextClient(
        error=RuntimeError(
            "temporary API failure"
        )
    )

    judge = LLMFactualSupportJudge(
        client=client,
        prompt_builder=cast(
            FactualSupportPromptBuilder,
            StubPromptBuilder(),
        ),
    )

    with pytest.raises(
        FactualSupportJudgeError,
        match="model request failed",
    ):
        judge.evaluate(
            instance=build_instance(),
            analysis=build_analysis(),
            edit_plan=build_edit_plan(
                Action.KEEP,
            ),
            output_text="测试输出",
        )
def test_source_supported_claim_does_not_require_action_match() -> None:
    """
    A SOURCE_SUPPORTED claim is justified by source evidence.

    An unnecessary instance-level KEEP reference must not invalidate
    the claim when the actual span action is KEEP_WITH_ATTRIBUTION.
    """

    payload = valid_supported_payload()

    payload["claims"][0]["relevant_actions"] = [
        "KEEP"
    ]

    judge, _ = build_judge(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    result = judge.evaluate(
        instance=build_instance(),
        analysis=build_analysis(),
        edit_plan=build_edit_plan(
            Action.KEEP_WITH_ATTRIBUTION,
        ),
        output_text=(
            "她是本院第一位女性院长"
        ),
    )

    assert (
        result.claims[0].label
        == FactualSupportLabel.SOURCE_SUPPORTED
    )


def test_licensed_reframing_requires_action_reference() -> None:
    payload = valid_supported_payload()

    payload["claims"][0].update(
        {
            "claim": "领导能力不应由性别决定。",
            "kind": "NORMATIVE_REFRAMING",
            "label": "LICENSED_REFRAMING",
            "evidence": [],
            "relevant_actions": [],
            "rationale": (
                "这是去除性别本质化判断的重构。"
            ),
        }
    )

    judge, _ = build_judge(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    with pytest.raises(
        FactualSupportJudgeError,
        match="does not cite any relevant EditPlan action",
    ):
        judge.evaluate(
            instance=build_instance(),
            analysis=build_analysis(),
            edit_plan=build_edit_plan(
                Action.REFRAME_PROPOSITION,
            ),
            output_text=(
                "领导能力不应由性别决定。"
            ),
        )


def test_non_factual_paraphrase_action_must_exist_in_plan() -> None:
    payload = valid_supported_payload()

    payload["claims"][0].update(
        {
            "claim": "每位学生都应提交自己的作业。",
            "kind": "NON_FACTUAL_LANGUAGE",
            "label": "NON_FACTUAL_PARAPHRASE",
            "evidence": [
                "每位学生都应提交他的作业"
            ],
            "relevant_actions": [
                "REPLACE_GENERIC_FORM"
            ],
            "rationale": (
                "使用中性反身形式替换泛指男性形式。"
            ),
        }
    )

    judge, _ = build_judge(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    with pytest.raises(
        FactualSupportJudgeError,
        match="not present in the EditPlan",
    ):
        judge.evaluate(
            instance=build_instance(),
            analysis=build_analysis(),
            edit_plan=build_edit_plan(
                Action.KEEP,
            ),
            output_text=(
                "每位学生都应提交自己的作业。"
            ),
        )