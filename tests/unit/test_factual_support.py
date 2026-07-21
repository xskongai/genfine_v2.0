from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from genfine.domain.enums import Action
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)
from genfine.factual_support import (
    ClaimKind,
    FactualClaimAssessment,
    FactualSupportJudge,
    FactualSupportLabel,
    FactualSupportResult,
    FactualSupportStatus,
)


def build_supported_claim() -> FactualClaimAssessment:
    return FactualClaimAssessment(
        claim_id="c1",
        claim="她是本院第一位女性院长",
        kind=ClaimKind.SPECIFIC_ENTITY_FACT,
        label=FactualSupportLabel.SOURCE_SUPPORTED,
        evidence=[
            "她是本院第一位女性院长",
        ],
        relevant_actions=[
            Action.KEEP,
        ],
        rationale="该事实在输入中直接出现。",
        confidence=1.0,
    )


def test_source_supported_claim_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        FactualClaimAssessment(
            claim_id="c1",
            claim="她是本院第一位女性院长",
            kind=ClaimKind.SPECIFIC_ENTITY_FACT,
            label=(
                FactualSupportLabel.SOURCE_SUPPORTED
            ),
            evidence=[],
            relevant_actions=[
                Action.KEEP,
            ],
            rationale="声称有来源支持，但没有提供证据。",
            confidence=1.0,
        )


def test_licensed_reframing_is_not_unsupported() -> None:
    claim = FactualClaimAssessment(
        claim_id="c1",
        claim="领导适任性不应由性别决定",
        kind=ClaimKind.NORMATIVE_REFRAMING,
        label=FactualSupportLabel.LICENSED_REFRAMING,
        evidence=[],
        relevant_actions=[
            Action.REFRAME_PROPOSITION,
        ],
        rationale=(
            "这是去除性别本质化判断所允许的纠偏表达。"
        ),
        confidence=0.95,
    )

    result = FactualSupportResult(
        instance_id="example_1",
        status=FactualSupportStatus.EVALUATED,
        claims=[claim],
        unsupported_factual_insertion=False,
        judge_name="fake/factual-support",
        prompt_version="0.1",
    )

    assert not result.unsupported_factual_insertion
    assert result.unsupported_claim_count == 0


def test_unsupported_claim_sets_instance_flag() -> None:
    claim = FactualClaimAssessment(
        claim_id="c1",
        claim="她为学院发展作出了重要贡献",
        kind=ClaimKind.SPECIFIC_ENTITY_FACT,
        label=(
            FactualSupportLabel
            .UNSUPPORTED_FACTUAL_INSERTION
        ),
        evidence=[],
        relevant_actions=[
            Action.REFRAME_PROPOSITION,
        ],
        rationale=(
            "输入与上下文均未提供该人物作出该贡献的证据，"
            "重构动作也不许可增加具体成就。"
        ),
        confidence=1.0,
    )

    result = FactualSupportResult(
        instance_id="zh_seed_0013",
        status=FactualSupportStatus.EVALUATED,
        claims=[claim],
        unsupported_factual_insertion=True,
        judge_name="fake/factual-support",
        prompt_version="0.1",
    )

    assert result.unsupported_factual_insertion
    assert result.unsupported_claim_count == 1


def test_inconsistent_instance_flag_is_rejected() -> None:
    unsupported_claim = FactualClaimAssessment(
        claim_id="c1",
        claim="她的领导能力出色",
        kind=ClaimKind.SPECIFIC_ENTITY_FACT,
        label=(
            FactualSupportLabel
            .UNSUPPORTED_FACTUAL_INSERTION
        ),
        evidence=[],
        relevant_actions=[
            Action.REFRAME_PROPOSITION,
        ],
        rationale="输入没有支持该人物能力评价。",
        confidence=1.0,
    )

    with pytest.raises(ValidationError):
        FactualSupportResult(
            instance_id="zh_seed_0013",
            status=FactualSupportStatus.EVALUATED,
            claims=[unsupported_claim],
            unsupported_factual_insertion=False,
            judge_name="fake/factual-support",
            prompt_version="0.1",
        )


def test_no_output_result_is_not_applicable() -> None:
    result = FactualSupportResult.no_output(
        instance_id="zh_seed_0012",
        judge_name="fake/factual-support",
        prompt_version="0.1",
    )

    assert (
        result.status
        == FactualSupportStatus.NOT_APPLICABLE_NO_OUTPUT
    )
    assert result.claims == []
    assert not result.unsupported_factual_insertion


class FakeFactualSupportJudge(FactualSupportJudge):
    @property
    def name(self) -> str:
        return "fake/factual-support"

    @property
    def prompt_version(self) -> str:
        return "0.1"

    def evaluate(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> FactualSupportResult:
        del analysis
        del edit_plan

        if output_text is None:
            return FactualSupportResult.no_output(
                instance_id=instance.instance_id,
                judge_name=self.name,
                prompt_version=self.prompt_version,
            )

        return FactualSupportResult(
            instance_id=instance.instance_id,
            status=FactualSupportStatus.EVALUATED,
            claims=[
                build_supported_claim(),
            ],
            unsupported_factual_insertion=False,
            judge_name=self.name,
            prompt_version=self.prompt_version,
        )


def test_fake_judge_implements_contract() -> None:
    judge = FakeFactualSupportJudge()

    instance = cast(
        DatasetInstance,
        type(
            "FakeInstance",
            (),
            {"instance_id": "example_1"},
        )(),
    )

    result = judge.evaluate(
        instance=instance,
        analysis=cast(AnalysisResult, object()),
        edit_plan=cast(EditPlan, object()),
        output_text="她是本院第一位女性院长",
    )

    assert result.instance_id == "example_1"
    assert result.judge_name == judge.name
    assert len(result.claims) == 1


def test_evaluated_result_requires_claims() -> None:
    with pytest.raises(ValidationError):
        FactualSupportResult(
            instance_id="example_1",
            status=FactualSupportStatus.EVALUATED,
            claims=[],
            unsupported_factual_insertion=False,
            judge_name="fake/factual-support",
            prompt_version="0.1",
        )


def test_claim_ids_must_be_unique() -> None:
    claim_1 = build_supported_claim()

    claim_2 = claim_1.model_copy(
        update={
            "claim": "另一条事实",
        }
    )

    with pytest.raises(ValidationError):
        FactualSupportResult(
            instance_id="example_1",
            status=FactualSupportStatus.EVALUATED,
            claims=[
                claim_1,
                claim_2,
            ],
            unsupported_factual_insertion=False,
            judge_name="fake/factual-support",
            prompt_version="0.1",
        )