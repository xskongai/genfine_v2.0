from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from genfine.domain.enums import (
    Action,
    EditScope,
    InstanceAction,
)
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)
from genfine.factual_support import (
    FactualSupportPromptBuilder,
    FactualSupportPromptError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prompts"
    / "factual_support_v0.1.yaml"
)


def build_fake_inputs():
    instance = SimpleNamespace(
        instance_id="example_1",
        language="zh",
        context=SimpleNamespace(
            preceding_context="前文介绍了张教授的任职经历。",
            target_text=(
                "她是本院第一位女性院长，"
                "因此女性天生更擅长管理"
            ),
            following_context="",
        ),
    )

    analysis = SimpleNamespace(
        spans=[
            SimpleNamespace(
                span_id="s1",
                text="第一位女性院长",
            ),
            SimpleNamespace(
                span_id="s2",
                text="女性天生更擅长管理",
            ),
        ]
    )

    protected_fact = SimpleNamespace(
        model_dump=lambda **_: {
            "fact_id": "f1",
            "description": "她是本院第一位女性院长",
            "must_preserve": True,
        }
    )

    edit_plan = SimpleNamespace(
        instance_action=InstanceAction.SPAN_LEVEL_EDIT,
        edit_scope=EditScope.SPAN,
        span_decisions=[
            SimpleNamespace(
                span_id="s1",
                action=Action.KEEP,
                reason_code=(
                    "HISTORICAL_GENDER_INFORMATION"
                ),
                constraints=[
                    "Preserve the historical status."
                ],
            ),
            SimpleNamespace(
                span_id="s2",
                action=Action.REFRAME_PROPOSITION,
                reason_code=(
                    "ENDORSED_GENDER_STEREOTYPE"
                ),
                constraints=[
                    "Remove the gender generalization."
                ],
            ),
        ],
        protected_facts=[
            protected_fact,
        ],
        global_constraints=[
            "Preserve supported facts.",
            "Do not invent new facts.",
        ],
    )

    return (
        cast(DatasetInstance, instance),
        cast(AnalysisResult, analysis),
        cast(EditPlan, edit_plan),
    )


def test_prompt_builder_loads_yaml() -> None:
    builder = FactualSupportPromptBuilder.from_yaml(
        PROMPT_PATH
    )

    assert builder.version == "0.1"
    assert builder.system_instruction
    assert (
        "transformation plan"
        in builder.system_instruction.lower()
    )


def test_prompt_separates_evidence_from_plan() -> None:
    builder = FactualSupportPromptBuilder.from_yaml(
        PROMPT_PATH
    )

    instance, analysis, edit_plan = build_fake_inputs()

    raw_prompt = builder.build_input(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
        output_text=(
            "她是本院第一位女性院长，"
            "她的领导能力出色。"
        ),
    )

    payload = json.loads(raw_prompt)

    assert payload["source_evidence"]["target_text"] == (
        "她是本院第一位女性院长，"
        "因此女性天生更擅长管理"
    )

    assert (
        payload["evidence_policy"]
        ["transformation_plan_is_not_factual_evidence"]
        is True
    )

    actions = {
        item["action"]
        for item in (
            payload["transformation_plan"]
            ["action_licenses"]
        )
    }

    assert actions == {
        "KEEP",
        "REFRAME_PROPOSITION",
    }


def test_prompt_contains_output_and_schema() -> None:
    builder = FactualSupportPromptBuilder.from_yaml(
        PROMPT_PATH
    )

    instance, analysis, edit_plan = build_fake_inputs()

    raw_prompt = builder.build_input(
        instance=instance,
        analysis=analysis,
        edit_plan=edit_plan,
        output_text="领导能力不应由性别决定。",
    )

    payload = json.loads(raw_prompt)

    assert payload["rewritten_output"] == (
        "领导能力不应由性别决定。"
    )

    schema = payload["required_output_schema"]

    assert schema["instance_id"] == "example_1"
    assert schema["status"] == "EVALUATED"
    assert "claims" in schema
    assert (
        "unsupported_factual_insertion"
        in schema
    )


def test_prompt_rejects_missing_output() -> None:
    builder = FactualSupportPromptBuilder.from_yaml(
        PROMPT_PATH
    )

    instance, analysis, edit_plan = build_fake_inputs()

    with pytest.raises(
        FactualSupportPromptError,
        match="without output_text",
    ):
        builder.build_input(
            instance=instance,
            analysis=analysis,
            edit_plan=edit_plan,
            output_text=None,
        )