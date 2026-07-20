#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 7/20/26
Description: test_models
"""
import pytest
from pydantic import ValidationError

from genfine.domain.enums import (
    Action,
    BiasStatus,
    EditScope,
    Explicitness,
    FunctionLabel,
    GenderCueType,
    GenderDimension,
    GenderSource,
    GenderValue,
    InstanceAction,
    Necessity,
    NecessityReason,
    ProtectedFactType,
    ReferentType,
    SourceReliability,
    SpeakerStance,
    TaskMode,
)
from genfine.domain.models import (
    AnalysisResult,
    AnalysisSpan,
    BiasAnnotation,
    DatasetInstance,
    DecisionAnnotation,
    NecessityAnnotation,
    ProtectedFact,
    Referent,
    SpanActionAnnotation,
    TextContext,
)


def build_valid_instance() -> DatasetInstance:
    text = "我姐姐打电话说她会回来"

    return DatasetInstance(
        instance_id="zh_000001",
        language="zh",
        task_mode=TaskMode.INCLUSIVE_REWRITING,
        context=TextContext(
            target_text=text,
            genre="personal_conversation",
            speaker="first_person",
            communicative_goal="report_event",
        ),
        gold_analysis=AnalysisResult(
            referents=[
                Referent(
                    referent_id="person_1",
                    referent_type=ReferentType.SPECIFIC_PERSON,
                    number="SINGULAR",
                    specificity="KNOWN",
                    identity_relevance=True,
                    description="说话者的姐姐",
                )
            ],
            spans=[
                AnalysisSpan(
                    span_id="s1",
                    text="姐姐",
                    start=1,
                    end=3,
                    cue_type=GenderCueType.KINSHIP_TERM,
                    explicitness=Explicitness.EXPLICIT,
                    gender_dimension=GenderDimension.SOCIAL_GENDER_ROLE,
                    gender_value=GenderValue.FEMALE,
                    referent_id="person_1",
                    source=GenderSource.EXPLICITLY_STATED,
                    source_reliability=SourceReliability.CONFIRMED,
                    functions=[
                        FunctionLabel.KINSHIP_RELATION,
                        FunctionLabel.IDENTIFICATION,
                    ],
                    necessity=NecessityAnnotation(
                        status=Necessity.ESSENTIAL,
                        reasons=[
                            NecessityReason.KINSHIP_RELATION,
                            NecessityReason.REFERENT_IDENTIFICATION,
                        ],
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.NONE,
                        mechanisms=[],
                    ),
                    stance=SpeakerStance.NEUTRAL_REPORT,
                ),
                AnalysisSpan(
                    span_id="s2",
                    text="她",
                    start=7,
                    end=8,
                    cue_type=GenderCueType.PRONOUN,
                    explicitness=Explicitness.EXPLICIT,
                    gender_dimension=GenderDimension.GENDER_REFERENCE,
                    gender_value=GenderValue.FEMALE,
                    referent_id="person_1",
                    source=GenderSource.CONTEXTUALLY_INFERRED,
                    source_reliability=SourceReliability.SUPPORTED,
                    functions=[
                        FunctionLabel.COREFERENCE,
                        FunctionLabel.TOPIC_CONTINUITY,
                    ],
                    necessity=NecessityAnnotation(
                        status=Necessity.RELEVANT,
                        reasons=[
                            NecessityReason.COREFERENCE_RESOLUTION,
                            NecessityReason.DISCOURSE_COHERENCE,
                        ],
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.NONE,
                        mechanisms=[],
                    ),
                    stance=SpeakerStance.NEUTRAL_REPORT,
                ),
            ],
            protected_facts=[
                ProtectedFact(
                    fact_id="f1",
                    fact_type=ProtectedFactType.KINSHIP_RELATION,
                    description="person_1 是说话者的姐姐",
                    source_span_ids=["s1"],
                    must_preserve=True,
                )
            ],
            speaker_stance=SpeakerStance.NEUTRAL_REPORT,
            context_sufficient=True,
        ),
        gold_decision=DecisionAnnotation(
            instance_action=InstanceAction.KEEP,
            edit_scope=EditScope.NONE,
            span_actions=[
                SpanActionAnnotation(
                    span_id="s1",
                    action=Action.KEEP,
                ),
                SpanActionAnnotation(
                    span_id="s2",
                    action=Action.KEEP,
                ),
            ],
        ),
        gold_output=text,
    )


def test_valid_instance_can_be_created() -> None:
    instance = build_valid_instance()

    assert instance.instance_id == "zh_000001"
    assert len(instance.gold_analysis.spans) == 2
    assert instance.gold_decision.instance_action == InstanceAction.KEEP


def test_instance_can_round_trip_through_json() -> None:
    instance = build_valid_instance()

    serialized = instance.model_dump_json()
    restored = DatasetInstance.model_validate_json(serialized)

    assert restored == instance

def test_span_end_must_be_greater_than_start() -> None:
    instance = build_valid_instance()

    with pytest.raises(
        ValidationError,
        match="span end must be greater than start",
    ):
        instance.gold_analysis.spans[0].end = 1


def test_span_text_must_match_offsets() -> None:
    instance = build_valid_instance()

    payload = instance.model_dump()

    # “姐姐”正确位置是 [1:3]，这里故意改成 [0:3]
    payload["gold_analysis"]["spans"][0]["start"] = 0

    with pytest.raises(
        ValidationError,
        match="span text mismatch",
    ):
        DatasetInstance.model_validate(payload)


def test_duplicate_span_ids_are_rejected() -> None:
    instance = build_valid_instance()

    duplicated_span = instance.gold_analysis.spans[1].model_copy(
        update={"span_id": "s1"}
    )

    with pytest.raises(
        ValidationError,
        match="duplicate span_id",
    ):
        AnalysisResult(
            referents=instance.gold_analysis.referents,
            spans=[
                instance.gold_analysis.spans[0],
                duplicated_span,
            ],
            protected_facts=[],
        )


def test_unknown_referent_is_rejected() -> None:
    instance = build_valid_instance()

    invalid_span = instance.gold_analysis.spans[0].model_copy(
        update={"referent_id": "missing_person"}
    )

    with pytest.raises(
        ValidationError,
        match="unknown referent",
    ):
        AnalysisResult(
            referents=[],
            spans=[invalid_span],
            protected_facts=[],
        )


def test_bias_none_cannot_have_bias_mechanisms() -> None:
    from genfine.domain.enums import BiasMechanism

    with pytest.raises(
        ValidationError,
        match="must be empty",
    ):
        BiasAnnotation(
            status=BiasStatus.NONE,
            mechanisms=[
                BiasMechanism.GENERIC_MALE_DEFAULT
            ],
        )


def test_abstain_instance_must_not_have_gold_output() -> None:
    instance = build_valid_instance()

    with pytest.raises(
        ValidationError,
        match="gold_output must be null",
    ):
        DatasetInstance(
            instance_id=instance.instance_id,
            language=instance.language,
            task_mode=instance.task_mode,
            context=instance.context,
            gold_analysis=instance.gold_analysis,
            gold_decision=DecisionAnnotation(
                instance_action=InstanceAction.ABSTAIN,
                edit_scope=EditScope.NONE,
                span_actions=[],
            ),
            gold_output="不应提供改写结果",
        )