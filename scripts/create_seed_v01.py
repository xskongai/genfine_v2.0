#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 7/20/26
Description: create_seed_v01
"""

from __future__ import annotations

import json
from pathlib import Path

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
    DatasetMetadata,
    DecisionAnnotation,
    NecessityAnnotation,
    ProtectedFact,
    Referent,
    SpanActionAnnotation,
    TextContext,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "seed" / "seed_v0.1.jsonl"


def build_kinship_keep_instance() -> DatasetInstance:
    """
    Necessary gender information:

    “姐姐” expresses a specific kinship relation.
    “她” maintains coreference with the same person.

    Both spans should be preserved.
    """

    text = "我姐姐打电话说她会回来"

    return DatasetInstance(
        instance_id="zh_seed_0001",
        language="zh",
        task_mode=TaskMode.INCLUSIVE_REWRITING,
        context=TextContext(
            preceding_context="",
            target_text=text,
            following_context="",
            genre="personal_conversation",
            speaker="first_person_speaker",
            audience="general",
            communicative_goal="report_a_personal_event",
            cultural_context="modern_chinese",
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
                        rationale=(
                            "删除或替换“姐姐”会丢失明确的亲属关系信息。"
                        ),
                        confidence=1.0,
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.NONE,
                        mechanisms=[],
                        rationale="具体、真实的亲属关系不构成性别偏见。",
                        confidence=1.0,
                    ),
                    stance=SpeakerStance.NEUTRAL_REPORT,
                    confidence=1.0,
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
                        rationale="“她”与前文“姐姐”形成自然、明确的指代关系。",
                        confidence=1.0,
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.NONE,
                        mechanisms=[],
                        rationale="代词与明确的具体人物指称一致。",
                        confidence=1.0,
                    ),
                    stance=SpeakerStance.NEUTRAL_REPORT,
                    confidence=1.0,
                ),
            ],
            protected_facts=[
                ProtectedFact(
                    fact_id="f1",
                    fact_type=ProtectedFactType.KINSHIP_RELATION,
                    description="person_1 是说话者的姐姐",
                    source_span_ids=["s1"],
                    must_preserve=True,
                    verification_hint="改写后仍应保留“姐姐”这一亲属关系。",
                ),
                ProtectedFact(
                    fact_id="f2",
                    fact_type=ProtectedFactType.COREFERENCE,
                    description="“她”和“姐姐”指向同一个人",
                    source_span_ids=["s1", "s2"],
                    must_preserve=True,
                    verification_hint="改写后不能改变两个表达的指称关系。",
                ),
            ],
            speaker_stance=SpeakerStance.NEUTRAL_REPORT,
            context_sufficient=True,
            analysis_notes=(
                "该句包含明确且必要的具体人物性别信息，不应机械中性化。"
            ),
        ),
        gold_decision=DecisionAnnotation(
            instance_action=InstanceAction.KEEP,
            edit_scope=EditScope.NONE,
            span_actions=[
                SpanActionAnnotation(
                    span_id="s1",
                    action=Action.KEEP,
                    reason_code="ESSENTIAL_KINSHIP_INFORMATION",
                    rationale="亲属关系属于受保护的事实信息。",
                ),
                SpanActionAnnotation(
                    span_id="s2",
                    action=Action.KEEP,
                    reason_code="VALID_SPECIFIC_COREFERENCE",
                    rationale="代词与明确的具体人物身份一致。",
                ),
            ],
        ),
        gold_output=text,
        metadata=DatasetMetadata(
            source_type="controlled_natural_example",
            scenario_category="necessary_kinship_and_coreference",
            annotation_version="v0.1",
            notes="GenFINE 第一条种子数据。",
        ),
    )


def write_jsonl(
    instances: list[DatasetInstance],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instance_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8") as file:
        for instance in instances:
            if instance.instance_id in instance_ids:
                raise ValueError(
                    f"Duplicate instance_id: {instance.instance_id}"
                )

            instance_ids.add(instance.instance_id)

            payload = instance.model_dump(
                mode="json",
                exclude_none=True,
            )

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def main() -> None:
    instances = [
        build_kinship_keep_instance(),
    ]

    write_jsonl(instances, OUTPUT_PATH)

    print(
        f"Wrote {len(instances)} instance(s) to "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()