
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
                    required_output_phrases=["姐姐"],
                    verification_hint="改写后仍应保留“姐姐”这一亲属关系。",
                ),
                ProtectedFact(
                    fact_id="f2",
                    fact_type=ProtectedFactType.COREFERENCE,
                    description="“她”和“姐姐”指向同一个人",
                    source_span_ids=["s1", "s2"],
                    must_preserve=True,
                    required_output_phrases=["姐姐", "她"],
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

def build_historical_visibility_instance() -> DatasetInstance:
    """
    Historical and representational gender information should be preserved.

    Removing “女性” would erase the fact that this appointment was a
    historical first for women in the institution.
    """

    text = "她是本院第一位女性院士"

    return DatasetInstance(
        instance_id="zh_seed_0002",
        language="zh",
        task_mode=TaskMode.INCLUSIVE_REWRITING,
        context=TextContext(
            preceding_context="前文正在介绍李教授的学术经历。",
            target_text=text,
            following_context="",
            genre="institutional_biography",
            speaker="institutional_author",
            audience="general_public",
            communicative_goal="describe_a_historical_achievement",
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
                    description="前文提到的李教授",
                )
            ],
            spans=[
                AnalysisSpan(
                    span_id="s1",
                    text="她",
                    start=0,
                    end=1,
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
                        rationale="“她”与前文具体人物形成指代关系。",
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.NONE,
                        mechanisms=[],
                        rationale="具体人物的准确代词不构成偏见。",
                    ),
                    stance=SpeakerStance.NEUTRAL_REPORT,
                ),
                AnalysisSpan(
                    span_id="s2",
                    text="第一位女性院士",
                    start=4,
                    end=11,
                    cue_type=GenderCueType.OCCUPATION_MARKER,
                    explicitness=Explicitness.EXPLICIT,
                    gender_dimension=GenderDimension.SOCIAL_GENDER_ROLE,
                    gender_value=GenderValue.FEMALE,
                    referent_id="person_1",
                    source=GenderSource.DOCUMENTED_RECORD,
                    source_reliability=SourceReliability.CONFIRMED,
                    functions=[
                        FunctionLabel.HISTORICAL_SIGNIFICANCE,
                        FunctionLabel.REPRESENTATIONAL_VISIBILITY,
                        FunctionLabel.UNDERREPRESENTED_IDENTITY,
                    ],
                    necessity=NecessityAnnotation(
                        status=Necessity.ESSENTIAL,
                        reasons=[
                            NecessityReason.TRUTH_CONDITIONAL,
                            NecessityReason.HISTORICAL_SIGNIFICANCE,
                            NecessityReason.REPRESENTATIONAL_RELEVANCE,
                        ],
                        rationale=(
                            "删除“女性”会丢失该任命具有历史突破意义的事实。"
                        ),
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.NONE,
                        mechanisms=[],
                        rationale="性别用于表达历史代表性，而非贬低或刻板化。",
                    ),
                    stance=SpeakerStance.NEUTRAL_REPORT,
                ),
            ],
            protected_facts=[
                ProtectedFact(
                    fact_id="f1",
                    fact_type=ProtectedFactType.HISTORICAL_STATUS,
                    description="该人物是本院第一位女性院士",
                    source_span_ids=["s2"],
                    must_preserve=True,
                    required_output_phrases=["第一位女性院士"],
                    verification_hint=(
                        "改写结果必须保留“第一位女性院士”的历史事实。"
                    ),
                )
            ],
            speaker_stance=SpeakerStance.NEUTRAL_REPORT,
            context_sufficient=True,
            analysis_notes="这是性别信息具有历史和代表性功能的案例。",
        ),
        gold_decision=DecisionAnnotation(
            instance_action=InstanceAction.KEEP,
            edit_scope=EditScope.NONE,
            span_actions=[
                SpanActionAnnotation(
                    span_id="s1",
                    action=Action.KEEP,
                    reason_code="VALID_SPECIFIC_COREFERENCE",
                ),
                SpanActionAnnotation(
                    span_id="s2",
                    action=Action.KEEP,
                    reason_code="HISTORICAL_GENDER_INFORMATION",
                ),
            ],
        ),
        gold_output=text,
        metadata=DatasetMetadata(
            source_type="controlled_natural_example",
            scenario_category="historical_representational_visibility",
            annotation_version="v0.1",
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


from genfine.domain.enums import (
    Action,
    BiasMechanism,
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

def build_generic_male_instance() -> DatasetInstance:
    """
    A masculine pronoun is used generically for all students.

    The requirement itself should be preserved, while the generic masculine
    form should be replaced.
    """

    text = "每位学生都应提交他的作业"
    gold_output = "每位学生都应提交自己的作业"

    return DatasetInstance(
        instance_id="zh_seed_0003",
        language="zh",
        task_mode=TaskMode.INCLUSIVE_REWRITING,
        context=TextContext(
            target_text=text,
            genre="school_policy",
            speaker="school_administrator",
            audience="all_students",
            communicative_goal="state_an_assignment_requirement",
            cultural_context="modern_chinese",
        ),
        gold_analysis=AnalysisResult(
            referents=[
                Referent(
                    referent_id="generic_student",
                    referent_type=ReferentType.GENERIC_PERSON,
                    number="DISTRIBUTIVE_SINGULAR",
                    specificity="GENERIC",
                    identity_relevance=False,
                    description="泛指任意一名学生",
                )
            ],
            spans=[
                AnalysisSpan(
                    span_id="s1",
                    text="他",
                    start=8,
                    end=9,
                    cue_type=GenderCueType.GENERIC_GENDER_FORM,
                    explicitness=Explicitness.EXPLICIT,
                    gender_dimension=GenderDimension.GENDER_REFERENCE,
                    gender_value=GenderValue.MALE,
                    referent_id="generic_student",
                    source=GenderSource.AUTHOR_ASSIGNED,
                    source_reliability=SourceReliability.CONFIRMED,
                    functions=[
                        FunctionLabel.GENERIC_REFERENCE,
                    ],
                    necessity=NecessityAnnotation(
                        status=Necessity.INCIDENTAL,
                        reasons=[
                            NecessityReason.NO_COMMUNICATIVE_RELEVANCE,
                        ],
                        rationale=(
                            "规则适用于所有学生，男性代词不是表达要求所必需的。"
                        ),
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.IMPLICIT,
                        mechanisms=[
                            BiasMechanism.GENERIC_MALE_DEFAULT,
                        ],
                        rationale=(
                            "使用男性代词代表所有学生，形成男性默认表达。"
                        ),
                    ),
                    stance=SpeakerStance.NOT_APPLICABLE,
                )
            ],
            protected_facts=[
                ProtectedFact(
                    fact_id="f1",
                    fact_type=ProtectedFactType.OTHER,
                    description="每位学生都应提交属于自己的作业",
                    source_span_ids=[],
                    must_preserve=True,
                    required_output_phrases=[
                        "每位学生",
                        "提交",
                        "作业",
                    ],
                    verification_hint=(
                        "改写不能改变学生需要提交本人作业的要求。"
                    ),
                )
            ],
            speaker_stance=SpeakerStance.NEUTRAL_REPORT,
            context_sufficient=True,
            analysis_notes=(
                "应替换泛指男性形式，但保持原有义务和所有关系。"
            ),
        ),
        gold_decision=DecisionAnnotation(
            instance_action=InstanceAction.EDIT,
            edit_scope=EditScope.SPAN,
            span_actions=[
                SpanActionAnnotation(
                    span_id="s1",
                    action=Action.REPLACE_GENERIC_FORM,
                    reason_code="GENERIC_MALE_DEFAULT",
                    rationale="将泛指“他”替换为不指定性别的反身表达。",
                )
            ],
        ),
        gold_output=gold_output,
        metadata=DatasetMetadata(
            source_type="controlled_natural_example",
            scenario_category="generic_male_default",
            annotation_version="v0.1",
        ),
    )


def build_rejected_quote_instance() -> DatasetInstance:
    """
    The text contains an explicitly biased proposition, but the current
    speaker rejects it.

    The quotation and the rejection stance must both be preserved.
    """

    text = "“女人不适合领导”是错误观念"

    return DatasetInstance(
        instance_id="zh_seed_0004",
        language="zh",
        task_mode=TaskMode.INCLUSIVE_REWRITING,
        context=TextContext(
            target_text=text,
            genre="educational_commentary",
            speaker="educator",
            audience="general_public",
            communicative_goal="criticize_a_gender_stereotype",
            cultural_context="modern_chinese",
        ),
        gold_analysis=AnalysisResult(
            referents=[
                Referent(
                    referent_id="women_as_group",
                    referent_type=ReferentType.GENERIC_GROUP,
                    number="PLURAL",
                    specificity="GENERIC",
                    identity_relevance=True,
                    description="被偏见命题概括的女性群体",
                )
            ],
            spans=[
                AnalysisSpan(
                    span_id="s1",
                    text="女人不适合领导",
                    start=1,
                    end=8,
                    cue_type=GenderCueType.STEREOTYPICAL_ATTRIBUTE,
                    explicitness=Explicitness.EXPLICIT,
                    gender_dimension=GenderDimension.SOCIAL_GENDER_ROLE,
                    gender_value=GenderValue.FEMALE,
                    referent_id="women_as_group",
                    source=GenderSource.QUOTED_SOURCE,
                    source_reliability=SourceReliability.CONFIRMED,
                    functions=[
                        FunctionLabel.DIRECT_QUOTATION,
                        FunctionLabel.BIAS_CRITIQUE,
                        FunctionLabel.GROUP_GENERALIZATION,
                    ],
                    necessity=NecessityAnnotation(
                        status=Necessity.ESSENTIAL,
                        reasons=[
                            NecessityReason.QUOTATION_FIDELITY,
                            NecessityReason.SPEAKER_VOICE,
                        ],
                        rationale=(
                            "该命题是被批评的对象，删除后会破坏批评内容。"
                        ),
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.EXPLICIT,
                        mechanisms=[
                            BiasMechanism.ESSENTIALIST_GENERALIZATION,
                            BiasMechanism.COMPETENCE_OR_AGENCY_DENIAL,
                        ],
                        rationale=(
                            "引语本身否定女性的领导能力，但当前说话者明确反对。"
                        ),
                    ),
                    stance=SpeakerStance.REJECT,
                )
            ],
            protected_facts=[
                ProtectedFact(
                    fact_id="f1",
                    fact_type=ProtectedFactType.QUOTATION_CONTENT,
                    description="被批评的原始命题是“女人不适合领导”",
                    source_span_ids=["s1"],
                    must_preserve=True,
                    required_output_phrases=["女人不适合领导"],
                    verification_hint="不得把被批评的引语改成其他命题。",
                ),
                ProtectedFact(
                    fact_id="f2",
                    fact_type=ProtectedFactType.SPEAKER_STANCE,
                    description="当前说话者认为该观念是错误的",
                    source_span_ids=["s1"],
                    must_preserve=True,
                    required_output_phrases=["错误观念"],
                    verification_hint="不得把否定立场改成赞同或中立陈述。",
                ),
            ],
            speaker_stance=SpeakerStance.REJECT,
            context_sufficient=True,
            analysis_notes=(
                "偏见表达的存在与当前说话者是否支持偏见必须分开。"
            ),
        ),
        gold_decision=DecisionAnnotation(
            instance_action=InstanceAction.KEEP,
            edit_scope=EditScope.NONE,
            span_actions=[
                SpanActionAnnotation(
                    span_id="s1",
                    action=Action.KEEP_WITH_ATTRIBUTION,
                    reason_code="REJECTED_BIASED_QUOTATION",
                    rationale="保留被批评命题及明确的否定立场。",
                )
            ],
        ),
        gold_output=text,
        metadata=DatasetMetadata(
            source_type="controlled_natural_example",
            scenario_category="rejected_biased_quotation",
            annotation_version="v0.1",
        ),
    )


def build_preserve_ambiguity_instance() -> DatasetInstance:
    """
    The source text identifies a surgeon but supplies no gender information.

    The system must not infer gender from the occupation.
    """

    text = "外科医生完成手术后离开了手术室"

    return DatasetInstance(
        instance_id="zh_seed_0005",
        language="zh",
        task_mode=TaskMode.INCLUSIVE_REWRITING,
        context=TextContext(
            target_text=text,
            genre="event_report",
            speaker="reporter",
            audience="general_public",
            communicative_goal="report_a_medical_event",
            cultural_context="modern_chinese",
        ),
        gold_analysis=AnalysisResult(
            referents=[
                Referent(
                    referent_id="surgeon_1",
                    referent_type=ReferentType.SPECIFIC_PERSON,
                    number="SINGULAR",
                    specificity="ROLE_IDENTIFIED",
                    identity_relevance=False,
                    description="执行该手术的外科医生",
                )
            ],
            spans=[
                AnalysisSpan(
                    span_id="s1",
                    text="外科医生",
                    start=0,
                    end=4,
                    cue_type=GenderCueType.NO_GENDER_CUE,
                    explicitness=Explicitness.NONE,
                    gender_dimension=GenderDimension.NOT_APPLICABLE,
                    gender_value=GenderValue.UNSPECIFIED,
                    referent_id="surgeon_1",
                    source=GenderSource.UNKNOWN,
                    source_reliability=SourceReliability.UNKNOWN,
                    functions=[
                        FunctionLabel.IDENTIFICATION,
                    ],
                    necessity=NecessityAnnotation(
                        status=Necessity.NOT_APPLICABLE,
                        reasons=[],
                        rationale="原文没有提供需要判断必要性的性别信息。",
                    ),
                    bias=BiasAnnotation(
                        status=BiasStatus.NONE,
                        mechanisms=[],
                        rationale="职业词“外科医生”本身未指定性别。",
                    ),
                    stance=SpeakerStance.NEUTRAL_REPORT,
                )
            ],
            protected_facts=[
                ProtectedFact(
                    fact_id="f1",
                    fact_type=ProtectedFactType.OTHER,
                    description="执行手术的人是一名外科医生",
                    source_span_ids=["s1"],
                    must_preserve=True,
                    required_output_phrases=[
                        "外科医生",
                        "手术",
                    ],
                    verification_hint="不能改变人物的职业角色。",
                ),
                ProtectedFact(
                    fact_id="f2",
                    fact_type=ProtectedFactType.OTHER,
                    description="原文没有说明该外科医生的性别",
                    source_span_ids=["s1"],
                    must_preserve=True,
                    forbidden_output_phrases=[
                        "他",
                        "她",
                        "男性",
                        "女性",
                        "男外科医生",
                        "女外科医生",
                    ],
                    verification_hint=(
                        "改写不得新增“他”“她”“男性”或“女性”等信息。"
                    ),
                ),
            ],
            speaker_stance=SpeakerStance.NEUTRAL_REPORT,
            context_sufficient=True,
            analysis_notes=(
                "系统需要保持原文的性别未指定状态，不能根据职业推断。"
            ),
        ),
        gold_decision=DecisionAnnotation(
            instance_action=InstanceAction.KEEP,
            edit_scope=EditScope.NONE,
            span_actions=[
                SpanActionAnnotation(
                    span_id="s1",
                    action=Action.PRESERVE_AMBIGUITY,
                    reason_code="NO_GROUNDED_GENDER_INFORMATION",
                    rationale="保持原文未指定性别的状态。",
                )
            ],
        ),
        gold_output=text,
        metadata=DatasetMetadata(
            source_type="controlled_natural_example",
            scenario_category="preserve_gender_ambiguity",
            annotation_version="v0.1",
        ),
    )

def main() -> None:
    instances = [
        build_kinship_keep_instance(),
        build_historical_visibility_instance(),
        build_generic_male_instance(),
        build_rejected_quote_instance(),
        build_preserve_ambiguity_instance(),
    ]

    write_jsonl(instances, OUTPUT_PATH)

    print(
        f"Wrote {len(instances)} instance(s) to "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )



if __name__ == "__main__":
    main()