#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 7/20/26
Description: models
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class GenFineBaseModel(BaseModel):
    """
    Base class shared by all GenFINE domain models.

    extra="forbid" is intentional:
    misspelled or unexpected fields should fail validation instead of being
    silently ignored.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class TextContext(GenFineBaseModel):
    """Input text together with its discourse and task context."""

    preceding_context: str = ""
    target_text: str = Field(min_length=1)
    following_context: str = ""

    genre: str = "unspecified"
    speaker: str = "unspecified"
    audience: str = "unspecified"
    communicative_goal: str = "unspecified"

    temporal_context: str | None = None
    cultural_context: str | None = None


class Referent(GenFineBaseModel):
    """An entity or group referred to by one or more analysis spans."""

    referent_id: str = Field(min_length=1)
    referent_type: ReferentType

    number: str = "UNKNOWN"
    specificity: str = "UNKNOWN"
    identity_relevance: bool = False

    description: str | None = None


class NecessityAnnotation(GenFineBaseModel):
    """Context-sensitive necessity annotation."""

    status: Necessity
    reasons: list[NecessityReason] = Field(default_factory=list)

    rationale: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class BiasAnnotation(GenFineBaseModel):
    """Bias status and the mechanisms through which it is expressed."""

    status: BiasStatus
    mechanisms: list[BiasMechanism] = Field(default_factory=list)

    rationale: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bias_mechanisms(self) -> BiasAnnotation:
        if self.status == BiasStatus.NONE and self.mechanisms:
            raise ValueError(
                "bias mechanisms must be empty when bias status is NONE"
            )

        return self


class AnalysisSpan(GenFineBaseModel):
    """
    A text span relevant to gender-information analysis.

    This can be a short gender cue such as “她”, or a larger proposition such
    as “女性天生更善于管理” when the complete proposition is the edit target.
    """

    span_id: str = Field(min_length=1)

    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    cue_type: GenderCueType
    explicitness: Explicitness

    gender_dimension: GenderDimension = GenderDimension.UNKNOWN_DIMENSION
    gender_value: GenderValue = GenderValue.UNKNOWN

    referent_id: str | None = None

    source: GenderSource = GenderSource.UNKNOWN
    source_reliability: SourceReliability = SourceReliability.UNKNOWN

    functions: list[FunctionLabel] = Field(default_factory=list)

    necessity: NecessityAnnotation
    bias: BiasAnnotation

    stance: SpeakerStance = SpeakerStance.NOT_APPLICABLE

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_offsets(self) -> AnalysisSpan:
        if self.end <= self.start:
            raise ValueError(
                f"span end must be greater than start: "
                f"start={self.start}, end={self.end}"
            )

        return self


class ProtectedFact(GenFineBaseModel):
    """A fact or semantic property that a rewrite must preserve."""

    fact_id: str = Field(min_length=1)
    fact_type: ProtectedFactType

    description: str = Field(min_length=1)

    source_span_ids: list[str] = Field(default_factory=list)
    must_preserve: bool = True

    # MVP lexical verification.
    # Every required phrase must appear in the output.
    required_output_phrases: list[str] = Field(
        default_factory=list
    )

    # None of the forbidden phrases may appear in the output.
    forbidden_output_phrases: list[str] = Field(
        default_factory=list
    )

    verification_hint: str | None = None


class AnalysisResult(GenFineBaseModel):
    """Complete structured analysis of one input instance."""

    referents: list[Referent] = Field(default_factory=list)
    spans: list[AnalysisSpan] = Field(default_factory=list)
    protected_facts: list[ProtectedFact] = Field(default_factory=list)

    speaker_stance: SpeakerStance = SpeakerStance.NOT_APPLICABLE
    context_sufficient: bool = True

    analysis_notes: str | None = None

    @model_validator(mode="after")
    def validate_unique_ids_and_references(self) -> AnalysisResult:
        referent_ids = [item.referent_id for item in self.referents]
        span_ids = [item.span_id for item in self.spans]
        fact_ids = [item.fact_id for item in self.protected_facts]

        self._raise_for_duplicates("referent_id", referent_ids)
        self._raise_for_duplicates("span_id", span_ids)
        self._raise_for_duplicates("fact_id", fact_ids)

        known_referents = set(referent_ids)
        known_spans = set(span_ids)

        for span in self.spans:
            if (
                span.referent_id is not None
                and span.referent_id not in known_referents
            ):
                raise ValueError(
                    f"span {span.span_id!r} refers to unknown referent "
                    f"{span.referent_id!r}"
                )

        for fact in self.protected_facts:
            unknown_span_ids = (
                set(fact.source_span_ids) - known_spans
            )

            if unknown_span_ids:
                raise ValueError(
                    f"protected fact {fact.fact_id!r} refers to unknown "
                    f"span ids: {sorted(unknown_span_ids)}"
                )

        return self

    @staticmethod
    def _raise_for_duplicates(
        field_name: str,
        values: list[str],
    ) -> None:
        duplicates = {
            value for value in values if values.count(value) > 1
        }

        if duplicates:
            raise ValueError(
                f"duplicate {field_name} values: {sorted(duplicates)}"
            )


class SpanActionAnnotation(GenFineBaseModel):
    """Gold or predicted action for one analysis span."""

    span_id: str = Field(min_length=1)
    action: Action

    reason_code: str | None = None
    rationale: str | None = None


class DecisionAnnotation(GenFineBaseModel):
    """Gold decision labels stored in the dataset."""

    instance_action: InstanceAction
    edit_scope: EditScope

    span_actions: list[SpanActionAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_span_actions(self) -> DecisionAnnotation:
        span_ids = [item.span_id for item in self.span_actions]

        duplicates = {
            span_id
            for span_id in span_ids
            if span_ids.count(span_id) > 1
        }

        if duplicates:
            raise ValueError(
                f"duplicate span actions: {sorted(duplicates)}"
            )

        return self


class DatasetMetadata(GenFineBaseModel):
    """Dataset provenance and annotation metadata."""

    source_type: str = "constructed"
    source_reference: str | None = None

    scenario_category: str | None = None
    context_pair_id: str | None = None
    minimal_pair_id: str | None = None

    annotator_ids: list[str] = Field(default_factory=list)
    annotation_version: str = "v0.1"

    notes: str | None = None


class DatasetInstance(GenFineBaseModel):
    """One complete labeled GenFINE dataset instance."""

    instance_id: str = Field(min_length=1)
    language: str = Field(default="zh", min_length=2)

    task_mode: TaskMode = TaskMode.INCLUSIVE_REWRITING
    context: TextContext

    gold_analysis: AnalysisResult
    gold_decision: DecisionAnnotation

    gold_output: str | None = None

    metadata: DatasetMetadata = Field(default_factory=DatasetMetadata)

    @model_validator(mode="after")
    def validate_instance_consistency(self) -> DatasetInstance:
        target_text = self.context.target_text

        analysis_span_ids = {
            span.span_id for span in self.gold_analysis.spans
        }

        for span in self.gold_analysis.spans:
            if span.end > len(target_text):
                raise ValueError(
                    f"span {span.span_id!r} ends at {span.end}, "
                    f"but target text length is {len(target_text)}"
                )

            actual_text = target_text[span.start:span.end]

            if actual_text != span.text:
                raise ValueError(
                    f"span text mismatch for {span.span_id!r}: "
                    f"expected {span.text!r}, found {actual_text!r} "
                    f"at offsets [{span.start}:{span.end}]"
                )

        decision_span_ids = {
            item.span_id
            for item in self.gold_decision.span_actions
        }

        unknown_decision_spans = (
            decision_span_ids - analysis_span_ids
        )

        if unknown_decision_spans:
            raise ValueError(
                "gold decision contains unknown span ids: "
                f"{sorted(unknown_decision_spans)}"
            )

        if (
            self.gold_decision.instance_action
            == InstanceAction.ABSTAIN
            and self.gold_output is not None
        ):
            raise ValueError(
                "gold_output must be null when instance_action is ABSTAIN"
            )

        if (
            self.gold_decision.instance_action
            != InstanceAction.ABSTAIN
            and self.gold_output is None
        ):
            raise ValueError(
                "gold_output is required unless instance_action is ABSTAIN"
            )

        return self


class SpanDecision(GenFineBaseModel):
    """Decision produced by the rule-based policy engine."""

    span_id: str = Field(min_length=1)
    action: Action

    rule_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)

    priority: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    constraints: list[str] = Field(default_factory=list)


class EditPlan(GenFineBaseModel):
    """Executable editing instructions produced by the policy layer."""

    instance_id: str = Field(min_length=1)
    original_text: str = Field(min_length=1)

    instance_action: InstanceAction
    edit_scope: EditScope

    span_decisions: list[SpanDecision] = Field(default_factory=list)
    protected_facts: list[ProtectedFact] = Field(default_factory=list)

    global_constraints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_edit_plan(self) -> EditPlan:
        span_ids = [
            decision.span_id for decision in self.span_decisions
        ]

        duplicates = {
            span_id
            for span_id in span_ids
            if span_ids.count(span_id) > 1
        }

        if duplicates:
            raise ValueError(
                f"duplicate span decisions: {sorted(duplicates)}"
            )

        return self


class VerificationIssue(GenFineBaseModel):
    """One violation or warning detected after rewriting."""

    checker: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)

    severity: str = "ERROR"
    span_text: str | None = None


class VerificationResult(GenFineBaseModel):
    """Aggregated result from independent output verifiers."""

    passed: bool

    protected_facts_preserved: bool = True
    action_compliant: bool = True
    unsupported_gender_inserted: bool = False

    issues: list[VerificationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_passed_status(self) -> VerificationResult:
        expected_passed = (
            self.protected_facts_preserved
            and self.action_compliant
            and not self.unsupported_gender_inserted
            and not any(
                issue.severity == "ERROR"
                for issue in self.issues
            )
        )

        if self.passed != expected_passed:
            raise ValueError(
                "passed is inconsistent with verification fields"
            )

        return self


class ModelInfo(GenFineBaseModel):
    """Information about a model used during one pipeline run."""

    provider: str
    model_name: str

    prompt_version: str | None = None
    temperature: float | None = None


class RunRecord(GenFineBaseModel):
    """One complete pipeline output record."""

    instance_id: str
    original_text: str

    predicted_analysis: AnalysisResult | None = None
    edit_plan: EditPlan | None = None

    output_text: str | None = None
    verification: VerificationResult | None = None

    analysis_model: ModelInfo | None = None
    rewrite_model: ModelInfo | None = None

    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)