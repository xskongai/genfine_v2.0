from __future__ import annotations

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from genfine.domain.enums import Action


class ClaimKind(str, Enum):
    """
    Coarse semantic type of an output claim.

    This taxonomy is intentionally small in v0.1. It distinguishes
    concrete factual assertions from task-licensed reframing without
    attempting to model every possible proposition type.
    """

    SPECIFIC_ENTITY_FACT = "SPECIFIC_ENTITY_FACT"
    GENERAL_FACT = "GENERAL_FACT"

    STUDY_SCOPE_OR_QUALIFICATION = (
        "STUDY_SCOPE_OR_QUALIFICATION"
    )

    ATTRIBUTION_OR_STANCE = (
        "ATTRIBUTION_OR_STANCE"
    )

    NORMATIVE_REFRAMING = (
        "NORMATIVE_REFRAMING"
    )

    NON_FACTUAL_LANGUAGE = (
        "NON_FACTUAL_LANGUAGE"
    )

    UNCERTAIN = "UNCERTAIN"


class FactualSupportLabel(str, Enum):
    """
    Support status of one atomic output claim or change.
    """

    SOURCE_SUPPORTED = "SOURCE_SUPPORTED"

    LICENSED_REFRAMING = (
        "LICENSED_REFRAMING"
    )

    NON_FACTUAL_PARAPHRASE = (
        "NON_FACTUAL_PARAPHRASE"
    )

    UNSUPPORTED_FACTUAL_INSERTION = (
        "UNSUPPORTED_FACTUAL_INSERTION"
    )

    UNCERTAIN = "UNCERTAIN"


class FactualSupportStatus(str, Enum):
    """
    Whether semantic faithfulness evaluation was applicable.
    """

    EVALUATED = "EVALUATED"

    NOT_APPLICABLE_NO_OUTPUT = (
        "NOT_APPLICABLE_NO_OUTPUT"
    )


class FactualClaimAssessment(BaseModel):
    """
    Assessment of one atomic claim or meaningful output change.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    claim_id: str = Field(
        min_length=1,
    )

    claim: str = Field(
        min_length=1,
    )

    kind: ClaimKind

    label: FactualSupportLabel

    evidence: list[str] = Field(
        default_factory=list,
    )

    relevant_actions: list[Action] = Field(
        default_factory=list,
    )

    rationale: str = Field(
        min_length=1,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_supported_evidence(
        self,
    ) -> "FactualClaimAssessment":
        """
        SOURCE_SUPPORTED claims must identify their source evidence.
        """

        if (
            self.label
            == FactualSupportLabel.SOURCE_SUPPORTED
            and not self.evidence
        ):
            raise ValueError(
                "SOURCE_SUPPORTED claims require at least "
                "one evidence item."
            )

        return self


class FactualSupportResult(BaseModel):
    """
    Complete factual-support judgment for one rewritten output.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    instance_id: str = Field(
        min_length=1,
    )

    status: FactualSupportStatus

    claims: list[FactualClaimAssessment] = Field(
        default_factory=list,
    )

    unsupported_factual_insertion: bool

    judge_name: str = Field(
        min_length=1,
    )

    prompt_version: str = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_result_consistency(
        self,
    ) -> "FactualSupportResult":
        """
        Validate relationships between status, claims and the
        instance-level unsupported-insertion flag.
        """

        has_unsupported_claim = any(
            claim.label
            == (
                FactualSupportLabel
                .UNSUPPORTED_FACTUAL_INSERTION
            )
            for claim in self.claims
        )

        if (
            self.status
            == (
                FactualSupportStatus
                .NOT_APPLICABLE_NO_OUTPUT
            )
        ):
            if self.claims:
                raise ValueError(
                    "NOT_APPLICABLE_NO_OUTPUT results cannot "
                    "contain claim assessments."
                )

            if self.unsupported_factual_insertion:
                raise ValueError(
                    "A result without output cannot contain an "
                    "unsupported factual insertion."
                )

            return self

        if not self.claims:
            raise ValueError(
                "EVALUATED results must contain at least "
                "one claim assessment."
            )

        claim_ids = [
            claim.claim_id
            for claim in self.claims
        ]

        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(
                "Claim identifiers must be unique within "
                "one factual-support result."
            )

        if (
            self.unsupported_factual_insertion
            != has_unsupported_claim
        ):
            raise ValueError(
                "unsupported_factual_insertion must equal "
                "whether at least one claim is labelled "
                "UNSUPPORTED_FACTUAL_INSERTION."
            )

        return self

    @property
    def unsupported_claim_count(self) -> int:
        """
        Number of claims labelled as unsupported factual insertions.
        """

        return sum(
            claim.label
            == (
                FactualSupportLabel
                .UNSUPPORTED_FACTUAL_INSERTION
            )
            for claim in self.claims
        )

    @property
    def uncertain_claim_count(self) -> int:
        """
        Number of claims that require manual review.
        """

        return sum(
            claim.label
            == FactualSupportLabel.UNCERTAIN
            for claim in self.claims
        )

    @classmethod
    def no_output(
        cls,
        *,
        instance_id: str,
        judge_name: str,
        prompt_version: str,
    ) -> "FactualSupportResult":
        """
        Construct a result for ABSTAIN or other no-output cases.

        No model-based factual-support evaluation is needed when no
        rewritten output exists.
        """

        return cls(
            instance_id=instance_id,
            status=(
                FactualSupportStatus
                .NOT_APPLICABLE_NO_OUTPUT
            ),
            claims=[],
            unsupported_factual_insertion=False,
            judge_name=judge_name,
            prompt_version=prompt_version,
        )