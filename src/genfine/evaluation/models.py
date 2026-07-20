
from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)


class MetricValue(BaseModel):
    """One metric with its numerator, denominator and computed value."""

    model_config = ConfigDict(
        extra="forbid",
    )

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @computed_field
    @property
    def value(self) -> float:
        if self.denominator == 0:
            return 0.0

        return self.numerator / self.denominator


class EvaluationSummary(BaseModel):
    """Complete evaluation summary for one pipeline run."""

    model_config = ConfigDict(
        extra="forbid",
    )

    dataset_instance_count: int = Field(ge=0)
    evaluated_instance_count: int = Field(ge=0)

    successful_instances: int = Field(ge=0)
    failed_instances: int = Field(ge=0)

    analysis_exact_match: MetricValue
    span_action_accuracy: MetricValue
    instance_action_accuracy: MetricValue
    edit_scope_accuracy: MetricValue
    output_exact_match: MetricValue

    verification_coverage: MetricValue
    verification_pass_rate: MetricValue
    protected_fact_preservation_rate: MetricValue
    action_compliance_rate: MetricValue
    unsupported_gender_insertion_rate: MetricValue

    over_neutralization_rate: MetricValue
    under_correction_rate: MetricValue

    metadata: dict[str, object] = Field(
        default_factory=dict
    )