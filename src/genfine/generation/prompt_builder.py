from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)


class RewritePromptError(ValueError):
    """Raised when a rewrite prompt cannot be loaded or built."""


class RewritePromptConfig(BaseModel):
    """Versioned rewrite prompt configuration."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    version: str = Field(min_length=1)
    system_instruction: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)


class RewritePromptBuilder:
    """Build a constrained rewrite request from an EditPlan."""

    def __init__(
        self,
        config: RewritePromptConfig,
    ) -> None:
        self.config = config

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "RewritePromptBuilder":
        prompt_path = Path(path)

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Rewrite prompt file does not exist: "
                f"{prompt_path}"
            )

        try:
            raw_data = yaml.safe_load(
                prompt_path.read_text(
                    encoding="utf-8"
                )
            )
        except yaml.YAMLError as exc:
            raise RewritePromptError(
                f"Invalid YAML in {prompt_path}: {exc}"
            ) from exc

        try:
            config = RewritePromptConfig.model_validate(
                raw_data
            )
        except ValidationError as exc:
            raise RewritePromptError(
                f"Invalid rewrite prompt configuration: "
                f"{exc}"
            ) from exc

        return cls(config)

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def system_instruction(self) -> str:
        return self.config.system_instruction

    def build_input(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
    ) -> str:
        span_by_id = {
            span.span_id: span
            for span in analysis.spans
        }

        span_instructions: list[dict[str, Any]] = []

        for decision in edit_plan.span_decisions:
            span = span_by_id.get(
                decision.span_id
            )

            if span is None:
                raise RewritePromptError(
                    f"Edit decision refers to unknown span "
                    f"{decision.span_id!r}"
                )

            span_instructions.append(
                {
                    "span_id": span.span_id,
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "action": decision.action.value,
                    "reason_code": decision.reason_code,
                    "constraints": decision.constraints,
                }
            )

        protected_facts = [
            {
                "fact_id": fact.fact_id,
                "fact_type": fact.fact_type.value,
                "description": fact.description,
                "verification_hint": (
                    fact.verification_hint
                ),
            }
            for fact in edit_plan.protected_facts
        ]

        request_payload = {
            "language": instance.language,
            "task_mode": instance.task_mode.value,
            "context": {
                "preceding_context": (
                    instance.context.preceding_context
                ),
                "target_text": (
                    instance.context.target_text
                ),
                "following_context": (
                    instance.context.following_context
                ),
                "genre": instance.context.genre,
                "speaker": instance.context.speaker,
                "audience": instance.context.audience,
                "communicative_goal": (
                    instance.context.communicative_goal
                ),
            },
            "edit_plan": {
                "instance_action": (
                    edit_plan.instance_action.value
                ),
                "edit_scope": (
                    edit_plan.edit_scope.value
                ),
                "span_instructions": (
                    span_instructions
                ),
                "protected_facts": (
                    protected_facts
                ),
                "global_constraints": (
                    edit_plan.global_constraints
                ),
            },
        }

        serialized_payload = json.dumps(
            request_payload,
            ensure_ascii=False,
            indent=2,
        )

        return (
            "请严格根据以下结构化编辑计划处理文本。\n\n"
            f"{serialized_payload}\n\n"
            f"{self.config.output_contract}"
        )