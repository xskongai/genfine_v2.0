from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)


class FactualSupportPromptError(ValueError):
    """Raised when a factual-support prompt cannot be built."""


@dataclass(frozen=True)
class FactualSupportPromptBuilder:
    """
    Build an action-aware factual-support evaluation prompt.

    Source text is factual evidence.

    Analysis and EditPlan fields are transformation metadata only.
    """

    version: str
    system_instruction: str

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "FactualSupportPromptBuilder":
        prompt_path = Path(path)

        try:
            payload = yaml.safe_load(
                prompt_path.read_text(
                    encoding="utf-8",
                )
            )
        except FileNotFoundError as exc:
            raise FactualSupportPromptError(
                f"Prompt file not found: {prompt_path}"
            ) from exc
        except yaml.YAMLError as exc:
            raise FactualSupportPromptError(
                f"Invalid prompt YAML: {prompt_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise FactualSupportPromptError(
                "Factual-support prompt must be a YAML object."
            )

        version = payload.get("version")
        system_instruction = payload.get(
            "system_instruction"
        )

        if not isinstance(version, str) or not version:
            raise FactualSupportPromptError(
                "Prompt field 'version' must be a non-empty string."
            )

        if (
            not isinstance(system_instruction, str)
            or not system_instruction.strip()
        ):
            raise FactualSupportPromptError(
                "Prompt field 'system_instruction' must be "
                "a non-empty string."
            )

        return cls(
            version=version,
            system_instruction=system_instruction.strip(),
        )

    def build_input(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> str:
        if output_text is None:
            raise FactualSupportPromptError(
                "Factual-support prompt cannot be built "
                "without output_text. Use the no-output result "
                "instead of calling the semantic judge."
            )

        span_by_id = {
            span.span_id: span
            for span in analysis.spans
        }

        action_licenses: list[dict[str, Any]] = []

        for decision in edit_plan.span_decisions:
            span = span_by_id.get(
                decision.span_id
            )

            action_licenses.append(
                {
                    "span_id": decision.span_id,
                    "span_text": (
                        span.text
                        if span is not None
                        else None
                    ),
                    "action": decision.action.value,
                    "reason_code": decision.reason_code,
                    "constraints": list(
                        decision.constraints
                    ),
                }
            )

        source_evidence = {
            "preceding_context": (
                instance.context.preceding_context or ""
            ),
            "target_text": (
                instance.context.target_text
            ),
            "following_context": (
                instance.context.following_context or ""
            ),
        }

        transformation_plan = {
            "instance_action": (
                edit_plan.instance_action.value
            ),
            "edit_scope": edit_plan.edit_scope.value,
            "action_licenses": action_licenses,
            "planning_protected_facts": [
                fact.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for fact in edit_plan.protected_facts
            ],
            "global_constraints": list(
                edit_plan.global_constraints
            ),
        }

        output_schema = {
            "instance_id": instance.instance_id,
            "status": "EVALUATED",
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "one atomic output claim",
                    "kind": (
                        "SPECIFIC_ENTITY_FACT | "
                        "GENERAL_FACT | "
                        "STUDY_SCOPE_OR_QUALIFICATION | "
                        "ATTRIBUTION_OR_STANCE | "
                        "NORMATIVE_REFRAMING | "
                        "NON_FACTUAL_LANGUAGE | "
                        "UNCERTAIN"
                    ),
                    "label": (
                        "SOURCE_SUPPORTED | "
                        "LICENSED_REFRAMING | "
                        "NON_FACTUAL_PARAPHRASE | "
                        "UNSUPPORTED_FACTUAL_INSERTION | "
                        "UNCERTAIN"
                    ),
                    "evidence": [
                        "source evidence phrase"
                    ],
                    "relevant_actions": [
                        "REFRAME_PROPOSITION"
                    ],
                    "rationale": (
                        "brief source- and action-aware reason"
                    ),
                    "confidence": 0.0,
                }
            ],
            "unsupported_factual_insertion": False,
        }

        payload = {
            "task": (
                "Evaluate whether the rewritten output "
                "contains unsupported factual insertions."
            ),
            "instance_id": instance.instance_id,
            "language": instance.language,
            "evidence_policy": {
                "only_source_evidence_is_factual_evidence": True,
                "transformation_plan_is_not_factual_evidence": True,
                "external_knowledge_allowed": False,
            },
            "source_evidence": source_evidence,
            "transformation_plan": transformation_plan,
            "rewritten_output": output_text,
            "required_output_schema": output_schema,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )