from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import ValidationError

from genfine.domain.enums import Action
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)
from genfine.factual_support.base import (
    FactualSupportJudge,
    FactualSupportJudgeError,
)
from genfine.factual_support.models import (
    FactualSupportLabel,
    FactualSupportResult,
)
from genfine.factual_support.prompt_builder import (
    FactualSupportPromptBuilder,
)


class TextGenerationClient(Protocol):
    """
    Minimal client interface required by the semantic judge.
    """

    model: str

    def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        ...


class LLMFactualSupportJudge(FactualSupportJudge):
    """
    Evaluate unsupported factual insertion with an LLM.

    The judge is provider-independent. Any client implementing
    generate_text() can be used.

    The judge performs semantic evaluation only. It does not rewrite
    text, alter the EditPlan or retry generation.
    """

    def __init__(
        self,
        *,
        client: TextGenerationClient,
        prompt_builder: FactualSupportPromptBuilder,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder

    @property
    def name(self) -> str:
        model_name = getattr(
            self.client,
            "model",
            self.client.__class__.__name__,
        )

        return f"llm-factual-support/{model_name}"

    @property
    def prompt_version(self) -> str:
        return self.prompt_builder.version

    def evaluate(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> FactualSupportResult:
        if output_text is None:
            return FactualSupportResult.no_output(
                instance_id=instance.instance_id,
                judge_name=self.name,
                prompt_version=self.prompt_version,
            )

        prompt_input = self.prompt_builder.build_input(
            instance=instance,
            analysis=analysis,
            edit_plan=edit_plan,
            output_text=output_text,
        )

        try:
            raw_output = self.client.generate_text(
                instructions=(
                    self.prompt_builder.system_instruction
                ),
                input_text=prompt_input,
            )
        except Exception as exc:
            raise FactualSupportJudgeError(
                "Factual-support model request failed for "
                f"{instance.instance_id!r}: {exc}"
            ) from exc

        payload = self._parse_json_output(
            raw_output=raw_output,
            instance_id=instance.instance_id,
        )

        self._validate_instance_id(
            payload=payload,
            expected_instance_id=instance.instance_id,
        )

        # These two fields are controlled by the system rather than
        # trusted from model output.
        payload["judge_name"] = self.name
        payload["prompt_version"] = self.prompt_version

        try:
            result = FactualSupportResult.model_validate(
                payload
            )
        except ValidationError as exc:
            raise FactualSupportJudgeError(
                "LLM returned an invalid factual-support "
                f"result for {instance.instance_id!r}: {exc}"
            ) from exc

        self._validate_relevant_actions(
            result=result,
            edit_plan=edit_plan,
        )

        return result

    @classmethod
    def _parse_json_output(
        cls,
        *,
        raw_output: str,
        instance_id: str,
    ) -> dict[str, Any]:
        normalized = cls._strip_code_fence(
            raw_output
        )

        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError as exc:
            preview = normalized[:500]

            raise FactualSupportJudgeError(
                "LLM returned invalid factual-support JSON "
                f"for {instance_id!r}: {exc}. "
                f"Output preview: {preview!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise FactualSupportJudgeError(
                "Factual-support output must be a JSON object "
                f"for {instance_id!r}, received "
                f"{type(payload).__name__}."
            )

        return payload

    @staticmethod
    def _validate_instance_id(
        *,
        payload: dict[str, Any],
        expected_instance_id: str,
    ) -> None:
        returned_instance_id = payload.get(
            "instance_id"
        )

        if returned_instance_id != expected_instance_id:
            raise FactualSupportJudgeError(
                "Factual-support result instance_id mismatch: "
                f"expected {expected_instance_id!r}, "
                f"received {returned_instance_id!r}."
            )

    @classmethod
    def _validate_relevant_actions(
        cls,
        *,
        result: FactualSupportResult,
        edit_plan: EditPlan,
    ) -> None:
        """
        Validate action references only when an EditPlan action is used
        to justify an output transformation.

        SOURCE_SUPPORTED claims are justified by source evidence rather
        than by the EditPlan. Therefore an unnecessary or imprecise
        relevant_actions entry must not invalidate an otherwise
        source-supported claim.

        LICENSED_REFRAMING must cite at least one action that actually
        occurs in the saved EditPlan.

        NON_FACTUAL_PARAPHRASE may omit relevant_actions. When it cites
        actions, those actions must occur in the saved EditPlan.
        """

        plan_actions: set[Action] = {
            decision.action
            for decision in edit_plan.span_decisions
        }

        for claim in result.claims:
            if (
                claim.label
                == FactualSupportLabel.LICENSED_REFRAMING
            ):
                if not claim.relevant_actions:
                    raise FactualSupportJudgeError(
                        f"Claim {claim.claim_id!r} is labelled "
                        "LICENSED_REFRAMING but does not cite "
                        "any relevant EditPlan action."
                    )

                cls._require_actions_in_plan(
                    claim_id=claim.claim_id,
                    relevant_actions=set(
                        claim.relevant_actions
                    ),
                    plan_actions=plan_actions,
                )

                continue

            if (
                claim.label
                == FactualSupportLabel.NON_FACTUAL_PARAPHRASE
                and claim.relevant_actions
            ):
                cls._require_actions_in_plan(
                    claim_id=claim.claim_id,
                    relevant_actions=set(
                        claim.relevant_actions
                    ),
                    plan_actions=plan_actions,
                )

    @staticmethod
    def _require_actions_in_plan(
        *,
        claim_id: str,
        relevant_actions: set[Action],
        plan_actions: set[Action],
    ) -> None:
        unsupported_actions = (
            relevant_actions
            - plan_actions
        )

        if not unsupported_actions:
            return

        action_names = sorted(
            action.value
            for action in unsupported_actions
        )

        raise FactualSupportJudgeError(
            f"Claim {claim_id!r} references "
            "actions that are not present in the "
            f"EditPlan: {action_names}."
        )

    @staticmethod
    def _strip_code_fence(
        raw_output: str,
    ) -> str:
        """
        Remove one surrounding Markdown code fence.

        No semantic repair or label correction is performed.
        """

        output = raw_output.strip()

        if not output.startswith("```"):
            return output

        lines = output.splitlines()

        if len(lines) < 3:
            return output

        if lines[-1].strip() != "```":
            return output

        return "\n".join(
            lines[1:-1]
        ).strip()