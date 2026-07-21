from __future__ import annotations

from genfine.domain.enums import InstanceAction
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
)
from genfine.generation.base import (
    Rewriter,
    RewriterError,
)
from genfine.generation.prompt_builder import (
    RewritePromptBuilder,
)
from genfine.providers.openai_client import (
    OpenAIClientError,
    OpenAITextClient,
)


class GPTRewriter(Rewriter):
    """
    Execute a GenFINE EditPlan using an OpenAI text model.

    KEEP and ABSTAIN instances are handled deterministically and do not make
    API requests.
    """

    def __init__(
        self,
        *,
        client: OpenAITextClient,
        prompt_builder: RewritePromptBuilder,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder

    @property
    def name(self) -> str:
        return f"openai/{self.client.model}"

    def rewrite(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
    ) -> str | None:
        self._validate_inputs(
            instance=instance,
            edit_plan=edit_plan,
        )

        # KEEP is deterministic: returning the original text avoids cost and
        # prevents an unnecessary model call from causing over-neutralization.
        if (
            edit_plan.instance_action
            == InstanceAction.KEEP
        ):
            return instance.context.target_text

        if (
            edit_plan.instance_action
            == InstanceAction.ABSTAIN
        ):
            return None

        prompt_input = (
            self.prompt_builder.build_input(
                instance=instance,
                analysis=analysis,
                edit_plan=edit_plan,
            )
        )

        try:
            output = self.client.generate_text(
                instructions=(
                    self.prompt_builder.system_instruction
                ),
                input_text=prompt_input,
            )
        except OpenAIClientError as exc:
            raise RewriterError(
                f"GPT rewriting failed for "
                f"{instance.instance_id!r}: {exc}"
            ) from exc

        output = self._normalize_output(
            output
        )

        if not output:
            raise RewriterError(
                "GPT rewriter returned an empty output"
            )

        return output

    @staticmethod
    def _validate_inputs(
        *,
        instance: DatasetInstance,
        edit_plan: EditPlan,
    ) -> None:
        if (
            edit_plan.instance_id
            != instance.instance_id
        ):
            raise RewriterError(
                "edit plan instance_id does not match "
                "the dataset instance"
            )

        if (
            edit_plan.original_text
            != instance.context.target_text
        ):
            raise RewriterError(
                "edit plan original_text does not match "
                "the input target_text"
            )

    @staticmethod
    def _normalize_output(
        output: str,
    ) -> str:
        """
        Apply only minimal normalization.

        Do not aggressively rewrite model output here, because doing so would
        hide instruction-following errors from later evaluation.
        """

        output = output.strip()

        prefixes = (
            "改写结果：",
            "改写结果:",
            "最终结果：",
            "最终结果:",
        )

        for prefix in prefixes:
            if output.startswith(prefix):
                output = output[
                    len(prefix):
                ].strip()
                break

        return output