from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Protocol

from pydantic import ValidationError

from genfine.analysis.base import (
    Analyzer,
    AnalyzerError,
)
from genfine.analysis.prompt_builder import (
    AnalysisPromptBuilder,
)
from genfine.domain.enums import (
    BiasMechanism,
    FunctionLabel,
)
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
)


class TextGenerationClient(Protocol):
    """
    Minimal interface required by LLMAnalyzer.

    OpenAITextClient, a future Qwen client and a local-model
    client can all implement this interface.
    """

    model: str

    def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        ...


class LLMAnalyzer(Analyzer):
    """
    Produce a structured AnalysisResult using a text-generation model.

    The analyzer is model-independent. Its client only needs to
    implement generate_text().
    """

    def __init__(
        self,
        *,
        client: TextGenerationClient,
        prompt_builder: AnalysisPromptBuilder,
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

        return f"llm/{model_name}"

    def analyze(
        self,
        instance: DatasetInstance,
    ) -> AnalysisResult:
        prompt_input = self.prompt_builder.build_input(
            instance=instance,
        )

        try:
            raw_output = self.client.generate_text(
                instructions=(
                    self.prompt_builder.system_instruction
                ),
                input_text=prompt_input,
            )
        except Exception as exc:
            raise AnalyzerError(
                "LLM analysis request failed for "
                f"{instance.instance_id!r}: {exc}"
            ) from exc

        payload = self._parse_json_output(
            raw_output=raw_output,
            instance_id=instance.instance_id,
        )

        payload = self._align_span_offsets(
            instance=instance,
            payload=payload,
        )

        analysis = self._validate_analysis_payload(
            payload=payload,
            instance_id=instance.instance_id,
        )

        self._validate_span_offsets(
            instance=instance,
            analysis=analysis,
        )

        return analysis

    @classmethod
    def _validate_analysis_payload(
        cls,
        *,
        payload: dict[str, Any],
        instance_id: str,
    ) -> AnalysisResult:
        """
        Validate once, apply two deterministic schema repairs on failure,
        then validate exactly one more time.

        The repair step never invents semantic labels or retries the model.
        """

        try:
            return AnalysisResult.model_validate(payload)
        except ValidationError as first_error:
            repaired_payload = cls._repair_schema_payload(
                payload=payload,
            )

            try:
                return AnalysisResult.model_validate(
                    repaired_payload
                )
            except ValidationError as second_error:
                raise AnalyzerError(
                    "LLM returned an invalid AnalysisResult for "
                    f"{instance_id!r}. "
                    f"Initial validation error: {first_error}. "
                    f"Validation error after minimal repair: "
                    f"{second_error}"
                ) from second_error

    @classmethod
    def _repair_schema_payload(
        cls,
        *,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply only the two approved deterministic repairs.

        The input payload is not mutated. A repaired deep copy is returned.
        """

        repaired = deepcopy(payload)
        repaired.pop("confidence", None)

        spans = repaired.get("spans")
        if isinstance(spans, list):
            movable_values = cls._movable_bias_mechanism_values()

            for span in spans:
                cls._repair_span_functions(
                    span=span,
                    movable_values=movable_values,
                )

        return repaired

    @staticmethod
    def _movable_bias_mechanism_values() -> set[str]:
        bias_values = {
            member.value
            for member in BiasMechanism
        }
        function_values = {
            member.value
            for member in FunctionLabel
        }

        return bias_values - function_values

    @classmethod
    def _repair_span_functions(
        cls,
        *,
        span: Any,
        movable_values: set[str],
    ) -> None:
        if not isinstance(span, dict):
            return

        functions = span.get("functions")
        bias = span.get("bias")

        if (
            not isinstance(functions, list)
            or not isinstance(bias, dict)
        ):
            return

        retained, moved = cls._partition_functions(
            functions=functions,
            movable_values=movable_values,
        )

        if not moved:
            return

        mechanisms = bias.get("mechanisms", [])
        if not isinstance(mechanisms, list):
            return

        span["functions"] = retained
        bias["mechanisms"] = cls._dedupe_preserving_order(
            [*mechanisms, *moved]
        )

    @staticmethod
    def _partition_functions(
        *,
        functions: list[Any],
        movable_values: set[str],
    ) -> tuple[list[Any], list[str]]:
        retained: list[Any] = []
        moved: list[str] = []

        for value in functions:
            if (
                isinstance(value, str)
                and value in movable_values
            ):
                moved.append(value)
            else:
                retained.append(value)

        return retained, moved

    @staticmethod
    def _dedupe_preserving_order(
        values: list[Any],
    ) -> list[Any]:
        unique: list[Any] = []

        for value in values:
            if value not in unique:
                unique.append(value)

        return unique

    @classmethod
    def _parse_json_output(
        cls,
        *,
        raw_output: str,
        instance_id: str,
    ) -> dict:
        normalized = cls._strip_code_fence(
            raw_output
        )

        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError as exc:
            preview = normalized[:500]

            raise AnalyzerError(
                "LLM returned invalid JSON for "
                f"{instance_id!r}: {exc}. "
                f"Output preview: {preview!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise AnalyzerError(
                "LLM analysis output must be a JSON object "
                f"for {instance_id!r}, received "
                f"{type(payload).__name__}"
            )

        return payload

    @staticmethod
    def _strip_code_fence(
        raw_output: str,
    ) -> str:
        """
        Apply one minimal repair: remove a surrounding Markdown fence.

        Other malformed output remains visible as an analyzer error.
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

    @staticmethod
    def _validate_span_offsets(
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
    ) -> None:
        target_text = instance.context.target_text

        for span in analysis.spans:
            if span.end > len(target_text):
                raise AnalyzerError(
                    f"Span {span.span_id!r} ends outside "
                    f"target_text: end={span.end}, "
                    f"length={len(target_text)}"
                )

            actual_text = target_text[
                span.start:span.end
            ]

            if actual_text != span.text:
                raise AnalyzerError(
                    "Span text mismatch for "
                    f"{span.span_id!r}: expected "
                    f"{span.text!r}, found "
                    f"{actual_text!r} at offsets "
                    f"[{span.start}:{span.end}]"
                )

    @staticmethod
    def _find_all_occurrences(
            *,
            target_text: str,
            span_text: str,
    ) -> list[int]:
        """Return every start position of span_text in target_text."""

        positions: list[int] = []
        search_from = 0

        while True:
            position = target_text.find(
                span_text,
                search_from,
            )

            if position == -1:
                break

            positions.append(position)
            search_from = position + 1

        return positions

    @classmethod
    def _align_span_offsets(
        cls,
        *,
        instance: DatasetInstance,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Recalculate span offsets deterministically from span.text.

        The input payload is not mutated. A deep-copied aligned payload
        is returned.
        """

        aligned = deepcopy(payload)
        spans = aligned.get("spans")

        if isinstance(spans, list):
            target_text = instance.context.target_text

            for index, span in enumerate(spans):
                cls._align_single_span(
                    instance=instance,
                    target_text=target_text,
                    span=span,
                    index=index,
                )

        return aligned

    @classmethod
    def _align_single_span(
        cls,
        *,
        instance: DatasetInstance,
        target_text: str,
        span: Any,
        index: int,
    ) -> None:
        if not isinstance(span, dict):
            return

        span_text = span.get("text")
        if not isinstance(span_text, str) or not span_text:
            return

        positions = cls._find_all_occurrences(
            target_text=target_text,
            span_text=span_text,
        )

        if not positions:
            raise AnalyzerError(
                f"Span {span.get('span_id', index)!r} "
                f"text {span_text!r} does not occur in "
                f"target_text for {instance.instance_id!r}"
            )

        resolved_start = cls._resolve_span_start(
            instance=instance,
            span=span,
            span_text=span_text,
            positions=positions,
            index=index,
        )

        span["start"] = resolved_start
        span["end"] = resolved_start + len(span_text)

    @staticmethod
    def _resolve_span_start(
        *,
        instance: DatasetInstance,
        span: dict[str, Any],
        span_text: str,
        positions: list[int],
        index: int,
    ) -> int:
        if len(positions) == 1:
            return positions[0]

        provided_start = span.get("start")
        if (
            isinstance(provided_start, int)
            and provided_start in positions
        ):
            return provided_start

        raise AnalyzerError(
            f"Span {span.get('span_id', index)!r} "
            f"text {span_text!r} occurs multiple times "
            f"at {positions}, but no valid occurrence "
            f"was identified for {instance.instance_id!r}"
        )

