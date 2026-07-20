from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from genfine.analysis.base import (
    Analyzer,
    AnalyzerError,
)
from genfine.analysis.prompt_builder import (
    AnalysisPromptBuilder,
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

        try:
            analysis = AnalysisResult.model_validate(
                payload
            )
        except ValidationError as exc:
            raise AnalyzerError(
                "LLM returned an invalid AnalysisResult for "
                f"{instance.instance_id!r}: {exc}"
            ) from exc

        self._validate_span_offsets(
            instance=instance,
            analysis=analysis,
        )

        return analysis

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
            payload: dict,
    ) -> dict:
        """
        Recalculate span offsets deterministically from span.text.

        LLM-provided offsets are treated only as hints. Python string
        matching is the source of truth.
        """

        spans = payload.get("spans")

        if not isinstance(spans, list):
            return payload

        target_text = instance.context.target_text

        for index, span in enumerate(spans):
            if not isinstance(span, dict):
                continue

            span_text = span.get("text")

            if (
                    not isinstance(span_text, str)
                    or not span_text
            ):
                continue

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

            provided_start = span.get("start")

            if len(positions) == 1:
                resolved_start = positions[0]

            elif (
                    isinstance(provided_start, int)
                    and provided_start in positions
            ):
                resolved_start = provided_start

            else:
                raise AnalyzerError(
                    f"Span {span.get('span_id', index)!r} "
                    f"text {span_text!r} occurs multiple times "
                    f"at {positions}, but no valid occurrence "
                    "was identified"
                )

            span["start"] = resolved_start
            span["end"] = (
                    resolved_start + len(span_text)
            )

        return payload