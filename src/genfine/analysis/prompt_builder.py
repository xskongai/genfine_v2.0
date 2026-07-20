from __future__ import annotations

import json
from pathlib import Path

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
)


class AnalysisPromptError(RuntimeError):
    """Raised when an analysis prompt cannot be loaded or built."""


class AnalysisPromptConfig(BaseModel):
    """Configuration loaded from analyze_v0.1.yaml."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    system_instruction: str = Field(min_length=1)
    output_contract: str = Field(min_length=1)


class AnalysisPromptBuilder:
    """Build a structured analysis request for an LLM."""

    def __init__(
        self,
        config: AnalysisPromptConfig,
    ) -> None:
        self.config = config

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "AnalysisPromptBuilder":
        prompt_path = Path(path)

        if not prompt_path.exists():
            raise FileNotFoundError(
                "Analysis prompt file does not exist: "
                f"{prompt_path}"
            )

        try:
            raw_data = yaml.safe_load(
                prompt_path.read_text(
                    encoding="utf-8",
                )
            )
        except yaml.YAMLError as exc:
            raise AnalysisPromptError(
                f"Invalid YAML in {prompt_path}: {exc}"
            ) from exc

        try:
            config = AnalysisPromptConfig.model_validate(
                raw_data
            )
        except ValidationError as exc:
            raise AnalysisPromptError(
                "Invalid analysis prompt configuration: "
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
    ) -> str:
        request_payload = {
            "instance_id": instance.instance_id,
            "language": instance.language,
            "task_mode": instance.task_mode.value,
            "offset_convention": {
                "type": "python_character_offset",
                "interval": "[start, end)",
                "validation": (
                    "target_text[start:end] must equal span.text"
                ),
            },
            "context": instance.context.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "analysis_result_json_schema": (
                AnalysisResult.model_json_schema()
            ),
        }

        serialized_payload = json.dumps(
            request_payload,
            ensure_ascii=False,
            indent=2,
        )

        return (
            "请分析以下 GenFINE 实例，并严格按照给出的 "
            "AnalysisResult JSON Schema 返回结果。\n\n"
            f"{serialized_payload}\n\n"
            f"{self.config.output_contract}"
        )