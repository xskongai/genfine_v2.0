from __future__ import annotations

import json
from pathlib import Path

import pytest

from genfine.analysis import (
    AnalyzerError,
    AnalysisPromptBuilder,
    AnalysisPromptConfig,
    LLMAnalyzer,
)
from genfine.data.loader import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.2.jsonl"
)


class FakeTextClient:
    model = "fake-analysis-model"

    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.last_instructions: str | None = None
        self.last_input_text: str | None = None

    def generate_text(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> str:
        self.last_instructions = instructions
        self.last_input_text = input_text

        return self.response


def build_prompt_builder() -> AnalysisPromptBuilder:
    return AnalysisPromptBuilder(
        AnalysisPromptConfig(
            version="test",
            system_instruction=(
                "Return a valid AnalysisResult JSON object."
            ),
            output_contract=(
                "Output JSON only."
            ),
        )
    )


def load_first_instance():
    return load_dataset(SEED_PATH)[0]


def gold_response(
    instance,
) -> str:
    return json.dumps(
        instance.gold_analysis.model_dump(
            mode="json",
            exclude_none=True,
        ),
        ensure_ascii=False,
    )


def test_llm_analyzer_parses_valid_analysis() -> None:
    instance = load_first_instance()

    client = FakeTextClient(
        gold_response(instance)
    )

    analyzer = LLMAnalyzer(
        client=client,
        prompt_builder=build_prompt_builder(),
    )

    result = analyzer.analyze(instance)

    assert result == instance.gold_analysis
    assert analyzer.name == "llm/fake-analysis-model"

    assert client.last_instructions is not None
    assert client.last_input_text is not None

    assert instance.context.target_text in (
        client.last_input_text
    )

    assert "analysis_result_json_schema" in (
        client.last_input_text
    )


def test_llm_analyzer_accepts_json_code_fence() -> None:
    instance = load_first_instance()

    response = (
        "```json\n"
        f"{gold_response(instance)}\n"
        "```"
    )

    analyzer = LLMAnalyzer(
        client=FakeTextClient(response),
        prompt_builder=build_prompt_builder(),
    )

    result = analyzer.analyze(instance)

    assert result == instance.gold_analysis


def test_llm_analyzer_rejects_invalid_json() -> None:
    instance = load_first_instance()

    analyzer = LLMAnalyzer(
        client=FakeTextClient(
            "This is not JSON."
        ),
        prompt_builder=build_prompt_builder(),
    )

    with pytest.raises(
        AnalyzerError,
        match="invalid JSON",
    ):
        analyzer.analyze(instance)


def test_llm_analyzer_rejects_invalid_schema() -> None:
    instance = load_first_instance()

    response = json.dumps(
        {
            "referents": [],
            "spans": [
                {
                    "span_id": "s1",
                    "text": "姐姐",
                }
            ],
            "protected_facts": [],
        },
        ensure_ascii=False,
    )

    analyzer = LLMAnalyzer(
        client=FakeTextClient(response),
        prompt_builder=build_prompt_builder(),
    )

    with pytest.raises(
        AnalyzerError,
        match="invalid AnalysisResult",
    ):
        analyzer.analyze(instance)


def test_llm_analyzer_repairs_span_offsets() -> None:
    instance = load_first_instance()

    payload = instance.gold_analysis.model_dump(
        mode="json",
        exclude_none=True,
    )

    # “姐姐”的正确位置是 [1:3]，故意提供错误位置。
    payload["spans"][0]["start"] = 0
    payload["spans"][0]["end"] = 2

    analyzer = LLMAnalyzer(
        client=FakeTextClient(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        ),
        prompt_builder=build_prompt_builder(),
    )

    result = analyzer.analyze(instance)

    span = result.spans[0]

    assert span.text == "姐姐"
    assert span.start == 1
    assert span.end == 3

def test_llm_analyzer_rejects_missing_span_text() -> None:
    instance = load_first_instance()

    payload = instance.gold_analysis.model_dump(
        mode="json",
        exclude_none=True,
    )

    payload["spans"][0]["text"] = "不存在的内容"

    analyzer = LLMAnalyzer(
        client=FakeTextClient(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        ),
        prompt_builder=build_prompt_builder(),
    )

    with pytest.raises(
        AnalyzerError,
        match="does not occur in target_text",
    ):
        analyzer.analyze(instance)

def test_prompt_builder_loads_yaml() -> None:
    builder = AnalysisPromptBuilder.from_yaml(
        PROJECT_ROOT
        / "configs"
        / "prompts"
        / "analyze_v0.1.yaml"
    )

    assert builder.version == "0.1"
    assert builder.system_instruction