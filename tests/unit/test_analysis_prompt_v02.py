from __future__ import annotations

from pathlib import Path

from genfine.analysis import AnalysisPromptBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prompts"
    / "analyze_v0.2.yaml"
)


def test_analysis_prompt_v02_loads() -> None:
    builder = AnalysisPromptBuilder.from_yaml(
        PROMPT_PATH
    )

    assert builder.version == "0.2"
    assert builder.system_instruction


def test_analysis_prompt_v02_defines_decision_unit() -> None:
    builder = AnalysisPromptBuilder.from_yaml(
        PROMPT_PATH
    )

    instruction = builder.system_instruction

    assert "能够独立接受一个编辑决策" in instruction
    assert "完整偏见命题" in instruction
    assert "统计、医学和范围限定命题" in instruction
    assert "同一句中的混合功能" in instruction


def test_analysis_prompt_v02_guards_schema_fields() -> None:
    builder = AnalysisPromptBuilder.from_yaml(
        PROMPT_PATH
    )

    instruction = builder.system_instruction

    assert "FunctionLabel 与 BiasMechanism 必须分开" in instruction
    assert "顶层不得输出 confidence" in instruction
    assert "不得创造 Schema 中不存在的标签" in instruction
