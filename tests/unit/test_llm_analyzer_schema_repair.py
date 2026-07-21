from __future__ import annotations

import copy

import pytest

from genfine.analysis.base import AnalyzerError
from genfine.analysis.llm_analyzer import LLMAnalyzer


def minimal_analysis_payload() -> dict:
    return {
        "referents": [],
        "spans": [],
        "protected_facts": [],
        "speaker_stance": "NOT_APPLICABLE",
        "context_sufficient": True,
    }


def test_minimal_repair_removes_only_top_level_confidence() -> None:
    payload = minimal_analysis_payload()
    payload["confidence"] = 1.0
    payload["analysis_notes"] = "keep this field"

    repaired = LLMAnalyzer._repair_schema_payload(
        payload=payload,
    )

    assert "confidence" not in repaired
    assert repaired["analysis_notes"] == "keep this field"
    assert payload["confidence"] == 1.0


def test_minimal_repair_moves_bias_mechanism_from_functions() -> None:
    payload = minimal_analysis_payload()
    payload["spans"] = [
        {
            "span_id": "s1",
            "text": "女人不适合做工程师",
            "start": 0,
            "end": 9,
            "cue_type": "STEREOTYPICAL_ATTRIBUTE",
            "explicitness": "EXPLICIT",
            "gender_dimension": "SOCIAL_GENDER_ROLE",
            "gender_value": "FEMALE",
            "source": "QUOTED_SOURCE",
            "source_reliability": "CONFIRMED",
            "functions": [
                "DIRECT_QUOTATION",
                "STEREOTYPE_ASSOCIATION",
            ],
            "necessity": {
                "status": "ESSENTIAL",
                "reasons": ["QUOTATION_FIDELITY"],
            },
            "bias": {
                "status": "EXPLICIT",
                "mechanisms": [
                    "COMPETENCE_OR_AGENCY_DENIAL",
                    "STEREOTYPE_ASSOCIATION",
                ],
            },
            "stance": "REPORT",
        }
    ]

    original = copy.deepcopy(payload)

    repaired = LLMAnalyzer._repair_schema_payload(
        payload=payload,
    )

    repaired_span = repaired["spans"][0]

    assert repaired_span["functions"] == [
        "DIRECT_QUOTATION"
    ]
    assert repaired_span["bias"]["mechanisms"] == [
        "COMPETENCE_OR_AGENCY_DENIAL",
        "STEREOTYPE_ASSOCIATION",
    ]
    assert payload == original


def test_validation_accepts_top_level_confidence_after_repair() -> None:
    payload = minimal_analysis_payload()
    payload["confidence"] = 1.0

    analysis = LLMAnalyzer._validate_analysis_payload(
        payload=payload,
        instance_id="repair_case",
    )

    assert analysis.spans == []
    assert analysis.context_sufficient is True


def test_unknown_function_is_not_guessed_or_removed() -> None:
    payload = minimal_analysis_payload()
    payload["spans"] = [
        {
            "span_id": "s1",
            "text": "她",
            "start": 0,
            "end": 1,
            "cue_type": "PRONOUN",
            "explicitness": "EXPLICIT",
            "gender_dimension": "GENDER_REFERENCE",
            "gender_value": "FEMALE",
            "source": "EXPLICITLY_STATED",
            "source_reliability": "CONFIRMED",
            "functions": ["NOT_A_REAL_LABEL"],
            "necessity": {
                "status": "RELEVANT",
                "reasons": [],
            },
            "bias": {
                "status": "NONE",
                "mechanisms": [],
            },
            "stance": "NOT_APPLICABLE",
        }
    ]

    with pytest.raises(
        AnalyzerError,
        match="Validation error after minimal repair",
    ):
        LLMAnalyzer._validate_analysis_payload(
            payload=payload,
            instance_id="unknown_label_case",
        )
