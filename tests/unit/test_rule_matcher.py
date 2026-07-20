from pathlib import Path

import pytest

from genfine.data.loader import load_dataset
from genfine.policy.rule_matcher import (
    RuleMatchError,
    match_condition,
    resolve_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.1.jsonl"
)


def build_context() -> dict:
    instance = load_dataset(SEED_PATH)[2]
    span = instance.gold_analysis.spans[0]

    return {
        "span": span,
        "analysis": instance.gold_analysis,
        "task_mode": instance.task_mode,
    }


def test_resolve_nested_enum_path() -> None:
    context = build_context()

    result = resolve_path(
        context,
        "span.necessity.status",
    )

    assert result.value == "INCIDENTAL"


def test_eq_operator() -> None:
    context = build_context()

    assert match_condition(
        context,
        {
            "path": "span.necessity.status",
            "op": "eq",
            "value": "INCIDENTAL",
        },
    )


def test_contains_operator() -> None:
    context = build_context()

    assert match_condition(
        context,
        {
            "path": "span.bias.mechanisms",
            "op": "contains",
            "value": "GENERIC_MALE_DEFAULT",
        },
    )


def test_intersects_operator() -> None:
    context = build_context()

    assert match_condition(
        context,
        {
            "path": "span.functions",
            "op": "intersects",
            "value": [
                "GENERIC_REFERENCE",
                "COREFERENCE",
            ],
        },
    )


def test_all_group() -> None:
    context = build_context()

    assert match_condition(
        context,
        {
            "all": [
                {
                    "path": "span.necessity.status",
                    "op": "eq",
                    "value": "INCIDENTAL",
                },
                {
                    "path": "span.bias.mechanisms",
                    "op": "contains",
                    "value": "GENERIC_MALE_DEFAULT",
                },
            ]
        },
    )


def test_any_group() -> None:
    context = build_context()

    assert match_condition(
        context,
        {
            "any": [
                {
                    "path": "span.stance",
                    "op": "eq",
                    "value": "REJECT",
                },
                {
                    "path": "span.functions",
                    "op": "contains",
                    "value": "GENERIC_REFERENCE",
                },
            ]
        },
    )


def test_not_group() -> None:
    context = build_context()

    assert match_condition(
        context,
        {
            "not": {
                "path": "span.bias.status",
                "op": "eq",
                "value": "NONE",
            }
        },
    )


def test_unknown_path_fails_loudly() -> None:
    context = build_context()

    with pytest.raises(
        RuleMatchError,
        match="unknown rule path",
    ):
        match_condition(
            context,
            {
                "path": "span.missing_field",
                "op": "eq",
                "value": "x",
            },
        )