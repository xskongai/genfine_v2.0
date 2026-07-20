#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 7/20/26
Description: test_enums
"""

# pip install -e .
from genfine.domain.enums import (
    Action,
    BiasMechanism,
    FunctionLabel,
    GenderValue,
    InstanceAction,
    Necessity,
)


def test_enum_is_string_serializable() -> None:
    assert Action.KEEP == "KEEP"
    assert str(Action.KEEP) == "KEEP"
    assert Necessity.ESSENTIAL.value == "ESSENTIAL"


def test_core_labels_exist() -> None:
    assert FunctionLabel.KINSHIP_RELATION.value == "KINSHIP_RELATION"
    assert FunctionLabel.HISTORICAL_SIGNIFICANCE.value == (
        "HISTORICAL_SIGNIFICANCE"
    )

    assert BiasMechanism.GENERIC_MALE_DEFAULT.value == (
        "GENERIC_MALE_DEFAULT"
    )
    assert BiasMechanism.ESSENTIALIST_GENERALIZATION.value == (
        "ESSENTIALIST_GENERALIZATION"
    )

    assert Action.PRESERVE_AMBIGUITY.value == "PRESERVE_AMBIGUITY"
    assert InstanceAction.SPAN_LEVEL_EDIT.value == "SPAN_LEVEL_EDIT"


def test_unknown_and_unspecified_are_distinct() -> None:
    assert GenderValue.UNKNOWN != GenderValue.UNSPECIFIED