from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any


class RuleMatchError(ValueError):
    """Raised when a rule condition is invalid or cannot be evaluated."""


SUPPORTED_OPERATORS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "contains",
    "intersects",
    "is_true",
    "is_false",
}


def normalize_value(value: Any) -> Any:
    """Convert enums and nested collections into comparable plain values."""

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {
            key: normalize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple | list | set | frozenset):
        return [
            normalize_value(item)
            for item in value
        ]

    return value


def resolve_path(
    root: Any,
    path: str,
) -> Any:
    """
    Resolve a dotted path from dictionaries or object attributes.

    Example:
        span.necessity.status
    """

    if not path:
        raise RuleMatchError("condition path cannot be empty")

    current = root

    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise RuleMatchError(
                    f"unknown rule path {path!r}: "
                    f"mapping has no key {part!r}"
                )

            current = current[part]
            continue

        if not hasattr(current, part):
            raise RuleMatchError(
                f"unknown rule path {path!r}: "
                f"{type(current).__name__} has no attribute {part!r}"
            )

        current = getattr(current, part)

    return current


def validate_condition(
    condition: dict[str, Any],
) -> None:
    """Validate the structure of a rule condition recursively."""

    if not isinstance(condition, dict):
        raise RuleMatchError(
            "each condition must be a mapping"
        )

    if "always" in condition:
        if set(condition) != {"always"}:
            raise RuleMatchError(
                "'always' cannot be combined with other condition keys"
            )

        if not isinstance(condition["always"], bool):
            raise RuleMatchError(
                "'always' must be a boolean"
            )

        return

    group_keys = {
        key
        for key in ("all", "any", "not")
        if key in condition
    }

    if group_keys:
        if len(group_keys) != 1:
            raise RuleMatchError(
                "a condition can contain only one of all/any/not"
            )

        group_key = next(iter(group_keys))

        if set(condition) != {group_key}:
            raise RuleMatchError(
                f"{group_key!r} cannot be combined with leaf keys"
            )

        nested = condition[group_key]

        if group_key == "not":
            validate_condition(nested)
            return

        if not isinstance(nested, list) or not nested:
            raise RuleMatchError(
                f"{group_key!r} must contain a non-empty list"
            )

        for item in nested:
            validate_condition(item)

        return

    required_keys = {"path", "op"}

    if not required_keys.issubset(condition):
        raise RuleMatchError(
            "leaf conditions require 'path' and 'op'"
        )

    allowed_keys = {"path", "op", "value"}

    unknown_keys = set(condition) - allowed_keys

    if unknown_keys:
        raise RuleMatchError(
            f"unknown condition keys: {sorted(unknown_keys)}"
        )

    operator = condition["op"]

    if operator not in SUPPORTED_OPERATORS:
        raise RuleMatchError(
            f"unsupported operator: {operator!r}"
        )

    if operator not in {"is_true", "is_false"}:
        if "value" not in condition:
            raise RuleMatchError(
                f"operator {operator!r} requires a value"
            )


def match_condition(
    context: Any,
    condition: dict[str, Any],
) -> bool:
    """Evaluate one condition or condition group."""

    validate_condition(condition)

    if "always" in condition:
        return condition["always"]

    if "all" in condition:
        return all(
            match_condition(context, item)
            for item in condition["all"]
        )

    if "any" in condition:
        return any(
            match_condition(context, item)
            for item in condition["any"]
        )

    if "not" in condition:
        return not match_condition(
            context,
            condition["not"],
        )

    actual = normalize_value(
        resolve_path(
            context,
            condition["path"],
        )
    )

    operator = condition["op"]
    expected = normalize_value(
        condition.get("value")
    )

    if operator == "eq":
        return actual == expected

    if operator == "ne":
        return actual != expected

    if operator == "in":
        _require_collection(
            expected,
            operator=operator,
            side="expected",
        )

        return actual in expected

    if operator == "not_in":
        _require_collection(
            expected,
            operator=operator,
            side="expected",
        )

        return actual not in expected

    if operator == "contains":
        _require_collection(
            actual,
            operator=operator,
            side="actual",
            allow_string=True,
        )

        return expected in actual

    if operator == "intersects":
        _require_collection(
            actual,
            operator=operator,
            side="actual",
        )

        _require_collection(
            expected,
            operator=operator,
            side="expected",
        )

        return bool(
            set(actual).intersection(expected)
        )

    if operator == "is_true":
        return actual is True

    if operator == "is_false":
        return actual is False

    raise RuleMatchError(
        f"operator was validated but not implemented: {operator}"
    )


def _require_collection(
    value: Any,
    *,
    operator: str,
    side: str,
    allow_string: bool = False,
) -> None:
    """Ensure an operator receives a collection-like value."""

    if allow_string and isinstance(value, str):
        return

    if isinstance(value, str):
        raise RuleMatchError(
            f"{operator!r} requires {side} to be a collection, "
            "not a string"
        )

    if not isinstance(value, Sequence | set | frozenset):
        raise RuleMatchError(
            f"{operator!r} requires {side} to be a collection"
        )