from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from genfine.domain.enums import Action
from genfine.policy.rule_matcher import (
    RuleMatchError,
    validate_condition,
)


class RuleConfigError(ValueError):
    """Raised when a decision-rule configuration is invalid."""


class DecisionRule(BaseModel):
    """One declarative policy rule."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    priority: int = 0
    description: str = ""

    when: dict[str, Any]

    action: Action
    reason_code: str = Field(min_length=1)

    constraints: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_when_expression(
        self,
    ) -> "DecisionRule":
        try:
            validate_condition(self.when)
        except RuleMatchError as exc:
            raise ValueError(
                f"invalid condition in rule {self.id!r}: {exc}"
            ) from exc

        return self


class DecisionRuleSet(BaseModel):
    """Versioned collection of decision rules."""

    model_config = ConfigDict(
        extra="forbid",
    )

    version: str = Field(min_length=1)
    rules: list[DecisionRule] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_rules(
        self,
    ) -> "DecisionRuleSet":
        rule_ids = [
            rule.id
            for rule in self.rules
        ]

        duplicates = {
            rule_id
            for rule_id in rule_ids
            if rule_ids.count(rule_id) > 1
        }

        if duplicates:
            raise ValueError(
                f"duplicate rule ids: {sorted(duplicates)}"
            )

        if not any(
            rule.when.get("always") is True
            for rule in self.rules
        ):
            raise ValueError(
                "rule set must contain an always=true fallback rule"
            )

        return self


def load_rule_set(
    path: str | Path,
) -> DecisionRuleSet:
    """Load and validate a YAML decision-rule file."""

    rule_path = Path(path)

    if not rule_path.exists():
        raise FileNotFoundError(
            f"Decision rule file does not exist: {rule_path}"
        )

    try:
        raw_data = yaml.safe_load(
            rule_path.read_text(
                encoding="utf-8"
            )
        )
    except yaml.YAMLError as exc:
        raise RuleConfigError(
            f"Invalid YAML in {rule_path}: {exc}"
        ) from exc

    if raw_data is None:
        raise RuleConfigError(
            f"Rule file is empty: {rule_path}"
        )

    try:
        return DecisionRuleSet.model_validate(
            raw_data
        )
    except ValidationError as exc:
        raise RuleConfigError(
            f"Invalid rule configuration in {rule_path}: "
            f"{exc}"
        ) from exc