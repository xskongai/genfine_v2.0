from __future__ import annotations

from pathlib import Path
from typing import Any

from genfine.domain.enums import Action, TaskMode
from genfine.domain.models import (
    AnalysisResult,
    AnalysisSpan,
    SpanDecision,
)
from genfine.policy.rule_loader import (
    DecisionRule,
    DecisionRuleSet,
    load_rule_set,
)
from genfine.policy.rule_matcher import (
    match_condition,
)


class DecisionEngineError(RuntimeError):
    """Raised when the policy engine cannot produce a decision."""


class DecisionEngine:
    """
    Deterministic policy engine mapping structured analysis to span actions.
    """

    def __init__(
        self,
        rule_set: DecisionRuleSet,
    ) -> None:
        self.rule_set = rule_set

        # Python sorting is stable. Rules with equal priority retain
        # their YAML order.
        self.rules = sorted(
            rule_set.rules,
            key=lambda rule: rule.priority,
            reverse=True,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "DecisionEngine":
        return cls(
            load_rule_set(path)
        )

    @property
    def version(self) -> str:
        return self.rule_set.version

    def decide_span(
        self,
        *,
        span: AnalysisSpan,
        analysis: AnalysisResult,
        task_mode: TaskMode,
    ) -> SpanDecision:
        """Select the highest-priority matching rule for one span."""

        context = {
            "span": span,
            "analysis": analysis,
            "task_mode": task_mode,
        }

        for rule in self.rules:
            if match_condition(
                context,
                rule.when,
            ):
                return self._build_decision(
                    span=span,
                    analysis=analysis,
                    rule=rule,
                )

        raise DecisionEngineError(
            f"No decision rule matched span "
            f"{span.span_id!r}. "
            "The rule configuration should contain a fallback rule."
        )

    def decide_analysis(
        self,
        *,
        analysis: AnalysisResult,
        task_mode: TaskMode,
    ) -> list[SpanDecision]:
        """Produce one decision for every analysis span."""

        return [
            self.decide_span(
                span=span,
                analysis=analysis,
                task_mode=task_mode,
            )
            for span in analysis.spans
        ]

    def _build_decision(
        self,
        *,
        span: AnalysisSpan,
        analysis: AnalysisResult,
        rule: DecisionRule,
    ) -> SpanDecision:
        constraints = list(rule.constraints)

        constraints.extend(
            self._protected_fact_constraints(
                span=span,
                analysis=analysis,
            )
        )

        constraints.extend(
            self._dynamic_action_constraints(
                span=span,
                action=rule.action,
            )
        )

        return SpanDecision(
            span_id=span.span_id,
            action=rule.action,
            rule_id=rule.id,
            reason_code=rule.reason_code,
            priority=rule.priority,
            confidence=self._decision_confidence(
                span
            ),
            constraints=_deduplicate(
                constraints
            ),
        )

    @staticmethod
    def _decision_confidence(
        span: AnalysisSpan,
    ) -> float:
        """
        A policy decision cannot be more confident than its weakest
        underlying structured judgment.
        """

        return min(
            span.confidence,
            span.necessity.confidence,
            span.bias.confidence,
        )

    @staticmethod
    def _protected_fact_constraints(
        *,
        span: AnalysisSpan,
        analysis: AnalysisResult,
    ) -> list[str]:
        constraints: list[str] = []

        for fact in analysis.protected_facts:
            if not fact.must_preserve:
                continue

            is_global_fact = not fact.source_span_ids
            belongs_to_span = (
                span.span_id
                in fact.source_span_ids
            )

            if is_global_fact or belongs_to_span:
                constraints.append(
                    f"Preserve protected fact "
                    f"{fact.fact_id}: "
                    f"{fact.description}"
                )

        return constraints

    @staticmethod
    def _dynamic_action_constraints(
        *,
        span: AnalysisSpan,
        action: Action,
    ) -> list[str]:
        if action == Action.KEEP:
            return [
                f"Preserve the meaning and function of "
                f"the span {span.text!r}."
            ]

        if action == Action.KEEP_WITH_ATTRIBUTION:
            return [
                f"Preserve the quoted or reported span "
                f"{span.text!r}.",
                "Preserve the distinction between quoted content "
                "and the current speaker's stance.",
            ]

        if action == Action.PRESERVE_AMBIGUITY:
            return [
                "Do not add any unsupported gender identity, "
                "pronoun or gender marker."
            ]

        if action == Action.REPLACE_GENERIC_FORM:
            return [
                f"Replace the generic gender form {span.text!r} "
                "with a natural inclusive form."
            ]

        if action == Action.UNMARK:
            return [
                f"Remove only the unnecessary gender marking "
                f"within {span.text!r}."
            ]

        if action == Action.REFRAME_PROPOSITION:
            return [
                f"Reframe the biased proposition {span.text!r} "
                "without deleting surrounding protected facts."
            ]

        if action == Action.ADD_SCOPE_OR_QUALIFICATION:
            return [
                f"Add appropriate scope or qualification to "
                f"{span.text!r}."
            ]

        if action == Action.ABSTAIN:
            return [
                "Do not generate a definitive rewrite."
            ]

        return []


def _deduplicate(
    values: list[str],
) -> list[str]:
    """Deduplicate strings while preserving their original order."""

    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result