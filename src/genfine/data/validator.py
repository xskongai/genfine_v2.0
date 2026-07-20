#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 7/20/26
Description: validator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from genfine.data.loader import LoadedInstance
from genfine.domain.enums import (
    Action,
    EditScope,
    InstanceAction,
)


Severity = Literal["ERROR", "WARNING"]


KEEP_LIKE_ACTIONS = {
    Action.KEEP,
    Action.KEEP_WITH_ATTRIBUTION,
    Action.PRESERVE_AMBIGUITY,
}


@dataclass(frozen=True)
class ValidationIssue:
    """One dataset-level validation error or warning."""

    severity: Severity
    code: str
    message: str

    instance_id: str | None = None
    line_number: int | None = None


@dataclass
class DatasetValidationReport:
    """Aggregated dataset validation result."""

    instance_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == "ERROR"
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == "WARNING"
        ]

    @property
    def passed(self) -> bool:
        return not self.errors


class DatasetValidator:
    """Validate relationships that involve complete dataset instances."""

    def validate(
        self,
        records: list[LoadedInstance],
    ) -> DatasetValidationReport:
        report = DatasetValidationReport(
            instance_count=len(records)
        )

        self._validate_unique_instance_ids(
            records,
            report,
        )

        for record in records:
            self._validate_record(
                record,
                report,
            )

        return report

    def _validate_unique_instance_ids(
        self,
        records: list[LoadedInstance],
        report: DatasetValidationReport,
    ) -> None:
        first_occurrence: dict[str, LoadedInstance] = {}

        for record in records:
            instance_id = record.instance.instance_id

            if instance_id not in first_occurrence:
                first_occurrence[instance_id] = record
                continue

            original = first_occurrence[instance_id]

            report.issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="DUPLICATE_INSTANCE_ID",
                    message=(
                        f"instance_id {instance_id!r} is duplicated; "
                        f"first occurrence is at line "
                        f"{original.line_number}"
                    ),
                    instance_id=instance_id,
                    line_number=record.line_number,
                )
            )

    def _validate_record(
        self,
        record: LoadedInstance,
        report: DatasetValidationReport,
    ) -> None:
        self._validate_span_action_coverage(
            record,
            report,
        )

        self._validate_instance_action(
            record,
            report,
        )

        self._validate_edit_scope(
            record,
            report,
        )

        self._validate_gold_output(
            record,
            report,
        )

        self._validate_metadata(
            record,
            report,
        )

    def _validate_span_action_coverage(
        self,
        record: LoadedInstance,
        report: DatasetValidationReport,
    ) -> None:
        instance = record.instance

        analysis_span_ids = {
            span.span_id
            for span in instance.gold_analysis.spans
        }

        decision_span_ids = {
            item.span_id
            for item in instance.gold_decision.span_actions
        }

        missing_decisions = (
            analysis_span_ids - decision_span_ids
        )

        if missing_decisions:
            self._add_issue(
                report=report,
                record=record,
                severity="ERROR",
                code="MISSING_SPAN_ACTION",
                message=(
                    "analysis spans have no gold action: "
                    f"{sorted(missing_decisions)}"
                ),
            )

        extra_decisions = (
            decision_span_ids - analysis_span_ids
        )

        if extra_decisions:
            self._add_issue(
                report=report,
                record=record,
                severity="ERROR",
                code="UNKNOWN_SPAN_ACTION",
                message=(
                    "gold actions refer to unknown spans: "
                    f"{sorted(extra_decisions)}"
                ),
            )

    def _validate_instance_action(
        self,
        record: LoadedInstance,
        report: DatasetValidationReport,
    ) -> None:
        instance = record.instance

        actions = [
            item.action
            for item in instance.gold_decision.span_actions
        ]

        expected = infer_instance_action(actions)
        actual = instance.gold_decision.instance_action

        if actual != expected:
            self._add_issue(
                report=report,
                record=record,
                severity="ERROR",
                code="INSTANCE_ACTION_MISMATCH",
                message=(
                    f"instance_action is {actual.value}, "
                    f"but span actions imply {expected.value}"
                ),
            )

    def _validate_edit_scope(
        self,
        record: LoadedInstance,
        report: DatasetValidationReport,
    ) -> None:
        decision = record.instance.gold_decision

        if (
            decision.instance_action
            in {
                InstanceAction.KEEP,
                InstanceAction.ABSTAIN,
            }
            and decision.edit_scope != EditScope.NONE
        ):
            self._add_issue(
                report=report,
                record=record,
                severity="ERROR",
                code="INVALID_KEEP_EDIT_SCOPE",
                message=(
                    f"{decision.instance_action.value} requires "
                    "edit_scope=NONE"
                ),
            )

        if (
            decision.instance_action == InstanceAction.EDIT
            and decision.edit_scope == EditScope.NONE
        ):
            self._add_issue(
                report=report,
                record=record,
                severity="ERROR",
                code="MISSING_EDIT_SCOPE",
                message="EDIT requires a non-NONE edit scope",
            )

        if (
            decision.instance_action
            == InstanceAction.SPAN_LEVEL_EDIT
            and decision.edit_scope
            not in {
                EditScope.SPAN,
                EditScope.MULTI_SPAN,
                EditScope.SENTENCE,
            }
        ):
            self._add_issue(
                report=report,
                record=record,
                severity="ERROR",
                code="INVALID_SPAN_LEVEL_SCOPE",
                message=(
                    "SPAN_LEVEL_EDIT requires SPAN, MULTI_SPAN "
                    "or SENTENCE edit scope"
                ),
            )

    def _validate_gold_output(
        self,
        record: LoadedInstance,
        report: DatasetValidationReport,
    ) -> None:
        instance = record.instance

        original = instance.context.target_text
        output = instance.gold_output
        action = instance.gold_decision.instance_action

        if action == InstanceAction.KEEP:
            if output != original:
                self._add_issue(
                    report=report,
                    record=record,
                    severity="ERROR",
                    code="KEEP_OUTPUT_CHANGED",
                    message=(
                        "instance_action is KEEP, but gold_output "
                        "differs from target_text"
                    ),
                )

        elif action in {
            InstanceAction.EDIT,
            InstanceAction.SPAN_LEVEL_EDIT,
        }:
            if output == original:
                self._add_issue(
                    report=report,
                    record=record,
                    severity="ERROR",
                    code="EDIT_OUTPUT_UNCHANGED",
                    message=(
                        f"instance_action is {action.value}, "
                        "but gold_output is unchanged"
                    ),
                )

        elif action == InstanceAction.ABSTAIN:
            if output is not None:
                self._add_issue(
                    report=report,
                    record=record,
                    severity="ERROR",
                    code="ABSTAIN_HAS_OUTPUT",
                    message=(
                        "ABSTAIN instances must not have gold_output"
                    ),
                )

    def _validate_metadata(
        self,
        record: LoadedInstance,
        report: DatasetValidationReport,
    ) -> None:
        metadata = record.instance.metadata

        if not metadata.scenario_category:
            self._add_issue(
                report=report,
                record=record,
                severity="WARNING",
                code="MISSING_SCENARIO_CATEGORY",
                message="metadata.scenario_category is empty",
            )

        if not metadata.annotation_version:
            self._add_issue(
                report=report,
                record=record,
                severity="WARNING",
                code="MISSING_ANNOTATION_VERSION",
                message="metadata.annotation_version is empty",
            )

    @staticmethod
    def _add_issue(
        *,
        report: DatasetValidationReport,
        record: LoadedInstance,
        severity: Severity,
        code: str,
        message: str,
    ) -> None:
        report.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                instance_id=record.instance.instance_id,
                line_number=record.line_number,
            )
        )


def infer_instance_action(
    actions: list[Action],
) -> InstanceAction:
    """
    Aggregate span-level actions into an instance-level action.

    KEEP_WITH_ATTRIBUTION and PRESERVE_AMBIGUITY do not modify the surface
    text, so they are treated as KEEP-like actions.
    """

    if not actions:
        return InstanceAction.KEEP

    if Action.ABSTAIN in actions:
        return InstanceAction.ABSTAIN

    keep_actions = [
        action
        for action in actions
        if action in KEEP_LIKE_ACTIONS
    ]

    edit_actions = [
        action
        for action in actions
        if action not in KEEP_LIKE_ACTIONS
    ]

    if not edit_actions:
        return InstanceAction.KEEP

    unique_edit_actions = set(edit_actions)

    if keep_actions:
        return InstanceAction.SPAN_LEVEL_EDIT

    if len(unique_edit_actions) > 1:
        return InstanceAction.SPAN_LEVEL_EDIT

    return InstanceAction.EDIT