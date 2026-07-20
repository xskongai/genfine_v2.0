from __future__ import annotations

import re
from collections import Counter

from genfine.domain.enums import (
    Action,
    Explicitness,
    GenderSource,
    GenderValue,
)
from genfine.domain.models import (
    AnalysisResult,
    DatasetInstance,
    EditPlan,
    VerificationIssue,
)
from genfine.verification.base import OutputChecker


GENDER_MARKER_VALUES: dict[str, GenderValue] = {
    "非二元性别": GenderValue.NON_BINARY,
    "非二元": GenderValue.NON_BINARY,

    "男外科医生": GenderValue.MALE,
    "女外科医生": GenderValue.FEMALE,

    "他们": GenderValue.MALE,
    "她们": GenderValue.FEMALE,

    "男性": GenderValue.MALE,
    "女性": GenderValue.FEMALE,
    "男人": GenderValue.MALE,
    "女人": GenderValue.FEMALE,
    "男孩": GenderValue.MALE,
    "女孩": GenderValue.FEMALE,

    "先生": GenderValue.MALE,
    "女士": GenderValue.FEMALE,

    "父亲": GenderValue.MALE,
    "母亲": GenderValue.FEMALE,
    "哥哥": GenderValue.MALE,
    "姐姐": GenderValue.FEMALE,
    "弟弟": GenderValue.MALE,
    "妹妹": GenderValue.FEMALE,
    "丈夫": GenderValue.MALE,
    "妻子": GenderValue.FEMALE,

    "他": GenderValue.MALE,
    "她": GenderValue.FEMALE,
    "男": GenderValue.MALE,
    "女": GenderValue.FEMALE,
}


UNRELIABLE_GENDER_SOURCES = {
    GenderSource.UNKNOWN,
    GenderSource.MODEL_INFERRED,
    GenderSource.STEREOTYPE_INFERRED,
}


_MARKER_PATTERN = re.compile(
    "|".join(
        re.escape(marker)
        for marker in sorted(
            GENDER_MARKER_VALUES,
            key=len,
            reverse=True,
        )
    )
)


class GenderInsertionChecker(OutputChecker):
    """Detect gender markers newly introduced without source grounding."""

    @property
    def name(self) -> str:
        return "gender_insertion"

    def check(
        self,
        *,
        instance: DatasetInstance,
        analysis: AnalysisResult,
        edit_plan: EditPlan,
        output_text: str | None,
    ) -> list[VerificationIssue]:
        if output_text is None:
            return []

        original = instance.context.target_text

        original_markers = extract_gender_markers(
            original
        )

        output_markers = extract_gender_markers(
            output_text
        )

        introduced_markers = {
            marker: output_count
            - original_markers.get(marker, 0)
            for marker, output_count
            in output_markers.items()
            if (
                output_count
                > original_markers.get(marker, 0)
            )
        }

        if not introduced_markers:
            return []

        strict_ambiguity = any(
            decision.action
            == Action.PRESERVE_AMBIGUITY
            for decision in edit_plan.span_decisions
        )

        grounded_values = {
            span.gender_value
            for span in analysis.spans
            if (
                span.explicitness
                != Explicitness.NONE
                and span.gender_value
                not in {
                    GenderValue.UNKNOWN,
                    GenderValue.UNSPECIFIED,
                    GenderValue.NOT_APPLICABLE,
                }
                and span.source
                not in UNRELIABLE_GENDER_SOURCES
            )
        }

        issues: list[VerificationIssue] = []

        for marker, added_count in introduced_markers.items():
            marker_value = (
                GENDER_MARKER_VALUES[marker]
            )

            if (
                not strict_ambiguity
                and marker_value in grounded_values
            ):
                # Example: source “女人” rewritten as “女性”.
                # The lexical marker changed, but the gender value was grounded.
                continue

            severity = (
                "ERROR"
                if strict_ambiguity
                else "WARNING"
            )

            code = (
                "UNSUPPORTED_GENDER_INSERTION"
                if strict_ambiguity
                else "POSSIBLE_UNSUPPORTED_GENDER_INSERTION"
            )

            issues.append(
                VerificationIssue(
                    checker=self.name,
                    code=code,
                    message=(
                        f"Output introduced gender marker "
                        f"{marker!r} {added_count} time(s) "
                        "without sufficient source grounding."
                    ),
                    severity=severity,
                    span_text=marker,
                )
            )

        return issues


def extract_gender_markers(
    text: str,
) -> Counter[str]:
    """
    Extract non-overlapping gender markers.

    Longest markers are matched first by the compiled regular expression,
    avoiding double-counting “他们” as both “他们” and “他”.
    """

    return Counter(
        match.group(0)
        for match in _MARKER_PATTERN.finditer(
            text
        )
    )