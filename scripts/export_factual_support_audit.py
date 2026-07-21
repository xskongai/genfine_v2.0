from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "instance_id",
    "claim_id",
    "original_text",
    "output_text",
    "changed",
    "instance_action",
    "edit_scope",
    "plan_actions",
    "predicted_kind",
    "predicted_label",
    "predicted_evidence",
    "relevant_actions",
    "rationale",
    "confidence",

    # Assistant pre-audit; not human Gold.
    "assistant_suggested_label",
    "assistant_claim_atomic",
    "assistant_claim_coverage_complete",
    "assistant_evidence_valid",
    "assistant_action_license_valid",
    "assistant_label_correct",
    "assistant_review_status",
    "assistant_notes",

    # Fill these columns manually.
    "human_label",
    "human_claim_atomic",
    "human_claim_coverage_complete",
    "human_evidence_valid",
    "human_action_license_valid",
    "human_label_correct",
    "human_notes",
]


REVIEW_OVERRIDES: dict[
    tuple[str, str],
    dict[str, str],
] = {
    (
        "zh_seed_0004",
        "c1",
    ): {
        "assistant_claim_atomic": "REVIEW",
        "assistant_review_status": "REVIEW_ATOMICITY",
        "assistant_notes": (
            "当前 Claim 同时包含被引用的偏见命题和"
            "“错误观念”这一拒绝立场。可考虑分别审查"
            "引语内容与当前说话者立场。"
        ),
    },
    (
        "zh_seed_0009",
        "c1",
    ): {
        "assistant_claim_atomic": "FALSE",
        "assistant_label_correct": "REVIEW",
        "assistant_review_status": (
            "REVIEW_LABEL_AND_ATOMICITY"
        ),
        "assistant_notes": (
            "该 Claim 至少包含两个可独立判断的命题："
            "“细心是护理工作的一个重要特质”以及"
            "“具备该特质的人适合护理工作”。"
            "LICENSED_REFRAMING 有一定合理性，但需要"
            "人工判断是否加入了过强的职业适任性主张。"
        ),
    },
    (
        "zh_seed_0010",
        "c1",
    ): {
        "assistant_evidence_valid": "PARTIAL",
        "assistant_review_status": "REVIEW_EVIDENCE",
        "assistant_notes": (
            "SOURCE_SUPPORTED 标签总体合理，但 evidence "
            "只引用了单中心和样本有限的信息，没有同时引用"
            "目标文本中的不良反应观察，证据覆盖不完整。"
        ),
    },
    (
        "zh_seed_0015",
        "c1",
    ): {
        "assistant_claim_atomic": "REVIEW",
        "assistant_review_status": "REVIEW_ATOMICITY",
        "assistant_notes": (
            "当前 Claim 将“报告进行了记录”这一归属事实"
            "与引语内容合在一起。可考虑分别审查报告归属"
            "和被记录的命题。"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export GenFINE factual-support claims "
            "for manual auditing."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="runs/factual_support_v0.1.1.jsonl",
    )

    parser.add_argument(
        "--output",
        default=(
            "runs/"
            "factual_support_human_audit_v0.1.1.csv"
        ),
    )

    return parser.parse_args()


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: "
                    f"{exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Record at {path}:{line_number} "
                    "must be a JSON object."
                )

            records.append(payload)

    return records


def encode_list(value: Any) -> str:
    if not value:
        return ""

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def build_default_review(
    *,
    claim: dict[str, Any],
) -> dict[str, str]:
    label = str(
        claim["label"]
    )

    action_license = (
        "TRUE"
        if label == "LICENSED_REFRAMING"
        else "N/A"
    )

    return {
        "assistant_suggested_label": label,
        "assistant_claim_atomic": "TRUE",
        "assistant_claim_coverage_complete": "TRUE",
        "assistant_evidence_valid": "TRUE",
        "assistant_action_license_valid": action_license,
        "assistant_label_correct": "TRUE",
        "assistant_review_status": "PASS",
        "assistant_notes": "",
    }


def build_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in records:
        result = record.get(
            "factual_support"
        )

        if not isinstance(result, dict):
            continue

        claims = result.get(
            "claims",
            [],
        )

        for claim in claims:
            instance_id = str(
                record["instance_id"]
            )

            claim_id = str(
                claim["claim_id"]
            )

            review = build_default_review(
                claim=claim
            )

            review.update(
                REVIEW_OVERRIDES.get(
                    (
                        instance_id,
                        claim_id,
                    ),
                    {},
                )
            )

            row = {
                "instance_id": instance_id,
                "claim_id": claim_id,
                "original_text": record.get(
                    "original_text",
                    "",
                ),
                "output_text": record.get(
                    "output_text",
                    "",
                ),
                "changed": record.get(
                    "changed",
                    False,
                ),
                "instance_action": record.get(
                    "instance_action",
                    "",
                ),
                "edit_scope": record.get(
                    "edit_scope",
                    "",
                ),
                "plan_actions": encode_list(
                    record.get(
                        "plan_actions"
                    )
                ),
                "predicted_kind": claim.get(
                    "kind",
                    "",
                ),
                "predicted_label": claim.get(
                    "label",
                    "",
                ),
                "predicted_evidence": encode_list(
                    claim.get(
                        "evidence"
                    )
                ),
                "relevant_actions": encode_list(
                    claim.get(
                        "relevant_actions"
                    )
                ),
                "rationale": claim.get(
                    "rationale",
                    "",
                ),
                "confidence": claim.get(
                    "confidence",
                    "",
                ),
                **review,

                # Human fields intentionally remain blank.
                "human_label": "",
                "human_claim_atomic": "",
                "human_claim_coverage_complete": "",
                "human_evidence_valid": "",
                "human_action_license_valid": "",
                "human_label_correct": "",
                "human_notes": "",
            }

            rows.append(row)

    return rows


def write_csv(
    *,
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # utf-8-sig allows Excel to display Chinese correctly.
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    records = load_jsonl(
        input_path
    )

    rows = build_rows(
        records
    )

    if len(rows) != 17:
        raise ValueError(
            "Expected 17 claims from factual_support_v0.1.1, "
            f"but exported {len(rows)}."
        )

    write_csv(
        path=output_path,
        rows=rows,
    )

    review_count = sum(
        row["assistant_review_status"]
        != "PASS"
        for row in rows
    )

    print(f"Claims exported: {len(rows)}")
    print(f"Pre-audit PASS: {len(rows) - review_count}")
    print(f"Needs human review: {review_count}")
    print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())