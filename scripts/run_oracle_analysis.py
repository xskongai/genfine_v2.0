from __future__ import annotations

import argparse
import json
from pathlib import Path

from genfine.analysis import OracleAnalyzer
from genfine.data.loader import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Oracle analysis results "
            "for a GenFINE dataset."
        )
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="data/seed/seed_v0.1.jsonl",
        help="Input GenFINE JSONL dataset.",
    )

    parser.add_argument(
        "--output",
        default="runs/oracle_analysis_v0.1.jsonl",
        help="Output analysis JSONL path.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    instances = load_dataset(dataset_path)
    analyzer = OracleAnalyzer()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for instance in instances:
            analysis = analyzer.analyze(instance)

            payload = {
                "instance_id": instance.instance_id,
                "original_text": (
                    instance.context.target_text
                ),
                "analyzer": analyzer.name,
                "analysis": analysis.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
            }

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

            file.write("\n")

    print(
        f"Wrote {len(instances)} Oracle analyses "
        f"to {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    raise SystemExit(main())