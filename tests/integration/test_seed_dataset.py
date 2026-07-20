from pathlib import Path

from genfine.domain.models import DatasetInstance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = (
    PROJECT_ROOT
    / "data"
    / "seed"
    / "seed_v0.1.jsonl"
)


def load_seed_instances() -> list[DatasetInstance]:
    instances: list[DatasetInstance] = []

    for line_number, line in enumerate(
        SEED_PATH.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            instance = DatasetInstance.model_validate_json(line)
        except Exception as exc:
            raise AssertionError(
                f"Invalid seed instance at line {line_number}"
            ) from exc

        instances.append(instance)

    return instances


def test_seed_file_exists() -> None:
    assert SEED_PATH.exists()


def test_seed_file_contains_valid_instances() -> None:
    instances = load_seed_instances()

    assert len(instances) == 5


def test_seed_instance_ids_are_unique() -> None:
    instances = load_seed_instances()
    instance_ids = [
        instance.instance_id
        for instance in instances
    ]

    assert len(instance_ids) == len(set(instance_ids))


def test_first_seed_instance_is_keep_case() -> None:
    instances = load_seed_instances()
    first_instance = instances[0]

    assert first_instance.instance_id == "zh_seed_0001"
    assert (
        first_instance.context.target_text
        == "我姐姐打电话说她会回来"
    )
    assert (
        first_instance.gold_decision.instance_action.value
        == "KEEP"
    )


def test_all_span_offsets_match_text() -> None:
    instances = load_seed_instances()

    for instance in instances:
        target_text = instance.context.target_text

        for span in instance.gold_analysis.spans:
            assert (
                target_text[span.start:span.end]
                == span.text
            )

def test_seed_dataset_covers_core_instance_actions() -> None:
    instances = load_seed_instances()

    actions = {
        instance.gold_decision.instance_action.value
        for instance in instances
    }

    assert "KEEP" in actions
    assert "EDIT" in actions


def test_seed_dataset_covers_core_span_actions() -> None:
    instances = load_seed_instances()

    span_actions = {
        span_action.action.value
        for instance in instances
        for span_action in instance.gold_decision.span_actions
    }

    assert "KEEP" in span_actions
    assert "KEEP_WITH_ATTRIBUTION" in span_actions
    assert "REPLACE_GENERIC_FORM" in span_actions
    assert "PRESERVE_AMBIGUITY" in span_actions


def test_edit_instances_change_the_text() -> None:
    instances = load_seed_instances()

    for instance in instances:
        if instance.gold_decision.instance_action.value == "EDIT":
            assert (
                instance.gold_output
                != instance.context.target_text
            )


def test_keep_instances_preserve_the_text() -> None:
    instances = load_seed_instances()

    for instance in instances:
        if instance.gold_decision.instance_action.value == "KEEP":
            assert (
                instance.gold_output
                == instance.context.target_text
            )


def test_scenario_categories_are_unique() -> None:
    instances = load_seed_instances()

    categories = [
        instance.metadata.scenario_category
        for instance in instances
    ]

    assert len(categories) == len(set(categories))