from pathlib import Path

from genfine.data.loader import load_dataset, load_dataset_records
from genfine.data.validator import DatasetValidator
from genfine.domain.enums import InstanceAction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = PROJECT_ROOT / "data" / "seed" / "seed_v0.2.jsonl"


def test_seed_v02_has_fifteen_instances() -> None:
    instances = load_dataset(SEED_PATH)

    assert len(instances) == 15
    assert len({item.instance_id for item in instances}) == 15


def test_seed_v02_dataset_validation_passes() -> None:
    records = load_dataset_records(SEED_PATH)
    report = DatasetValidator().validate(records)

    assert report.errors == []


def test_all_protected_facts_are_machine_checkable() -> None:
    instances = load_dataset(SEED_PATH)

    missing = [
        (instance.instance_id, fact.fact_id)
        for instance in instances
        for fact in instance.gold_analysis.protected_facts
        if (
            fact.must_preserve
            and not fact.required_output_phrases
            and not fact.forbidden_output_phrases
        )
    ]

    assert missing == []


def test_context_flip_pair_uses_same_text_but_different_actions() -> None:
    instances = {
        item.instance_id: item
        for item in load_dataset(SEED_PATH)
    }

    routine = instances["zh_seed_0006"]
    representation = instances["zh_seed_0007"]

    assert (
        routine.context.target_text
        == representation.context.target_text
    )

    assert (
        routine.gold_decision.instance_action
        == InstanceAction.EDIT
    )

    assert (
        representation.gold_decision.instance_action
        == InstanceAction.KEEP
    )


def test_seed_v02_contains_all_instance_action_types() -> None:
    instances = load_dataset(SEED_PATH)

    actions = {
        item.gold_decision.instance_action
        for item in instances
    }

    assert actions == {
        InstanceAction.KEEP,
        InstanceAction.EDIT,
        InstanceAction.SPAN_LEVEL_EDIT,
        InstanceAction.ABSTAIN,
    }
