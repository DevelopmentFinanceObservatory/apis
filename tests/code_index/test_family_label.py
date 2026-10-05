"""Test `family_label` against the packaged indexes.

A family label leads with the identity's measure partition label and follows with its bound
slots' code labels. Measure-axis slots and the slots the identity's five-tuple pins add nothing,
because the partition label already says them: "imputed multilateral ODA" names the imputed
aid type and official development assistance."""

from __future__ import annotations

import pytest

from code_index import default_index_path, load_index
from code_index.labels import family_label
from code_index.shared import get_measure_index
from statvar_members.enumerate import five_tuple_from_identity
from tests.code_index.aid_identity import education_aid_identity, imputed_education_aid_identity


@pytest.fixture(scope="module")
def index():
    return load_index(default_index_path())


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        (
            education_aid_identity(),
            "gross disbursements, education, official development assistance",
        ),
        (imputed_education_aid_identity(), "imputed multilateral ODA, education"),
    ],
    ids=["crs", "imputed"],
)
def test_a_family_label_leads_with_the_measure_partition_label(index, identity, expected):
    assert family_label(identity, index=index, measure_index=get_measure_index()) == expected


def test_the_imputed_label_leaves_out_every_slot_its_five_tuple_pins(index):
    identity = imputed_education_aid_identity()
    pinned = index.five_tuple_pins[five_tuple_from_identity(identity["shape"])]
    bound = {slot["property"] for slot in identity["slots"] if slot["binding"]["state"] == "value"}
    assert pinned.keys() <= bound

    label = family_label(identity, index=index, measure_index=get_measure_index())

    assert label is not None
    _, _, slots = label.partition(", ")
    for value in pinned.values():
        row = index.code_by_dcid(value)
        assert row is not None
        assert row.label not in slots


def test_a_family_with_no_labelled_partition_names_its_slots_alone(index):
    identity = education_aid_identity()
    identity["shape"]["measured_property"] = "unlabelledProperty"
    purpose = next(slot for slot in identity["slots"] if slot["property"] == "crsPurpose")
    purpose["binding"] = {"state": "set", "refs": ["CRSPurposeCode110", "CRSPurposeCode120"]}

    label = family_label(identity, index=index, measure_index=get_measure_index())

    assert label == "education/health, official development assistance"


def test_a_family_with_nothing_labelled_has_no_label(index):
    identity = {
        "shape": {"population_type": "Person", "measured_property": "count"},
        "slots": [{"property": "gender", "binding": {"state": "unbound"}}],
    }

    assert family_label(identity, index=index, measure_index=get_measure_index()) is None
