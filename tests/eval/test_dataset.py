"""Tests for case requests, resubmit identities and Langfuse item ids."""

import re

import pytest

from search.eval.cases import Case, Resubmit, validate_case
from search.eval.dataset import item_id_for_case, resubmit_identity
from search.models import BindingValue, GraphRef, IdentityBindingValue, Slot, SlotKey, SlotValue

_RESUBMIT_CASE = {
    "id": "resubmit-tza",
    "query": "dataset_resubmit: IMF debt disbursements to Tanzania, 2015 to 2024",
    "door": "dataset_resubmit",
    "include_observations": True,
    "why": "The IMF is the creditor and Tanzania the debtor.",
    "resubmit": {
        "variable_text": "debt disbursements from the IMF, 2015 to 2024",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "amount",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {
                "key": {
                    "axis": "how",
                    "property": {"ref": "financialFlowType", "label": "financial flow type"},
                    "label": "financial flow type",
                },
                "binding": {
                    "kind": "value",
                    "value": {
                        "ref": {"ref": "Disbursement", "label": "disbursement"},
                        "value_kind": "enum_value",
                    },
                },
            },
            {
                "key": {"axis": "when", "property": None, "label": "when"},
                "binding": {
                    "kind": "value",
                    "value": {
                        "value_kind": "time_window",
                        "time_window": {"start_year": 2015, "end_year": 2024},
                    },
                },
            },
        ],
        "entity_dcids": ["country/TZA"],
        "origin_dcids": ["InternationalMonetaryFund"],
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": "USD", "prices": "constant", "base_year": 2024},
            }
        },
    },
    "expect": {
        "definite": {"measure": {"stat_vars": ["Amount_FinancialFlow_Disbursement_Nominal"]}}
    },
}


def _case(raw: dict) -> Case:
    return validate_case(raw, families={})


def test_resubmit_identity_carries_roles_window_and_facets():
    identity = resubmit_identity(_case(_RESUBMIT_CASE).resubmit)
    dumped = identity.model_dump(mode="json")
    assert dumped["entities"] == {
        "destination": ["country/TZA"],
        "origin": ["InternationalMonetaryFund"],
    }
    assert dumped["requested_window"] == {"kind": "range", "start_year": 2015, "end_year": 2024}
    assert dumped["facets"]["unit"]["request"] == {
        "currency": "USD",
        "prices": "constant",
        "base_year": 2024,
    }
    assert [slot["property"] for slot in dumped["slots"]] == ["financialFlowType"]


def test_item_id_for_case_is_scoped_by_dataset_name():
    assert (
        item_id_for_case(_case(_RESUBMIT_CASE), "search-crs")
        == "search-crs:search-case-resubmit-tza"
    )


_COUNT_PERSON_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "count",
    "stat_type": "measuredValue",
}


def _source_slot(dcid: str) -> Slot:
    return Slot(
        key=SlotKey(axis="source", property=None, label="data source"),
        binding=BindingValue(
            value=SlotValue(ref=GraphRef(ref=dcid, label=dcid), value_kind="source")
        ),
    )


def _resubmit(**fields) -> Resubmit:
    return Resubmit.model_validate(
        {"variable_text": "population", "shape": _COUNT_PERSON_SHAPE, "slots": [], **fields}
    )


def test_resubmit_identity_keys_subject_entities_under_observation_about():
    identity = resubmit_identity(_resubmit(entity_dcids=["country/KEN"]))

    assert identity.entities == {"observationAbout": ["country/KEN"]}


def test_resubmit_identity_reads_legacy_provenance_as_the_source_slot():
    """Legacy provenance and a source slot produce the same identity and keep the facets."""
    facets = {
        "provenance": IdentityBindingValue(ref="dc/base/WorldDevelopmentIndicators"),
        "measurementMethod": IdentityBindingValue(ref="dc/measurementMethod/MethodA"),
        "unit": IdentityBindingValue(ref="Count_Person"),
    }
    legacy = resubmit_identity(_resubmit(entity_dcids=["country/KEN"], facets=facets))
    source = resubmit_identity(
        _resubmit(
            entity_dcids=["country/KEN"],
            slots=[_source_slot("dc/base/WorldDevelopmentIndicators")],
            facets={key: value for key, value in facets.items() if key != "provenance"},
        )
    )

    assert legacy == source
    assert legacy.facets == facets


def test_resubmit_identity_refuses_a_source_slot_beside_legacy_provenance():
    with pytest.raises(ValueError, match=re.escape("source-axis slot and facets.provenance")):
        resubmit_identity(
            _resubmit(
                entity_dcids=["country/KEN"],
                slots=[_source_slot("dc/base/SomeSource")],
                facets={"provenance": IdentityBindingValue(ref="dc/base/OtherSource")},
            )
        )
