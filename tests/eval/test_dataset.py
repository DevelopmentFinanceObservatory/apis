"""Tests for case requests and Langfuse item ids."""

import pytest

from search.eval.cases import Case, validate_case
from search.eval.dataset import item_id_for_case, resubmit_identity

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
                    "property": {"dcid": "financialFlowType", "label": "financial flow type"},
                    "label": "financial flow type",
                },
                "binding": {
                    "kind": "value",
                    "value": {
                        "ref": {"dcid": "Disbursement", "label": "disbursement"},
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
        "stat_var_dcids": ["Amount_FinancialFlow_Disbursement_Nominal"],
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
    base_spec_id, identity = resubmit_identity(_case(_RESUBMIT_CASE).resubmit)
    dumped = identity.model_dump(mode="json")
    assert base_spec_id is not None
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


def test_resubmit_identity_refuses_a_payload_with_no_anchor():
    raw = _RESUBMIT_CASE | {
        "resubmit": _RESUBMIT_CASE["resubmit"]
        | {"stat_var_dcids": None, "entity_dcids": None, "origin_dcids": None}
    }
    with pytest.raises(ValueError, match="identity definition"):
        resubmit_identity(_case(raw).resubmit)


def test_item_id_for_case_is_scoped_by_dataset_name():
    assert (
        item_id_for_case(_case(_RESUBMIT_CASE), "search-crs")
        == "search-crs:search-case-resubmit-tza"
    )
