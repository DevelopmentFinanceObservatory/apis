"""Each group A check passes, fails and stands aside (n/a) on recorded and minimal responses.

The table rows read `check, expect, response, verdict`. Recorded responses come from the
sentence panel; a hand-built response covers what the panel lacks (operations, unit
requests, declined and catalogue answers, unsupported parts, full no-data bodies).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from search.eval.checks import CHECKS, PART_CHECKS, ScoreContext, Verdict, WireSchema
from search.eval.gate import Observation, Outcome, judge
from tests.eval.checks.conftest import (
    IDS,
    make_case,
    no_members_door,
    panel,
    panel_members,
    response_schema,
    schema_listing_members,
    serve,
)

PASS, FAIL, NA, U = Verdict.PASS, Verdict.FAIL, Verdict.NA, Verdict.UNVERIFIABLE
UNREACHED = Verdict.UNREACHED

CRS_001_ENTITIES = [
    {"dcid": "country/DEU", "role": "origin"},
    {"dcid": "country/KEN", "role": "destination"},
]
OECD_CRS = "provenance/OECDCreditorReportingSystem"
CRS_001_FIRST_STAT_VAR = (
    "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110"
    "_CRSPurposeCode11120_Nominal"
)


def answer(kind: str, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, "scope": {}, "diagnostics": {"warnings": []}, **fields}


def with_spec_identity(row_id: str, **identity: Any) -> dict[str, Any]:
    """A recorded definite response with its first spec's identity fields replaced."""
    response = panel(row_id)
    response["parts"][0]["spec"]["definition"]["identity"].update(identity)
    return response


def with_operations(row_id: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    response = panel(row_id)
    response["parts"][0]["operations"] = operations
    return response


def with_swapped_provenance(row_id: str) -> dict[str, Any]:
    """A recorded candidates response with its two candidates' provenance facets swapped."""
    response = panel(row_id)
    first, second = (
        spec["resolved"]["facets"] for spec in response["parts"][0]["candidates"]["specs"]
    )
    first["provenance"], second["provenance"] = second["provenance"], first["provenance"]
    return response


def with_provenance_of(row_id: str, source_id: str) -> dict[str, Any]:
    """A recorded definite response carrying another response's first provenance facet."""
    response = panel(row_id)
    donor = panel(source_id)["parts"][0]["candidates"]["specs"][0]["resolved"]["facets"]
    response["parts"][0]["spec"]["resolved"]["facets"]["provenance"] = donor["provenance"]
    return response


def with_ids_provenance(row_id: str) -> dict[str, Any]:
    """A recorded definite response whose provenance facet names IDS."""
    response = panel(row_id)
    facet = response["parts"][0]["spec"]["resolved"]["facets"]["provenance"]
    facet["ref"] = IDS
    facet["data_commons"]["id"] = "dfo"
    return response


def with_bound_provenance(response: dict[str, Any]) -> dict[str, Any]:
    """``response`` with its first spec's identity binding the provenance its facets name."""
    spec = response["parts"][0]["spec"]
    ref = spec["resolved"]["facets"]["provenance"]["ref"]
    facets = spec["definition"]["identity"].setdefault("facets", {})
    facets["provenance"] = {"state": "value", "ref": ref}
    return response


def as_candidates(row_id: str, *, second_dataset: str | None = None) -> dict[str, Any]:
    """A recorded definite response recast as a candidates part: its spec, then a copy that
    serves only ``Other_StatVar``, carrying ``second_dataset`` as its provenance when given."""
    response = panel(row_id)
    part = response["parts"][0]
    spec = part.pop("spec")
    second = copy.deepcopy(spec)
    second["spec_id"] = second["definition"]["spec_id"] = f"{spec['spec_id']}_other"
    serve(second["spec_id"], ["Other_StatVar"])
    if second_dataset is not None:
        second["resolved"]["facets"]["provenance"]["ref"] = second_dataset
    part["status"] = response["status"] = "candidates"
    part["candidates"] = {"specs": [spec, second]}
    return response


def no_data(**body: Any) -> dict[str, Any]:
    part = {"status": "no_data", "no_data": body}
    return {"kind": "resolution", "status": "no_data", "parts": [part], "diagnostics": {}}


def unsupported(code: str) -> dict[str, Any]:
    identity = {
        "identity_version": "spec1",
        "shape": {"kind": "statvar_schema", "population_type": "FinancialFlow"},
        "slots": [{"property": "crsPurpose", "binding": {"state": "unbound"}}],
        "entities": {"origin": ["country/FRA"]},
        "requested_window": {"kind": "all"},
        "facets": {"unit": {"state": "value", "request": {"currency": "XDR", "prices": "current"}}},
    }
    part = {
        "status": "unsupported",
        "unsupported": {
            "code": code,
            "pointer": "/identity/facets/unit",
            "detail": "not supported",
            "definition": {"schema_version": 1, "identity": identity},
        },
    }
    return {"kind": "resolution", "status": "unsupported", "parts": [part], "diagnostics": {}}


WDI_GDP = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
IMF_GDP_PER_CAPITA = (
    "ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal"
    "_AsAFractionOf_Count_Person"
)


def problem(code: str) -> dict[str, Any]:
    """An RFC 9457 problem document, as a door returns it with a non-2xx status."""
    return {
        "type": f"/errors/{code}",
        "title": "Not implemented",
        "status": 501,
        "detail": "The XDR unit cannot be bound on this deployment.",
        "code": code,
        "pointer": "/input/entries/0/identity/facets/unit",
    }


UNIT_EUR = {"currency": "EUR", "prices": "constant", "base_year": 2022}

ROWS = [
    # disposition: kind, status, scope place. Every case asserts it, so it is never n/a.
    ("disposition", {"definite": {}}, panel("crs-001"), PASS),
    ("disposition", {"no_data": {}, "candidates": {}}, panel("crs-001"), FAIL),
    ("disposition", {"mixed": {}}, panel("ids-038"), PASS),
    ("disposition", {"kind": "catalogue", "scope_place": None}, answer("catalogue"), PASS),
    (
        "disposition",
        {"kind": "catalogue", "scope_place": "country/KEN"},
        answer("catalogue"),
        FAIL,
    ),
    ("disposition", {"kind": "catalogue"}, panel("crs-001"), FAIL),
    # A door that refuses the request answers with a problem document.
    (
        "disposition",
        {"kind": "problem", "problem": {"code": "facet_binding_unsupported"}},
        problem("facet_binding_unsupported"),
        PASS,
    ),
    (
        "disposition",
        {"kind": "problem", "problem": {"code": "facet_binding_unsupported"}},
        problem("unit_request_unsupported"),
        FAIL,
    ),
    ("disposition", {"definite": {}}, problem("facet_binding_unsupported"), FAIL),
    (
        "disposition",
        {"kind": ["resolution", "problem"], "definite": {}, "problem": {"code": "x"}},
        panel("crs-001"),
        PASS,
    ),
    ("disposition", {"kind": "problem", "problem": {"code": "x"}}, panel("crs-001"), FAIL),
    # measure: shape keys, bound and open slots, StatVars, families.
    (
        "measure",
        {
            "definite": {
                "measure": {
                    "shape": {"population_type": "FinancialFlow", "measurement_denominator": None},
                    "bound": {
                        "financialFlowType": "Disbursement",
                        "crsTypeOfFlow": "CRSTypeOfFlow10",
                    },
                    "open": ["crsPurpose"],
                }
            }
        },
        panel("crs-001"),
        PASS,
    ),
    (
        "measure",
        {"definite": {"measure": {"family": "crs.disbursement_gross"}}},
        panel("crs-001"),
        PASS,
    ),
    ("measure", {"definite": {"measure": {"family": "crs.commitment"}}}, panel("crs-001"), FAIL),
    (
        "measure",
        {"definite": {"measure": {"family": ["crs.commitment", "crs.disbursement_gross"]}}},
        panel("crs-001"),
        PASS,
    ),
    (
        "measure",
        {
            "definite": {
                "measure": {
                    "shape": {"population_type": "FinancialFlow"},
                    "bound": {"crsPurpose": "CRSPurposeCode12220"},
                }
            }
        },
        panel("crs-001"),
        FAIL,
    ),
    (
        "measure",
        {
            "definite": {
                "measure": {"shape": {"population_type": "FinancialFlow"}, "open": ["nettingBasis"]}
            }
        },
        panel("crs-001"),
        FAIL,
    ),
    # A set binding matches in any order.
    (
        "measure",
        {
            "definite": {
                "measure": {
                    "family": "crs.disbursement_gross",
                    "bound": {"crsPurpose": ["CRSPurposeCode120", "CRSPurposeCode110"]},
                }
            }
        },
        panel("crs-048"),
        PASS,
    ),
    (
        "measure",
        {
            "definite": {
                "measure": {
                    "family": "crs.disbursement_gross",
                    "bound": {"crsPurpose": "CRSPurposeCode110"},
                }
            }
        },
        panel("crs-048"),
        FAIL,
    ),
    (
        "measure",
        {
            "definite": {
                "measure": {
                    "stat_vars": [
                        "Saving_EconomicUnit_GrossOfFixedCapitalConsumption_Nominal_AsAFractionOf_ValueAdded_EconomicProduction_IncludingProductTaxes_GrossOfFixedCapitalConsumption_Nominal"
                    ]
                }
            }
        },
        panel("weo-033"),
        PASS,
    ),
    ("measure", {"definite": {"measure": {"stat_vars": ["Count_Person"]}}}, panel("weo-033"), FAIL),
    # weo-034's identity marks the PerCapita denominator unresolved; the dcid is resolved.
    (
        "measure",
        {"definite": {"measure": {"shape": {"measurement_denominator": "PerCapita"}}}},
        panel("weo-034"),
        PASS,
    ),
    (
        "measure",
        {"definite": {"measure": {"shape": {"measurement_denominator": "Count_Person"}}}},
        panel("weo-034"),
        FAIL,
    ),
    (
        "measure",
        {"definite": {"measure": {"shape": {"measurement_denominator": None}}}},
        panel("weo-034"),
        FAIL,
    ),
    # A resolved denominator reads from the identity's ref.
    (
        "measure",
        {"candidates": {"measure": {"shape": {"measurement_denominator": "Count_Person"}}}},
        panel("weo-002"),
        PASS,
    ),
    ("measure", {"definite": {}}, panel("crs-001"), NA),
    # A status the expectation does not accept leaves the check unscored: unreached when a
    # block asserts it, n/a when none does.
    ("measure", {"definite": {"measure": {"family": "wdi.gdp"}}}, panel("weo-045"), UNREACHED),
    ("measure", {"definite": {}}, panel("weo-045"), NA),
    # An accepted kind other than resolution carries no parts to score.
    (
        "measure",
        {
            "kind": ["resolution", "catalogue"],
            "definite": {"measure": {"family": "crs.commitment"}},
        },
        answer("catalogue"),
        NA,
    ),
    # A candidates part passes when one candidate is the measure.
    (
        "measure",
        {"candidates": {"measure": {"family": "weo.gdp_per_capita"}}},
        panel("weo-002"),
        PASS,
    ),
    ("measure", {"candidates": {"measure": {"family": "crs.commitment"}}}, panel("weo-002"), FAIL),
    # Under candidates, each pinned StatVar is served by its own candidate; extras pass.
    (
        "measure",
        {"candidates": {"measure": {"stat_vars": [IMF_GDP_PER_CAPITA, WDI_GDP]}}},
        panel("weo-002"),
        PASS,
    ),
    ("measure", {"candidates": {"measure": {"stat_vars": [WDI_GDP]}}}, panel("weo-030"), PASS),
    (
        "measure",
        {"candidates": {"measure": {"stat_vars": [WDI_GDP, "Count_Person"]}}},
        panel("weo-002"),
        FAIL,
    ),
    (
        "measure",
        {
            "candidates": {
                "measure": {
                    "shape": {"population_type": "EconomicProduction"},
                    "stat_vars": [WDI_GDP],
                }
            }
        },
        panel("weo-030"),
        PASS,
    ),
    (
        "measure",
        {
            "candidates": {
                "measure": {"shape": {"population_type": "FinancialFlow"}, "stat_vars": [WDI_GDP]}
            }
        },
        panel("weo-030"),
        FAIL,
    ),
    # An unsupported part's definition carries the identity, without StatVars.
    (
        "measure",
        {
            "unsupported": {
                "measure": {"shape": {"population_type": "FinancialFlow"}, "open": ["crsPurpose"]}
            }
        },
        unsupported("unit_request_unsupported"),
        PASS,
    ),
    (
        "measure",
        {"unsupported": {"measure": {"stat_vars": ["X"]}}},
        unsupported("unit_request_unsupported"),
        FAIL,
    ),
    # A no-data part without a flagged match carries no measure.
    (
        "measure",
        {"no_data": {"measure": {"family": "crs.disbursement_gross"}}},
        panel("crs-013"),
        NA,
    ),
    # entities: the exact (dcid, role) set; a group read as members.
    ("entities", {"definite": {"entities": CRS_001_ENTITIES}}, panel("crs-001"), PASS),
    ("entities", {"definite": {"entities": CRS_001_ENTITIES[:1]}}, panel("crs-001"), FAIL),
    (
        "entities",
        {"definite": {"entities": [{"dcid": "country/KEN", "role": "observationAbout"}]}},
        panel("crs-001"),
        FAIL,
    ),
    (
        "entities",
        {
            "definite": {
                "entities": [
                    {"dcid": "EuropeanUnion", "role": "origin", "expands": True},
                    {"dcid": "country/ETH", "role": "destination"},
                ]
            }
        },
        panel("crs-006"),
        PASS,
    ),
    (
        "entities",
        {
            "definite": {
                "entities": [
                    {"dcid": "africa", "role": "origin", "expands": True},
                    {"dcid": "country/ETH", "role": "destination"},
                ]
            }
        },
        panel("crs-006"),
        FAIL,
    ),
    # weo-036 records G20 with no graph node: a label names it, case and spaces aside.
    (
        "entities",
        {
            "definite": {
                "entities": [{"label": " g20 ", "role": "observationAbout", "expands": True}]
            }
        },
        panel("weo-036"),
        PASS,
    ),
    (
        "entities",
        {"definite": {"entities": [{"label": "G7", "role": "observationAbout", "expands": True}]}},
        panel("weo-036"),
        FAIL,
    ),
    (
        "entities",
        {"definite": {"entities": [{"dcid": "G20", "role": "observationAbout", "expands": True}]}},
        panel("weo-036"),
        FAIL,
    ),
    # A label never matches a group the graph holds a node for.
    (
        "entities",
        {
            "definite": {
                "entities": [
                    {"label": "European Union", "role": "origin", "expands": True},
                    {"dcid": "country/ETH", "role": "destination"},
                ]
            }
        },
        panel("crs-006"),
        FAIL,
    ),
    (
        "entities",
        {"unsupported": {"entities": [{"dcid": "country/FRA", "role": "origin"}]}},
        unsupported("x"),
        PASS,
    ),
    # An unsupported answer records no group.
    (
        "entities",
        {"unsupported": {"entities": [{"dcid": "country/FRA", "role": "origin", "expands": True}]}},
        unsupported("x"),
        FAIL,
    ),
    ("entities", {"definite": {}}, panel("crs-001"), NA),
    # provenance: the family's, or an override.
    (
        "provenance",
        {"definite": {"measure": {"family": "crs.disbursement_gross"}}},
        panel("crs-001"),
        PASS,
    ),
    (
        "provenance",
        {"definite": {"provenance": {"origin": "dfo", "dataset": OECD_CRS}}},
        panel("crs-001"),
        PASS,
    ),
    ("provenance", {"definite": {"provenance": {"origin": "base"}}}, panel("crs-001"), FAIL),
    (
        "provenance",
        {"definite": {"provenance": {"dataset": "dc/base/UN_SDG"}}},
        panel("crs-001"),
        FAIL,
    ),
    ("provenance", {"candidates": {"provenance": {"origin": "dfo"}}}, panel("weo-002"), PASS),
    (
        "provenance",
        {"definite": {"measure": {"shape": {"population_type": "FinancialFlow"}}}},
        panel("crs-001"),
        NA,
    ),
    ("provenance", {"unsupported": {"provenance": {"origin": "dfo"}}}, unsupported("x"), NA),
    # With several families, the provenance is the one of the family the answer matches.
    (
        "provenance",
        {"candidates": {"measure": {"family": ["wdi.gdp", "weo.gdp_per_capita"]}}},
        panel("weo-002"),
        PASS,
    ),
    (
        "provenance",
        {"candidates": {"measure": {"family": ["wdi.gdp", "weo.gdp_per_capita"]}}},
        with_swapped_provenance("weo-002"),
        FAIL,
    ),
    (
        "provenance",
        {"definite": {"measure": {"family": ["crs.disbursement_gross", "wdi.gdp"]}}},
        with_provenance_of("crs-001", "weo-002"),
        FAIL,
    ),
    # On a definite part a family pairs only when its StatVar pin matches too.
    (
        "provenance",
        {
            "definite": {
                "measure": {"family": ["crs.disbursement_gross", "ids.disbursement_unserved"]}
            }
        },
        with_ids_provenance("crs-001"),
        FAIL,
    ),
    # Under candidates, measure passes through the IDS family's pin, and the OECD
    # provenance belongs to a family whose pin fails.
    (
        "provenance",
        {
            "candidates": {
                "measure": {"family": ["crs.disbursement_unserved", "ids.disbursement_served"]}
            }
        },
        as_candidates("crs-001"),
        FAIL,
    ),
    # The candidate serving the IDS family's pin carries OECD; the IDS candidate does not
    # serve it.
    (
        "provenance",
        {
            "candidates": {
                "measure": {"family": ["crs.disbursement_unserved", "ids.disbursement_served"]}
            }
        },
        as_candidates("crs-001", second_dataset=IDS),
        FAIL,
    ),
    (
        "provenance",
        {"candidates": {"measure": {"family": "crs.disbursement_pin_unconfirmed"}}},
        as_candidates("crs-001"),
        U,
    ),
    (
        "provenance",
        {"definite": {"measure": {"family": "crs.disbursement_gl_dataset"}}},
        panel("crs-001"),
        U,
    ),
    # A block's own provenance replaces the family's: the IDS candidate must be the measure.
    (
        "provenance",
        {
            "candidates": {
                "measure": {
                    "shape": {"population_type": "FinancialFlow"},
                    "stat_vars": [CRS_001_FIRST_STAT_VAR],
                },
                "provenance": {"dataset": IDS},
            }
        },
        as_candidates("crs-001", second_dataset=IDS),
        FAIL,
    ),
    (
        "provenance",
        {"unsupported": {"measure": {"family": "crs.flow"}}},
        unsupported("x"),
        NA,
    ),
    # bound reads the identity's provenance facet; a block naming only bound keeps the
    # family's origin and dataset.
    (
        "provenance",
        {
            "definite": {
                "measure": {"family": "crs.disbursement_gross"},
                "provenance": {"bound": False},
            }
        },
        panel("crs-001"),
        PASS,
    ),
    ("provenance", {"definite": {"provenance": {"bound": True}}}, panel("crs-001"), FAIL),
    (
        "provenance",
        {
            "definite": {
                "measure": {"family": "crs.disbursement_gross"},
                "provenance": {"bound": True},
            }
        },
        with_bound_provenance(panel("crs-001")),
        PASS,
    ),
    (
        "provenance",
        {"definite": {"provenance": {"origin": "dfo", "bound": False}}},
        with_bound_provenance(panel("crs-001")),
        FAIL,
    ),
    (
        "provenance",
        {
            "definite": {
                "measure": {"family": "crs.disbursement_gross"},
                "provenance": {"bound": True},
            }
        },
        with_bound_provenance(with_ids_provenance("crs-001")),
        FAIL,
    ),
    # A family whose slots are unconfirmed pairs as unverifiable.
    (
        "provenance",
        {
            "definite": {
                "measure": {"family": ["crs.disbursement_gross", "ids.disbursement_unconfirmed"]}
            }
        },
        with_ids_provenance("crs-001"),
        U,
    ),
    # An answer matching no family is held to any family's provenance.
    (
        "provenance",
        {"definite": {"measure": {"family": ["crs.commitment", "wdi.gdp"]}}},
        panel("crs-001"),
        PASS,
    ),
    # request: window and unit.
    ("request", {"definite": {"window": {"kind": "all"}}}, panel("crs-001"), PASS),
    ("request", {"definite": {"window": {"kind": "latest"}}}, panel("crs-001"), FAIL),
    (
        "request",
        {"definite": {"window": {"kind": "range", "start": 2015, "end": 2026}}},
        panel("weo-033"),
        PASS,
    ),
    ("request", {"definite": {"window": {"kind": "range", "start": 2015}}}, panel("weo-033"), FAIL),
    (
        "request",
        {"definite": {"unit": UNIT_EUR}},
        with_spec_identity("crs-001", facets={"unit": {"state": "value", "request": UNIT_EUR}}),
        PASS,
    ),
    (
        "request",
        {"definite": {"unit": {**UNIT_EUR, "currency": "USD"}}},
        with_spec_identity("crs-001", facets={"unit": {"state": "value", "request": UNIT_EUR}}),
        FAIL,
    ),
    (
        "request",
        {"definite": {"unit": "none"}},
        with_spec_identity("crs-001", facets={"unit": {"state": "value", "request": UNIT_EUR}}),
        FAIL,
    ),
    ("request", {"definite": {"unit": UNIT_EUR}}, panel("crs-001"), FAIL),
    # An omitted unit field is not asserted.
    (
        "request",
        {"definite": {"unit": {"currency": "EUR"}}},
        with_spec_identity("crs-001", facets={"unit": {"state": "value", "request": UNIT_EUR}}),
        PASS,
    ),
    ("request", {"definite": {}}, panel("crs-001"), NA),
    # operations: verb and, where set, axis and target year, in order.
    ("operations", {"definite": {"operations": []}}, panel("crs-001"), PASS),
    (
        "operations",
        {"definite": {"operations": [{"verb": "rank", "over": "origin"}]}},
        with_operations("crs-001", [{"verb": "rank", "phrase": "top donors", "over": "origin"}]),
        PASS,
    ),
    (
        "operations",
        {"definite": {"operations": [{"verb": "rank"}]}},
        with_operations("crs-001", [{"verb": "rank", "phrase": "top", "over": "origin"}]),
        PASS,
    ),
    (
        "operations",
        {"definite": {"operations": [{"verb": "rank", "over": "destination"}]}},
        with_operations("crs-001", [{"verb": "rank", "phrase": "top", "over": "origin"}]),
        FAIL,
    ),
    (
        "operations",
        {"definite": {"operations": [{"verb": "forecast", "target_year": 2030}]}},
        with_operations(
            "crs-001", [{"verb": "forecast", "phrase": "in 2030", "target_year": 2031}]
        ),
        FAIL,
    ),
    (
        "operations",
        {"definite": {"operations": []}},
        with_operations("crs-001", [{"verb": "growth", "phrase": "change"}]),
        FAIL,
    ),
    ("operations", {"definite": {}}, panel("crs-001"), NA),
    # warnings: required codes on the part or the request; extras pass.
    ("warnings", {"definite": {"warnings": ["role_scope_defaulted"]}}, panel("crs-002"), PASS),
    ("warnings", {"definite": {"warnings": ["binding_folded"]}}, panel("crs-002"), FAIL),
    (
        "warnings",
        {"no_data": {"warnings": ["un_system_unavailable"]}},
        {
            **no_data(reason="no_observations"),
            "diagnostics": {
                "warnings": [{"code": "un_system_unavailable", "severity": "warn", "message": "m"}]
            },
        },
        PASS,
    ),
    ("warnings", {"definite": {}}, panel("crs-002"), NA),
    # refusal: no-data reason, category and gate; unsupported code; declined reason.
    ("refusal", {"no_data": {"reason": "no_observations"}}, panel("crs-013"), PASS),
    (
        "refusal",
        {"no_data": {"reason": ["entity_not_resolved", "no_observations"]}},
        panel("crs-013"),
        PASS,
    ),
    ("refusal", {"no_data": {"reason": "entity_not_resolved"}}, panel("crs-013"), FAIL),
    (
        "refusal",
        {"no_data": {"reason": "no_observations", "category": "absent"}},
        no_data(reason="no_observations", category="absent"),
        PASS,
    ),
    (
        "refusal",
        {"no_data": {"category": "unresolved"}},
        no_data(reason="no_observations", category="absent"),
        FAIL,
    ),
    (
        "refusal",
        {"no_data": {"gate": "no_member_has_data_at_place"}},
        no_data(reason="no_observations", category="absent"),
        FAIL,
    ),
    (
        "refusal",
        {"unsupported": {"code": "unit_request_unsupported"}},
        unsupported("unit_request_unsupported"),
        PASS,
    ),
    (
        "refusal",
        {"unsupported": {"code": "unit_request_unsupported"}},
        unsupported("facet_binding_unsupported"),
        FAIL,
    ),
    (
        "refusal",
        {"kind": "declined", "declined": {"reason": "out_of_scope"}},
        answer("declined", reason="out_of_scope"),
        PASS,
    ),
    (
        "refusal",
        {"kind": "declined", "declined": {"reason": "out_of_scope"}},
        answer("declined", reason="requires_analysis"),
        FAIL,
    ),
    (
        "refusal",
        {"kind": ["catalogue", "declined"], "declined": {"reason": "out_of_scope"}},
        answer("catalogue"),
        NA,
    ),
    ("refusal", {"definite": {}, "no_data": {"reason": "no_observations"}}, panel("crs-001"), NA),
    # A declined-only case answered with a resolution never reads the reason.
    (
        "refusal",
        {"kind": "declined", "declined": {"reason": "out_of_scope"}},
        panel("crs-001"),
        UNREACHED,
    ),
    # A resolution in a status the case does not accept reaches no block, so the declined
    # reason reads unreached, as for dq-ext-042 and dq-ids-054; one in an accepted status
    # reached the resolution block instead.
    (
        "refusal",
        {
            "kind": ["declined", "resolution"],
            "declined": {"reason": "out_of_scope"},
            "definite": {},
        },
        panel("weo-002"),
        UNREACHED,
    ),
    (
        "refusal",
        {
            "kind": ["declined", "resolution"],
            "declined": {"reason": "out_of_scope"},
            "definite": {},
        },
        panel("crs-001"),
        NA,
    ),
    # A problem document returns no part for the asserted no-data block to score.
    (
        "refusal",
        {"no_data": {"reason": "no_observations"}},
        problem("facet_binding_unsupported"),
        UNREACHED,
    ),
    # candidates: every family pairs with its own candidate.
    (
        "candidates",
        {"candidates": {"include_families": ["wdi.gdp", "weo.gdp_per_capita"]}},
        panel("weo-002"),
        PASS,
    ),
    (
        "candidates",
        {"candidates": {"include_families": ["wdi.gdp", "crs.commitment"]}},
        panel("weo-002"),
        FAIL,
    ),
    # Both IMF families match only the one IMF candidate, so they cannot both be included.
    (
        "candidates",
        {"candidates": {"include_families": ["weo.gdp_per_capita", "weo.gdp_any"]}},
        panel("weo-002"),
        FAIL,
    ),
    # weo-030 offers two IMF readings, so both IMF families pair.
    (
        "candidates",
        {"candidates": {"include_families": ["weo.gdp_per_capita", "weo.gdp_any"]}},
        panel("weo-030"),
        PASS,
    ),
    (
        "candidates",
        {"definite": {}, "candidates": {"include_families": ["wdi.gdp"]}},
        panel("crs-001"),
        NA,
    ),
    # parts: each expected part in order, with every part check.
    (
        "parts",
        {
            "parts": [
                {"definite": {"measure": {"family": "crs.disbursement_gross"}}},
                {"definite": {"warnings": ["binding_folded"]}},
            ]
        },
        panel("crs-048"),
        PASS,
    ),
    (
        "parts",
        {"parts": [{"definite": {}}, {"definite": {"warnings": ["role_scope_defaulted"]}}]},
        panel("crs-048"),
        FAIL,
    ),
    (
        "parts",
        {"parts": [{"definite": {}}, {"no_data": {"reason": "role_not_resolved"}}]},
        panel("ids-038"),
        PASS,
    ),
    ("parts", {"parts": [{"definite": {}}, {"definite": {}}]}, panel("ids-038"), FAIL),
    (
        "parts",
        {"parts": [{"definite": {}}, {"definite": {}}, {"definite": {}}]},
        panel("crs-048"),
        FAIL,
    ),
    ("parts", {"definite": {}}, panel("crs-048"), NA),
    # Another kind the case accepts carries no parts to expect.
    (
        "parts",
        {
            "kind": ["resolution", "declined"],
            "declined": {"reason": "out_of_scope"},
            "parts": [{"definite": {}}, {"definite": {}}],
        },
        answer("declined", reason="out_of_scope"),
        NA,
    ),
]


@pytest.mark.parametrize(
    ("check", "expect", "response", "verdict"),
    ROWS,
    ids=[f"{row[0]}-{i}-{row[3]}" for i, row in enumerate(ROWS)],
)
def test_check_verdicts(check, expect, response, verdict, ctx: ScoreContext):
    result = CHECKS[check](make_case(expect), response, ctx)
    assert result.verdict == verdict, result.detail


@pytest.mark.parametrize(("listed", "verdict"), [(True, PASS), (False, FAIL)])
def test_an_app_listing_its_members_on_the_wire_is_scored_from_them(listed: bool, verdict):
    """An app with no members door is scored from the StatVars its own wire lists, so the
    base side of a paired run reads no door it lacks."""
    response = panel("weo-033")
    spec = response["parts"][0]["spec"]
    served = panel_members(spec["spec_id"], spec["definition"])
    spec["resolved"]["stat_vars"]["items"] = [
        {"ref": stat_var} for stat_var in (served if listed else ["Count_Person"])
    ]
    ctx = ScoreContext(schema=schema_listing_members(), members=no_members_door)

    result = CHECKS["measure"](
        make_case({"definite": {"measure": {"stat_vars": served}}}), response, ctx
    )

    assert result.verdict == verdict, result.detail


def test_a_failed_members_read_leaves_candidates_to_infrastructure():
    """A family that pins its StatVars pairs only through the members door, so a door that
    answers 404 makes the check an infrastructure loss."""
    ctx = ScoreContext(
        schema=WireSchema.from_document({"response": response_schema()}), members=no_members_door
    )
    case = make_case({"candidates": {"include_families": ["crs.disbursement_unserved"]}})

    result = CHECKS["candidates"](case, as_candidates("crs-001"), ctx)

    assert (result.verdict, result.infra) == (U, True), result.detail


GROUP_A = (
    "disposition",
    "measure",
    "entities",
    "provenance",
    "request",
    "operations",
    "refusal",
    "warnings",
    "candidates",
    "parts",
)


def test_every_check_has_pass_fail_and_na_rows():
    """Group A's rows; ``test_group_b.py`` and ``test_materialisation.py`` cover group B."""
    covered = {(row[0], row[3]) for row in ROWS}
    for check in GROUP_A:
        verdicts = {PASS, FAIL} if check == "disposition" else {PASS, FAIL, NA}
        assert verdicts <= {v for c, v in covered if c == check}, check


def test_unit_none_passes_when_the_wire_omits_the_unit_facet(ctx: ScoreContext):
    """`exclude_none` drops the identity's facets map; the schema still declares it, so the
    read is None and `unit: none` holds."""
    response = panel("crs-001")
    assert "facets" not in response["parts"][0]["spec"]["definition"]["identity"]
    case = make_case({"definite": {"unit": "none"}})
    assert CHECKS["request"](case, response, ctx).verdict == PASS


def test_an_unconfirmed_value_is_unverifiable(ctx: ScoreContext):
    case = make_case(
        {
            "definite": {
                "measure": {
                    "family": "crs.disbursement_gross",
                    "bound": {"crsPurpose": {"gl": "the health purpose codes"}},
                }
            }
        }
    )
    assert CHECKS["measure"](case, panel("crs-001"), ctx).verdict == U


def test_a_failure_names_the_part_and_the_field(ctx: ScoreContext):
    case = make_case({"definite": {"measure": {"family": "crs.commitment"}}})
    result = CHECKS["measure"](case, panel("crs-001"), ctx)
    assert result.detail.startswith("parts[0]: shape population_type")


def test_a_shared_field_applies_under_each_accepted_status(ctx: ScoreContext):
    case = make_case({"entities": CRS_001_ENTITIES, "definite": {}, "no_data": {}})
    assert CHECKS["entities"](case, panel("crs-001"), ctx).verdict == PASS
    assert CHECKS["entities"](case, panel("crs-013"), ctx).verdict == NA


COMMITMENT = {"measure": {"family": "crs.commitment"}}
DISBURSEMENT = {"measure": {"family": "crs.disbursement_gross"}}


def with_part_status(row_id: str, index: int, status: str) -> dict[str, Any]:
    response = panel(row_id)
    response["parts"][index]["status"] = status
    return response


def with_part_from(row_id: str, index: int, source_id: str) -> dict[str, Any]:
    """A recorded response with the part at ``index`` replaced by ``source_id``'s first part."""
    response = panel(row_id)
    response["parts"][index] = panel(source_id)["parts"][0]
    return response


@pytest.mark.parametrize(
    ("expect", "head", "base", "outcome"),
    [
        pytest.param(
            {"definite": COMMITMENT},
            panel("crs-001"),
            panel("weo-002"),
            Outcome.PRE_EXISTING,
            id="other-status",
        ),
        pytest.param(
            {"definite": COMMITMENT},
            panel("crs-001"),
            answer("catalogue"),
            Outcome.PRE_EXISTING,
            id="other-kind",
        ),
        pytest.param(
            {"definite": COMMITMENT},
            panel("crs-001"),
            problem("x"),
            Outcome.PRE_EXISTING,
            id="problem",
        ),
        pytest.param(
            {"parts": [{"definite": {}}, {"definite": COMMITMENT}]},
            panel("crs-048"),
            panel("crs-001"),
            Outcome.PRE_EXISTING,
            id="missing-part",
        ),
        # The head fails part 0, which the base reached and passed; the base left only
        # part 1 unscored, so the failure is the head's own.
        pytest.param(
            {"parts": [{"definite": DISBURSEMENT}, {"definite": DISBURSEMENT}]},
            with_part_from("crs-048", 0, "weo-001"),
            with_part_status("crs-048", 1, "no_data"),
            Outcome.REGRESSION,
            id="other-part-reached",
        ),
    ],
)
def test_a_head_that_fixes_the_disposition_and_fails_the_measure(
    expect, head, base, outcome, ctx: ScoreContext
):
    """dq-ext-005's shape: the base's answer never reaches the expected block, so its
    measure never runs, and the head reaches it with the wrong measure. The failure is
    pre-existing unless the head failed on a part the base reached."""
    case = make_case(expect)
    head_result = CHECKS["measure"](case, head, ctx)
    base_result = CHECKS["measure"](case, base, ctx)
    assert (head_result.verdict, base_result.verdict) == (FAIL, UNREACHED)
    head_sample = Observation(head_result.verdict, parts=head_result.parts)
    base_sample = Observation(base_result.verdict, parts=base_result.parts)

    judgement = judge(
        check="measure",
        head=[head_sample] * 3,
        base=[base_sample] * 3,
        confirm=(head_sample, base_sample),
    )

    assert judgement.outcome == outcome


# A status block asserting each part check, answered by ASSERTED_UNPAIRED's response,
# whose status the block's expectation does not accept.
ASSERTING = {
    "measure": {"definite": COMMITMENT},
    "provenance": {"definite": {"provenance": {"origin": "dfo"}}},
    "entities": {"definite": {"entities": CRS_001_ENTITIES}},
    "request": {"definite": {"window": {"kind": "all"}}},
    "operations": {"definite": {"operations": []}},
    "warnings": {"definite": {"warnings": ["role_scope_defaulted"]}},
    "skills": {"definite": {"skills": {"present": True}}},
    "candidates": {"candidates": {"include_families": ["crs.flow"]}},
    "refusal": {"no_data": {"reason": "no_observations"}},
}


def test_every_part_check_has_an_asserting_block():
    assert set(ASSERTING) == set(PART_CHECKS)


@pytest.mark.parametrize("check", sorted(ASSERTING))
def test_an_unpaired_part_is_unreached_only_for_a_check_its_blocks_assert(check, ctx: ScoreContext):
    """The asserting block reads unreached; a block under the same status asserting another
    check alone, on the same unpaired answer, reads n/a."""
    status = next(iter(ASSERTING[check]))
    unpaired = panel("weo-002") if status == "definite" else panel("crs-001")
    other = {"entities": CRS_001_ENTITIES} if check == "warnings" else {"warnings": []}
    assert CHECKS[check](make_case(ASSERTING[check]), unpaired, ctx).verdict == UNREACHED
    assert CHECKS[check](make_case({status: other}), unpaired, ctx).verdict == NA


@pytest.mark.parametrize("check", ["disposition", "parts", "measure"])
def test_every_check_names_a_problem_document_by_its_kind(check, ctx: ScoreContext):
    case = make_case({"parts": [{"definite": COMMITMENT}, {"definite": {}}]})
    result = CHECKS[check](case, problem("facet_binding_unsupported"), ctx)
    assert result.detail.startswith("kind: got 'problem'"), result.detail
