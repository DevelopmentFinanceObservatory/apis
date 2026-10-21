"""Candidate `member_count` reports the resolved population and sorts candidates broadest first.

The fixture contrasts an index-sourced FinancialObligation/amount shape with two members and a
non-indexed shape with one. Both recall one member, so recall counts cannot produce this ordering.
"""

from __future__ import annotations

import asyncio

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_QUERY = (
    "grant equivalent disbursements, financial obligation commitments, and a fabricated "
    "test measure, to Kenya"
)
_KEN_DCID = "country/KEN"

# Elected FinancialFlow/grantEquivalent shape.
_POP_TYPE_0 = "FinancialFlow"
_MEAS_PROP_0 = "grantEquivalent"
_STAT_TYPE_0 = "measuredValue"
_MEAS_QUAL_0 = "Nominal"

_RECALLED_SV0 = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_EXTRA_SV0 = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)
_PROP_PURPOSE = "crsPurpose"
_EXTRA_PURPOSE = "CRSPurposeCode51010"

# Index-sourced FinancialObligation/amount alternate.
_POP_TYPE_1 = "FinancialObligation"
_MEAS_PROP_1 = "amount"
_STAT_TYPE_1 = "measuredValue"
_MEAS_QUAL_1 = "Nominal"

_RECALLED_SV1 = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow30_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)
_EXTRA_SV1 = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)
# Omitting crsTypeOfFlow's value arc leaves it unbound, so the differing sibling stays eligible.
_SHAPE1_BOUND_PROPS = (
    "crsCooperationModality",
    _PROP_PURPOSE,
    "crsTypeOfFinance",
    "financialObligationType",
)

# Fabricated five-tuple absent from the fixture index.
_POP_TYPE_2 = "TestBarePopulation"
_MEAS_PROP_2 = "testBareMeasure"
_STAT_TYPE_2 = "measuredValue"
_MEAS_QUAL_2 = "Nominal"
_BARE_SV2 = "TestBareMeasure_TestBarePopulation_Nominal"


def _real_row(dcid: str):
    index = get_index()
    return next(r for r in index.statvars if r.dcid == dcid)


def _sv_node(
    dcid: str,
    *,
    pop_type: str,
    meas_prop: str,
    stat_type: str,
    meas_qual: str,
    constraint_props: list[str],
) -> dict:
    row = _real_row(dcid)
    return {
        "label": dcid,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": dcid}]},
            "populationType": {"nodes": [{"dcid": pop_type}]},
            "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
            "statType": {"nodes": [{"dcid": stat_type}]},
            "measurementQualifier": {"nodes": [{"dcid": meas_qual}]},
            # The member lists every key of its index row, as the graph serves it, so the
            # index decides its family; only `constraint_props` carry a value arc.
            "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
            **{p: {"nodes": [{"dcid": row.constraint_properties[p]}]} for p in constraint_props},
        },
    }


def _synthetic_sv_node(
    dcid: str, *, label: str, pop_type: str, meas_prop: str, stat_type: str, meas_qual: str
) -> dict:
    """Build an unconstrained node for a five-tuple absent from the fixture index."""
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": label}]},
            "populationType": {"nodes": [{"dcid": pop_type}]},
            "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
            "statType": {"nodes": [{"dcid": stat_type}]},
            "measurementQualifier": {"nodes": [{"dcid": meas_qual}]},
            "constraintProperties": {"nodes": []},
        },
    }


_recalled1_row = _real_row(_RECALLED_SV1)

_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV0: _sv_node(
        _RECALLED_SV0,
        pop_type=_POP_TYPE_0,
        meas_prop=_MEAS_PROP_0,
        stat_type=_STAT_TYPE_0,
        meas_qual=_MEAS_QUAL_0,
        constraint_props=list(_real_row(_RECALLED_SV0).constraint_properties),
    ),
    _EXTRA_SV0: {"label": "test index-only sibling 0", "type": "StatisticalVariable"},
    _EXTRA_PURPOSE: {"label": "CRS purpose 51010"},
    _RECALLED_SV1: _sv_node(
        _RECALLED_SV1,
        pop_type=_POP_TYPE_1,
        meas_prop=_MEAS_PROP_1,
        stat_type=_STAT_TYPE_1,
        meas_qual=_MEAS_QUAL_1,
        constraint_props=list(_SHAPE1_BOUND_PROPS),
    ),
    _EXTRA_SV1: {"label": "test index-only sibling 1", "type": "StatisticalVariable"},
    # standard_bindings_from_arcs' own "value" bindings need each bound
    # value dcid to resolve too (build_slot falls back to unbound
    # otherwise) -- one node per _SHAPE1_BOUND_PROPS value, all real.
    **{
        v: {"label": v}
        for p, v in _recalled1_row.constraint_properties.items()
        if p in _SHAPE1_BOUND_PROPS
    },
    # The query names this reading's words, so its nomination is a rival (status rule 2(b)).
    _BARE_SV2: _synthetic_sv_node(
        _BARE_SV2,
        label="Fabricated test measure",
        pop_type=_POP_TYPE_2,
        meas_prop=_MEAS_PROP_2,
        stat_type=_STAT_TYPE_2,
        meas_qual=_MEAS_QUAL_2,
    ),
}

_DETECT = {
    _QUERY: {
        "svs": [_RECALLED_SV0, _RECALLED_SV1, _BARE_SV2],
        "cosine_scores": [0.9, 0.85, 0.8],
    }
}

_OBS = {
    f"{_EXTRA_SV0}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}],
    f"{_EXTRA_SV1}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}],
    f"{_BARE_SV2}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}],
}


class _ChooseShape0AlsoPlausibleShapes12LLM:
    """Elects shape 0 (FinancialFlow/grantEquivalent), binding crsPurpose to
    _EXTRA_SV0's own real value, and names shapes 1 (FinancialObligation/
    amount) and 2 (the fabricated non-indexed shape) as also_plausible --
    genuine alternates, materialised through
    region.candidates._materialise_alt_shapes/standard_bindings_from_arcs
    rather than through this bind call at all."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_PURPOSE,
                    kind="value",
                    value_dcids=[_EXTRA_PURPOSE],
                )
            ],
            also_plausible=[1, 2],
            ask=None,
        ), None


def _run():
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs=_OBS)
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_ChooseShape0AlsoPlausibleShapes12LLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def _spec_for(specs, sv_dcid: str):
    return next(
        spec for spec in specs if any(sv.ref.ref == sv_dcid for sv in spec.resolved.members)
    )


def test_broad_candidate_reports_its_real_population_in_menu_order():
    """The two-member index-sourced population reports its real member count and keeps its
    menu place ahead of the one-member non-indexed shape."""
    result = _run()
    assert result.status == "candidates"
    specs = result.specs
    assert len(specs) == 3

    broad_spec = _spec_for(specs, _EXTRA_SV1)
    narrow_spec = _spec_for(specs, _BARE_SV2)

    broad_count = broad_spec.resolved.shape.member_count
    narrow_count = narrow_spec.resolved.shape.member_count

    # The fixture index has exactly two members matching the four bound axes.
    assert broad_count == 2
    assert narrow_count == 1
    assert broad_count > narrow_count

    assert specs.index(broad_spec) < specs.index(narrow_spec), (
        "menu order: the candidate ranked first on the menu comes first"
    )


def test_non_indexed_candidate_reports_the_same_count_as_before():
    """A non-indexed, unconstrained shape keeps its single recalled member."""
    result = _run()
    assert result.status == "candidates"
    bare_spec = _spec_for(result.specs, _BARE_SV2)
    assert bare_spec.resolved.shape.member_count == 1
