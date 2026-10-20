"""Each alternate candidate uses its own materialised member facts to resolve StatVars and slot
values.

The elected grantEquivalent shape binds crsPurpose. The alternate FinancialObligation/amount
shape binds four declared axes, leaving crsTypeOfFlow open so its index-only sibling can survive.
"""

from __future__ import annotations

import asyncio

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.binders import Binder
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from tests.engine._binders import ReplyBinder, elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "grant equivalent disbursements and financial obligation commitments to Kenya"
_KEN_DCID = "country/KEN"

# Elected FinancialFlow/grantEquivalent shape.
_POP_TYPE_1 = "FinancialFlow"
_MEAS_PROP_1 = "grantEquivalent"
_STAT_TYPE_1 = "measuredValue"
_MEAS_QUAL_1 = "Nominal"

_RECALLED_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode11110_Nominal"
)
_EXTRA_SV = (
    "GrantEquivalent_FinancialFlow_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA01_CRSPurposeCode51010_Nominal"
)
_PROP_PURPOSE = "crsPurpose"
_EXTRA_PURPOSE = "CRSPurposeCode51010"

# Alternate FinancialObligation/amount shape.
_POP_TYPE_2 = "FinancialObligation"
_MEAS_PROP_2 = "amount"
_STAT_TYPE_2 = "measuredValue"
_MEAS_QUAL_2 = "Nominal"

_RECALLED_SV2 = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)
_EXTRA_SV2 = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow30_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)
# Leave the recalled member's crsTypeOfFlow value arc out so the differing sibling can survive.
_SHAPE2_BOUND_PROPS = (
    "crsCooperationModality",
    _PROP_PURPOSE,
    "crsTypeOfFinance",
    "financialObligationType",
)


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


_recalled2_row = _real_row(_RECALLED_SV2)

_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _RECALLED_SV: _sv_node(
        _RECALLED_SV,
        pop_type=_POP_TYPE_1,
        meas_prop=_MEAS_PROP_1,
        stat_type=_STAT_TYPE_1,
        meas_qual=_MEAS_QUAL_1,
        constraint_props=list(_real_row(_RECALLED_SV).constraint_properties),
    ),
    _EXTRA_SV: {"label": "test index-only sibling 1", "type": "StatisticalVariable"},
    _EXTRA_PURPOSE: {"label": "CRS purpose 51010"},
    _RECALLED_SV2: _sv_node(
        _RECALLED_SV2,
        pop_type=_POP_TYPE_2,
        meas_prop=_MEAS_PROP_2,
        stat_type=_STAT_TYPE_2,
        meas_qual=_MEAS_QUAL_2,
        constraint_props=list(_SHAPE2_BOUND_PROPS),
    ),
    _EXTRA_SV2: {"label": "test index-only sibling 2", "type": "StatisticalVariable"},
    # standard_bindings_from_arcs' own "value" bindings need each bound
    # value dcid to resolve too (build_slot falls back to unbound
    # otherwise) -- one node per _SHAPE2_BOUND_PROPS value, all real.
    **{
        v: {"label": v}
        for p, v in _recalled2_row.constraint_properties.items()
        if p in _SHAPE2_BOUND_PROPS
    },
}

_DETECT = {_QUERY: {"svs": [_RECALLED_SV, _RECALLED_SV2], "cosine_scores": [0.9, 0.85]}}

_OBS = {
    f"{_EXTRA_SV}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}],
    f"{_EXTRA_SV2}|{_KEN_DCID}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}],
}


class _ChooseShape1AlsoPlausibleShape2LLM:
    """Elect the grantEquivalent shape, bind crsPurpose, and mark the amount shape as plausible."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(
            0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_PURPOSE,
                    kind="value",
                    value_dcids=[_EXTRA_PURPOSE],
                )
            ],
            also_plausible=[1],
        ), None


def _run(*, binder: Binder | None = None):
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
            binder=binder,
            llm=_ChooseShape1AlsoPlausibleShape2LLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def test_alternate_shapes_index_only_winner_renders_its_slot_values_on_the_wire():
    """The alternate's index-only StatVar must carry all four bound slot values."""
    result = _run(binder=ReplyBinder())
    assert result.status == "candidates"
    assert len(result.specs) == 2
    alt_spec = next(
        spec
        for spec in result.specs
        if any(sv.ref.ref == _EXTRA_SV2 for sv in spec.resolved.members)
    )
    stat_var = next(sv for sv in alt_spec.resolved.members if sv.ref.ref == _EXTRA_SV2)
    rendered = {sv.key.property.ref: sv.value.ref.ref for sv in stat_var.slot_values}
    expected = {p: _recalled2_row.constraint_properties[p] for p in _SHAPE2_BOUND_PROPS}
    assert rendered == expected
