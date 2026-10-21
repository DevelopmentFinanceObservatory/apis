"""Checks Search's CRS member digest against `enumerate_members` for two real index members.
Four shared axes are bound; `crsTypeOfFlow` distinguishes the two-member family.
"""

from __future__ import annotations

import asyncio
import hashlib

import rfc8785

from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.clients.graph import Facet
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from series_coverage.schema import RoleNames
from statvar_members import FiveTuple, enumerate_members
from tests.engine._fake_coverage import FakeCoverageService, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import door_roles_for_test_query
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver
from tests.statvar_members.conftest import FakeMemberArcReader

_QUERY = "financial obligation commitments to Kenya"
_KEN_DCID = "country/KEN"

_POP_TYPE = "FinancialObligation"
_MEAS_PROP = "amount"
_STAT_TYPE = "measuredValue"
_MEAS_QUAL = "Nominal"

# The two real dcids test_candidates_member_count_ordering.py's own shape 1
# uses: real members of the fixture CRS index, sharing every
# constraint property but crsTypeOfFlow (30 vs 10).
_RECALLED_SV = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow30_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)
_EXTRA_SV = (
    "Amount_FinancialObligation_Commitment_CRSTypeOfFlow10_CRSTypeOfFinance110_"
    "CRSCooperationModalityA02_CRSPurposeCode12110_Nominal"
)

_PROP_MODALITY = "crsCooperationModality"
_PROP_PURPOSE = "crsPurpose"
_PROP_FINANCE = "crsTypeOfFinance"
_PROP_OBLIGATION_TYPE = "financialObligationType"
_PROP_FLOW = "crsTypeOfFlow"

# The four axes both real rows share (real values, read off the fixture
# index below rather than hand-typed, so a corpus refresh that moves them
# fails this fixture loudly instead of silently). crsTypeOfFlow stays
# unbound: it's the one axis the two dcids differ on.
_BOUND_PROPS = (_PROP_MODALITY, _PROP_PURPOSE, _PROP_FINANCE, _PROP_OBLIGATION_TYPE)

_ROLE_RECIPIENT = "financingRecipient"


def _real_row(dcid: str):
    index = get_index()
    return next(r for r in index.statvars if r.dcid == dcid)


def test_fixture_matches_the_real_index():
    """The fixture index members share four bound axes and differ on crsTypeOfFlow."""
    recalled_row = _real_row(_RECALLED_SV)
    extra_row = _real_row(_EXTRA_SV)
    target = (_POP_TYPE, _MEAS_PROP, _STAT_TYPE, _MEAS_QUAL, None)
    for row in (recalled_row, extra_row):
        five_tuple = (
            row.population_type or "",
            row.measured_property,
            row.stat_type or "",
            row.measurement_qualifier,
            row.measurement_denominator,
        )
        assert five_tuple == target
    for prop in _BOUND_PROPS:
        assert recalled_row.constraint_properties[prop] == extra_row.constraint_properties[prop]
    assert (
        recalled_row.constraint_properties[_PROP_FLOW]
        != extra_row.constraint_properties[_PROP_FLOW]
    )


def _sv_arcs_from_real_row(dcid: str) -> dict:
    row = _real_row(dcid)
    constraint_props = list(row.constraint_properties)
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": dcid}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "measurementQualifier": {"nodes": [{"dcid": _MEAS_QUAL}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraint_props]},
        "observationProperties": {"nodes": [{"dcid": p} for p in row.observation_properties]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in row.constraint_properties.items()},
    }


def _facet(*, obs_count: int = 3, facet_id: str = "f1") -> Facet:
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"],
        values=[0.0],
        facet_id=facet_id,
    )


def _bound_value_nodes() -> dict[str, dict]:
    """One node per bound value dcid: build_slot falls back to unbound when a
    bound value dcid doesn't resolve, so each of the four bound values below
    needs its own graph node to ground as a "value" binding."""
    recalled_row = _real_row(_RECALLED_SV)
    return {
        recalled_row.constraint_properties[prop]: {
            "label": recalled_row.constraint_properties[prop]
        }
        for prop in _BOUND_PROPS
    }


def _nodes() -> dict:
    return {
        _KEN_DCID: {"label": "Kenya", "type": "Country"},
        _RECALLED_SV: {
            "label": "test recalled member",
            "type": "StatisticalVariable",
            "arcs": _sv_arcs_from_real_row(_RECALLED_SV),
        },
        _EXTRA_SV: {
            "label": "test index-only sibling",
            "type": "StatisticalVariable",
            "arcs": _sv_arcs_from_real_row(_EXTRA_SV),
        },
        **_bound_value_nodes(),
    }


_DETECT = {_QUERY: {"svs": [_RECALLED_SV], "cosine_scores": [0.9]}}

# Only _EXTRA_SV carries an observation at Kenya (as recipient), so the
# role-bearing resolve elects it -- the admitted population (both dcids)
# does not depend on which one this probe confirms.
_OBS = {_EXTRA_SV: {((_ROLE_RECIPIENT, _KEN_DCID),): [_facet()]}}


class _BindAllFourSharedAxesLLM:
    """Binds the four shared axes and leaves `crsTypeOfFlow` unbound."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        recalled_row = _real_row(_RECALLED_SV)
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=prop,
                    kind="value",
                    value_dcids=[recalled_row.constraint_properties[prop]],
                )
                for prop in _BOUND_PROPS
            ],
            also_plausible=[],
            ask=None,
        ), None


def _run(monkeypatch, llm=None):
    # Only `_EXTRA_SV` has coverage, so it wins observation confirmation.
    # Member admission still includes both dcids.
    _auto_coverage_index_1 = patch_coverage(
        monkeypatch,
        [(_EXTRA_SV, "f1", "country/USA", _KEN_DCID, [2020])],
        role_names={_EXTRA_SV: RoleNames(donor="financingProvider", recipient=_ROLE_RECIPIENT)},
    )
    graph = FakeGraph(nodes=_nodes(), detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs={})
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            entity_roles=door_roles_for_test_query(_QUERY, ["Kenya"]),
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations=_OBS, availability={}),
            llm=llm if llm is not None else _BindAllFourSharedAxesLLM(),
            base_timing={},
            coverage=FakeCoverageService(_auto_coverage_index_1).snapshot(),
        )
    )


def _independent_digest_via_enumerate_members() -> str:
    """The same two dcids, resolved from scratch through
    statvar_members.enumerate_members rather than search's own resolve
    machinery: a FakeMemberArcReader fed the two real dcids' own real arcs
    (read off the fixture index, mirroring _sv_arcs_from_real_row above),
    queried with the identical FiveTuple and the identical four "value"
    slot bindings, crsTypeOfFlow left ("unbound",)."""
    recalled_row = _real_row(_RECALLED_SV)
    extra_row = _real_row(_EXTRA_SV)
    arcs = {
        dcid: {
            "typeOf": ("StatisticalVariable",),
            "populationType": (_POP_TYPE,),
            "measuredProperty": (_MEAS_PROP,),
            "statType": (_STAT_TYPE,),
            "measurementQualifier": (_MEAS_QUAL,),
            "measurementDenominator": (),
            "constraintProperties": tuple(row.constraint_properties),
            "observationProperties": row.observation_properties,
            **{p: (v,) for p, v in row.constraint_properties.items()},
        }
        for dcid, row in ((_RECALLED_SV, recalled_row), (_EXTRA_SV, extra_row))
    }
    reader = FakeMemberArcReader(arcs=arcs)
    five_tuple = FiveTuple(
        population_type=_POP_TYPE,
        measured_property=_MEAS_PROP,
        stat_type=_STAT_TYPE,
        measurement_qualifier=_MEAS_QUAL,
    )
    slots = {prop: ("value", recalled_row.constraint_properties[prop]) for prop in _BOUND_PROPS}
    slots[_PROP_FLOW] = ("unbound",)

    group = enumerate_members(five_tuple=five_tuple, slots=slots, graph=reader)

    member_dcids = [member.stat_var for member in group.members]
    assert sorted(member_dcids) == sorted([_RECALLED_SV, _EXTRA_SV])
    canonical = rfc8785.dumps(sorted(member_dcids))
    return hashlib.sha256(canonical).hexdigest()


def test_search_member_digest_agrees_with_enumerate_members_for_a_crs_family(monkeypatch):
    result = _run(monkeypatch)
    assert result.status == "definite"
    spec = result.specs[0]
    shape = spec.resolved.shape

    assert shape.member_count == 2
    assert shape.member_digest is not None

    assert shape.member_digest == _independent_digest_via_enumerate_members()


def test_stat_vars_lists_the_confirmed_subset_while_the_digest_names_the_whole_binding(
    monkeypatch,
):
    """`resolved.stat_vars` counts the members confirmed at the place, although the
    binding is wider. The population a mint's
    `trace.family_member_digest` states is `resolved.shape`'s, which `member_digest`
    hashes, members without observations included (C75, C75a).
    """
    result = _run(monkeypatch)
    resolved = result.specs[0].resolved

    assert [item.ref.ref for item in resolved.members] == [_EXTRA_SV]
    assert resolved.stat_vars.member_count == 1

    assert resolved.shape.member_count == 2
    assert resolved.shape.member_digest == _independent_digest_via_enumerate_members()


class _AlsoNamesASourceLLM(_BindAllFourSharedAxesLLM):
    """The same drafts, with a source-axis value the prompt never offered."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        output, usage = await super().generate_structured(
            prompt=prompt, system=system, schema=schema
        )
        made_up = SlotBindingDraft(
            axis="source",
            property_dcid=None,
            kind="value",
            value_dcids=["provenance/ONEImputedMultilateralSectors"],
        )
        return output.model_copy(update={"bindings": [*output.bindings, made_up]}), usage


def test_a_source_draft_the_model_writes_leaves_the_family_unchanged(monkeypatch):
    """The imputed dataset publishes no member of this family, so electing by the model's
    source draft would resolve nothing. The draft is dropped, and the spec and its family
    are those of the resolve without it."""
    plain = _run(monkeypatch).specs[0]

    result = _run(monkeypatch, llm=_AlsoNamesASourceLLM())

    assert result.status == "definite"
    spec = result.specs[0]
    assert spec.spec_id == plain.spec_id
    assert spec.resolved.shape.member_digest == plain.resolved.shape.member_digest
    assert spec.resolved.shape.member_count == 2
    assert spec.definition.identity.facets == plain.definition.identity.facets
