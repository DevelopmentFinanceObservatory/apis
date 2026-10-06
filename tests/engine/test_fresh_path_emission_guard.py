"""Checks fresh resolution's rule that every winning StatVar asserts each bound property.

Fixtures cover a silent aggregate, a matching asserter, and siblings split across two bound axes.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import (
    NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    NO_MEMBER_HAS_DATA_AT_PLACE,
)
from tests.engine._builders import sv_arcs as _sv_arcs
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "female test incidence in Kenya"

_PROP_GENDER = "gender"
_VAL_FEMALE = "Female"

_SV_BARE = "TEST/FreshGenderIncidence_Bare"
_SV_FEMALE = "TEST/FreshGenderIncidence_Female"

_KEN_DCID = "country/KEN"

_POP_TYPE = "TestFreshIncidenceEvent"
_MEAS_PROP = "testFreshIncidence"
_STAT_TYPE = "measuredValue"


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _SV_BARE: {
        "label": "Test fresh incidence (all genders)",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            pop_type=_POP_TYPE,
            meas_prop=_MEAS_PROP,
            stat_type=_STAT_TYPE,
            label="Test fresh incidence (all genders)",
            gender=None,
        ),
    },
    _SV_FEMALE: {
        "label": "Test fresh incidence, female",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            pop_type=_POP_TYPE,
            meas_prop=_MEAS_PROP,
            stat_type=_STAT_TYPE,
            label="Test fresh incidence, female",
            gender=_VAL_FEMALE,
        ),
    },
    _VAL_FEMALE: {"label": "Female", "type": "GenderType"},
}

_DETECT = {
    _QUERY: {
        "svs": [_SV_BARE, _SV_FEMALE],
        "cosine_scores": [0.9, 0.85],
    },
}


class _GenderBindLLM:
    """Drafts a `gender=Female` value binding."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_GENDER,
                    kind="value",
                    value_dcids=[_VAL_FEMALE],
                ),
            ],
            also_plausible=[],
            ask=None,
        ), None


def _run(*, obs: dict) -> object:
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs=obs)
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
            llm=_GenderBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def test_silent_sibling_only_has_data_returns_no_data_with_flagged_match():
    """A silent aggregate cannot answer the bound claim when the asserting StatVar has no data."""
    obs = {
        f"{_SV_BARE}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]
    }
    result = _run(obs=obs)

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is not None
    flagged_spec = result.flagged_match[0]
    assert [sv.ref.dcid for sv in flagged_spec.resolved.members] == [_SV_FEMALE]
    assert flagged_spec.resolved.available_at_place is False
    assert flagged_spec.resolved.confidence == "low"
    assert result.gate == NO_MEMBER_HAS_DATA_AT_PLACE


def test_asserting_member_has_data_returns_definite_with_binding_intact():
    """The asserting member's data yields a definite answer with the binding intact."""
    obs = {
        f"{_SV_FEMALE}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]
    }
    result = _run(obs=obs)

    assert result.status == "definite"
    spec = result.specs[0]
    assert [sv.ref.dcid for sv in spec.resolved.members] == [_SV_FEMALE]
    gender_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.dcid == _PROP_GENDER
    )
    assert gender_slot.binding.kind == "value"


def test_both_have_data_asserting_member_wins():
    """The asserting member wins when both siblings have observations."""
    obs = {
        f"{_SV_BARE}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
        f"{_SV_FEMALE}|{_KEN_DCID}": [
            {"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}
        ],
    }
    result = _run(obs=obs)

    assert result.status == "definite"
    spec = result.specs[0]
    assert [sv.ref.dcid for sv in spec.resolved.members] == [_SV_FEMALE]


def test_coherence_invariant_bound_slot_never_inapplicable():
    """A bound property is excluded from `inapplicable_slots`."""
    obs = {
        f"{_SV_FEMALE}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}]
    }
    result = _run(obs=obs)

    assert result.status == "definite"
    spec = result.specs[0]
    assert _PROP_GENDER not in spec.resolved.inapplicable_slots


# Split-ownership: two bound VALUE axes, each asserted by a different,
# non-overlapping sibling -- two independently-bound axes whose asserters
# are disjoint, closed by the unconditional assert-every-bound-property rule.

_PROP_RESIDENCE = "placeOfResidenceClassification"
_VAL_URBAN = "Urban"

_SV_A = "TEST/SplitOwnership_GenderOnly"  # asserts gender=Female only
_SV_B = "TEST/SplitOwnership_ResidenceOnly"  # asserts residence=Urban only

_SPLIT_QUERY = "female urban test incidence in Kenya"


def _split_sv_arcs(*, label: str, prop: str, value: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value}]},
    }


_SPLIT_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _SV_A: {
        "label": "Test fresh incidence, female (any residence)",
        "type": "StatisticalVariable",
        "arcs": _split_sv_arcs(
            label="Test fresh incidence, female (any residence)",
            prop=_PROP_GENDER,
            value=_VAL_FEMALE,
        ),
    },
    _SV_B: {
        "label": "Test fresh incidence, urban (any gender)",
        "type": "StatisticalVariable",
        "arcs": _split_sv_arcs(
            label="Test fresh incidence, urban (any gender)",
            prop=_PROP_RESIDENCE,
            value=_VAL_URBAN,
        ),
    },
    _VAL_FEMALE: {"label": "Female", "type": "GenderType"},
    _VAL_URBAN: {"label": "Urban", "type": "HumanSettlementClassificationEnum"},
}

_SPLIT_DETECT = {
    _SPLIT_QUERY: {
        "svs": [_SV_A, _SV_B],
        "cosine_scores": [0.9, 0.85],
    },
}

_SPLIT_OBS = {
    f"{_SV_A}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    f"{_SV_B}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


class _SplitOwnershipBindLLM:
    """Drafts gender and residence bindings split across the two siblings."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_GENDER,
                    kind="value",
                    value_dcids=[_VAL_FEMALE],
                ),
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_RESIDENCE,
                    kind="value",
                    value_dcids=[_VAL_URBAN],
                ),
            ],
            also_plausible=[],
            ask=None,
        ), None


def test_split_ownership_neither_partial_asserter_wins():
    """Neither sibling can answer both bound axes, despite both having data."""
    graph = FakeGraph(
        nodes=_SPLIT_NODES, detect=_SPLIT_DETECT, resolve={"Kenya": _KEN_DCID}, obs=_SPLIT_OBS
    )
    result = asyncio.run(
        resolve_variable(
            _SPLIT_QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_SPLIT_QUERY,
            role_query=_SPLIT_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_SplitOwnershipBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is not None
    flagged_spec = result.flagged_match[0]
    assert flagged_spec.resolved.available_at_place is False
    assert flagged_spec.resolved.confidence == "low"
    assert result.gate == NO_MEMBER_HAS_DATA_AT_PLACE


# Same split-ownership fixture, driven with NO place entity at all
# (entities=[]) -- the no-entity branch of standard_resolve() must
# apply the same assert-every-bound-property eligibility rule as the
# entity-probe, not just SV existence.

_SPLIT_NO_ENTITY_QUERY = "female urban test incidence rate"

_SPLIT_NO_ENTITY_DETECT = {
    _SPLIT_NO_ENTITY_QUERY: {
        "svs": [_SV_A, _SV_B],
        "cosine_scores": [0.9, 0.85],
    },
}


def test_split_ownership_no_entity_neither_partial_asserter_wins():
    """Without a place, neither partial asserter can back both bound properties."""
    graph = FakeGraph(nodes=_SPLIT_NODES, detect=_SPLIT_NO_ENTITY_DETECT, resolve={}, obs={})
    result = asyncio.run(
        resolve_variable(
            _SPLIT_NO_ENTITY_QUERY,
            entities=[],
            date_request=None,
            detect_query=_SPLIT_NO_ENTITY_QUERY,
            role_query=_SPLIT_NO_ENTITY_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_SplitOwnershipBindLLM(),
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "no_matching_slice"
    assert result.flagged_match is not None
    flagged_spec = result.flagged_match[0]
    # No place entity here to be unavailable AT -- distinct from the
    # entity-probe's own flagged-unavailable carrier (available_at_place=False,
    # confidence="low"), which the sibling test above (with entities=["Kenya"])
    # exercises. available_at_place/confidence both stay at their documented
    # not-assessed default.
    assert flagged_spec.resolved.available_at_place is None
    assert flagged_spec.resolved.confidence is None
    assert result.gate == NO_MEMBER_ASSERTS_BOUND_PROPERTIES
    # The representative is the best partial asserter (_bind_aware_order over
    # the full member set): both SVs tie at match-count 1, so insertion order
    # (gender-only inserted first) decides.
    assert flagged_spec.resolved.members[0].ref.dcid == _SV_A
