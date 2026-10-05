"""A resubmit must not serve a StatVar that fails to assert a bound property.
The fixtures cover a lone non-asserting member, a valid co-answer, and the
best partial match, including the no-entity gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from materialise.place_scope import DefaultPlaceScope
from search.engine.region import promote_ground, promote_materialise, resolve_spec_resubmit
from search.engine.region.codes import NO_MEMBER_HAS_DATA_AT_PLACE
from search.engine.retrieval.retrieve import Materialised
from search.models import (
    BindingSet,
    BindingValue,
    CoverageBare,
    IdentityWindowRange,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    make_base,
    subject_entity,
)
from tests.engine._builders import gref as _gref
from tests.engine._builders import prop_node as _prop_node
from tests.engine._builders import sv_arcs as _sv_arcs
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_PROP_GENDER = "gender"
_VAL_FEMALE = "Female"

_SV_BARE = "TEST/GenderIncidence_Bare"
_SV_FEMALE = "TEST/GenderIncidence_Female"

_KEN_DCID = "country/KEN"

_POP_TYPE = "TestIncidenceEvent"
_MEAS_PROP = "testIncidence"
_STAT_TYPE = "measuredValue"
# Mirrors _sv_arcs' own populationType/measuredProperty/statType below --
# resolve_spec_resubmit's standard path re-derives the shape from the posted
# resolved_stat_var_dcids' real arcs and requires it to match the posted
# shape structurally.
_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": _POP_TYPE,
    "measured_property": _MEAS_PROP,
    "stat_type": _STAT_TYPE,
}


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _SV_BARE: {
        "label": "Test incidence (all genders)",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            pop_type=_POP_TYPE,
            meas_prop=_MEAS_PROP,
            stat_type=_STAT_TYPE,
            label="Test incidence (all genders)",
            gender=None,
        ),
    },
    _SV_FEMALE: {
        "label": "Test incidence, female",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            pop_type=_POP_TYPE,
            meas_prop=_MEAS_PROP,
            stat_type=_STAT_TYPE,
            label="Test incidence, female",
            gender=_VAL_FEMALE,
        ),
    },
    _VAL_FEMALE: {"label": "Female", "type": "GenderType"},
}

# Only the bare, non-asserting aggregate has data at Kenya; the explicit
# Female sibling has none.
_OBS = {
    f"{_SV_BARE}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


def _guard_graph(*, with_obs: bool = True) -> FakeGraph:
    return FakeGraph(nodes=_NODES, obs=_OBS if with_obs else {})


def _resubmit_input() -> ResubmitIdentityPayload:
    entities = [subject_entity(_KEN_DCID)]
    gender_bound_slot = Slot(
        key=SlotKey(axis="how", property=_gref(_PROP_GENDER), label="gender"),
        binding=BindingValue(value=SlotValue(ref=_gref(_VAL_FEMALE), value_kind="enum_value")),
    )
    base_spec_id, base_identity = make_base(
        _SHAPE,
        slots=[gender_bound_slot],
        entities=entities,
    )
    return ResubmitIdentityPayload(
        identity=base_identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        resolved_stat_var_dcids=[_SV_FEMALE, _SV_BARE],
    )


def test_bound_gender_with_only_bare_aggregate_data_returns_no_data():
    """SV_BARE has data but lacks gender=Female; SV_FEMALE asserts it but has
    none. The resubmit returns no_data instead of serving either member."""
    region = resolve_spec_resubmit(
        inp=_resubmit_input(),
        graph=_guard_graph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )
    assert region.status == "no_data"
    assert region.no_data_reason == "no_observations"
    assert region.specs == ()


def test_no_data_response_flags_the_explicit_member_as_the_match():
    """The unavailable carrier names SV_FEMALE, the member that asserts the
    bound gender, rather than SV_BARE, which has observations."""
    region = resolve_spec_resubmit(
        inp=_resubmit_input(),
        graph=_guard_graph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )
    assert region.flagged_match is not None
    flagged_spec = region.flagged_match[0]
    assert [sv.ref for sv in flagged_spec.resolved.stat_vars.items] == [_SV_FEMALE]
    # outcomes.py's documented flagged_match contract: "its available_at_place is
    # False and confidence is low" -- the carrier Spec must actually carry those
    # fields, not ship the None/None "not assessed" default.
    assert flagged_spec.resolved.available_at_place is False
    assert flagged_spec.resolved.confidence == "low"
    assert region.gate == NO_MEMBER_HAS_DATA_AT_PLACE


def test_no_data_anywhere_still_returns_no_data():
    """With no observations for either sibling, the result remains no_data."""
    region = resolve_spec_resubmit(
        inp=_resubmit_input(),
        graph=_guard_graph(with_obs=False),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )
    assert region.status == "no_data"
    assert region.no_data_reason == "no_observations"
    assert region.specs == ()
    assert region.flagged_match is not None
    flagged_spec = region.flagged_match[0]
    assert [sv.ref for sv in flagged_spec.resolved.stat_vars.items] == [_SV_FEMALE]


# ---------------------------------------------------------------------------
# Regression: mixed presence must not fabricate a no_data when an asserting
# co-answer has real observations.
# ---------------------------------------------------------------------------

_IND_DCID = "country/IND"

_URBAN_SV = "Count_Person_Urban_Female"  # pre-existing fixture: gender=Female, residence=Urban
_RURAL_SV = "Count_Person_Rural_Female"  # pre-existing fixture: gender=Female, residence=Rural
_RURAL_ADULT_SV = "Count_Person_Rural_Female_Adult"  # synthetic: + age=Years18Onwards


def _mixed_presence_graph() -> FakeGraph:
    """Build three India siblings: Urban_Female and Rural_Adult assert age;
    Rural_Female asserts residence only. All three have observations."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())

    # Both pre-existing fixture SVs carry a "name" arc but a null top-level
    # "label" -- FakeGraph.node_labels_batch only reads the top-level field.
    for dcid in (_URBAN_SV, _RURAL_SV):
        if nodes[dcid].get("label") is None:
            name_nodes = nodes[dcid]["arcs"]["name"]["nodes"]
            nodes[dcid]["label"] = name_nodes[-1]["value"]

    # "Years18Onwards" has no top-level fixture entry, unlike "Female"/"Urban"/
    # "Rural" -- without it, node_labels_batch can't resolve it and grounding
    # would silently drop the binding for want of a label.
    nodes["Years18Onwards"] = {"label": "Years18Onwards", "type": None}

    # Patch age=Years18Onwards onto the Urban sibling -- the scenario needs
    # `age` genuinely bound in base_identity (asserted by the first posted
    # resolved StatVar, so the pre-filter seed-vs-base check passes), while
    # the Rural (non-Adult) sibling stays silent on it.
    nodes[_URBAN_SV]["arcs"]["age"] = {"nodes": [_prop_node("Years18Onwards", ["Class"])]}
    nodes[_URBAN_SV]["arcs"]["constraintProperties"]["nodes"].append(
        _prop_node("age", ["Property"])
    )

    nodes[_RURAL_ADULT_SV] = {
        "label": "Adult rural female population",
        "type": "StatisticalVariable",
        "arcs": {
            "statType": {"nodes": [_prop_node("measuredValue", ["Property"])]},
            "typeOf": {"nodes": [_prop_node("StatisticalVariable", ["Class"])]},
            "measuredProperty": {"nodes": [_prop_node("count", ["Property"])]},
            "populationType": {"nodes": [_prop_node("Person", ["Class"])]},
            "gender": {"nodes": [_prop_node("Female", ["GenderType"])]},
            "placeOfResidenceClassification": {
                "nodes": [_prop_node("Rural", ["HumanSettlementClassificationEnum"])]
            },
            "age": {"nodes": [_prop_node("Years18Onwards", ["Class"])]},
            "constraintProperties": {
                "nodes": [
                    _prop_node("gender", ["Property"]),
                    _prop_node("placeOfResidenceClassification", ["Property"]),
                    _prop_node("age", ["Property"]),
                ]
            },
            "name": {
                "nodes": [
                    {
                        "provenanceId": "dc/base/HumanReadableStatVars",
                        "value": "Adult rural female population",
                    }
                ]
            },
        },
    }

    _obs = [
        {
            "earliestDate": "2001",
            "latestDate": "2021",
            "obsCount": 3,
            "provenanceId": "dc/base/test",
        }
    ]
    obs[f"{_URBAN_SV}|{_IND_DCID}"] = _obs
    obs[f"{_RURAL_SV}|{_IND_DCID}"] = _obs
    obs[f"{_RURAL_ADULT_SV}|{_IND_DCID}"] = _obs

    return FakeGraph(nodes=nodes, obs=obs)


def test_mixed_presence_asserting_coanswer_with_data_returns_definite():
    """Rural_Female has observations but omits bound age. Urban_Female and
    Rural_Adult assert both age and residence, so the answer stays definite
    and serves only those asserting members."""
    entities = [subject_entity(_IND_DCID)]
    age_slot = Slot(
        key=SlotKey(axis="how", property=_gref("age"), label="age"),
        binding=BindingValue(value=SlotValue(ref=_gref("Years18Onwards"), value_kind="enum_value")),
    )
    gender_slot = Slot(
        key=SlotKey(axis="how", property=_gref("gender"), label="gender"),
        binding=BindingValue(value=SlotValue(ref=_gref("Female"), value_kind="enum_value")),
    )
    residence_slot = Slot(
        key=SlotKey(
            axis="how",
            property=_gref("placeOfResidenceClassification", "place of residence"),
            label="place of residence",
        ),
        binding=BindingSet(
            values=[
                SlotValue(ref=_gref("Urban"), value_kind="enum_value"),
                SlotValue(ref=_gref("Rural"), value_kind="enum_value"),
            ]
        ),
    )
    base_spec_id, base_identity = make_base(
        PERSON_COUNT_SHAPE,
        slots=[age_slot, gender_slot, residence_slot],
        entities=entities,
    )
    inp = ResubmitIdentityPayload(
        identity=base_identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        # Unedited echo of the original mint's own confirmed set.
        resolved_stat_var_dcids=[_URBAN_SV, _RURAL_SV, _RURAL_ADULT_SV],
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=_mixed_presence_graph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
    )
    assert region.status == "definite", region.no_data_reason
    spec = region.specs[0]
    served = {sv.ref for sv in spec.resolved.stat_vars.items}
    assert served == {_URBAN_SV, _RURAL_ADULT_SV}, (
        f"expected the answer backed only by the members asserting every bound "
        f"claim (age + residence), got {served!r}. Count_Person_Rural_Female must "
        "never back the answer despite having real observations: it is silent on "
        "the bound `age` claim."
    )


# ---------------------------------------------------------------------------
# No-edits split-ownership, 3 bound properties, asymmetric partial matchers:
# pins the guard's has_data short-circuit against the ranking mismatch its
# own explicit_sv fallback (posted-order first element) would otherwise
# introduce relative to the shared resolver's own best-partial-matcher pick.
# ---------------------------------------------------------------------------

_PROP_RESIDENCE = "placeOfResidenceClassification"
_PROP_AGE = "age"
_VAL_URBAN = "Urban"
_VAL_ADULT = "Adult"

_SV_HIGH = "TEST/PartialMatch_GenderResidence"  # asserts 2 of 3: gender, residence
_SV_LOW = "TEST/PartialMatch_AgeOnly"  # asserts 1 of 3: age

_PARTIAL_POP_TYPE = "TestPartialIncidenceEvent"
_PARTIAL_MEAS_PROP = "testPartialIncidence"
# Mirrors _partial_sv_arcs' own populationType/measuredProperty/statType
# below -- resolve_spec_resubmit's standard path re-derives the shape from
# the posted resolved_stat_var_dcids' real arcs and requires it to match the
# posted shape structurally.
_PARTIAL_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": _PARTIAL_POP_TYPE,
    "measured_property": _PARTIAL_MEAS_PROP,
    "stat_type": _STAT_TYPE,
}


def _partial_sv_arcs(*, label: str, constraints: dict[str, str]) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": _PARTIAL_POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _PARTIAL_MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{prop: {"nodes": [{"dcid": val}]} for prop, val in constraints.items()},
    }


_PARTIAL_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _SV_HIGH: {
        "label": "Test partial incidence, female urban",
        "type": "StatisticalVariable",
        "arcs": _partial_sv_arcs(
            label="Test partial incidence, female urban",
            constraints={_PROP_GENDER: _VAL_FEMALE, _PROP_RESIDENCE: _VAL_URBAN},
        ),
    },
    _SV_LOW: {
        "label": "Test partial incidence, adult",
        "type": "StatisticalVariable",
        "arcs": _partial_sv_arcs(
            label="Test partial incidence, adult",
            constraints={_PROP_AGE: _VAL_ADULT},
        ),
    },
    _VAL_FEMALE: {"label": "Female", "type": "GenderType"},
    _VAL_URBAN: {"label": "Urban", "type": "Thing"},
    _VAL_ADULT: {"label": "Adult", "type": "Thing"},
}

# Both posted siblings carry real observations at Kenya -- neither asserts
# every bound property, so the outcome must be an honest flagged no_data, but
# WHICH sibling is named as flagged_match matters: it must be SV_HIGH (the
# best partial matcher, asserting 2 of 3 bound properties), matching whatever
# a fresh mint of this same scenario would name via the shared resolver's own
# ranking -- not SV_LOW, which only the guard's posted-order fallback would
# pick were the short-circuit missing.
_PARTIAL_OBS = {
    f"{_SV_HIGH}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    f"{_SV_LOW}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


def test_no_edits_three_property_partial_match_names_the_best_partial_matcher():
    """SV_HIGH asserts two of three bound properties; SV_LOW asserts one. LOW is
    posted first to expose ordering errors. The flagged no_data carrier must name
    SV_HIGH, the best partial match."""
    entities = [subject_entity(_KEN_DCID)]
    gender_slot = Slot(
        key=SlotKey(axis="how", property=_gref(_PROP_GENDER), label="gender"),
        binding=BindingValue(value=SlotValue(ref=_gref(_VAL_FEMALE), value_kind="enum_value")),
    )
    residence_slot = Slot(
        key=SlotKey(
            axis="how",
            property=_gref(_PROP_RESIDENCE, "place of residence"),
            label="place of residence",
        ),
        binding=BindingValue(value=SlotValue(ref=_gref(_VAL_URBAN), value_kind="enum_value")),
    )
    age_slot = Slot(
        key=SlotKey(axis="how", property=_gref(_PROP_AGE), label="age"),
        binding=BindingValue(value=SlotValue(ref=_gref(_VAL_ADULT), value_kind="enum_value")),
    )
    base_spec_id, base_identity = make_base(
        _PARTIAL_SHAPE,
        slots=[gender_slot, residence_slot, age_slot],
        entities=entities,
    )
    inp = ResubmitIdentityPayload(
        identity=base_identity,
        base_spec_id=base_spec_id,
        base_identity=base_identity,
        # LOW-first posted order: exposes the guard's naive posted-order
        # fallback were the has_data short-circuit not in place.
        resolved_stat_var_dcids=[_SV_LOW, _SV_HIGH],
    )
    graph = FakeGraph(nodes=_PARTIAL_NODES, obs=_PARTIAL_OBS)
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
    )

    assert region.status == "no_data"
    assert region.no_data_reason == "no_observations"
    assert region.flagged_match is not None
    flagged_spec = region.flagged_match[0]
    assert [sv.ref for sv in flagged_spec.resolved.stat_vars.items] == [_SV_HIGH]


_AGE_SLOT = Slot(
    key=SlotKey(axis="how", property=_gref("age"), label="age"),
    binding=BindingValue(value=SlotValue(ref=_gref("Years18Onwards"), value_kind="enum_value")),
)
_GENDER_SLOT = Slot(
    key=SlotKey(axis="how", property=_gref("gender"), label="gender"),
    binding=BindingValue(value=SlotValue(ref=_gref("Female"), value_kind="enum_value")),
)
_RESIDENCE_SLOT = Slot(
    key=SlotKey(
        axis="how",
        property=_gref("placeOfResidenceClassification", "place of residence"),
        label="place of residence",
    ),
    binding=BindingSet(
        values=[
            SlotValue(ref=_gref("Urban"), value_kind="enum_value"),
            SlotValue(ref=_gref("Rural"), value_kind="enum_value"),
        ]
    ),
)


def _point_years(years: range) -> list[dict]:
    return [
        {
            "earliestDate": str(years[0]),
            "latestDate": str(years[-1]),
            "obsCount": len(years),
            "provenanceId": "dc/base/test",
            "dates": [str(y) for y in years],
            "values": [1.0] * len(years),
        }
    ]


def test_asserting_members_with_data_only_outside_the_years_answer_for_the_window(
    monkeypatch,
):
    """The first probe's winner, Rural_Female, is silent on the bound age, so the resubmit
    re-runs over Urban_Female and Rural_Adult, which assert every bound property and hold
    2001 to 2005 alone. Their re-run is a window carrier: the answer is for the window and
    names their span, never "no member has data at the place"."""
    real = promote_materialise.run_materialise
    calls: list[int] = []

    def _first_probe_finds_only_the_silent_member(shape, *args, **kwargs):
        # Both resolvers serve only asserting members, so the first probe's answer stands in
        # for an edit-reshaped survivor set whose winner is the silent member.
        calls.append(1)
        if len(calls) == 1:
            return Materialised(
                sv_dcids=[_RURAL_SV],
                has_data=True,
                coverage=CoverageBare(has_data=True),
            )
        return real(shape, *args, **kwargs)

    monkeypatch.setattr(
        promote_materialise, "run_materialise", _first_probe_finds_only_the_silent_member
    )
    graph = _mixed_presence_graph()
    graph._obs[f"{_URBAN_SV}|{_IND_DCID}"] = _point_years(range(2001, 2006))
    graph._obs[f"{_RURAL_ADULT_SV}|{_IND_DCID}"] = _point_years(range(2001, 2006))
    base_spec_id, base_identity = make_base(
        PERSON_COUNT_SHAPE,
        slots=[_AGE_SLOT, _GENDER_SLOT, _RESIDENCE_SLOT],
        entities=[subject_entity(_IND_DCID)],
    )
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=base_identity.model_copy(
                update={"requested_window": IdentityWindowRange(start_year=2019, end_year=2021)}
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_URBAN_SV, _RURAL_SV, _RURAL_ADULT_SV],
        ),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
    )

    assert len(calls) == 2
    assert region.no_data_reason == "no_observations"
    assert region.gate is None
    assert region.window_gap_fact is not None
    assert (region.window_gap_fact.asked, region.window_gap_fact.data_years) == (
        "2019 to 2021",
        (2001, 2005),
    )


def test_asserting_members_answer_over_their_own_default_scope(monkeypatch):
    """No place is posted. The first probe's winner, Rural_Female, is silent on the bound age
    and holds data in India and Kenya; Urban_Female and Rural_Adult assert every bound
    property and hold data in India alone. The resubmit re-runs over those two, and the
    answer's reads are built over the default scope of the members it serves: India,
    without the country only the silent member has data in."""
    real = promote_materialise.run_materialise
    calls: list[int] = []

    def _first_probe_finds_only_the_silent_member(shape, *args, **kwargs):
        # Both resolvers serve only asserting members, so the first probe's answer stands in
        # for an edit-reshaped survivor set whose winner is the silent member.
        calls.append(1)
        if len(calls) == 1:
            return Materialised(
                sv_dcids=[_RURAL_SV],
                has_data=True,
                coverage=CoverageBare(has_data=True),
                default_scope=DefaultPlaceScope(places=(_IND_DCID, _KEN_DCID), every_place=False),
            )
        return real(shape, *args, **kwargs)

    monkeypatch.setattr(
        promote_materialise, "run_materialise", _first_probe_finds_only_the_silent_member
    )
    real_entities = promote_ground._observation_entities
    read_over: list[dict[str, list[str]]] = []

    def _recorded_entities(shape, **kwargs):
        entities, warning = real_entities(shape, **kwargs)
        read_over.append(entities)
        return entities, warning

    monkeypatch.setattr(promote_ground, "_observation_entities", _recorded_entities)
    graph = _mixed_presence_graph()
    graph._obs[f"{_RURAL_SV}|{_KEN_DCID}"] = graph._obs[f"{_RURAL_SV}|{_IND_DCID}"]
    graph._children = {"Earth": {"Country": [_IND_DCID, _KEN_DCID]}}
    base_spec_id, base_identity = make_base(
        PERSON_COUNT_SHAPE, slots=[_AGE_SLOT, _GENDER_SLOT, _RESIDENCE_SLOT], entities=[]
    )
    region = resolve_spec_resubmit(
        inp=ResubmitIdentityPayload(
            identity=base_identity,
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_URBAN_SV, _RURAL_SV, _RURAL_ADULT_SV],
        ),
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        coverage=empty_coverage_snapshot(),
    )

    assert len(calls) == 2
    assert region.status == "definite", region.no_data_reason
    (spec,) = region.specs
    assert {sv.ref for sv in spec.resolved.stat_vars.items} == {_URBAN_SV, _RURAL_ADULT_SV}
    assert read_over == [{"observationAbout": [_IND_DCID]}]
