"""Tests for nearest-real relaxation, confidence, and option-coverage enrichment.
The two synthetic members each match one of two bindings, so the original query
has no data and relaxing either binding can produce a suggestion.
"""

from __future__ import annotations

import dataclasses
import functools

import pytest

from code_index.shared import get_index
from materialise import place_scope
from materialise.graph import GraphReadError, SubjectPage
from materialise.place_scope import DefaultPlaceScope, DefaultPresence, scope_of_places
from search.engine.bind import SlotBindingDraft
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.errors import GraphInfraError
from search.engine.extract import DateRequest
from search.engine.place import ConsultedPresence
from search.engine.region import grounding as grounding_module
from search.engine.region.codes import ENTITY_LABEL_UNRESOLVED, OPTION_COVERAGE_PROBE_FAILED
from search.engine.region.grounding import _suggest_nearest_real
from search.engine.region.place_context import PlaceContext, PlaceMode, UnsettledPlace
from search.engine.region.place_roles import PlaceAssignment, where_binding
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.engine.retrieval import role_resolve as role_resolve_module
from search.engine.retrieval.bound_values import RoleClassWalks
from search.engine.retrieval.coverage import citable_source_ids
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.retrieval.read_plan import facets_served
from search.engine.retrieval.retrieve import Materialised
from search.engine.shape import ShapeDraft, SlotKeyDraft, is_role_bearing
from search.engine.spec_build import NamedGroup
from search.models import CoverageBare, EntityGroup, EntityRoleKey, TimeWindow
from series_coverage.schema import FacetMetadata, RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine import _answer_invariants_world as fixture_world
from tests.engine._builders import bound_place, subject_place_roles
from tests.engine._fake_coverage import (
    CoverageRow,
    FakeCoverageService,
    empty_coverage_snapshot,
    patch_coverage,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph
from tests.materialise._scope_availability import ScopeAvailability
from tests.materialise.conftest import FakeGraphReader

_RECIPIENT = "country/XYZ"

_POP_TYPE = "TestRelaxPopType"
_MEAS_PROP = "testRelaxMeasure"
_STAT_TYPE = "measuredValue"

_PROP_A = "TestPropA"
_PROP_B = "TestPropB"
_VAL_A1 = "ValA1"
_VAL_A_OTHER = "ValAOther"
_VAL_B1 = "ValB1"
_VAL_B_OTHER = "ValBOther"

_SV_RELAX_A = "TEST/RelaxA"
_SV_RELAX_B = "TEST/RelaxB"


def _sv_arcs(*, label: str, prop_a: str, prop_b: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": _POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _STAT_TYPE}]},
        "constraintProperties": {"nodes": [{"dcid": _PROP_A}, {"dcid": _PROP_B}]},
        _PROP_A: {"nodes": [{"dcid": prop_a}]},
        _PROP_B: {"nodes": [{"dcid": prop_b}]},
    }


_SV_ARC_FACTS = {
    _SV_RELAX_A: _sv_arcs(label="Relax A", prop_a=_VAL_A_OTHER, prop_b=_VAL_B1),
    _SV_RELAX_B: _sv_arcs(label="Relax B", prop_a=_VAL_A1, prop_b=_VAL_B_OTHER),
}

_NODES = {
    _RECIPIENT: {"label": "Testland", "type": "Country"},
    _SV_RELAX_A: {
        "label": "Relax A",
        "type": "StatisticalVariable",
        "arcs": _SV_ARC_FACTS[_SV_RELAX_A],
    },
    _SV_RELAX_B: {
        "label": "Relax B",
        "type": "StatisticalVariable",
        "arcs": _SV_ARC_FACTS[_SV_RELAX_B],
    },
    _VAL_A1: {"label": "Value A1"},
    _VAL_A_OTHER: {"label": "Value A Other"},
    _VAL_B1: {"label": "Value B1"},
    _VAL_B_OTHER: {"label": "Value B Other"},
}

_BINDINGS = [
    SlotBindingDraft(axis="how", property_dcid=_PROP_A, kind="value", value_dcids=[_VAL_A1]),
    SlotBindingDraft(axis="what", property_dcid=_PROP_B, kind="value", value_dcids=[_VAL_B1]),
]

_ROLES: dict[str, EntityRoleKey] = {_RECIPIENT: "observationAbout"}

_GROUP = NamedGroup(label="the test group", dcid=None, member_dcids=frozenset({_RECIPIENT}))


def _shape() -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_relax_shape",
        label="test relax shape",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="how", property_dcid=_PROP_A, label="Prop A"),
            SlotKeyDraft(axis="what", property_dcid=_PROP_B, label="Prop B"),
        ),
        sv_arc_facts=_SV_ARC_FACTS,
        slot_taxonomy={
            f"how:{_PROP_A}": [_VAL_A1, _VAL_A_OTHER],
            f"what:{_PROP_B}": [_VAL_B1, _VAL_B_OTHER],
        },
        representative_score=0.95,
    )


def _patch_coverage_snapshot(
    monkeypatch,
    rows: list[CoverageRow],
    *,
    role_names: dict,
    facet_metadata: dict | None = None,
) -> CoverageSnapshot:
    """Return a snapshot over the patched coverage index."""
    index = patch_coverage(monkeypatch, rows, role_names=role_names, facet_metadata=facet_metadata)
    return FakeCoverageService(index).snapshot()


def _obs_graph(*, a_has_data: bool, b_has_data: bool) -> FakeGraph:
    obs = {}
    if a_has_data:
        obs[f"{_SV_RELAX_A}|{_RECIPIENT}"] = [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
        ]
    if b_has_data:
        obs[f"{_SV_RELAX_B}|{_RECIPIENT}"] = [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
        ]
    return FakeGraph(nodes=_NODES, obs=obs, detect={}, resolve={})


# ---------------------------------------------------------------------------
# Relaxation capture + perturbation: relax a different binding, relaxation moves.
# ---------------------------------------------------------------------------


def test_relaxation_names_the_dropped_binding_when_a_relaxes():
    graph = _obs_graph(a_has_data=True, b_has_data=False)
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        named_groups=(_GROUP,),
        coverage=empty_coverage_snapshot(),
    )

    assert warnings == []
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.resolved.entity_groups == {
        "observationAbout": EntityGroup(ref=None, label="the test group")
    }
    assert suggestion.relaxation.axis == "how"
    assert suggestion.relaxation.property_dcid == _PROP_A
    assert suggestion.relaxation.dropped_value_dcids == [_VAL_A1]
    assert [sv.ref.ref for sv in suggestion.resolved.members] == [_SV_RELAX_A]


def test_relaxation_moves_to_the_other_binding_when_data_moves():
    """Moving the data to the other member makes relaxation move from propA
    to propB, with both original bindings unchanged.
    """
    graph = _obs_graph(a_has_data=False, b_has_data=True)
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert warnings == []
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.relaxation.axis == "what"
    assert suggestion.relaxation.property_dcid == _PROP_B
    assert suggestion.relaxation.dropped_value_dcids == [_VAL_B1]
    assert [sv.ref.ref for sv in suggestion.resolved.members] == [_SV_RELAX_B]


def test_untouched_binding_keeps_the_caller_supplied_provenance():
    """The suggestion drops propA but preserves the caller's provenance for
    untouched propB.
    """
    graph = _obs_graph(a_has_data=True, b_has_data=False)
    suggestions, _, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        text_recognized_properties=frozenset({_PROP_B}),
        coverage=empty_coverage_snapshot(),
    )
    assert len(suggestions) == 1
    filters = suggestions[0].resolved.trace.slot_filters
    prop_b_filter = next(
        f for f in filters if f.key.property is not None and f.key.property.ref == _PROP_B
    )
    assert prop_b_filter.provenance == "text_recognized"


def test_no_suggestions_when_neither_relaxation_finds_data():
    graph = _obs_graph(a_has_data=False, b_has_data=False)
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    assert suggestions == []
    assert warnings == []


_REGION = "TestRelaxRegion"
_OTHER_MEMBER = "country/XYA"
_REGION_SET = PlaceSet(
    name="Test region",
    kind="region",
    node=PlaceNode(dcid=_REGION, graph_type="Continent"),
    members=(_RECIPIENT, _OTHER_MEMBER),
    member_type="Country",
)


def _region_suggestions(held_at: str):
    """The suggestions for the two bindings asked at the test region, where relaxing
    property A finds data at `held_at` alone. The original answer held no data at the
    region's node or its members, so its context binds the members, and so does its
    where slot."""
    graph = FakeGraph(
        nodes=_NODES,
        obs={
            f"{_SV_RELAX_A}|{held_at}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ]
        },
        detect={},
        resolve={},
    )
    shape = dataclasses.replace(
        _shape(),
        slot_keys=(
            *_shape().slot_keys,
            SlotKeyDraft(axis="where", property_dcid=None, label="place"),
        ),
    )
    where = where_binding(list(_REGION_SET.members))
    assert where is not None
    unsettled = UnsettledPlace(
        shape,
        PlaceAssignment(sets={"observationAbout": (_REGION_SET,)}),
        get_index(),
        admitted=frozenset({("observationAbout", "Continent"), ("observationAbout", "Country")}),
    )
    suggestions, _warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=shape,
        bindings=[*_BINDINGS, where],
        place=unsettled.context({"observationAbout": _REGION_SET.members}),
        unsettled=unsettled,
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    return suggestions


@pytest.mark.parametrize("held_at", [_REGION, _OTHER_MEMBER])
def test_a_relaxation_settles_its_sets_on_its_own_read(held_at):
    """The relaxed series holds data at the region's node alone, or at one member alone.
    Its suggestion binds the node in the first case and the members in the second,
    whatever the original answer bound, and its where slot holds the places it binds."""
    (suggestion,) = _region_suggestions(held_at)

    places = [_REGION] if held_at == _REGION else sorted(_REGION_SET.members)
    assert suggestion.definition.identity.entities == {"observationAbout": places}
    (where,) = [slot.binding for slot in suggestion.resolved.slots if slot.key.axis == "where"]
    values = [where.value] if where.kind == "value" else where.values
    assert sorted(value.ref.ref for value in values) == places


def test_a_relaxation_over_the_place_bound_is_not_suggested(search_settings):
    """The relaxed series holds data at a member and none at the node, so the relaxation
    binds the region's two members. Under a bound of one place it is not suggested."""
    search_settings(max_child_places=1)

    assert _region_suggestions(_OTHER_MEMBER) == []


# The same `roles` dict feeds each binding iteration, so a missing label can duplicate warnings.


def test_label_miss_warning_is_deduped_across_two_surviving_suggestions():
    """Both siblings have data under the default `n_max=2`; the missing
    recipient label produces one warning for two suggestions.
    """
    label_miss_nodes = {**_NODES, _RECIPIENT: {"type": "Country"}}  # no ->name arc
    obs = {
        f"{_SV_RELAX_A}|{_RECIPIENT}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
        ],
        f"{_SV_RELAX_B}|{_RECIPIENT}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
        ],
    }
    graph = FakeGraph(nodes=label_miss_nodes, obs=obs, detect={}, resolve={})
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert len(suggestions) == 2
    entity_warnings = [w for w in warnings if w.code == ENTITY_LABEL_UNRESOLVED]
    assert len(entity_warnings) == 1
    assert _RECIPIENT in entity_warnings[0].message


# ---------------------------------------------------------------------------
# Shaping: confidence is computed for nearest_real suggestions.
# ---------------------------------------------------------------------------


def test_suggestion_carries_confidence():
    graph = _obs_graph(a_has_data=True, b_has_data=False)
    suggestions, _, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    assert len(suggestions) == 1
    resolved = suggestions[0].resolved
    assert resolved.available_at_place is True
    assert resolved.confidence in ("high", "medium")


# ---------------------------------------------------------------------------
# Option-coverage enrichment: opt-in, fail-open.
# ---------------------------------------------------------------------------


def _raise_option_coverage_probe_failure(**_kwargs):
    raise RuntimeError("simulated option_coverage failure")


def test_option_coverage_probe_failure_is_fail_open(monkeypatch):
    """A coverage-probe error leaves the suggestion and reports a warning."""
    monkeypatch.setattr(
        grounding_module, "standard_option_coverage", _raise_option_coverage_probe_failure
    )
    graph = _obs_graph(a_has_data=True, b_has_data=False)
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        include_option_coverage=True,
        coverage=empty_coverage_snapshot(),
    )
    assert len(suggestions) == 1
    assert any(w.code == OPTION_COVERAGE_PROBE_FAILED for w in warnings)


def test_real_materialise_transport_failure_is_fail_open():
    """A real materialise graph transport error leaves the suggestion and
    returns `OPTION_COVERAGE_PROBE_FAILED`.
    """

    class _RaisingGraphReader(FakeGraphReader):
        def observation_facets_batch(self, *, stat_var, entities, deadline: float | None = None):
            raise GraphReadError("simulated transport error")

        def observation_facets_batch_many(
            self, *, stat_vars, entities, needs_dates: bool = False, deadline: float | None = None
        ):
            raise GraphReadError("simulated transport error")

    graph = _obs_graph(a_has_data=True, b_has_data=False)
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        include_option_coverage=True,
        materialise_graph=_RaisingGraphReader(),
        coverage=empty_coverage_snapshot(),
    )
    assert len(suggestions) == 1
    assert any(w.code == OPTION_COVERAGE_PROBE_FAILED for w in warnings)


def test_option_coverage_off_by_default_is_a_no_op(monkeypatch):
    """The default skips `standard_option_coverage`."""
    monkeypatch.setattr(
        grounding_module, "standard_option_coverage", _raise_option_coverage_probe_failure
    )
    graph = _obs_graph(a_has_data=True, b_has_data=False)
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    assert len(suggestions) == 1
    assert warnings == []


# Relaxing propA on a subject-only shape suggests what the relaxed query's own first answer
# serves (`standard.standard_resolve`): Relax A, whether or not its propC value resolves. A
# value that does not resolve is reported in `unresolved_slots`.

_PROP_C = "TestPropC"
_VAL_C = "TestValC"


def _sv_arcs_with_c(*, label: str, prop_a: str, prop_b: str, c_node: dict) -> dict:
    arcs = _sv_arcs(label=label, prop_a=prop_a, prop_b=prop_b)
    arcs["constraintProperties"]["nodes"].append({"dcid": _PROP_C})
    arcs[_PROP_C] = {"nodes": [c_node]}
    return arcs


def _shape_with_c(c_node: dict) -> tuple[ShapeDraft, dict]:
    """The relaxation shape with Relax A asserting propC as `c_node`, and its graph nodes.
    An empty `c_node` is a value node with no dcid or value, which does not resolve."""
    sv_arc_facts = {
        _SV_RELAX_A: _sv_arcs_with_c(
            label="Relax A", prop_a=_VAL_A_OTHER, prop_b=_VAL_B1, c_node=c_node
        ),
        _SV_RELAX_B: _sv_arcs(label="Relax B", prop_a=_VAL_A1, prop_b=_VAL_B_OTHER),
    }
    nodes = {
        **_NODES,
        **{
            sv: {"label": label, "type": "StatisticalVariable", "arcs": sv_arc_facts[sv]}
            for sv, label in ((_SV_RELAX_A, "Relax A"), (_SV_RELAX_B, "Relax B"))
        },
    }
    shape = ShapeDraft(
        shape_id="test_relax_shape_with_c",
        label="test relax shape with c",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="how", property_dcid=_PROP_A, label="Prop A"),
            SlotKeyDraft(axis="what", property_dcid=_PROP_B, label="Prop B"),
            SlotKeyDraft(axis="what", property_dcid=_PROP_C, label="Prop C"),
        ),
        sv_arc_facts=sv_arc_facts,
        slot_taxonomy={
            f"how:{_PROP_A}": [_VAL_A1, _VAL_A_OTHER],
            f"what:{_PROP_B}": [_VAL_B1, _VAL_B_OTHER],
        },
        representative_score=0.95,
    )
    return shape, nodes


@pytest.mark.parametrize(
    ("c_node", "unresolved"),
    [
        pytest.param({"dcid": _VAL_C}, [], id="propC-resolves"),
        pytest.param({}, [_PROP_C], id="propC-does-not-resolve"),
    ],
)
def test_relaxation_suggests_what_the_relaxed_first_answer_serves(
    c_node: dict, unresolved: list[str]
):
    shape, nodes = _shape_with_c(c_node)
    obs = {
        f"{_SV_RELAX_A}|{_RECIPIENT}": [
            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
        ]
    }
    graph = FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={})
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=shape,
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )
    assert warnings == []
    assert not is_role_bearing(shape)
    assert [[sv.ref.ref for sv in suggestion.resolved.members] for suggestion in suggestions] == [
        [_SV_RELAX_A]
    ]
    assert suggestions[0].resolved.unresolved_slots == unresolved


# ---------------------------------------------------------------------------
# Role-bearing option-coverage probe: the probe entities are built through
# entities_by_role's by_axis form, keyed by the declared destination/origin
# role names, mirroring the same construction graph_confirm_resolve's own
# role-bearing branch uses for its primary read.
# ---------------------------------------------------------------------------

_DONOR = "country/FRA"
_DONOR_2 = "country/SWE"

_CREDITOR_DEBTOR_ROLE_NAMES = RoleNames(donor="creditor", recipient="debtor")


def _shape_role_bearing(declared_roles: tuple[str, ...]) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_relax_shape_role_bearing",
        label="test relax shape role bearing",
        pop_type_dcid=_POP_TYPE,
        meas_prop_dcid=_MEAS_PROP,
        stat_type_dcid=_STAT_TYPE,
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(
            SlotKeyDraft(axis="how", property_dcid=_PROP_A, label="Prop A"),
            SlotKeyDraft(axis="what", property_dcid=_PROP_B, label="Prop B"),
        ),
        sv_arc_facts={
            sv: {**arcs, "observationProperties": {"nodes": [{"dcid": r} for r in declared_roles]}}
            for sv, arcs in _SV_ARC_FACTS.items()
        },
        slot_taxonomy={
            f"how:{_PROP_A}": [_VAL_A1, _VAL_A_OTHER],
            f"what:{_PROP_B}": [_VAL_B1, _VAL_B_OTHER],
        },
        representative_score=0.95,
        declared_roles=declared_roles,
    )


def test_role_bearing_probe_entities_are_keyed_by_the_declared_names(monkeypatch):
    """The option-coverage probe receives entities keyed by `debtor` and
    `creditor`, the shape's declared role names.
    """
    captured: list[dict[str, list[str]]] = []

    def _record_entities(*, entities, slot_options, **_kwargs):
        captured.append(dict(entities))
        return slot_options

    monkeypatch.setattr(grounding_module, "standard_option_coverage", _record_entities)
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_RELAX_A, "f1", _DONOR, _RECIPIENT, [2020])],
        role_names={_SV_RELAX_A: _CREDITOR_DEBTOR_ROLE_NAMES},
    )

    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=FakeGraph(nodes=_NODES, obs={}, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}),
        shape_draft=_shape_role_bearing(("creditor", "debtor")),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"debtor": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
            origin_dcids=[_DONOR],
        ),
        presence=None,
        date_request=None,
        include_option_coverage=True,
        coverage=coverage,
    )

    assert len(suggestions) == 1
    assert warnings == []
    assert captured == [{"debtor": [_RECIPIENT], "creditor": [_DONOR]}]


def test_role_bearing_nearest_real_uses_every_provider_and_second_provider_data(monkeypatch):
    """Only the second provider has a coverage row; both providers must reach
    confirmation and option-coverage calls.
    """
    captured: list[dict[str, list[str]]] = []

    def _record_entities(*, entities, slot_options, **_kwargs):
        captured.append(dict(entities))
        return slot_options

    monkeypatch.setattr(grounding_module, "standard_option_coverage", _record_entities)
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_RELAX_A, "f1", _DONOR_2, _RECIPIENT, [2020])],
        role_names={_SV_RELAX_A: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=FakeGraph(
            nodes={
                **_NODES,
                _DONOR: {"label": "France", "type": "Country"},
                _DONOR_2: {"label": "Sweden", "type": "Country"},
            },
            obs={},
            detect={},
            resolve={},
        ),
        sdmx=FakeSdmxClient(observations={}),
        shape_draft=_shape_role_bearing(("creditor", "debtor")),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"debtor": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
            origin_dcids=[_DONOR, _DONOR_2],
        ),
        presence=None,
        date_request=None,
        coverage=coverage,
        include_option_coverage=True,
    )

    assert warnings == []
    assert len(suggestions) == 1
    assert [sv.ref.ref for sv in suggestions[0].resolved.members] == [_SV_RELAX_A]
    assert captured == [{"debtor": [_RECIPIENT], "creditor": [_DONOR, _DONOR_2]}]


def test_role_bearing_probe_degrades_on_an_entity_this_shape_has_no_role_for(monkeypatch):
    """A destination-only shape cannot place the bound donor. The test stubs
    primary resolution to reach the probe's `RoleEntityUnplaced` handling.
    """
    shape = _shape_role_bearing(("debtor",))

    def _stub_graph_confirm_resolve(**_kwargs):
        return Materialised(
            sv_dcids=[_SV_RELAX_A],
            has_data=True,
            coverage=CoverageBare(has_data=True),
            member_facts={_SV_RELAX_A: ArcsMemberFacts(arcs=_SV_ARC_FACTS[_SV_RELAX_A])},
        )

    monkeypatch.setattr(role_resolve_module, "graph_confirm_resolve", _stub_graph_confirm_resolve)

    def _fail_if_called(**_kwargs):
        raise AssertionError("standard_option_coverage must not be reached")

    monkeypatch.setattr(grounding_module, "standard_option_coverage", _fail_if_called)

    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=FakeGraph(nodes=_NODES, obs={}, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}),
        shape_draft=shape,
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"debtor": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
            origin_dcids=[_DONOR],
        ),
        presence=None,
        date_request=None,
        include_option_coverage=True,
        n_max=1,
        coverage=empty_coverage_snapshot(),
    )

    assert len(suggestions) == 1
    assert not any(w.code == OPTION_COVERAGE_PROBE_FAILED for w in warnings)


# ---------------------------------------------------------------------------
# Plural destinations: the primary probe (graph_confirm_resolve, called
# outside this module) confirms against every named destination. This
# relaxation probe's own graph_confirm_resolve call must do the same, not
# fall back to the scalar destination_dcid (the first named destination).
# ---------------------------------------------------------------------------

_RECIPIENT_2 = "country/ABC"


def test_relaxation_probe_uses_every_named_destination_not_just_the_first(monkeypatch):
    """The second of two named destinations has data, so both destinations
    must reach the relaxation probe.
    """
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [(_SV_RELAX_A, "f1", _DONOR, _RECIPIENT_2, [2020])],
        role_names={_SV_RELAX_A: _CREDITOR_DEBTOR_ROLE_NAMES},
    )
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=FakeGraph(nodes=_NODES, obs={}, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}),
        shape_draft=_shape_role_bearing(("creditor", "debtor")),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"debtor": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
            origin_dcids=[_DONOR],
            probe_dcids=[_RECIPIENT, _RECIPIENT_2],
        ),
        presence=None,
        date_request=None,
        coverage=coverage,
    )

    assert warnings == []
    assert len(suggestions) == 1
    assert [sv.ref.ref for sv in suggestions[0].resolved.members] == [_SV_RELAX_A]


def test_a_role_suggestion_cites_the_sources_its_plan_serves(monkeypatch):
    """The relaxed member holds IDS for 2010 to 2020 and a shorter Alternative series the
    2015 to 2020 plan does not elect: the suggestion's `resolved_sources` and source count
    state IDS alone, the source its plan serves."""
    ids, alt = "dc/base/Ids", "dc/base/Alt"
    coverage = _patch_coverage_snapshot(
        monkeypatch,
        [
            (_SV_RELAX_A, "f1", _DONOR, _RECIPIENT, list(range(2010, 2021))),
            (_SV_RELAX_A, "f9", _DONOR, _RECIPIENT, list(range(2016, 2019))),
        ],
        role_names={_SV_RELAX_A: _CREDITOR_DEBTOR_ROLE_NAMES},
        facet_metadata={
            "f1": FacetMetadata(provenance_id=ids),
            "f9": FacetMetadata(provenance_id=alt),
        },
    )
    suggestions, warnings, reads, _presence = _suggest_nearest_real(
        graph=FakeGraph(
            nodes={**_NODES, ids: {"label": "IDS"}, alt: {"label": "Alternative"}},
            obs={},
            detect={},
            resolve={},
        ),
        sdmx=FakeSdmxClient(observations={}),
        shape_draft=_shape_role_bearing(("creditor", "debtor")),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"debtor": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
            origin_dcids=[_DONOR],
        ),
        presence=None,
        date_request=DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False),
        coverage=coverage,
    )

    assert warnings == []
    (suggestion,) = suggestions
    (read,) = reads.values()
    assert citable_source_ids(facets_served(read.plan)) == [ids]
    assert [ref.ref for ref in suggestion.resolved.trace.resolved_sources] == [ids]
    dims = {d.label: d.count for d in suggestion.resolved.coverage.dimensions or []}
    assert dims["sources"] == 1


def test_a_relaxation_with_no_point_in_the_window_is_not_suggested():
    """The relaxed member holds 2010 to 2019 alone, so "since 2021" (2021 to the 2026
    clock) reads nothing: its plan serves no point and the relaxation offers no
    suggestion, rather than one whose coverage counts zero observations."""
    years = [str(y) for y in range(2010, 2020)]
    graph = FakeGraph(
        nodes=_NODES,
        obs={
            f"{_SV_RELAX_A}|{_RECIPIENT}": [
                {
                    "facetId": "fW",
                    "provenanceId": "dc/base/WorldDevelopmentIndicators",
                    "unit": "USD",
                    "earliestDate": "2010",
                    "latestDate": "2019",
                    "obsCount": 10,
                    "dates": years,
                    "values": [1.0] * 10,
                }
            ]
        },
        detect={},
        resolve={},
    )
    since = DateRequest(window=TimeWindow(start_year=2021, end_year=None), latest=False)
    suggestions, _warnings, _reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=since,
        coverage=empty_coverage_snapshot(),
    )
    assert suggestions == []


# ---------------------------------------------------------------------------
# Each relaxation answers with the family the mint's walk settles for its own query.
# ---------------------------------------------------------------------------

_SV_BOTH = "TEST/RelaxBoth"


def _sv_walked() -> dict:
    return _sv_arcs(label="Relax both", prop_a=_VAL_A1, prop_b=_VAL_B1)


def _walked_shape() -> ShapeDraft:
    """The answer's shape once the walk settled the original query: its one member asserts
    both bound values and holds no data."""
    return dataclasses.replace(
        _shape(),
        sv_arc_facts={_SV_BOTH: _sv_walked()},
    )


def _walk_graph(
    graph_class: type[fixture_world._MaterialiseGraph] = fixture_world._MaterialiseGraph,
) -> fixture_world._MaterialiseGraph:
    """The member graph the walk reads, through `graph_class`: the three members of the
    measured property."""
    members = {
        _SV_RELAX_A: (_VAL_A_OTHER, _VAL_B1),
        _SV_RELAX_B: (_VAL_A1, _VAL_B_OTHER),
        _SV_BOTH: (_VAL_A1, _VAL_B1),
    }
    return graph_class(
        arcs={
            sv: {
                "typeOf": ("StatisticalVariable",),
                "populationType": (_POP_TYPE,),
                "measuredProperty": (_MEAS_PROP,),
                "statType": (_STAT_TYPE,),
                "constraintProperties": (_PROP_A, _PROP_B),
                _PROP_A: (a,),
                _PROP_B: (b,),
            }
            for sv, (a, b) in members.items()
        },
        subjects={
            (_MEAS_PROP, "measuredProperty"): SubjectPage(dcids=tuple(members), truncated=False)
        },
    )


def _walked_nodes() -> dict:
    """The graph nodes of the three members the walk reads."""
    return {
        **_NODES,
        _SV_BOTH: {"label": "Relax both", "type": "StatisticalVariable", "arcs": _sv_walked()},
    }


def _suggest_over_walked_shape(walks: RoleClassWalks | None):
    nodes = _walked_nodes()
    obs = {
        f"{sv}|{_RECIPIENT}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}]
        for sv in (_SV_RELAX_A, _SV_RELAX_B)
    }
    return _suggest_nearest_real(
        graph=FakeGraph(nodes=nodes, obs=obs, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        shape_draft=_walked_shape(),
        bindings=_BINDINGS,
        place=bound_place(
            roles=_ROLES,
            entities={"observationAbout": [_RECIPIENT]},
            destination_dcid=_RECIPIENT,
        ),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
        role_class_walks=walks,
    )


def test_each_relaxation_answers_with_the_family_its_own_walk_settles(monkeypatch):
    """The answer's shape holds only the member its own bindings walked, so relaxing a
    binding over that shape finds no other member. Each relaxed query walks its own family,
    as its first answer does, and suggests the member with data. Both walks start at the same
    request depth, so the no-data path waits for one walk, not one per relaxation."""
    assert _suggest_over_walked_shape(None)[0] == []

    fixture_world.carry_logical_clock(monkeypatch)
    with fixture_world._recorded() as recorder:
        suggestions, _warnings, _reads, _presence = _suggest_over_walked_shape(
            RoleClassWalks(_walk_graph())
        )

    assert sorted(sv.ref.ref for s in suggestions for sv in s.resolved.members) == [
        _SV_RELAX_A,
        _SV_RELAX_B,
    ]
    walk_starts = [
        tick
        for client, method, tick in recorder.requests
        if (client, method) == ("materialise_graph", "node_subjects")
    ]
    assert len(walk_starts) == 2
    assert walk_starts[0] == walk_starts[1]


def _known_presence(
    countries: tuple[str, ...], held: dict[str, frozenset[str]]
) -> ConsultedPresence:
    """A presence whose StatVars each hold their `held` places as their every-place scope."""
    return ConsultedPresence(
        DefaultPresence(
            countries=countries,
            scopes={sv: scope_of_places(places, every_place=True) for sv, places in held.items()},
        )
    )


def _mode(scope: DefaultPlaceScope) -> PlaceMode:
    """The mode of a place context probed over `scope`."""
    return "every_place" if scope.every_place else "countries_fallback"


class _PresenceLogGraph(FakeGraph):
    """Records the StatVars of each presence read, each default-scope availability read
    (`availability_reads`) and each listing of Earth's countries."""

    @functools.cached_property
    def scope_availability(self) -> ScopeAvailability:
        return super().scope_availability

    @property
    def availability_reads(self) -> list[str]:
        return sorted(sv for read in self.scope_availability.reads for sv in read)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.presence_reads: list[list[str]] = []
        self.listings = 0

    def entities_with_data_by_variable(self, variable_dcids, entity_dcids, *, deadline=None):
        self.presence_reads.append(list(variable_dcids))
        return super().entities_with_data_by_variable(
            variable_dcids, entity_dcids, deadline=deadline
        )

    def child_places_batch(self, parent_dcids, child_type, *, deadline=None):
        self.listings += 1
        return super().child_places_batch(parent_dcids, child_type, deadline=deadline)


def _unsettling(monkeypatch, stat_var: str) -> None:
    """Make every default-scope read leave `stat_var` a countries fallback."""
    read = place_scope.read_default_presence

    def unsettled_read(**kwargs):
        return dataclasses.replace(read(**kwargs), unsettled=frozenset({stat_var}))

    monkeypatch.setattr(place_scope, "read_default_presence", unsettled_read)


@pytest.mark.parametrize("unsettled", [False, True], ids=["settled", "countries-fallback"])
def test_a_placeless_relaxation_reads_the_presence_of_the_members_its_walk_adds(
    monkeypatch, unsettled: bool
):
    """No place is named, and the request's presence read covered the answer's one walked
    member. Each relaxed query's walk adds a member with data, so one presence read tops up
    the request's presence with those members' availability, and each relaxation is
    suggested over its own member's scope."""
    nodes = {
        **_NODES,
        _SV_BOTH: {"label": "Relax both", "type": "StatisticalVariable", "arcs": _sv_walked()},
    }
    obs = {
        f"{sv}|{_RECIPIENT}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}]
        for sv in (_SV_RELAX_A, _SV_RELAX_B)
    }
    graph = _PresenceLogGraph(
        nodes=nodes, obs=obs, detect={}, resolve={}, children={"Earth": {"Country": [_RECIPIENT]}}
    )
    presence = _known_presence((_RECIPIENT,), {_SV_BOTH: frozenset()})
    scope = presence.consulted_union([_SV_BOTH])
    if unsettled:
        _unsettling(monkeypatch, _SV_RELAX_A)

    with recording_degraded_reads() as lost:
        suggestions, _warnings, reads, _presence = _suggest_nearest_real(
            graph=graph,
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            shape_draft=_walked_shape(),
            bindings=_BINDINGS,
            place=PlaceContext(
                roles=subject_place_roles(),
                probe_dcids=scope.places,
                mode=_mode(scope),
                scope_warning=None,
            ),
            presence=presence,
            date_request=None,
            coverage=empty_coverage_snapshot(),
            role_class_walks=RoleClassWalks(_walk_graph()),
        )

    assert sorted(sv.ref.ref for s in suggestions for sv in s.resolved.members) == [
        _SV_RELAX_A,
        _SV_RELAX_B,
    ]
    assert lost.degraded is unsettled
    assert graph.availability_reads == [_SV_RELAX_A, _SV_RELAX_B]
    assert (graph.presence_reads, graph.listings) == ([], 0)
    assert {read.place.probe_dcids for read in reads.values()} == {(_RECIPIENT,)}


def _recipient_place():
    return bound_place(
        roles=_ROLES, entities={"observationAbout": [_RECIPIENT]}, destination_dcid=_RECIPIENT
    )


@pytest.mark.parametrize(
    ("graph_class", "unverified"),
    [
        pytest.param(fixture_world._MaterialiseGraph, False, id="the_walk_settles"),
        pytest.param(fixture_world._UnwalkableGraph, True, id="the_walk_settles_nothing"),
    ],
)
def test_a_suggestion_whose_walk_settles_nothing_warns_membership_unverified(
    graph_class, unverified: bool
):
    """A relaxed query whose own member walk settles nothing answers with its recalled
    members and warns `membership_unverified`, so its suggestion does too."""
    obs = {
        f"{sv}|{_RECIPIENT}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}]
        for sv in (_SV_RELAX_A, _SV_RELAX_B)
    }
    suggestions, warnings, _reads, _presence = _suggest_nearest_real(
        graph=FakeGraph(nodes=_walked_nodes(), obs=obs, detect={}, resolve={}),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=_recipient_place(),
        presence=None,
        date_request=None,
        coverage=empty_coverage_snapshot(),
        role_class_walks=RoleClassWalks(_walk_graph(graph_class)),
    )

    assert len(suggestions) == 2
    assert [w.code for w in warnings] == (["membership_unverified"] if unverified else [])


def test_every_relaxation_is_walked_in_one_round_and_the_first_suggestions_are_kept(
    monkeypatch,
):
    """With one suggestion wanted, both relaxations are walked at once, and the first in
    query order that suggests is kept: Relax A when relaxing propA suggests it, else Relax B."""
    walked: list[str] = []
    walk = grounding_module.elected_role_class_shape

    def counted(shape, query, **kwargs):
        walked.append(
            next(prop for prop, binding in query.slots.items() if binding == ("unbound",))
        )
        return walk(shape, query, **kwargs)

    monkeypatch.setattr(grounding_module, "elected_role_class_shape", counted)
    fixture_world.carry_logical_clock(monkeypatch)

    def suggest(with_data: tuple[str, ...]):
        with fixture_world._recorded() as recorder:
            found = _suggest_nearest_real(
                graph=FakeGraph(
                    nodes=_walked_nodes(),
                    obs={
                        f"{sv}|{_RECIPIENT}": [
                            {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
                        ]
                        for sv in with_data
                    },
                    detect={},
                    resolve={},
                ),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                shape_draft=_shape(),
                bindings=_BINDINGS,
                place=_recipient_place(),
                presence=None,
                date_request=None,
                coverage=empty_coverage_snapshot(),
                role_class_walks=RoleClassWalks(_walk_graph()),
                n_max=1,
            )[0]
        walk_starts = [
            tick
            for client, method, tick in recorder.requests
            if (client, method) == ("materialise_graph", "node_subjects")
        ]
        return found, walk_starts

    for with_data, kept in (
        ((_SV_RELAX_A, _SV_RELAX_B), _SV_RELAX_A),
        ((_SV_RELAX_B,), _SV_RELAX_B),
    ):
        walked.clear()
        found, walk_starts = suggest(with_data)
        assert [[sv.ref.ref for sv in s.resolved.members] for s in found] == [[kept]]
        assert sorted(walked) == [_PROP_A, _PROP_B]
        assert len(walk_starts) == 2
        assert walk_starts[0] == walk_starts[1]


def test_a_place_less_suggestion_reads_its_own_members_default_scope():
    """The query names no place, so the answer reads the default place scope of the members
    its mint serves. Relax A holds data only in its own scope, which the original answer's
    scope does not reach; its suggestion reads that scope, as the relaxed query's first
    answer does, from the request's presence read."""
    a_land, b_land, original = "country/AAA", "country/BBB", "country/ORG"
    graph = FakeGraph(
        nodes={
            **_NODES,
            **{p: {"label": p, "type": "Country"} for p in (a_land, b_land, original)},
        },
        obs={
            f"{_SV_RELAX_A}|{a_land}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ]
        },
        detect={},
        resolve={},
    )
    presence = _known_presence(
        (a_land, b_land, original),
        {_SV_RELAX_A: frozenset({a_land}), _SV_RELAX_B: frozenset({b_land})},
    )
    # The answer's bindings hold both values no member holds together, so it serves no
    # member and its own scope is empty.
    scope = presence.consulted_union([])

    def suggest(place: PlaceContext, presence: DefaultPresence | None):
        return _suggest_nearest_real(
            graph=graph,
            sdmx=FakeSdmxClient(
                observations={}, availability={}, scope_availability=graph.scope_availability
            ),
            shape_draft=_shape(),
            bindings=_BINDINGS,
            place=place,
            presence=presence,
            date_request=None,
            coverage=empty_coverage_snapshot(),
        )

    bound_to_original = bound_place(
        roles={}, entities={}, destination_dcid=None, probe_dcids=[original]
    )
    assert suggest(bound_to_original, None)[0] == []
    default_scoped = PlaceContext(
        roles=subject_place_roles(), probe_dcids=scope.places, mode=_mode(scope), scope_warning=None
    )
    suggestions, _warnings, reads, _presence = suggest(default_scoped, presence)
    (suggestion,) = suggestions
    assert [sv.ref.ref for sv in suggestion.resolved.members] == [_SV_RELAX_A]
    plan = reads[suggestion.spec_id].plan
    assert {entity for series in plan.series for _role, entity in series.role_tuple} == {a_land}


def test_the_relaxed_queries_read_no_scope_their_requests_presence_holds():
    """Both relaxed queries name no place, and the request's presence read covers both
    members. No presence read or listing is made for the relaxations, and each suggestion
    reads its own member's scope."""
    a_land, b_land = "country/AAA", "country/BBB"
    graph = _PresenceLogGraph(
        nodes={**_NODES, **{p: {"label": p, "type": "Country"} for p in (a_land, b_land)}},
        obs={
            f"{_SV_RELAX_A}|{a_land}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ],
            f"{_SV_RELAX_B}|{b_land}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}
            ],
        },
        detect={},
        resolve={},
    )
    presence = _known_presence(
        (a_land, b_land), {_SV_RELAX_A: frozenset({a_land}), _SV_RELAX_B: frozenset({b_land})}
    )
    scope = presence.consulted_union([])

    suggestions, _warnings, reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_shape(),
        bindings=_BINDINGS,
        place=PlaceContext(
            roles=subject_place_roles(),
            probe_dcids=scope.places,
            mode=_mode(scope),
            scope_warning=None,
        ),
        presence=presence,
        date_request=None,
        coverage=empty_coverage_snapshot(),
    )

    assert graph.presence_reads == []
    assert graph.listings == 0
    places = {
        suggestion.resolved.members[0].ref.ref: {
            entity
            for series in reads[suggestion.spec_id].plan.series
            for _role, entity in series.role_tuple
        }
        for suggestion in suggestions
    }
    assert places == {_SV_RELAX_A: {a_land}, _SV_RELAX_B: {b_land}}


class _PresenceDownGraph(_PresenceLogGraph):
    """Fails the default scope's check of the places availability lists, as a graph failure
    does, and counts those checks (`place_checks`)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.place_checks = 0

    def node_arcs_many(self, dcids, *, properties, deadline=None):
        if _RECIPIENT in dcids:
            self.place_checks += 1
            raise GraphInfraError("the place check failed")
        return super().node_arcs_many(dcids, properties=properties, deadline=deadline)


def test_a_failed_presence_read_for_the_members_a_walk_adds_gives_no_suggestion():
    """As in `test_a_placeless_relaxation_reads_the_presence_of_the_members_its_walk_adds`,
    each walk adds a member the request's presence read did not cover, and the check of the
    places their availability lists fails. The suggestions are optional, so the failure gives none and
    raises nothing: the no-data answer stands as it is."""
    obs = {
        f"{sv}|{_RECIPIENT}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 3}]
        for sv in (_SV_RELAX_A, _SV_RELAX_B)
    }
    graph = _PresenceDownGraph(
        nodes=_walked_nodes(),
        obs=obs,
        detect={},
        resolve={},
        children={"Earth": {"Country": [_RECIPIENT]}},
    )
    presence = _known_presence((_RECIPIENT,), {_SV_BOTH: frozenset()})
    scope = presence.consulted_union([_SV_BOTH])

    suggestions, warnings, reads, _presence = _suggest_nearest_real(
        graph=graph,
        sdmx=FakeSdmxClient(
            observations={}, availability={}, scope_availability=graph.scope_availability
        ),
        shape_draft=_walked_shape(),
        bindings=_BINDINGS,
        place=PlaceContext(
            roles=subject_place_roles(),
            probe_dcids=scope.places,
            mode=_mode(scope),
            scope_warning=None,
        ),
        presence=presence,
        date_request=None,
        coverage=empty_coverage_snapshot(),
        role_class_walks=RoleClassWalks(_walk_graph()),
    )

    assert graph.place_checks == 1
    assert (suggestions, warnings, reads) == ([], [], {})
