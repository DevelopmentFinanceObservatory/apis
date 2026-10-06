"""Pins in-set narrowing, set reconstruction, mixed edits, and resubmit shape labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from origin_catalogue import BASE
from search.engine.region import related_read, resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.related import NO_RELATED
from search.engine.shape import STANDARD_SHAPE_LABEL
from search.engine.spec_id import build_spec_identity
from search.models import (
    BindingValue,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    subject_entity,
    where_slots_for,
)
from tests.engine._builders import (
    binding_set_slot as _binding_set_slot,
)
from tests.engine._builders import (
    prop_node as _prop_node,
)
from tests.engine._builders import (
    unbound_slot as _unbound_slot,
)
from tests.engine._builders import (
    value_slot as _value_slot,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _when_slot_value(start: int, end: int) -> Slot:
    return Slot(
        key=SlotKey(axis="when", property=None, label="time window"),
        binding=BindingValue(
            value=SlotValue(
                ref=None,
                value_kind="time_window",
                time_window=TimeWindow(start_year=start, end_year=end),
            )
        ),
    )


_PERSON_FEMALE_SV = "Count_Person_Female"
_PERSON_MALE_SV = "Count_Person_Male"
_PERSON_NONBINARY_SV = "Count_Person_NonBinary"
_PERSON_URBAN_FEMALE_SV = "Count_Person_Urban_Female"
_PERSON_RURAL_FEMALE_SV = "Count_Person_Rural_Female"
_ENTITY_DCID = "country/IND"

# The posted where-slot supplies the entity used by resubmit promotion.
_WHERE_SLOTS = where_slots_for([subject_entity(_ENTITY_DCID)])


def _edited_identity(slots: list[Slot]):
    return build_spec_identity(
        PERSON_COUNT_SHAPE,
        slots,
        entities=[subject_entity(_ENTITY_DCID)],
        open_roles=(),
    )[1]


class _CountingGraph(FakeGraph):
    """Record arc-batch calls to distinguish pure narrowing from sibling confirmation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_arcs_batch_calls: list[list[str]] = []

    def node_arcs_batch(
        self, dcids: list[str], *, deadline: float | None = None
    ) -> dict[str, dict | None]:
        self.node_arcs_batch_calls.append(list(dcids))
        return super().node_arcs_batch(dcids, deadline=deadline)


def _synthetic_gender_sv(gender_value: str, label: str) -> dict:
    """Build a synthetic gender-constrained StatVar with the fixture's arc shape."""
    return {
        "label": label,
        "type": "StatisticalVariable",
        "arcs": {
            "statType": {"nodes": [_prop_node("measuredValue", ["Property"])]},
            "typeOf": {"nodes": [_prop_node("StatisticalVariable", ["Class"])]},
            "description": {
                "nodes": [
                    {
                        "provenanceId": "dc/base/HumanReadableStatVars",
                        "value": f"The total number of {label.lower()}.",
                    }
                ]
            },
            "gender": {"nodes": [_prop_node(gender_value, ["GenderType"])]},
            "constraintProperties": {"nodes": [_prop_node("gender", ["Property"])]},
            "provenance": {"nodes": [_prop_node("dc/base/HumanReadableStatVars", ["Provenance"])]},
            "measuredProperty": {"nodes": [_prop_node("count", ["Property"])]},
            "memberOf": {
                "nodes": [
                    {
                        "name": f"Person With Gender = {gender_value}",
                        "types": ["StatVarGroup"],
                        "dcid": f"dc/g/Person_Gender-{gender_value}",
                        "provenanceId": "dc/base/HumanReadableStatVars",
                    }
                ]
            },
            "name": {
                "nodes": [
                    {
                        "provenanceId": "dc/base/HumanReadableStatVars",
                        "value": label,
                    }
                ]
            },
            "populationType": {"nodes": [_prop_node("Person", ["Class"])]},
        },
    }


def _narrow_graph(
    *, include_nonbinary: bool = False, include_place: bool = False
) -> _CountingGraph:
    """Build a graph with gender siblings and India observations for resubmit tests."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
    obs = json.loads((fixtures_dir / "graph_obs.json").read_text())

    nodes[_PERSON_MALE_SV] = _synthetic_gender_sv("Male", "Male population")
    if include_nonbinary:
        nodes[_PERSON_NONBINARY_SV] = _synthetic_gender_sv("NonBinary", "Nonbinary population")

    _obs = [
        {
            "earliestDate": "2001",
            "latestDate": "2021",
            "obsCount": 3,
            "importName": "IndiaCensus_Primary",
        }
    ]
    obs[f"{_PERSON_FEMALE_SV}|{_ENTITY_DCID}"] = _obs
    obs[f"{_PERSON_MALE_SV}|{_ENTITY_DCID}"] = _obs
    if include_place:
        obs[f"{_PERSON_URBAN_FEMALE_SV}|{_ENTITY_DCID}"] = _obs
        obs[f"{_PERSON_RURAL_FEMALE_SV}|{_ENTITY_DCID}"] = _obs

    return _CountingGraph(nodes=nodes, obs=obs)


def _family_reader(graph: FakeGraph):
    """The mint's walk reader over the person counts `_narrow_graph` holds."""
    return axis_confirming_reader(
        _PERSON_FEMALE_SV,
        _PERSON_MALE_SV,
        _PERSON_URBAN_FEMALE_SV,
        _PERSON_RURAL_FEMALE_SV,
        graph=graph,
    )


# ---------------------------------------------------------------------------
# Pure in-set narrow
# ---------------------------------------------------------------------------


class TestPureNarrow:
    @pytest.fixture(autouse=True)
    def _no_related_reads(self, monkeypatch):
        """These tests count the StatVar arc reads a narrow makes. The related-measures reads
        (`related.read_related`) read arcs of other StatVars beside them, so they are left
        out."""
        monkeypatch.setattr(related_read, "read_related", lambda **_kwargs: NO_RELATED)

    def test_narrow_to_single_in_set_value_resolves_requested_not_first_posted(self):
        """A gender=Male identity elects Male and reads its arcs once."""
        edited_slot = _value_slot("how", "gender", "gender", "Male", "Male")
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([edited_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        assert [sv.ref.dcid for sv in region.specs[0].resolved.members] == [_PERSON_MALE_SV]
        # The elected member's arcs are read once; no sibling confirmation follows.
        assert graph.node_arcs_batch_calls == [[_PERSON_MALE_SV]]

    def test_narrow_to_two_of_three_in_set_values_resolves_exactly_those_two(self):
        """A two-value gender set beside a third gender elects those two and keeps the set
        binding."""
        binding_set_slot = _binding_set_slot("how", "gender", "gender", ["Female", "Male"])
        graph = _narrow_graph(include_nonbinary=True)
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([binding_set_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        dcids = {sv.ref.dcid for sv in region.specs[0].resolved.members}
        assert dcids == {_PERSON_FEMALE_SV, _PERSON_MALE_SV}
        gender_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.dcid == "gender"
        )
        assert gender_slot.binding.kind == "set"
        assert {v.ref.dcid for v in gender_slot.binding.values if v.ref is not None} == {
            "Female",
            "Male",
        }
        # The second read grounds the requested values; it does not confirm siblings.
        assert graph.node_arcs_batch_calls == [
            [_PERSON_FEMALE_SV, _PERSON_MALE_SV],
            ["Female", "Male"],
        ]

    def test_requested_equal_to_union_is_plain_promote_not_edit(self):
        """An unchanged single-value binding keeps its posted SV without sibling lookup."""
        edited_slot = _value_slot("how", "gender", "gender", "Female", "Female")
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([edited_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        assert [sv.ref.dcid for sv in region.specs[0].resolved.members] == [_PERSON_FEMALE_SV]
        assert graph.node_arcs_batch_calls == [[_PERSON_FEMALE_SV]]

    def test_mixed_edit_with_any_out_of_set_value_elects_the_edited_family(self):
        """An out-of-set edit elects the edited identity's family. The graph holds no urban
        male count, so the edit answers `no_matching_slice`."""
        gender_slot = _value_slot("how", "gender", "gender", "Male", "Male")
        place_slot = _value_slot(
            "how",
            "placeOfResidenceClassification",
            "place of residence",
            "Urban",
            "Urban",
        )
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([gender_slot, place_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph, EngineOptions(materialise_graph=_family_reader(graph))),
        )
        assert region.status == "no_data"
        assert region.no_data_reason == "no_matching_slice"

    def test_cross_product_hole_two_in_set_props_no_sv_matches_both(self):
        """In-set values no posted StatVar carries together elect the edited identity's
        family, which holds no urban male count either, so the edit answers `no_data`."""
        gender_slot = _value_slot("how", "gender", "gender", "Male", "Male")
        place_slot = _value_slot(
            "how",
            "placeOfResidenceClassification",
            "place of residence",
            "Urban",
            "Urban",
        )
        graph = _narrow_graph(include_place=True)

        inp = ResubmitIdentityPayload(
            identity=_edited_identity([gender_slot, place_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph, EngineOptions(materialise_graph=_family_reader(graph))),
        )
        assert region.status == "no_data"
        assert region.no_data_reason == "no_matching_slice"


# ---------------------------------------------------------------------------
# Unchanged multi-value set survives resubmit (plurality reconstruction)
# ---------------------------------------------------------------------------


class TestUnchangedSetSurvives:
    """Unchanged multi-member bindings stay sets when all resolved StatVars assert the property."""

    def test_unchanged_two_member_set_echo_resolves_both_svs(self):
        """An unchanged two-member set keeps both StatVars and the set binding."""
        binding_set_slot = _binding_set_slot("how", "gender", "gender", ["Female", "Male"])
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([binding_set_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        dcids = {sv.ref.dcid for sv in region.specs[0].resolved.members}
        assert dcids == {_PERSON_FEMALE_SV, _PERSON_MALE_SV}
        gender_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.dcid == "gender"
        )
        assert gender_slot.binding.kind == "set"
        assert {v.ref.dcid for v in gender_slot.binding.values if v.ref is not None} == {
            "Female",
            "Male",
        }

    def test_unchanged_set_survives_alongside_when_slot_edit(self):
        """An unrelated time-window edit preserves the unchanged gender set."""
        binding_set_slot = _binding_set_slot("how", "gender", "gender", ["Female", "Male"])
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                [binding_set_slot, _when_slot_value(2005, 2015), *_WHERE_SLOTS]
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        dcids = {sv.ref.dcid for sv in region.specs[0].resolved.members}
        assert dcids == {_PERSON_FEMALE_SV, _PERSON_MALE_SV}
        gender_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.dcid == "gender"
        )
        assert gender_slot.binding.kind == "set"

    def test_mixed_presence_folded_total_unaffected(self):
        """A posted-unbound slot stays unbound when one StatVar lacks that property."""
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([_unbound_slot("how", "gender", "gender"), *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        gender_slot = next(
            (
                s
                for s in region.specs[0].resolved.slots
                if s.key.property is not None and s.key.property.dcid == "gender"
            ),
            None,
        )
        assert gender_slot is not None
        assert gender_slot.binding.kind == "unbound"

    def test_single_sv_promote_binding_stays_value_kind(self):
        """A single StatVar retains a value binding."""
        edited_slot = _value_slot("how", "gender", "gender", "Female", "Female")
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([edited_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        gender_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.dcid == "gender"
        )
        assert gender_slot.binding.kind == "value"


# Issue #629 pins shape labels to the representative StatVar's name arc.


class TestShapeLabel:
    """Pin shape labels to the representative StatVar's name arc (#629)."""

    def test_unedited_resubmit_labels_from_rep_sv_name_arc(self):
        """Resubmitting Female keeps the label from its `name` arc (#629)."""
        edited_slot = _value_slot("how", "gender", "gender", "Female", "Female")
        graph = _narrow_graph()
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([edited_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        shape = region.specs[0].resolved.shape
        assert shape.label == "Female population"
        assert shape.label != shape.shape_id

    def test_representative_without_name_arc_falls_back_to_standard_label_not_raw_id(self):
        """A missing name arc uses `STANDARD_SHAPE_LABEL`, not the shape ID (#629)."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        nodes = json.loads((fixtures_dir / "graph_nodes.json").read_text())
        obs = json.loads((fixtures_dir / "graph_obs.json").read_text())

        nameless_nonbinary = _synthetic_gender_sv("NonBinary", "Nonbinary population")
        del nameless_nonbinary["arcs"]["name"]
        nodes[_PERSON_NONBINARY_SV] = nameless_nonbinary
        obs[f"{_PERSON_NONBINARY_SV}|{_ENTITY_DCID}"] = [
            {
                "earliestDate": "2001",
                "latestDate": "2021",
                "obsCount": 3,
                "importName": "IndiaCensus_Primary",
            }
        ]
        graph = _CountingGraph(nodes=nodes, obs=obs)

        edited_slot = _value_slot("how", "gender", "gender", "NonBinary", "NonBinary")
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([edited_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph),
        )
        assert region.status == "definite"
        shape = region.specs[0].resolved.shape
        assert shape.label == STANDARD_SHAPE_LABEL
        assert shape.label != shape.shape_id

    def test_pinned_shape_names_its_upstream_source_beside_the_label(self, bind_catalogue):
        """A pinned base variable's shape names its source apart from its label."""
        census = provenance_row("dc/base/IndiaCensus", BASE, source="dc/s/IndiaCensus")
        bind_catalogue(
            restrictive_catalogue(
                provenances=[census], base_variables={_PERSON_FEMALE_SV: ([census.id], [])}
            )
        )
        edited_slot = _value_slot("how", "gender", "gender", "Female", "Female")
        inp = ResubmitIdentityPayload(
            identity=_edited_identity([edited_slot, *_WHERE_SLOTS]),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_narrow_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_narrow_graph()),
        )

        assert region.status == "definite"
        shape = region.specs[0].resolved.shape
        assert shape.label == "Female population"
        assert shape.upstream is not None
        assert shape.upstream.source.ref == census.source.ref
        assert shape.upstream.data_commons.id == BASE.id


# ---------------------------------------------------------------------------
# A requested value narrows the elected members as the menu counted the option
# ---------------------------------------------------------------------------


def _modality_arcs(code: str) -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": "crsCooperationModality"}]},
        "crsCooperationModality": {"nodes": [{"dcid": code}]},
    }


@pytest.mark.parametrize(
    ("edited", "narrowed"),
    [
        pytest.param(
            "CRSCooperationModalityA", ["sv_budget", "sv_general_budget"], id="rollup-and-leaf"
        ),
        pytest.param("CRSCooperationModalityA01", ["sv_general_budget"], id="leaf"),
        pytest.param("CRSCooperationModalityB", [], id="carried-by-no-member"),
    ],
)
def test_a_requested_value_keeps_the_members_carrying_it_or_a_descendant(edited, narrowed):
    """Elected members carry budget support (A) itself, general budget support (A01, under
    A), and project-type interventions (C01). A pick of A keeps both A members, as the menu
    counted A over its descendants. A value no member carries narrows to nothing, and the
    elected members stand."""
    from types import SimpleNamespace

    from search.engine.region.promote_survivors import _narrowed_members

    arcs = {
        "sv_budget": _modality_arcs("CRSCooperationModalityA"),
        "sv_general_budget": _modality_arcs("CRSCooperationModalityA01"),
        "sv_project": _modality_arcs("CRSCooperationModalityC01"),
    }
    state = SimpleNamespace(
        stat_var_dcids=list(arcs),
        confirmed_arcs=arcs,
        edits={"crsCooperationModality": [edited]},
        posted_bound_values={"crsCooperationModality": [edited]},
        widened_props=set(),
    )

    assert _narrowed_members(state) == narrowed
