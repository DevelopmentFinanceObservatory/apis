"""Pins admission of posted axes against the matching five-tuple family.

Unconfirmed axes stay unbound; missing graph wiring raises `EngineInfraError`.
Tests use the recorded Person/count family and an offline graph reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from search.engine.errors import EngineInfraError
from search.engine.region import resolve_spec_resubmit
from search.engine.region.promote_reread import _family_measurement_qualifier
from search.engine.region.promote_shape import _confirm_axes_against_family
from search.engine.region.state import EngineOptions
from search.engine.retrieval.facts import read_five_tuple
from search.engine.shape import identity_shape_for
from search.models import ResubmitIdentityPayload
from tests.engine._axis_doubles import AxisConfirmingGraphReader, axis_confirming_reader
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    make_base,
    make_identity,
    subject_entity,
)
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine.test_resubmit_standard_widen import _GENDER_FEMALE
from tests.fixtures import FakeGraph, _load_json

_ENTITY_DCID = "country/IND"
_FAMILY = ("Count_Person", "Count_Person_Male", "Count_Person_Female")

# This exact-family member asserts age without a denominator.
_EXACT_FAMILY_AGE_ASSERTER = "Count_Person_EducationalAttainment1StGrade"

# This AsAFractionOf sibling asserts age but belongs to another family.
_FOREIGN_DENOMINATOR_AGE_ASSERTER = (
    "Count_Person_20To34Years_BirthInThePast12Months_Female_AsAFractionOf_"
    "Count_Person_20To34Years_Female"
)


def test_provisional_qualifier_keeps_its_family_match_value():
    raw_arcs = {
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {
            "nodes": [{"dcid": "DanglingQualifier", "types": ["ProvisionalNode"]}]
        },
    }
    ft = read_five_tuple(raw_arcs)
    family_arcs = {
        "matching": {
            "populationType": ("Person",),
            "statType": ("measuredValue",),
            "measurementQualifier": ("DanglingQualifier",),
            "constraintProperties": ("gender",),
            "gender": ("Female",),
        },
        "qualified-neighbour": {
            "populationType": ("Person",),
            "statType": ("measuredValue",),
            "measurementQualifier": ("OtherQualifier",),
            "constraintProperties": ("age",),
            "age": ("Years25Onwards",),
        },
    }

    confirmed = _confirm_axes_against_family(
        ["gender", "age"],
        family_arcs,
        ft=ft,
        measurement_qualifiers=frozenset({_family_measurement_qualifier(raw_arcs)}),
    )

    assert ft.meas_qual_dcid is None
    assert confirmed == {"gender": ["Female"]}


def _family_reader():
    return axis_confirming_reader(*_FAMILY)


class TestUnassertedAxisResolvesUnboundAndInapplicable:
    """An unasserted age slot stays unbound and appears in `inapplicable_slots`."""

    def test_resolves_unbound_and_inapplicable_with_no_options(self):
        entities = [subject_entity(_ENTITY_DCID)]
        age_unbound = _unbound_slot("how", "age", "age")
        base_spec_id, _base_identity = make_base(
            PERSON_COUNT_SHAPE, slots=[age_unbound], entities=entities
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [age_unbound], entities=entities),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph(), EngineOptions(materialise_graph=_family_reader())),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        # A no-op post (identity == base) re-mints the base's own spec_id.
        assert spec.spec_id == base_spec_id
        age_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == "age"
        )
        assert age_slot.binding.kind == "unbound"
        assert "age" in spec.resolved.inapplicable_slots
        assert not any(
            so.key.property is not None and so.key.property.ref == "age"
            for so in spec.resolved.slot_options
        )


class TestUnassertedAxisAdmittedAndFilled:
    """Gender is admitted from male and female family members."""

    def test_gender_admitted_with_both_values_and_stable_spec_id(self):
        entities = [subject_entity(_ENTITY_DCID)]
        gender_unbound = _unbound_slot("how", "gender", "gender")
        base_spec_id, _base_identity = make_base(
            PERSON_COUNT_SHAPE, slots=[gender_unbound], entities=entities
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [gender_unbound], entities=entities),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph(), EngineOptions(materialise_graph=_family_reader())),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        assert spec.spec_id == base_spec_id
        gender_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == "gender"
        )
        assert gender_slot.binding.kind == "unbound"
        gender_options = next(
            so
            for so in spec.resolved.slot_options
            if so.key.property is not None and so.key.property.ref == "gender"
        )
        assert {opt.value.ref.ref for opt in gender_options.options} == {"Male", "Female"}


class TestBoundAssertedAxisUnaffected:
    """A bound gender value already asserted by the StatVar skips taxonomy lookup."""

    def test_bound_gender_resolves_without_materialise_graph(self):
        entities = [subject_entity(_ENTITY_DCID)]
        base_spec_id, _base_identity = make_base(
            PERSON_COUNT_SHAPE, slots=[_GENDER_FEMALE], entities=entities
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [_GENDER_FEMALE], entities=entities),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph(), EngineOptions(materialise_graph=None)),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        assert spec.spec_id == base_spec_id
        gender_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == "gender"
        )
        assert gender_slot.binding.kind == "value"


class TestMissingMaterialiseGraphRaises:
    """Missing graph wiring raises `EngineInfraError`: a standard identity elects its members
    by the walk over the materialise graph."""

    def test_none_materialise_graph_raises_engine_infra_error(self):
        entities = [subject_entity(_ENTITY_DCID)]
        gender_unbound = _unbound_slot("how", "gender", "gender")
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [gender_unbound], entities=entities),
        )
        with pytest.raises(EngineInfraError):
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=EngineOptions(materialise_graph=None),
            )


# A real fixture StatVar (graph_nodes.json) whose measuredProperty arc is a
# dcid-less ProvisionalNode -- read_five_tuple's meas_prop_dcid comes back
# "", the grammar's own "arc absent or unresolved" spelling.
_UNANCHORED_SV_DCID = "MortalityRate_Person_Upto4Years_AsFractionOf_Count_BirthEvent_LiveBirth"


class TestUnanchoredFiveTupleSkipsConfirmation:
    """An unanchored five-tuple skips family lookup and issues no graph reads."""

    def test_no_graph_read_for_an_unconfirmable_axis(self):
        graph = FakeGraph()
        ft = read_five_tuple(graph.node_arcs(_UNANCHORED_SV_DCID))
        assert ft.meas_prop_dcid == ""
        shape = identity_shape_for(
            pop_type_dcid=ft.pop_type_dcid,
            meas_prop_dcid=ft.meas_prop_dcid,
            stat_type_dcid=ft.stat_type_dcid,
            meas_qual_dcid=ft.meas_qual_dcid,
            meas_denom_dcid=ft.meas_denom_dcid,
            meas_denom_unresolved=ft.meas_denom_unresolved,
        )
        entities = [subject_entity(_ENTITY_DCID)]
        age_unbound = _unbound_slot("how", "age", "age")
        gender_unbound = _unbound_slot("how", "gender", "gender")
        inp = ResubmitIdentityPayload(
            identity=make_identity(shape, [age_unbound, gender_unbound], entities=entities),
        )
        reader = _CountingGraphReader(inner=_family_reader())
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph, EngineOptions(materialise_graph=reader)),
        )
        assert reader.node_subjects_calls == 0
        assert reader.node_arcs_many_calls == 0
        assert region.status == "no_data"


class TestForeignFamilyExcluded:
    """An age value from a different denominator family cannot admit the axis."""

    def test_axis_asserted_only_by_a_foreign_family_resolves_with_no_options(self):
        entities = [subject_entity(_ENTITY_DCID)]
        age_unbound = _unbound_slot("how", "age", "age")
        base_spec_id, _base_identity = make_base(
            PERSON_COUNT_SHAPE, slots=[age_unbound], entities=entities
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [age_unbound], entities=entities),
        )
        reader = axis_confirming_reader(*_FAMILY, _FOREIGN_DENOMINATOR_AGE_ASSERTER)
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph(), EngineOptions(materialise_graph=reader)),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        assert spec.spec_id == base_spec_id
        assert "age" in spec.resolved.inapplicable_slots
        assert not any(
            so.key.property is not None and so.key.property.ref == "age"
            for so in spec.resolved.slot_options
        )

    def test_foreign_family_value_absent_when_exact_family_confirms(self):
        entities = [subject_entity(_ENTITY_DCID)]
        age_unbound = _unbound_slot("how", "age", "age")
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [age_unbound], entities=entities),
        )
        reader = axis_confirming_reader(
            *_FAMILY, _EXACT_FAMILY_AGE_ASSERTER, _FOREIGN_DENOMINATOR_AGE_ASSERTER
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph(), EngineOptions(materialise_graph=reader)),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        age_options = next(
            so
            for so in spec.resolved.slot_options
            if so.key.property is not None and so.key.property.ref == "age"
        )
        age_values = {opt.value.ref.ref for opt in age_options.options}
        assert age_values == {"Years25Onwards"}
        assert "Years20To34" not in age_values


@dataclass
class _CountingGraphReader:
    """Counts family lookup calls made through a graph reader."""

    inner: object
    node_subjects_calls: int = field(default=0)
    node_arcs_many_calls: int = field(default=0)

    def node_subjects(self, *args, **kwargs):
        self.node_subjects_calls += 1
        return self.inner.node_subjects(*args, **kwargs)

    def node_arcs_many(self, *args, **kwargs):
        self.node_arcs_many_calls += 1
        return self.inner.node_arcs_many(*args, **kwargs)

    def node_names_many(self, *args, **kwargs):
        return self.inner.node_names_many(*args, **kwargs)

    def generation_stamp(self):
        return self.inner.generation_stamp()

    def node_label(self, *args, **kwargs):
        return self.inner.node_label(*args, **kwargs)

    def observation_facets_batch(self, *args, **kwargs):
        return self.inner.observation_facets_batch(*args, **kwargs)

    def default_facet_policy(self, *args, **kwargs):
        return self.inner.default_facet_policy(*args, **kwargs)

    def node_arcs(self, *args, **kwargs):
        return self.inner.node_arcs(*args, **kwargs)


class TestUnconfirmedAxisCallCount:
    """An unconfirmed axis costs one `node_subjects` and one `node_arcs_many` call beyond the
    mint's member walk, which a resubmit with no unconfirmed axis also runs."""

    @staticmethod
    def _family_reads(slots: list) -> tuple[int, int]:
        entities = [subject_entity(_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, slots, entities=entities),
        )
        reader = _CountingGraphReader(inner=_family_reader())
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph(), EngineOptions(materialise_graph=reader)),
        )
        assert region.status == "definite"
        return reader.node_subjects_calls, reader.node_arcs_many_calls

    def test_resolves_after_exactly_two_graph_calls(self):
        walk_subjects, walk_arcs = self._family_reads([])
        with_axis = self._family_reads([_unbound_slot("how", "age", "age")])

        assert with_axis == (walk_subjects + 1, walk_arcs + 1)


class TestConfirmedAxisSampleNeverDecidesClassification:
    """Sample values cannot change a confirmed constraint slot from the `how` axis."""

    def test_place_shaped_sample_values_keep_the_slot_on_how(self):
        entities = [subject_entity(_ENTITY_DCID)]
        gender_unbound = _unbound_slot("how", "gender", "gender")
        inp = ResubmitIdentityPayload(
            identity=make_identity(PERSON_COUNT_SHAPE, [gender_unbound], entities=entities),
        )
        place_shaped_member_arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": [{"dcid": "gender"}]},
            "gender": {"nodes": [{"dcid": "country/USA"}]},
        }
        # The family holds the total population, which the walk elects, beside the
        # place-shaped member the axis sample reads.
        reader = AxisConfirmingGraphReader(
            family_arcs={
                "synthetic-place-shaped-member": place_shaped_member_arcs,
                "Count_Person": _load_json("graph_nodes.json")["Count_Person"]["arcs"],
            },
            graph=FakeGraph(),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph(), EngineOptions(materialise_graph=reader)),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        gender_slot = next(
            s
            for s in spec.resolved.slots
            if s.key.property is not None and s.key.property.ref == "gender"
        )
        assert gender_slot.key.axis == "how"
        gender_options = next(
            so
            for so in spec.resolved.slot_options
            if so.key.property is not None and so.key.property.ref == "gender"
        )
        assert {opt.value.ref.ref for opt in gender_options.options} == {"country/USA"}
