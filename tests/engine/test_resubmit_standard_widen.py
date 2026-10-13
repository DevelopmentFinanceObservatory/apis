"""A resubmit's bound axes elect the members asserting them, and an axis posted unbound
widens."""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.spec_id import build_spec_identity
from search.models import ResubmitIdentityPayload
from tests.engine._base_identity import (
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    subject_entity,
    where_slots_for,
)
from tests.engine._builders import (
    binding_set_slot as _set_slot,
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
from tests.engine.test_path_c import (
    _PERSON_FEMALE_SV,
    _PERSON_MALE_SV,
    _PERSON_URBAN_FEMALE_SV,
    _REFINE_ENTITY_DCID,
    _standard_refine_graph,
)
from tests.fixtures import FakeGraph

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"

_ACTIVITY_SOURCE_BOUND = _value_slot(
    "how",
    "activitySource",
    "activity source",
    "GrossDomesticProduction",
    "Gross Domestic Production",
)
_ACTIVITY_SOURCE_UNBOUND = _unbound_slot("how", "activitySource", "activity source")

_GENDER_FEMALE = _value_slot("how", "gender", "gender", "Female", "Female")
_PLACE_URBAN = _value_slot(
    "how", "placeOfResidenceClassification", "place of residence", "Urban", "Urban"
)


def _edited_identity(shape: dict, slots: list, entities: list):
    return build_spec_identity(shape, slots, entities=entities, open_roles=())[1]


class TestWidening:
    """An unbound posted axis binds no value."""

    def test_an_unbound_axis_widens(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE, [_ACTIVITY_SOURCE_UNBOUND, *where_slots_for(entities)], entities
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        assert region.status == "definite"
        activity_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.ref == "activitySource"
        )
        assert activity_slot.binding.kind == "unbound"

    def test_a_bound_axis_promotes_its_member(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE, [_ACTIVITY_SOURCE_BOUND, *where_slots_for(entities)], entities
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        assert region.status == "definite"
        assert [sv.ref.ref for sv in region.specs[0].resolved.members] == [_GDP_SV_DCID]
        activity_slot = next(
            s
            for s in region.specs[0].resolved.slots
            if s.key.property is not None and s.key.property.ref == "activitySource"
        )
        assert activity_slot.binding.kind == "value"


class TestBoundAxes:
    """Bound axes elect the members asserting every bound value."""

    def test_two_bound_axes_promote_the_member_asserting_both(self):
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [_GENDER_FEMALE, _PLACE_URBAN, *where_slots_for(entities)],
                entities,
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_standard_refine_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_standard_refine_graph()),
        )
        assert region.status == "definite"
        assert [sv.ref.ref for sv in region.specs[0].resolved.members] == [_PERSON_URBAN_FEMALE_SV]


class TestMixedPresenceAndPlurality:
    """A member silent on a bound axis adds no claim, and a bound set elects each value's
    member."""

    def test_a_bound_gender_promotes_beside_a_folded_total(self):
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE, [_GENDER_FEMALE, *where_slots_for(entities)], entities
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_standard_refine_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_standard_refine_graph()),
        )
        assert region.status == "definite"

    def test_a_gender_set_promotes_both_members(self):
        entities = [subject_entity(_REFINE_ENTITY_DCID)]
        gender_set_slot = _set_slot("how", "gender", "gender", ["Female", "Male"])
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE, [gender_set_slot, *where_slots_for(entities)], entities
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_standard_refine_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_standard_refine_graph()),
        )
        assert region.status == "definite"
        dcids = {sv.ref.ref for sv in region.specs[0].resolved.members}
        assert dcids == {_PERSON_FEMALE_SV, _PERSON_MALE_SV}
