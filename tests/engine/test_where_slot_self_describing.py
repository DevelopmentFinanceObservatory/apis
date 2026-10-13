"""Where slots carry subject entities. Latest re-resolves the year on resubmit."""

from __future__ import annotations

from search.engine.bind import _build_universal_bind_prompt
from search.engine.region import resolve_spec_resubmit
from search.engine.region.pipeline import (
    resolve_variable,  # noqa: F401 -- import smoke, unused directly
)
from search.engine.shape import shape_draft_from
from search.engine.spec_id import compute_spec_id
from search.models import (
    BindingUnbound,
    BindingValue,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    RawTextInput,
    ResolveRequest,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.engine._base_identity import (
    DECLARING_SHAPE,
    GDP_SHAPE,
    make_base,
    subject_entity,
)
from tests.engine._builders import gref as _gref
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import offline_resolve, walk_options
from tests.fixtures import GDP_INDIA_PROVENANCES, FakeGraph, fake_graph_with_provenances


def _where_slot(slots: list[Slot]) -> Slot | None:
    return next((s for s in slots if s.key.axis == "where"), None)


_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_ENTITY_DCID = "country/IND"
_ACTIVITY_SOURCE_SLOT = _value_slot(
    "how", "activitySource", "activity source", "GrossDomesticProduction", "GDP"
)


class TestStandardWhereSlot:
    def test_raw_text_binds_subject_entity_into_where_slot(self):
        """A definite result binds the subject entity in a property-less where slot."""
        r = offline_resolve(ResolveRequest(input=RawTextInput(query="total population India")))
        assert r.parts[0].status == "definite"
        spec = r.parts[0].spec
        where = _where_slot(spec.resolved.slots)
        assert where is not None
        assert where.key.property is None
        assert isinstance(where.binding, BindingValue)
        assert where.binding.value.ref is not None
        assert where.binding.value.ref.ref == "country/IND"

    def test_round_trip_without_entity_dcids_fully_rematerialises(self):
        """The posted where slot supplies the entity when the base identity omits it."""
        inp = ResubmitIdentityPayload(
            identity=make_base(
                GDP_SHAPE,
                slots=[_ACTIVITY_SOURCE_SLOT],
                entities=[subject_entity(_ENTITY_DCID)],
            )[1],
            # The where slot supplies the omitted base entity.
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(FakeGraph()),
        )
        assert region.status == "definite"
        spec = region.specs[0]
        assert any(e.ref == _ENTITY_DCID for e in spec.resolved.entities)
        where = _where_slot(spec.resolved.slots)
        assert where is not None
        assert isinstance(where.binding, BindingValue)
        assert where.binding.value.ref.ref == _ENTITY_DCID


class TestWhereSlotDoesNotPerturbIdentity:
    def _subject_entity(self, dcid: str) -> Entity:
        return Entity(ref=_gref(dcid, dcid), entity_type=None, role=EntityRoleSubject())

    def test_bound_where_slot_same_id_as_no_where_slot_at_all(self):
        """Adding a matching where slot leaves the entity-anchored spec id unchanged."""
        entities = [self._subject_entity("country/IND")]
        slots_pre_13c = [_ACTIVITY_SOURCE_SLOT]
        slots_post_13c = [
            _ACTIVITY_SOURCE_SLOT,
            Slot(
                key=SlotKey(axis="where", property=None, label="place"),
                binding=BindingValue(
                    value=SlotValue(ref=_gref("country/IND", "India"), value_kind="entity")
                ),
            ),
        ]
        id_pre = compute_spec_id(GDP_SHAPE, slots_pre_13c, entities=entities, open_roles=())
        id_post = compute_spec_id(GDP_SHAPE, slots_post_13c, entities=entities, open_roles=())
        assert id_pre == id_post

    def test_where_slot_binding_value_does_not_affect_id_when_entities_fixed(self):
        """Changing the where-slot value leaves the id unchanged when entities stay fixed."""
        entities = [self._subject_entity("country/IND")]

        def _slots(where_dcid: str) -> list[Slot]:
            return [
                _ACTIVITY_SOURCE_SLOT,
                Slot(
                    key=SlotKey(axis="where", property=None, label="place"),
                    binding=BindingValue(
                        value=SlotValue(ref=_gref(where_dcid, where_dcid), value_kind="entity")
                    ),
                ),
            ]

        id_a = compute_spec_id(GDP_SHAPE, _slots("country/IND"), entities=entities, open_roles=())
        id_b = compute_spec_id(GDP_SHAPE, _slots("country/XXX"), entities=entities, open_roles=())
        assert id_a == id_b

    def test_named_where_slot_unaffected(self):
        """A directional where slot remains identity-bearing for distinct recipients."""
        directional = EntityRoleDirectional(
            role=_gref("FlowRecipient", "recipient"), axis="destination"
        )
        entities_eth = [
            Entity(ref=_gref("country/ETH", "Ethiopia"), entity_type=None, role=directional)
        ]
        entities_ken = [
            Entity(ref=_gref("country/KEN", "Kenya"), entity_type=None, role=directional)
        ]
        slots = [
            _value_slot("where", "FlowRecipient", "recipient", "country/ETH", "Ethiopia"),
        ]
        id_eth = compute_spec_id(DECLARING_SHAPE, slots, entities=entities_eth, open_roles=())
        slots_ken = [
            _value_slot("where", "FlowRecipient", "recipient", "country/KEN", "Kenya"),
        ]
        id_ken = compute_spec_id(DECLARING_SHAPE, slots_ken, entities=entities_ken, open_roles=())
        assert id_eth != id_ken


class TestBindPromptUnaffectedByWhereSlot:
    def test_where_slot_invisible_to_bind_prompt(self):
        """The property-less where slot is excluded from the universal bind prompt."""
        shape_with_where = shape_draft_from(
            shape_id="economicactivity_amount_measuredvalue_nominal",
            label="Gross domestic product",
            pop_type_dcid="EconomicActivity",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            constraint_props=["activitySource"],
            prop_labels={"activitySource": "activity source"},
            prop_observed_values={"activitySource": ["GrossDomesticProduction"]},
            recalled=False,
        )
        # Remove the property-less where slot for the prompt comparison.
        shape_without_where = shape_with_where.__class__(
            **{
                **shape_with_where.__dict__,
                "slot_keys": tuple(
                    s
                    for s in shape_with_where.slot_keys
                    if not (s.axis == "where" and s.property_dcid is None)
                ),
            }
        )
        assert len(shape_without_where.slot_keys) == len(shape_with_where.slot_keys) - 1

        prompt_with = _build_universal_bind_prompt("GDP", [shape_with_where])
        prompt_without = _build_universal_bind_prompt("GDP", [shape_without_where])
        assert prompt_with == prompt_without, (
            "adding the where slot changed the bind prompt -- the "
            "fixture-recording gate DOES apply after all"
        )


class TestLatestWindowSentinel:
    def test_latest_resubmit_reresolves_current_latest_not_pin(self):
        """The latest sentinel resolves to 2024 despite the posted 2015 window: the latest
        point of the facet a preview serves, the 65-observation series to 2024, rather than
        the 2026 end of a 12-observation facet."""
        inp_latest = ResubmitIdentityPayload(
            identity=make_base(
                GDP_SHAPE,
                slots=[_ACTIVITY_SOURCE_SLOT],
                entities=[subject_entity(_ENTITY_DCID)],
                requested_window="latest",
            )[1],
        )
        region_latest = resolve_spec_resubmit(
            inp=inp_latest,
            graph=fake_graph_with_provenances(GDP_INDIA_PROVENANCES),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(fake_graph_with_provenances(GDP_INDIA_PROVENANCES)),
        )
        assert region_latest.status == "definite"
        spec_latest = region_latest.specs[0]
        assert spec_latest.resolved.coverage.window == TimeWindow(start_year=2024, end_year=2024)
        assert spec_latest.resolved.trace.requested_window == "latest"

        inp_no_sentinel = ResubmitIdentityPayload(
            identity=make_base(
                GDP_SHAPE,
                slots=[_ACTIVITY_SOURCE_SLOT],
                entities=[subject_entity(_ENTITY_DCID)],
                requested_window=TimeWindow(start_year=2015, end_year=2015),
            )[1],
        )
        region_pinned = resolve_spec_resubmit(
            inp=inp_no_sentinel,
            graph=fake_graph_with_provenances(GDP_INDIA_PROVENANCES),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(fake_graph_with_provenances(GDP_INDIA_PROVENANCES)),
        )
        assert region_pinned.status == "definite"
        assert region_pinned.specs[0].resolved.coverage.window == TimeWindow(
            start_year=2015, end_year=2015
        )

    def test_latest_distinguishable_from_no_date_requested_in_spec_id(self):
        """With an unbound when slot, latest and no requested date produce distinct ids."""
        entities = [
            Entity(ref=_gref("country/IND", "India"), entity_type=None, role=EntityRoleSubject())
        ]
        when_slot = Slot(
            key=SlotKey(axis="when", property=None, label="time window"),
            binding=BindingUnbound(),
        )
        slots = [_ACTIVITY_SOURCE_SLOT, when_slot]
        id_latest = compute_spec_id(
            GDP_SHAPE,
            slots,
            entities=entities,
            requested_window="latest",
            open_roles=(),
        )
        id_none = compute_spec_id(
            GDP_SHAPE, slots, entities=entities, requested_window=None, open_roles=()
        )
        assert id_latest != id_none
