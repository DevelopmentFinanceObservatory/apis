"""Pins base-identity checks, removed parts, and resubmit edit accounting."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.errors import EngineInputError
from search.engine.identity_edit import validate_removed
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.spec_id import build_spec_identity
from search.models import (
    BaseIdentity,
    BindingValue,
    IdentityBindingExact,
    IdentityBindingUnbound,
    IdentityBindingValue,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    RemovedParts,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
    TimeWindow,
)
from tests.engine._base_identity import (
    DECLARING_SHAPE,
    GDP_SHAPE,
    PERSON_COUNT_SHAPE,
    destination_entity,
    make_base,
    origin_entity,
    shape_for,
    subject_entity,
    where_slots_for,
)
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine.test_path_c import _PERSON_URBAN_FEMALE_SV
from tests.fixtures import FakeGraph

_SCHEME_SLOT = _value_slot("what", "FlowScheme", "finance scheme", "ODAGrants", "ODA Grants")
_PURPOSE_SLOT = _value_slot("how", "FlowPurpose", "sector/purpose", "DAC/Health", "Health (Total)")
_PURPOSE_SLOT_EDITED = _value_slot(
    "how", "FlowPurpose", "sector/purpose", "DAC/BasicHealth", "Basic Health"
)
_PURPOSE_SLOT_UNBOUND = _unbound_slot("how", "FlowPurpose", "sector/purpose")

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"

# Only gender is posted for the missing-slot-key test below. The group's
# other constraint slot key is placeOfResidenceClassification.
_PERSON_ENTITY_DCID = "country/IND"
_GENDER_FEMALE_SLOT = _value_slot("how", "gender", "gender", "Female", "Female")


def _edited_identity(shape, slots, *, entities=None, facets=None):
    return build_spec_identity(shape, slots, entities=entities, explicit_facets=facets)[1]


class TestEditDiffs:
    def test_dropping_a_base_declared_slot_key_refuses_request_invalid(self):
        """Omitting a base-declared slot, even as unbound, raises `request_invalid`."""
        entities = [subject_entity(_PERSON_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE,
            slots=[
                _GENDER_FEMALE_SLOT,
                _unbound_slot("how", "placeOfResidenceClassification", "place of residence"),
            ],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                PERSON_COUNT_SHAPE,
                [_GENDER_FEMALE_SLOT, *where_slots_for(entities)],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_PERSON_URBAN_FEMALE_SV],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.pointer == "/input/identity/slots"
        assert "placeOfResidenceClassification" in str(exc_info.value)


# ---------------------------------------------------------------------------
# `facets.provenance` is translated to the source slot and included in edit accounting.
# ---------------------------------------------------------------------------

_PROVENANCE_DCID = "dc/base/WorldDevelopmentIndicators"
_EXTRA_PROVENANCE_DCID = "dc/base/India_NationalAccountsStatistics"

_MEASUREMENT_METHOD_A = "dc/measurementMethod/MethodA"
_MEASUREMENT_METHOD_B = "dc/measurementMethod/MethodB"


def _method_facet(provenance_dcid: str, measurement_method: str, *, dates: list[str]) -> dict:
    """Build a facet with measurement method, provenance, and date metadata."""
    return {
        "earliestDate": dates[0],
        "latestDate": dates[-1],
        "obsCount": len(dates),
        "dates": dates,
        "values": [1.0] * len(dates),
        "provenanceId": provenance_dcid,
        "measurementMethod": measurement_method,
    }


class TestMeasurementMethodNarrowsResponseMetadata:
    """A measurement-method binding narrows coverage, sources, and date range to matching facets."""

    def test_standard_promote_branch_narrows_the_same_way(self):
        """Standard promotion applies the same metadata narrowing."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        pair_key = f"{_GDP_SV_DCID}|{_GDP_ENTITY_DCID}"
        graph = FakeGraph(
            obs={
                pair_key: [
                    _method_facet(_PROVENANCE_DCID, _MEASUREMENT_METHOD_A, dates=["2020"]),
                    _method_facet(
                        _EXTRA_PROVENANCE_DCID, _MEASUREMENT_METHOD_B, dates=["2015", "2016"]
                    ),
                ]
            }
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE,
                [
                    _unbound_slot("how", "activitySource", "activity source"),
                    *where_slots_for(entities),
                ],
                entities=entities,
                facets={"measurementMethod": IdentityBindingValue(ref=_MEASUREMENT_METHOD_A)},
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=EngineOptions(include_observations=True),
        )
        assert region.status == "definite"
        resolved = region.specs[0].resolved

        assert resolved.coverage.observation_count == 1
        sources = {s.dcid for s in resolved.trace.resolved_sources}
        assert sources == {_PROVENANCE_DCID}
        date_range = resolved.stat_vars.items[0].data_date_range
        assert date_range is not None
        assert date_range.start == "2020"
        assert date_range.end == "2020"


class _CountingObsGraph(FakeGraph):
    """Count facet-batch reads used by bound-facet availability checks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obs_batch_calls = 0

    def observation_facets_batch(self, stat_vars, entities, needs_dates=False):
        self.obs_batch_calls += 1
        return super().observation_facets_batch(stat_vars, entities, needs_dates=needs_dates)


def _when_slot(start: int, end: int) -> Slot:
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


_INEXPRESSIBLE_UNIT_FACETS = {
    "unit": IdentityFacetValueRequest(request=IdentityUnitRequest(currency="XYZ", prices="current"))
}


class TestInexpressibleUnitHoistedBeforeGraphAccess:
    """A currency outside ISO 4217 raises `facet_binding_unsupported` (501) before graph reads."""

    def test_standard_promote_branch_refuses_before_any_graph_read(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE,
                [
                    _unbound_slot("how", "activitySource", "activity source"),
                    *where_slots_for(entities),
                ],
                entities=entities,
                facets=_INEXPRESSIBLE_UNIT_FACETS,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "facet_binding_unsupported"
        assert exc_info.value.status == 501
        assert exc_info.value.pointer == "/input/identity/facets/unit"

    def test_hoisted_check_preserves_facet_refusal_semantics_and_public_pointer(self):
        """The early check preserves refusal details and returns the nested request pointer."""
        from search.engine.facet_refusals import facet_refusal

        late = facet_refusal(
            FakeGraph(raise_on_call=True),
            ["Some/SV"],
            ["country/USA"],
            _INEXPRESSIBLE_UNIT_FACETS,
            deadline=None,
        )
        assert late is not None

        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                GDP_SHAPE,
                [
                    _unbound_slot("how", "activitySource", "activity source"),
                    *where_slots_for(entities),
                ],
                entities=entities,
                facets=_INEXPRESSIBLE_UNIT_FACETS,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        hoisted = exc_info.value

        assert hoisted.code == late.code
        assert hoisted.status == late.status
        assert hoisted.detail == late.detail
        assert late.pointer == "/input/facets/unit"
        assert hoisted.pointer == "/input/identity/facets/unit"


class TestPromotionViaBaseSpecId:
    def test_standard_promote_via_base_spec_id_no_ad_hoc_fields(self):
        """Standard promotion uses `base_spec_id` and `base_identity` for its base."""
        assert "stat_var_dcids" not in ResubmitIdentityPayload.model_fields
        assert "entity_dcids" not in ResubmitIdentityPayload.model_fields
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            # The schema requires the `activitySource` slot, even when unbound.
            identity=_edited_identity(
                GDP_SHAPE,
                [
                    _unbound_slot("how", "activitySource", "activity source"),
                    *where_slots_for(entities),
                ],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=FakeGraph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite"


class TestRefusals:
    def test_duplicate_edited_slot_property_refuses_before_any_graph_read(self):
        """The request model permits duplicate slot properties; the identity
        core rejects them before resolution can silently widen the ask."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(
            GDP_SHAPE,
            slots=[_PURPOSE_SLOT],
            entities=entities,
        )
        duplicate_unbound_slot = base_identity.slots[0].model_copy(
            update={"binding": IdentityBindingUnbound()}
        )
        edited_identity = BaseIdentity.model_validate(
            {
                **base_identity.model_dump(exclude_none=True),
                "slots": [*base_identity.slots, duplicate_unbound_slot],
            }
        )
        assert len(edited_identity.slots) == 2

        inp = ResubmitIdentityPayload(
            identity=edited_identity,
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.pointer == "/input/identity"

    @pytest.mark.parametrize(
        ("base_spec_id", "include_base_identity", "expected_code", "expected_status", "pointer"),
        [
            pytest.param(
                "spec_" + "0" * 32,
                False,
                "base_not_found",
                422,
                "/input/base_spec_id",
                id="missing-base-identity",
            ),
            pytest.param(
                None,
                True,
                "request_invalid",
                400,
                "/input/base_identity",
                id="missing-base-spec-id",
            ),
        ],
    )
    def test_incomplete_base_pair_refuses_exactly(
        self, base_spec_id, include_base_identity, expected_code, expected_status, pointer
    ):
        """Each complementary omission keeps its documented public refusal."""
        base_identity = None
        if include_base_identity:
            _, base_identity = make_base(
                DECLARING_SHAPE, entities=[destination_entity("country/ETH")]
            )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(DECLARING_SHAPE, [_SCHEME_SLOT, _PURPOSE_SLOT]),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == expected_code
        assert exc_info.value.status == expected_status
        assert exc_info.value.pointer == pointer

    def test_base_identity_not_hashing_to_base_spec_id_refuses_mismatch(self):
        _, base_identity = make_base(DECLARING_SHAPE, entities=[destination_entity("country/ETH")])
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(DECLARING_SHAPE, [_SCHEME_SLOT, _PURPOSE_SLOT]),
            base_spec_id="spec_" + "f" * 32,  # wrong hash, well-formed
            base_identity=base_identity,
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "base_identity_mismatch"

    def test_shape_change_with_a_base_refuses_shape_mismatch(self):
        """A shape change is a new construction, posted without a base -- when
        a base IS posted alongside a differing shape, that's a
        shape_mismatch, not a silent new construction."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(shape_for("some_other_shape"), [], entities=entities),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "shape_mismatch"
        assert exc_info.value.pointer == "/input/identity/shape"

    def test_removed_without_base_spec_id_refuses_request_invalid(self):
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(DECLARING_SHAPE, [_SCHEME_SLOT, _PURPOSE_SLOT]),
            removed=RemovedParts(entities=["origin"]),
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "request_invalid"

    def test_removed_naming_a_part_still_bound_refuses_request_invalid(self):
        """A posted entity cannot remain bound while `removed` also names its role."""
        entities = [destination_entity("country/ETH")]
        base_spec_id, base_identity = make_base(
            DECLARING_SHAPE,
            slots=[_SCHEME_SLOT, _PURPOSE_SLOT],
            entities=entities,
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(
                DECLARING_SHAPE,
                [_SCHEME_SLOT, _PURPOSE_SLOT, *where_slots_for(entities)],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            removed=RemovedParts(entities=["destination"]),
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "request_invalid"

    @pytest.mark.parametrize(
        ("shape", "base_slots", "entities", "missing_role"),
        [
            (
                GDP_SHAPE,
                [],
                [subject_entity(_GDP_ENTITY_DCID)],
                "observationAbout",
            ),
            (
                DECLARING_SHAPE,
                [_SCHEME_SLOT, _PURPOSE_SLOT],
                [destination_entity("country/ZMB")],
                "destination",
            ),
            (
                DECLARING_SHAPE,
                [_SCHEME_SLOT, _PURPOSE_SLOT],
                [origin_entity("country/FRA")],
                "origin",
            ),
        ],
    )
    def test_dropped_base_bound_entity_role_without_removed_refuses_edit_incomplete(
        self,
        shape,
        base_slots,
        entities,
        missing_role,
    ):
        """A base-bound entity role dropped from the posted identity, with no
        removed.entities acknowledgement, refuses edit_incomplete and names
        the exact missing role key."""
        base_spec_id, base_identity = make_base(shape, slots=base_slots, entities=entities)
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(shape, base_slots),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            # Resubmit's standard path requires a promote anchor before it
            # reaches edit accounting; the role check runs before this seed
            # is re-read or compared with the directional base shape.
            resolved_stat_var_dcids=[_GDP_SV_DCID],
            # No entity, no removed acknowledgement. The base-bound entity
            # silently vanishes unless this refuses.
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "edit_incomplete"
        assert exc_info.value.status == 422
        assert exc_info.value.pointer == "/input/identity/entities"
        assert missing_role in str(exc_info.value)

    def test_removed_duplicate_entity_entry_refuses_request_invalid(self):
        with pytest.raises(EngineInputError) as exc_info:
            validate_removed(RemovedParts(entities=["origin", "origin"]), "spec_" + "0" * 32)
        assert exc_info.value.code == "request_invalid"

    def test_removed_duplicate_facet_entry_refuses_request_invalid(self):
        with pytest.raises(EngineInputError) as exc_info:
            validate_removed(RemovedParts(facets=["provenance", "provenance"]), "spec_" + "0" * 32)
        assert exc_info.value.code == "request_invalid"

    def test_removed_unrecognized_entity_refuses_request_invalid(self):
        """An unknown role name raises `request_invalid` before graph resolution."""
        with pytest.raises(EngineInputError) as exc_info:
            validate_removed(RemovedParts(entities=["not_a_real_role"]), "spec_" + "0" * 32)
        assert exc_info.value.code == "request_invalid"

    def test_removed_unrecognized_facet_refuses_request_invalid(self):
        with pytest.raises(EngineInputError) as exc_info:
            validate_removed(RemovedParts(facets=["not_a_real_axis"]), "spec_" + "0" * 32)
        assert exc_info.value.code == "request_invalid"


# Search resubmit rejects exact bindings in both identities before graph access.


class TestExactBindingRefusal:
    def test_exact_slot_in_posted_identity_refuses_before_any_graph_read(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_spec_id, base_identity = make_base(GDP_SHAPE, slots=[_PURPOSE_SLOT], entities=entities)
        exact_slot = base_identity.slots[0].model_copy(
            update={"binding": IdentityBindingExact(ref="CRSPurposeCode100")}
        )
        edited_identity = BaseIdentity.model_validate(
            {**base_identity.model_dump(exclude_none=True), "slots": [exact_slot]}
        )
        inp = ResubmitIdentityPayload(
            identity=edited_identity,
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "exact_binding_unsupported"
        assert exc_info.value.status == 501
        assert exc_info.value.pointer == "/input/identity/slots/0/binding"

    def test_exact_slot_in_base_identity_alone_refuses_before_hash_verification(self):
        """Reject an exact base binding before checking its deliberately wrong base hash."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        _, base_identity = make_base(GDP_SHAPE, slots=[_PURPOSE_SLOT], entities=entities)
        exact_base_slot = base_identity.slots[0].model_copy(
            update={"binding": IdentityBindingExact(ref="CRSPurposeCode100")}
        )
        base_identity_with_exact = BaseIdentity.model_validate(
            {**base_identity.model_dump(exclude_none=True), "slots": [exact_base_slot]}
        )
        inp = ResubmitIdentityPayload(
            identity=_edited_identity(GDP_SHAPE, [_PURPOSE_SLOT], entities=entities),
            base_spec_id="spec_" + "f" * 32,  # wrong hash -- refused before verification
            base_identity=base_identity_with_exact,
            resolved_stat_var_dcids=[_GDP_SV_DCID],
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        assert exc_info.value.code == "exact_binding_unsupported"
        assert exc_info.value.status == 501
        assert exc_info.value.pointer == "/input/base_identity/slots/0/binding"


class TestRemovedValidation:
    def test_canonical_facet_and_entity_categories_both_accepted(self):
        result = validate_removed(
            RemovedParts(facets=["provenance"], entities=["origin", "destination"]),
            "spec_" + "0" * 32,
        )
        assert result == (frozenset({"provenance"}), frozenset({"origin", "destination"}))

    def test_order_within_a_category_does_not_matter(self):
        base_spec_id = "spec_" + "0" * 32
        forward = validate_removed(RemovedParts(entities=["origin", "destination"]), base_spec_id)
        backward = validate_removed(RemovedParts(entities=["destination", "origin"]), base_spec_id)
        assert forward == backward

    def test_none_returns_empty_sets(self):
        assert validate_removed(None, None) == (frozenset(), frozenset())

    def test_vacuous_removed_returns_empty_sets_without_requiring_base(self):
        """An explicitly posted but empty `removed` ({} on the wire) is a
        no-op, same as omitting the field -- it never requires
        base_spec_id."""
        assert validate_removed(RemovedParts(), None) == (frozenset(), frozenset())

    def test_removed_max_length_caps(self):
        """`RemovedParts` accepts at most 3 facets and 16 entities."""
        with pytest.raises(ValidationError):
            RemovedParts(facets=["a", "b", "c", "d"])
        with pytest.raises(ValidationError):
            RemovedParts(entities=["r"] * 17)


class TestBaseIdentityCaps:
    def test_slots_over_max_length_raises(self):
        """`BaseIdentity` accepts at most 64 slots."""
        oversized = [_value_slot("what", f"prop{i}", f"prop{i}", "v", "v") for i in range(65)]
        with pytest.raises(ValidationError):
            BaseIdentity(shape=shape_for("s"), slots=oversized)

    def test_entities_over_max_length_raises(self):
        """`BaseIdentity` accepts at most 16 entity roles."""
        oversized = [subject_entity(f"country/{i:03d}") for i in range(17)]
        with pytest.raises(ValidationError):
            BaseIdentity(shape=shape_for("s"), slots=[], entities=oversized)
