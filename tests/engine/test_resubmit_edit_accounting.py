"""Pins resubmit identity refusals, unit-request hoisting, and method narrowing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.errors import EngineInputError
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.engine.spec_id import build_spec_identity
from search.models import (
    BaseIdentity,
    IdentityBindingExact,
    IdentityBindingUnbound,
    IdentityBindingValue,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
    ResubmitIdentityPayload,
)
from tests.engine._base_identity import (
    GDP_SHAPE,
    make_identity,
    shape_for,
    subject_entity,
    where_slots_for,
)
from tests.engine._builders import unbound_slot as _unbound_slot
from tests.engine._builders import value_slot as _value_slot
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph

_PURPOSE_SLOT = _value_slot("how", "FlowPurpose", "sector/purpose", "DAC/Health", "Health (Total)")

_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"


def _edited_identity(shape, slots, *, entities=None, facets=None):
    return build_spec_identity(
        shape, slots, entities=entities, explicit_facets=facets, open_roles=()
    )[1]


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
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(graph, EngineOptions(include_observations=True)),
        )
        assert region.status == "definite"
        resolved = region.specs[0].resolved

        assert resolved.coverage.observation_count == 1
        sources = {s.dcid for s in resolved.trace.resolved_sources}
        assert sources == {_PROVENANCE_DCID}
        date_range = resolved.members[0].data_date_range
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


_INEXPRESSIBLE_UNIT_FACETS = {
    "unit": IdentityFacetValueRequest(request=IdentityUnitRequest(currency="XYZ", prices="current"))
}


class TestInexpressibleUnitHoistedBeforeGraphAccess:
    """A currency outside ISO 4217 raises `facet_binding_unsupported` (501) before graph reads."""

    def test_standard_promote_branch_refuses_before_any_graph_read(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
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
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=walk_options(FakeGraph(raise_on_call=True)),
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
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=walk_options(FakeGraph(raise_on_call=True)),
            )
        hoisted = exc_info.value

        assert hoisted.code == late.code
        assert hoisted.status == late.status
        assert hoisted.detail == late.detail
        assert late.pointer == "/input/facets/unit"
        assert hoisted.pointer == "/input/identity/facets/unit"


class TestRefusals:
    def test_duplicate_edited_slot_property_refuses_before_any_graph_read(self):
        """The request model permits duplicate slot properties; the identity
        core rejects them before resolution can silently widen the ask."""
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_identity = make_identity(
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
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=walk_options(FakeGraph(raise_on_call=True)),
            )
        assert exc_info.value.code == "request_invalid"
        assert exc_info.value.pointer == "/input/identity"


class TestExactBindingRefusal:
    def test_exact_slot_in_posted_identity_refuses_before_any_graph_read(self):
        entities = [subject_entity(_GDP_ENTITY_DCID)]
        base_identity = make_identity(GDP_SHAPE, slots=[_PURPOSE_SLOT], entities=entities)
        exact_slot = base_identity.slots[0].model_copy(
            update={"binding": IdentityBindingExact(ref="CRSPurposeCode100")}
        )
        edited_identity = BaseIdentity.model_validate(
            {**base_identity.model_dump(exclude_none=True), "slots": [exact_slot]}
        )
        inp = ResubmitIdentityPayload(
            identity=edited_identity,
        )
        with pytest.raises(EngineInputError) as exc_info:
            resolve_spec_resubmit(
                inp=inp,
                graph=FakeGraph(raise_on_call=True),
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
                options=walk_options(FakeGraph(raise_on_call=True)),
            )
        assert exc_info.value.code == "exact_binding_unsupported"
        assert exc_info.value.status == 501
        assert exc_info.value.pointer == "/input/identity/slots/0/binding"


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
