"""A widened activitySource remains open when the served GDP StatVar declares
GrossDomesticProduction. The resubmit reports that value in open_axis_facts.
"""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.models import GraphRef, ResubmitIdentityPayload
from tests.engine._base_identity import GDP_SHAPE, make_identity, subject_entity
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.engine.test_resubmit_standard_widen import (
    _ACTIVITY_SOURCE_BOUND,
    _ACTIVITY_SOURCE_UNBOUND,
    _GDP_ENTITY_DCID,
)
from tests.fixtures import FakeGraph


def test_a_widened_axis_the_served_member_still_declares_is_reported_open():
    """The widened activitySource stays unbound and reports the member's
    GrossDomesticProduction value as an open-axis fact."""
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    inp = ResubmitIdentityPayload(
        identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_UNBOUND], entities=entities),
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
    activity_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.ref == "activitySource"
    )
    assert activity_slot.binding.kind == "unbound"

    facts = spec.resolved.open_axis_facts
    assert facts is not None
    fact = next(f for f in facts if f.property == "activitySource")
    assert fact.varies is False
    assert fact.undeclared_count == 0
    # One served member, one declared value: the agreed value and the
    # served breadth both publish. No label is fetched for it, so it
    # falls back to the dcid, same as _tuple_ref's own dcid-as-label rule.
    assert fact.value == GraphRef(ref="GrossDomesticProduction", label="GrossDomesticProduction")
    assert fact.distinct_count == 1

    # The domain read survives on the region too, not just the wire Spec --
    # what a downstream same-shape merge would carry forward.
    assert region.open_axis_facts
    assert any(f.property == "activitySource" for f in region.open_axis_facts)


def test_a_request_with_no_open_axis_still_publishes_none():
    """A bound activitySource leaves no constraint axis open."""
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    inp = ResubmitIdentityPayload(
        identity=make_identity(GDP_SHAPE, [_ACTIVITY_SOURCE_BOUND], entities=entities),
    )
    region = resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )

    assert region.status == "definite"
    assert region.specs[0].resolved.open_axis_facts is None
    assert region.open_axis_facts == ()
