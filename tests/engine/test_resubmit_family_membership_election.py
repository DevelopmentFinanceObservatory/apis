"""A named family's membership comes from the packaged index election. Real
CRS health-purpose members pin equal counts with no seed or a truncated
200-member seed, and tolerate an elected member missing from the graph.
"""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import ResubmitIdentityPayload
from tests.engine._axis_doubles import axis_confirming_reader
from tests.engine._harness import walk_options
from tests.engine.test_resubmit_edit_detection import (
    _PURPOSE_ROLLUP,
    _coverage_for,
    _elected_members,
    _health_base,
    _health_graph,
    _member_rows,
    _sdmx_for,
)


def test_family_noop_resubmit_with_a_truncated_seed_resolves_the_same_member_count_as_no_seed(
    monkeypatch,
):
    """A 200-member seed must not truncate the family returned by the packaged
    election. With or without that seed, the response retains the full count."""
    _base_spec_id, base_identity = _health_base(purpose=_PURPOSE_ROLLUP)
    member_dcids = _elected_members(_PURPOSE_ROLLUP)
    member_rows = _member_rows(member_dcids)
    graph = _health_graph(member_rows, places=["country/KEN"])
    sdmx = _sdmx_for(member_dcids, "country/KEN")
    materialise_graph = axis_confirming_reader(*member_dcids, graph=graph)

    truncated_seed = list(member_dcids)[:200]
    assert len(truncated_seed) < len(member_dcids), (
        "fixture drift: expected the real family to exceed the 200-item serving cap"
    )

    inp_no_seed = ResubmitIdentityPayload(identity=base_identity)
    inp_truncated_seed = ResubmitIdentityPayload(
        identity=base_identity,
    )

    coverage = _coverage_for(member_dcids, "country/KEN", monkeypatch)
    region_no_seed = resolve_spec_resubmit(
        inp=inp_no_seed,
        graph=graph,
        sdmx=sdmx,
        coverage=coverage,
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )
    region_truncated_seed = resolve_spec_resubmit(
        inp=inp_truncated_seed,
        graph=graph,
        sdmx=sdmx,
        coverage=coverage,
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region_no_seed.status == "definite"
    assert region_truncated_seed.status == "definite"
    no_seed_count = region_no_seed.specs[0].resolved.stat_vars.member_count
    truncated_seed_count = region_truncated_seed.specs[0].resolved.stat_vars.member_count
    assert no_seed_count == len(member_dcids)
    assert truncated_seed_count == no_seed_count


def test_an_elected_member_absent_from_the_live_graph_is_dropped_not_fatal(monkeypatch):
    """An elected member absent from the graph is dropped, leaving the other
    real family members available for resolution."""
    _base_spec_id, base_identity = _health_base(purpose=_PURPOSE_ROLLUP)
    member_dcids = _elected_members(_PURPOSE_ROLLUP)
    present_dcids = list(member_dcids)[:-1]
    absent_dcid = member_dcids[-1]
    assert absent_dcid not in present_dcids

    member_rows = _member_rows(present_dcids)
    graph = _health_graph(member_rows, places=["country/KEN"])
    sdmx = _sdmx_for(present_dcids, "country/KEN")
    # The graph reader's own "family" scan never learns about absent_dcid
    # either -- it was never a fixture node -- matching a real absent index
    # row, which the live graph simply has nothing for.
    materialise_graph = axis_confirming_reader(*present_dcids, graph=graph)

    inp = ResubmitIdentityPayload(identity=base_identity)
    region = resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=sdmx,
        coverage=_coverage_for(present_dcids, "country/KEN", monkeypatch),
        options=walk_options(graph, EngineOptions(materialise_graph=materialise_graph)),
    )

    assert region.status == "definite", (
        region.no_data_reason if region.status != "definite" else None
    )
    served_refs = {sv.ref.ref for sv in region.specs[0].resolved.members}
    assert absent_dcid not in served_refs
