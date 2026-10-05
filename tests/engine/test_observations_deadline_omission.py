"""Deadline timeouts inside preview reads yield an `observations_timed_out`
omission instead of the builtin `TimeoutError` escaping `_build_observations`. Pipeline checkpoint timeouts propagate outside these catches; the
bulk `subject_observations` API has no deadline parameter.
"""

from __future__ import annotations

import asyncio

from search.engine.clients.graph import Facet
from search.engine.region.codes import (
    OBSERVATIONS_BULK_FETCH_FAILED,
    OBSERVATIONS_PROBE_FAILED,
    OBSERVATIONS_TIMED_OUT,
)
from series_coverage.schema import RoleNames
from series_coverage.shared import CoverageSnapshot
from tests.engine._fake_coverage import FakeCoverageService, empty_coverage_snapshot, patch_coverage
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    build_observations_planned,
    bulk_role_series_planned,
    harness_reader,
    sdmx_double,
    stub_preview_result,
)
from tests.fixtures import FakeGraph

# The deadline guard's `TimeoutError` matches the pipeline synthesis type.
_BUILTIN_TIMEOUT = TimeoutError

_SV = "TEST/DeadlineOmission_Flow"
_DONOR = "country/BRDonor"
_RECIPIENT = "country/BRReceiver"
_NODES: dict = {
    _SV: {
        "label": "Deadline Omission Flow",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Deadline Omission Flow"}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "observationProperties": {"nodes": [{"dcid": "provider"}, {"dcid": "recipient"}]},
        },
    },
    _DONOR: {"label": "Donor"},
    _RECIPIENT: {"label": "Recipient"},
}
_GRAPH = FakeGraph(nodes=_NODES, obs={})


def _facet(*, facet_id: str = "F1") -> Facet:
    return Facet(
        earliest_date="2020",
        latest_date="2020",
        obs_count=1,
        dates=["2020"],
        values=[1.0],
        provenance_id="dc/base/Src",
        facet_id=facet_id,
    )


def _obs_entities() -> dict[str, list[str]]:
    return {"provider": [_DONOR], "recipient": [_RECIPIENT]}


def _role_tuple() -> tuple[tuple[str, str], ...]:
    # "provider" < "recipient" lexicographically, so this is already the
    # sorted role tuple FakeSdmxClient/role_series key on.
    return (("provider", _DONOR), ("recipient", _RECIPIENT))


def _patch_role_coverage(monkeypatch, *, facet_id: str = "F1") -> CoverageSnapshot:
    """Build coverage rows for `_SV`'s donor-recipient pair. The role index is
    local; `role_series` performs the branch's network read.
    """
    index = patch_coverage(
        monkeypatch,
        [(_SV, facet_id, _DONOR, _RECIPIENT, [2020])],
        role_names={_SV: RoleNames(donor="provider", recipient="recipient")},
    )
    return FakeCoverageService(index).snapshot()


def test_deadline_synthesis_uses_the_same_builtin_type() -> None:
    """The deadline guard uses this Python's `asyncio.TimeoutError` type."""
    assert asyncio.TimeoutError is _BUILTIN_TIMEOUT


def test_per_pair_index_read_timeout_degrades_to_timed_out_omission(monkeypatch) -> None:
    """If the only pair times out, the preview returns a timed-out omission."""
    import search.engine.region.observations.fetch as region_fetch

    def _boom(*_args, **_kwargs):
        raise _BUILTIN_TIMEOUT("simulated per-pair deadline")

    monkeypatch.setattr(region_fetch, "materialise_preview", _boom)

    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": [_RECIPIENT]},
        graph=_GRAPH,
        materialise_graph=harness_reader([_RECIPIENT], stat_var=_SV),
        materialise_sdmx=sdmx_double(),
        sdmx=None,
        coverage=empty_coverage_snapshot(),
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_TIMED_OUT
    assert warn is not None
    assert warn.code == OBSERVATIONS_TIMED_OUT
    # One pair timed out; no pair failed transport.
    assert observations.omitted.dispatched == 1
    assert observations.omitted.hard_failed is None


def test_per_pair_all_fail_mixed_timeout_and_hard_failure_reports_probe_failed(
    monkeypatch,
) -> None:
    """When one pair times out and another fails transport, the omission is
    `observations_probe_failed` and reports 2 dispatched, 1 hard failure.
    """
    import search.engine.region.observations.fetch as region_fetch
    from materialise import GraphReadError

    timeout_entity = _RECIPIENT
    hard_fail_entity = "country/BRReceiverB"
    graph = FakeGraph(
        nodes={**_NODES, hard_fail_entity: {"label": "Recipient B"}},
        obs={},
    )

    def _mixed_failure(*args, **kwargs):
        request = args[0]
        if hard_fail_entity in request.role_bindings.get("observationAbout", ()):
            raise GraphReadError("simulated transport failure")
        raise _BUILTIN_TIMEOUT("simulated per-pair deadline")

    monkeypatch.setattr(region_fetch, "materialise_preview", _mixed_failure)

    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": [timeout_entity, hard_fail_entity]},
        graph=graph,
        materialise_graph=harness_reader([timeout_entity, hard_fail_entity], stat_var=_SV),
        materialise_sdmx=sdmx_double(),
        coverage=empty_coverage_snapshot(),
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_PROBE_FAILED
    assert observations.omitted.dispatched == 2
    assert observations.omitted.hard_failed == 1
    assert warn is not None
    assert warn.code == OBSERVATIONS_PROBE_FAILED


def test_per_pair_partial_timeout_warns_beside_delivered_series(monkeypatch) -> None:
    """A timed-out pair yields a warning beside the successful series."""
    import search.engine.region.observations.fetch as region_fetch

    first_entity = _RECIPIENT
    second_entity = "country/BRReceiverB"
    graph = FakeGraph(
        nodes={**_NODES, second_entity: {"label": "Recipient B"}},
        obs={},
    )

    def _one_times_out(*args, **kwargs):
        request = args[0]
        if second_entity in request.role_bindings.get("observationAbout", ()):
            raise _BUILTIN_TIMEOUT("simulated per-pair deadline")
        return stub_preview_result(
            sv=_SV,
            role_key="observationAbout",
            entity=first_entity,
            rows=(("2020", 5.0),),
            provenance="TEST/PpSrc",
        )

    monkeypatch.setattr(region_fetch, "materialise_preview", _one_times_out)

    observations, warn = build_observations_planned(
        [_SV],
        {"observationAbout": [first_entity, second_entity]},
        graph=graph,
        materialise_graph=harness_reader([first_entity, second_entity], stat_var=_SV),
        materialise_sdmx=sdmx_double(),
        coverage=empty_coverage_snapshot(),
    )
    assert observations.omitted is None
    assert len(observations.series) == 1
    assert warn is not None
    assert warn.code == OBSERVATIONS_TIMED_OUT
    assert "1" in warn.message


def test_bulk_role_series_narrowed_read_timeout_degrades_to_timed_out_omission(monkeypatch) -> None:
    """A timeout in the narrowed `role_series` read returns a timed-out
    omission. The coverage-built role index is a local lookup.
    """
    coverage = _patch_role_coverage(monkeypatch)
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple(): [_facet()]}},
        raise_series=_BUILTIN_TIMEOUT("simulated deadline"),
    )
    observations, warn = bulk_role_series_planned(
        [_SV], _obs_entities(), sdmx=sdmx, graph=_GRAPH, facet_filter=None, coverage=coverage
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_TIMED_OUT
    assert warn is not None
    assert warn.code == OBSERVATIONS_TIMED_OUT
    assert len(sdmx.series_calls) == 1  # the narrowed read timed out


def test_bulk_role_series_generic_failure_still_uses_bulk_fetch_failed(monkeypatch) -> None:
    """A transport error in `role_series` remains a bulk-fetch failure."""
    from search.engine.errors import GraphInfraError

    coverage = _patch_role_coverage(monkeypatch)
    sdmx = FakeSdmxClient(
        observations={_SV: {_role_tuple(): [_facet()]}},
        raise_series=GraphInfraError("simulated transport failure"),
    )
    observations, warn = bulk_role_series_planned(
        [_SV], _obs_entities(), sdmx=sdmx, graph=_GRAPH, facet_filter=None, coverage=coverage
    )
    assert observations.series == []
    assert observations.omitted is not None
    assert observations.omitted.reason == OBSERVATIONS_BULK_FETCH_FAILED
    assert warn is not None
    assert warn.code == OBSERVATIONS_BULK_FETCH_FAILED
