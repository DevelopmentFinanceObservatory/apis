"""Test source-filtered previews and full resolved-source provenance in both resubmit paths."""

from __future__ import annotations

import asyncio

import pytest

from origin_catalogue import current_catalogue
from search.engine.concurrency import scoped_concurrency
from search.engine.errors import EngineInputError
from search.engine.region import resolve_spec_resubmit
from search.engine.region.state import EngineOptions
from search.models import (
    BindingSet,
    BindingUnbound,
    BindingValue,
    CoverageExact,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    ProvenanceRef,
    ResubmitIdentityPayload,
    Slot,
    SlotKey,
    SlotValue,
)
from tests.engine._base_identity import (
    GDP_SHAPE,
    make_identity,
    subject_entity,
)
from tests.engine._builders import gref as _gref
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import materialise_double_from_obs, walk_options, with_member_arcs
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_slot_value(dcid: str) -> Slot:
    """Build a source slot with an invalid label. Resolution uses the graph label."""
    return Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingValue(value=SlotValue(ref=_gref(dcid, "IGNORED"), value_kind="source")),
    )


def _source_slot_set(dcids: list[str]) -> Slot:
    return Slot(
        key=SlotKey(axis="source", property=None, label="source"),
        binding=BindingSet(
            values=[SlotValue(ref=_gref(d, "IGNORED"), value_kind="source") for d in dcids]
        ),
    )


def _find_source_slot(slots: list[Slot]) -> Slot:
    return next(s for s in slots if s.key.axis == "source")


def _run_relaunch(region):
    """Run the streamed observations task in its required request scope (#624)."""

    async def _await() -> tuple:
        async with scoped_concurrency():
            task = region.observations_relaunch(region.specs[0].spec_id)
            return await task

    return asyncio.run(_await())


# Two provenance dcids supply facets for the same StatVar and entity.
_SOURCE_WDI = "dc/base/WorldDevelopmentIndicators"
_SOURCE_UNRELATED = "dc/base/India_NationalAccountsStatistics"

# GDP for India.


_GDP_SV_DCID = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GDP_ENTITY_DCID = "country/IND"
_GDP_PAIR = f"{_GDP_SV_DCID}|{_GDP_ENTITY_DCID}"
# Unconstrained dispatch picks the unrelated facet (2 observations vs WDI's 1).
# A posted WDI source binding returns WDI's own point.
_GDP_TWO_SOURCE_OBS = {
    _GDP_PAIR: [
        {
            "earliestDate": "2018",
            "latestDate": "2019",
            "obsCount": 2,
            "provenanceId": _SOURCE_UNRELATED,
            "importName": "India_NationalAccountsStatistics",
            "observations": [
                {"date": "2018", "value": 2.0},
                {"date": "2019", "value": 2.1},
            ],
        },
        {
            "earliestDate": "2020",
            "latestDate": "2020",
            "obsCount": 1,
            "provenanceId": _SOURCE_WDI,
            "importName": "WorldDevelopmentIndicators",
            "observations": [
                {"date": "2020", "value": 2.2},
            ],
        },
    ],
}


class _ScopeRecordingGraph(FakeGraph):
    """Records the bound catalogue's source scope at each facet read."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scopes: list[frozenset[str] | None] = []

    def observation_facets_batch(self, *args, **kwargs):
        self.scopes.append(current_catalogue().scope)
        return super().observation_facets_batch(*args, **kwargs)


def _resolve_standard(source_slot: Slot | None, graph: FakeGraph, *, stream: bool = False):
    # The declared activitySource key must be present, even when unbound.
    activity_source_slot = Slot(
        key=SlotKey(
            axis="how",
            property=_gref("activitySource", "activity source"),
            label="activity source",
        ),
        binding=BindingUnbound(),
    )
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    slots = [
        activity_source_slot,
        *([source_slot] if source_slot is not None else []),
    ]
    inp = ResubmitIdentityPayload(
        identity=make_identity(GDP_SHAPE, slots, entities=entities),
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_GDP_TWO_SOURCE_OBS)
    return resolve_spec_resubmit(
        inp=inp,
        graph=graph,
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            graph,
            EngineOptions(
                include_observations=not stream,
                stream_observations=stream,
                materialise_graph=with_member_arcs(materialise_graph, graph),
                materialise_sdmx=materialise_sdmx,
            ),
        ),
    )


class TestStandardPromoteSourceSwitch:
    def test_posted_source_binds_slot_and_filters_observations(self):
        """Binding WDI returns its point despite the unrelated facet's higher observation count."""
        graph = FakeGraph(obs=_GDP_TWO_SOURCE_OBS)
        region = _resolve_standard(_source_slot_value(_SOURCE_WDI), graph)
        assert region.status == "definite"
        spec = region.specs[0]
        source_slot = _find_source_slot(spec.resolved.slots)
        assert isinstance(source_slot.binding, BindingValue)
        assert source_slot.binding.value.ref.dcid == _SOURCE_WDI
        series = spec.resolved.observations.series[0]
        assert [p.date for p in series.points] == ["2020"]
        assert series.source_refs == [
            ProvenanceRef(dcid=_SOURCE_WDI, label="WorldDevelopmentIndicators"),
        ]

    def test_posted_source_dates_and_counts_the_bound_source(self):
        """Coverage counts the WDI point the preview serves, not the unrelated facet's two,
        and cites one source."""
        graph = FakeGraph(obs=_GDP_TWO_SOURCE_OBS)
        coverage = _resolve_standard(None, graph).specs[0].resolved.coverage
        assert isinstance(coverage, CoverageExact) and coverage.observation_count == 2

        region = _resolve_standard(_source_slot_value(_SOURCE_WDI), graph)

        coverage = region.specs[0].resolved.coverage
        assert isinstance(coverage, CoverageExact)
        assert coverage.observation_count == 1
        assert {d.label: d.count for d in coverage.dimensions or []}["sources"] == 1

    def test_resolved_sources_lists_the_pinned_source_alone(self):
        """A resubmit pinning WDI serves WDI alone, so `resolved_sources` lists it alone. The
        same definition with no pin serves and lists the unrelated source."""
        graph = FakeGraph(obs=_GDP_TWO_SOURCE_OBS)
        region = _resolve_standard(_source_slot_value(_SOURCE_WDI), graph)
        sources = [s.dcid for s in region.specs[0].resolved.trace.resolved_sources]
        assert sources == [_SOURCE_WDI]

        unpinned = _resolve_standard(None, graph)
        sources = [s.dcid for s in unpinned.specs[0].resolved.trace.resolved_sources]
        assert sources == [_SOURCE_UNRELATED]

    @pytest.mark.parametrize(
        ("source_slot", "scope"),
        [(None, None), (_source_slot_value(_SOURCE_WDI), frozenset({_SOURCE_WDI}))],
        ids=["unpinned", "pinned"],
    )
    def test_a_posted_source_scopes_every_facet_read_to_it(self, source_slot, scope):
        """The pin is a read scope (`resubmit_request.pinned_provenance_scope`), so admission
        drops other sources before DFO precedence and a base pin serves at a DFO place."""
        graph = _ScopeRecordingGraph(obs=_GDP_TWO_SOURCE_OBS)

        _resolve_standard(source_slot, graph)

        assert graph.scopes
        assert set(graph.scopes) == {scope}

    def test_no_source_posted_is_byte_identical_to_before_this_feature(self):
        graph = FakeGraph(obs=_GDP_TWO_SOURCE_OBS)
        region = _resolve_standard(None, graph)
        assert region.status == "definite"
        spec = region.specs[0]
        source_slot = _find_source_slot(spec.resolved.slots)
        assert isinstance(source_slot.binding, BindingUnbound)
        # Without a source binding, the 2-observation facet wins over WDI's 1.
        series = spec.resolved.observations.series[0]
        assert [p.date for p in series.points] == ["2018", "2019"]


class TestStandardPromoteSourceSwitchStreaming:
    """Exercise streamed resubmits through the observations relaunch task."""

    def test_stream_filters_identically_to_buffered(self):
        graph = FakeGraph(obs=_GDP_TWO_SOURCE_OBS)
        region = _resolve_standard(_source_slot_value(_SOURCE_WDI), graph, stream=True)
        assert region.specs[0].resolved.observations is None
        assert region.observations_relaunch is not None

        events, warns = _run_relaunch(region)
        assert warns == ()
        series = events[0].observations.series[0]
        assert [p.date for p in series.points] == ["2020"]
        assert series.source_refs == [
            ProvenanceRef(dcid=_SOURCE_WDI, label="WorldDevelopmentIndicators")
        ]

    def test_stream_no_source_posted_is_byte_identical_to_before_this_feature(self):
        graph = FakeGraph(obs=_GDP_TWO_SOURCE_OBS)
        region = _resolve_standard(None, graph, stream=True)
        events, warns = _run_relaunch(region)
        assert warns == ()
        series = events[0].observations.series[0]
        assert [p.date for p in series.points] == ["2018", "2019"]


# The unrelated source reports a census count; WDI's series declares no method.
_GDP_METHOD_OBS = {
    _GDP_PAIR: [
        {**_GDP_TWO_SOURCE_OBS[_GDP_PAIR][0], "measurementMethod": "Census"},
        _GDP_TWO_SOURCE_OBS[_GDP_PAIR][1],
    ],
}


def _resolve_standard_with_method(source_slot: Slot | None, method: IdentityFacetBinding):
    """Resubmit India's GDP with a posted source slot and a method pin."""
    activity_source_slot = Slot(
        key=SlotKey(
            axis="how",
            property=_gref("activitySource", "activity source"),
            label="activity source",
        ),
        binding=BindingUnbound(),
    )
    entities = [subject_entity(_GDP_ENTITY_DCID)]
    slots = [activity_source_slot, *([source_slot] if source_slot is not None else [])]
    inp = ResubmitIdentityPayload(
        identity=make_identity(
            GDP_SHAPE,
            slots,
            entities=entities,
            explicit_facets={"measurementMethod": method},
        ),
    )
    materialise_graph, materialise_sdmx = materialise_double_from_obs(_GDP_METHOD_OBS)
    return resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(obs=_GDP_METHOD_OBS),
        sdmx=FakeSdmxClient(observations={}, availability={}),
        coverage=empty_coverage_snapshot(),
        options=walk_options(
            FakeGraph(obs=_GDP_METHOD_OBS),
            EngineOptions(
                include_observations=True,
                materialise_graph=with_member_arcs(
                    materialise_graph, FakeGraph(obs=_GDP_METHOD_OBS)
                ),
                materialise_sdmx=materialise_sdmx,
            ),
        ),
    )


class TestStandardPromoteSourceSlotRefusal:
    """A posted source slot reaches the identity as the provenance facet pin
    (`spec_id._facets_member`), so the subject path's refusal judges it jointly with the
    other pins."""

    def test_a_pin_served_only_under_another_source_is_refused(self):
        with pytest.raises(EngineInputError) as exc_info:
            _resolve_standard_with_method(
                _source_slot_value(_SOURCE_WDI), IdentityBindingValue(ref="Census")
            )

        error = exc_info.value
        assert error.code == "facet_not_available"
        assert error.pointer == "/input/identity/facets/provenance"
        assert f"provenance: missing {_SOURCE_WDI}" in str(error)
        assert "measurementMethod: missing Census" in str(error)

    def test_a_pin_the_posted_source_carries_is_served(self):
        region = _resolve_standard_with_method(
            _source_slot_value(_SOURCE_WDI), IdentityFacetValueAbsent(ref=None)
        )

        assert region.status == "definite"
        series = region.specs[0].resolved.observations.series[0]
        assert [p.date for p in series.points] == ["2020"]

    def test_the_same_pin_without_a_source_slot_is_served(self):
        region = _resolve_standard_with_method(None, IdentityBindingValue(ref="Census"))

        assert region.status == "definite"
        series = region.specs[0].resolved.observations.series[0]
        assert [p.date for p in series.points] == ["2018", "2019"]
