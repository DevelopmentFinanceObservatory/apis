"""Origin catalogue members on the Search wire: provenance refs, shape origins, the
`withheld_by_registry` and `origin_catalogue_stale` warnings, and the SDMX role-series guard."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import get_args

import httpx
import pytest

from dc_graph import ArcTarget
from materialise.provenance_doc import read_source_citations
from origin_catalogue import (
    BASE,
    UN_SYSTEM,
    OriginId,
    UnBlockedPlaces,
    UnReadFailure,
    current_catalogue,
    note_un_blocked_places,
    recording_un_blocked_places,
)
from origin_catalogue.context import un_unavailability_record
from search.engine.clients.sdmx import (
    LiveSdmxClient,
    UpstreamVariableRoutedToSdmx,
    assert_dfo_role_variables,
)
from search.engine.clients.sdmx_cache import CachingSdmxClient
from search.engine.concurrency import scoped_concurrency
from search.engine.core import _with_stale_catalogue_warning
from search.engine.degraded_reads import recording_degraded_reads
from search.engine.departures import warning_key
from search.engine.errors import GraphInfraError
from search.engine.events import ResponseEvent
from search.engine.region import option_coverage, resubmit, withheld
from search.engine.region.codes import (
    NO_MEMBER_ASSERTS_BOUND_PROPERTIES,
    NO_MEMBER_HAS_DATA_AT_PLACE,
    OBSERVATIONS_NO_POINTS,
    ORIGIN_CATALOGUE_STALE,
    UN_SYSTEM_UNAVAILABLE,
    WITHHELD_BY_REGISTRY,
)
from search.engine.region.facets import _resolved_facets
from search.engine.region.observations import dispatch
from search.engine.region.resubmit_request import _rebind_source_slot
from search.engine.region.result import RegionResult
from search.engine.region.un_system import NO_LABELS, un_unavailable_warning
from search.engine.region.withheld import with_withheld_warning
from search.engine.shape import ShapeDraft
from search.engine.spec_build import build_shape_model
from search.heading.departure_fact import DepartureFact
from search.models import (
    BindingValue,
    DataCommonsId,
    DataCommonsRef,
    Entity,
    EntityRoleDirectional,
    EntityRoleSubject,
    GraphRef,
    ResolvedFacet,
    ResubmitIdentityPayload,
    SeriesFacetRef,
    StatVar,
    omitted_observations,
)
from search.models import SourceRef as WireSourceRef
from tests.engine._base_identity import make_identity, subject_entity
from tests.engine._builders import conjoin_spec, gref, shape_five_tuple
from tests.engine._fail_open_deadline import (
    FAIL_OPEN_REQUESTS,
    assert_fail_open_deadline,
    request_deadline,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph
from tests.fixtures.provenance import arc_targets
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)
from tests.render._helpers import _unbound_source_slot

_WDI = "dc/base/WorldDevelopmentIndicators"
_WDI_FACET = SeriesFacetRef(
    ref=_WDI,
    label="World Development Indicators",
    source=WireSourceRef(ref="dc/s/WorldBank", label="World Bank"),
    data_commons=DataCommonsRef(id="base", label="Data Commons by Google"),
    documentation=f"/api/graph/documentation/{_WDI}",
)


def test_data_commons_id_mirrors_the_catalogue_origins():
    assert get_args(DataCommonsId) == get_args(OriginId.__value__)


# ---------------------------------------------------------------------------
# Shape: data_commons and upstream
# ---------------------------------------------------------------------------


def _draft(members: list[str]) -> ShapeDraft:
    return ShapeDraft(
        shape_id="shape",
        label="Under-five mortality",
        pop_type_dcid="Person",
        meas_prop_dcid="mortalityRate",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={member: {} for member in members},
    )


def test_shape_lists_every_origin_holding_a_member_in_wire_order(bind_catalogue):
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row("dc/base/WHO", BASE),
                provenance_row("undata/p/WHO", UN_SYSTEM, segment="who"),
            ],
            dfo_variables=["Count_Person"],
            base_variables={"Count_Person": (["dc/base/WHO"], [])},
            un_mirror=un_mirror(
                {"undata/who/MORT": un_variable("undata/who/SERIES-MORT", name="Mortality")}
            ),
        )
    )
    members = ["undata/who/MORT", "Count_Person", "NotListed"]
    shape = build_shape_model(_draft(members), [], {}, member_count=3, members=members)
    assert shape.data_commons == [
        DataCommonsRef(id="dfo", label="Development Finance Observatory"),
        DataCommonsRef(id="base", label="Data Commons by Google"),
        DataCommonsRef(id="un_system", label="UN System Data Commons"),
    ]
    assert shape.upstream is None


def test_shape_of_a_member_no_table_lists_names_base(bind_catalogue):
    bind_catalogue(restrictive_catalogue(withheld_variables=["Withheld"]))
    listed_nowhere = build_shape_model(
        _draft(["NotListed"]), [], {}, member_count=1, members=["NotListed"]
    )
    withheld = build_shape_model(_draft(["Withheld"]), [], {}, member_count=1, members=["Withheld"])
    assert listed_nowhere.data_commons == [
        DataCommonsRef(id="base", label="Data Commons by Google")
    ]
    assert withheld.data_commons == []


def test_an_imputed_answer_names_the_upstream_source_of_its_own_members(bind_catalogue):
    """#1056: the recall pool holds a reported and an imputed member from different
    sources, and the answer serves the imputed one, so the shape names the imputed
    member's source and Data Commons. Read over the pool, it would name none."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row("dc/base/Reported", BASE, source="dc/s/Reported"),
                provenance_row("dc/base/Imputed", BASE, source="dc/s/Imputed"),
            ],
            base_variables={
                "Amount_Reported": (["dc/base/Reported"], []),
                "AttributedAmount_Imputed": (["dc/base/Imputed"], []),
            },
        )
    )
    pool = _draft(["Amount_Reported", "AttributedAmount_Imputed"])

    imputed = build_shape_model(pool, [], {}, member_count=1, members=["AttributedAmount_Imputed"])
    both = build_shape_model(pool, [], {}, member_count=2, members=list(pool.sv_arc_facts or ()))

    assert imputed.upstream is not None
    assert imputed.upstream.source.ref == "dc/s/Imputed"
    assert imputed.upstream.data_commons.id == "base"
    assert both.upstream is None


def test_shape_before_materialise_states_no_origin(bind_catalogue):
    """A shape built before materialise (`members=None`) states neither origin field,
    whatever its recall pool holds."""
    bind_catalogue(restrictive_catalogue(dfo_variables=["Count_Person"]))
    shape = build_shape_model(_draft(["Count_Person"]), [], {}, member_count=1, members=None)
    assert shape.data_commons == []
    assert shape.upstream is None


# ---------------------------------------------------------------------------
# resolved.facets.provenance
# ---------------------------------------------------------------------------


def test_resolved_provenance_carries_the_shared_series_members():
    facets = _resolved_facets([{"provenance": _WDI_FACET}, {"provenance": _WDI_FACET}], {})
    assert facets["provenance"] == ResolvedFacet(
        basis="fallback",
        ref=_WDI,
        label="World Development Indicators",
        source=_WDI_FACET.source,
        data_commons=_WDI_FACET.data_commons,
        documentation=_WDI_FACET.documentation,
    )


def test_resolved_provenance_per_series_drops_the_members():
    other = SeriesFacetRef(ref="dc/base/Other")
    facets = _resolved_facets([{"provenance": _WDI_FACET}, {"provenance": other}], {})
    assert facets["provenance"] == ResolvedFacet(basis="fallback", selection="per_series")


def test_resolved_facet_refuses_source_without_ref():
    with pytest.raises(ValueError, match="alongside ref"):
        ResolvedFacet(basis="fallback", selection="per_series", source=_WDI_FACET.source)


# ---------------------------------------------------------------------------
# withheld_by_registry
# ---------------------------------------------------------------------------


class _WithheldGraph:
    """Answers `withheld` and labels it from `labels`, or fails. With `stall` set, each
    re-read first waits up to 5 s for it, and with `label_stall` set, each label read. Each
    stall first calls `on_stall`, from the reading thread. `finished` is set once a stall's
    wait returns, so it reads False for an answer that did not wait for the stalled read."""

    def __init__(
        self,
        withheld: tuple[str, ...] = ("dc/base/Excluded",),
        fail: bool = False,
        stall: threading.Event | None = None,
        labels: dict[str, str] | None = None,
        label_stall: threading.Event | None = None,
        on_stall: Callable[[], None] | None = None,
    ):
        self.calls: list[tuple[list[str], list[str]]] = []
        self._labels = labels if labels is not None else {"dc/base/Excluded": "Excluded Survey"}
        self.deadlines: list[float | None] = []
        self._withheld = withheld
        self._fail = fail
        self._stall = stall
        self._label_stall = label_stall
        self._on_stall = on_stall
        self.finished = threading.Event()

    def withheld_provenances(self, variables, entities, *, deadline=None):
        self.calls.append((list(variables), list(entities)))
        self.deadlines.append(deadline)
        if self._stall is not None:
            self._announce_stall()
            self._stall.wait(5)
            self.finished.set()
        if self._fail:
            raise GraphInfraError("read failed")
        return self._withheld

    def _announce_stall(self) -> None:
        if self._on_stall is not None:
            self._on_stall()

    def node_arc_targets_many(self, dcids, *, properties, deadline=None):
        if self._label_stall is not None:
            self._announce_stall()
            self._label_stall.wait(5)
            self.finished.set()
        return {
            dcid: {"name": (ArcTarget(value=self._labels[dcid]),)} if dcid in self._labels else None
            for dcid in dcids
        }


class _CutWhenStalled:
    """Stands in for `withheld._await_bounded`: it cuts the await when a read reports its
    stall through `stalled`, where the real one cuts it at a clock time. A read that returns
    first is answered as usual. The cut leaves the stalled thread running, as the real one
    does. The real bound is tested in `tests/engine/test_request_deadline_clock.py`."""

    def __init__(self) -> None:
        self.deadlines: list[float] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stalled: asyncio.Event | None = None

    def stalled(self) -> None:
        """Called from the stalled read's thread."""
        assert self._loop is not None and self._stalled is not None
        self._loop.call_soon_threadsafe(self._stalled.set)

    async def __call__(self, coro, *, deadline: float):
        self.deadlines.append(deadline)
        self._loop = asyncio.get_running_loop()
        self._stalled = asyncio.Event()
        read = asyncio.ensure_future(coro)
        cut = asyncio.ensure_future(self._stalled.wait())
        await asyncio.wait({read, cut}, return_when=asyncio.FIRST_COMPLETED)
        cut.cancel()
        if read.done():
            return read.result()
        read.cancel()
        raise TimeoutError("stalled read cut")


def _subject(dcid: str) -> Entity:
    return Entity(ref=GraphRef(ref=dcid, label=dcid), role=EntityRoleSubject())


def _no_data(entities: list[Entity]) -> RegionResult:
    spec = conjoin_spec(shape_five_tuple("Person", "count"), "x", entities=entities)
    return RegionResult(
        variable_text="population",
        status="no_data",
        specs=(),
        no_data_reason="no_observations",
        warnings=(),
        timing_by_step={},
        flagged_match=(spec,),
        gate=NO_MEMBER_HAS_DATA_AT_PLACE,
        elected_stat_vars=[gref("Count_Person_A"), gref("Count_Person_B")],
    )


async def _with_warning_twice(
    first: RegionResult, second: RegionResult, graph: object, *, deadline: float | None = None
):
    async with scoped_concurrency():
        return (
            await with_withheld_warning(first, graph, deadline=deadline),
            await with_withheld_warning(second, graph, deadline=deadline),
        )


def test_withheld_warning_names_the_withheld_provenances_once_per_request():
    graph = _WithheldGraph()
    result = _no_data([_subject("country/KEN"), _subject("country/UGA")])

    first, second = asyncio.run(_with_warning_twice(result, result, graph))

    assert graph.calls == [(["Count_Person_A", "Count_Person_B"], ["country/KEN", "country/UGA"])]
    (warning,) = first.warnings
    assert warning.code == WITHHELD_BY_REGISTRY
    assert "dc/base/Excluded" in warning.message
    assert second.warnings == ()
    assert first.departure_facts == {
        warning_key(warning): DepartureFact(
            code="entity_unavailable",
            binding_ref="where:",
            asked="country/KEN, country/UGA",
            entities=("country/KEN", "country/UGA"),
            sources=("Excluded Survey",),
        )
    }


def test_a_withheld_source_no_label_names_is_left_out_of_the_departure():
    graph = _WithheldGraph(withheld=("dc/base/Excluded", "dc/base/Unnamed"))

    first, _ = asyncio.run(_with_warning_twice(*[_no_data([_subject("country/KEN")])] * 2, graph))

    (fact,) = first.departure_facts.values()
    assert fact.sources == ("Excluded Survey",)


def test_an_unlisted_withheld_source_is_named_as_its_citation_names_it():
    """A base provenance node is named for its import, so the departure names it by its parent
    Dataset, the name its citation carries, read under the re-read's fail-open bound."""
    who = "dc/base/WHO_GHO"
    graph = _WithheldGraph(withheld=(who,), labels={})
    arcs = {
        who: {"typeOf": ("Provenance",), "name": ("WHO_GHO",), "isPartOf": ("dc/d/WHO_GHO",)},
        "dc/d/WHO_GHO": {"typeOf": ("Dataset",), "name": ("WHO Global Health Observatory",)},
    }
    deadlines: list[float | None] = []

    def node_arcs_many(dcids, *, properties, deadline=None):
        return {dcid: arcs.get(dcid) for dcid in dcids}

    def node_arc_targets_many(dcids, *, properties, deadline=None):
        deadlines.append(deadline)
        return arc_targets(node_arcs_many(dcids, properties=properties), arcs)

    graph.node_arcs_many = node_arcs_many
    graph.node_arc_targets_many = node_arc_targets_many
    deadline = time.monotonic() + 60
    started = time.monotonic()

    first, _ = asyncio.run(
        _with_warning_twice(*[_no_data([_subject("country/KEN")])] * 2, graph, deadline=deadline)
    )

    (fact,) = first.departure_facts.values()
    (read_deadline,) = deadlines
    assert_fail_open_deadline(
        read_deadline, requested=deadline - withheld._HANDOFF_RESERVE_S, started=started
    )
    cited = read_source_citations(graph, [who], catalogue=current_catalogue())[who]
    assert fact.sources == (cited.name,) == ("WHO Global Health Observatory",)


def test_withheld_warning_skips_role_bearing_answers_and_definite_parts():
    graph = _WithheldGraph()
    donor = Entity(
        ref=GraphRef(ref="country/USA", label="USA"),
        role=EntityRoleDirectional(role=gref("origin"), axis="origin"),
    )
    role_bearing = _no_data([_subject("country/KEN"), donor])
    definite = dataclasses.replace(_no_data([_subject("country/KEN")]), status="definite")

    for result in (role_bearing, definite):
        (out, _) = asyncio.run(_with_warning_twice(result, result, graph))
        assert out.warnings == ()
    assert graph.calls == []


def test_withheld_warning_skips_no_data_answers_with_another_reason():
    # A qualifier refusal keeps the elected shape as `flagged_match`, but the shape
    # may hold admitted data at the place, so no re-read runs and nothing is claimed.
    graph = _WithheldGraph()
    base = _no_data([_subject("country/KEN")])
    qualifier_refusal = dataclasses.replace(base, no_data_reason="qualifier_not_bound", gate=None)
    no_slice = dataclasses.replace(
        base, no_data_reason="no_matching_slice", gate=NO_MEMBER_ASSERTS_BOUND_PROPERTIES
    )

    for result in (qualifier_refusal, no_slice):
        (out, _) = asyncio.run(_with_warning_twice(result, result, graph))
        assert out.warnings == ()
    assert graph.calls == []


@pytest.mark.parametrize("fail", [False, True], ids=["nothing-withheld", "read-fails"])
def test_withheld_warning_absent_when_nothing_is_withheld_or_the_read_fails(fail):
    """A failed re-read leaves the answer unwarned and notes the lost read, so no cache
    stores an answer that may have been withheld."""
    graph = _WithheldGraph(withheld=(), fail=fail)
    with recording_degraded_reads() as lost:
        (out, _) = asyncio.run(
            _with_warning_twice(_no_data([_subject("country/KEN")]), _no_data([]), graph)
        )
    assert out.warnings == ()
    assert lost.degraded is fail


def test_withheld_warning_reads_nothing_without_the_re_read():
    result = _no_data([_subject("country/KEN")])
    assert asyncio.run(with_withheld_warning(result, object(), deadline=None)) is result


def test_a_stalled_withheld_re_read_leaves_the_answer_unwarned_before_the_deadline(
    caplog, monkeypatch
):
    cut = _CutWhenStalled()
    monkeypatch.setattr(withheld, "_await_bounded", cut)
    stall = threading.Event()
    graph = _WithheldGraph(stall=stall, on_stall=cut.stalled)
    result = _no_data([_subject("country/KEN")])
    deadline = time.monotonic() + 1.0

    async def run() -> tuple[RegionResult, bool]:
        # Released here, since `asyncio.run` waits for the stalled read's thread on exit.
        try:
            out, _ = await _with_warning_twice(result, result, graph, deadline=deadline)
            return out, graph.finished.is_set()
        finally:
            stall.set()

    with (
        caplog.at_level(logging.INFO, logger="search.engine.region.withheld"),
        recording_degraded_reads() as lost,
    ):
        out, read_finished = asyncio.run(run())

    assert not read_finished, "the answer waited for the stalled re-read"
    assert out.warnings == ()
    assert lost.reads == ["withheld re-read"]
    bound = deadline - withheld._HANDOFF_RESERVE_S
    assert graph.deadlines == [bound]
    assert cut.deadlines == [bound]
    assert any("re-read skipped" in r.getMessage() for r in caplog.records)


class _RecordingAwait:
    """Wraps `withheld._await_bounded`, keeping the deadline each await was given."""

    def __init__(self) -> None:
        self.deadlines: list[float] = []
        self._await_bounded = withheld._await_bounded

    async def __call__(self, coro, *, deadline: float):
        self.deadlines.append(deadline)
        return await self._await_bounded(coro, deadline=deadline)


@FAIL_OPEN_REQUESTS
def test_the_re_read_and_the_label_read_each_run_under_the_fail_open_budget(
    request_in_s, monkeypatch
):
    """Both reads fail open, so each gets the budget from its own start, or the request
    deadline less the handoff reserve when that ends first."""
    awaited = _RecordingAwait()
    monkeypatch.setattr(withheld, "_await_bounded", awaited)
    graph = _WithheldGraph()
    result = _no_data([_subject("country/KEN")])
    requested = request_deadline(request_in_s)
    bound = None if requested is None else requested - withheld._HANDOFF_RESERVE_S
    started = time.monotonic()

    out, _ = asyncio.run(_with_warning_twice(result, result, graph, deadline=requested))

    assert out.warnings, "the re-read found the withheld provenances"
    (re_read,) = graph.deadlines
    assert_fail_open_deadline(re_read, requested=bound, started=started)
    assert len(awaited.deadlines) == 2
    for await_deadline in awaited.deadlines:
        assert_fail_open_deadline(await_deadline, requested=bound, started=started)


def test_a_stalled_source_label_read_keeps_the_warning_without_its_sources(caplog, monkeypatch):
    cut = _CutWhenStalled()
    monkeypatch.setattr(withheld, "_await_bounded", cut)
    label_stall = threading.Event()
    graph = _WithheldGraph(label_stall=label_stall, on_stall=cut.stalled)
    result = _no_data([_subject("country/KEN")])
    deadline = time.monotonic() + 1.0

    async def run() -> tuple[RegionResult, bool]:
        try:
            out, _ = await _with_warning_twice(result, result, graph, deadline=deadline)
            return out, graph.finished.is_set()
        finally:
            label_stall.set()

    with (
        caplog.at_level(logging.INFO, logger="search.engine.region.withheld"),
        recording_degraded_reads() as lost,
    ):
        out, read_finished = asyncio.run(run())

    assert not read_finished, "the answer waited for the stalled label read"
    assert lost.reads == ["withheld source labels"]
    (warning,) = out.warnings
    assert warning.code == WITHHELD_BY_REGISTRY
    (fact,) = out.departure_facts.values()
    assert (fact.code, fact.sources) == ("entity_unavailable", ())
    # The re-read and the label read each ran under the request deadline less the reserve.
    assert cut.deadlines == [deadline - withheld._HANDOFF_RESERVE_S] * 2
    assert any("source labels skipped" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# origin_catalogue_stale
# ---------------------------------------------------------------------------


def _response_event() -> ResponseEvent:
    from tests.engine._harness import make_request, offline_resolve

    return ResponseEvent(response=offline_resolve(make_request("population of Kenya")))


def test_stale_catalogue_adds_a_diagnostics_warning(bind_catalogue):
    event = _response_event()
    bind_catalogue(
        restrictive_catalogue(build_id="old-build", built_at=datetime(2020, 1, 1, tzinfo=UTC))
    )

    warned = _with_stale_catalogue_warning(event)

    assert isinstance(warned, ResponseEvent)
    (warning,) = [
        w for w in warned.response.diagnostics.warnings if w.code == ORIGIN_CATALOGUE_STALE
    ]
    assert "old-build" in warning.message
    assert "2020-01-01" in warning.message


def test_fresh_catalogue_passes_the_response_unchanged():
    event = _response_event()
    assert _with_stale_catalogue_warning(event) is event


# ---------------------------------------------------------------------------
# SDMX role-series guard
# ---------------------------------------------------------------------------


def test_role_series_guard_refuses_upstream_variables(bind_catalogue):
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("dc/base/WHO", BASE)],
            dfo_variables=["Shared_Var", "DFO_Flow"],
            base_variables={
                "Base_Var": (["dc/base/WHO"], []),
                "Shared_Var": (["dc/base/WHO"], []),
            },
        )
    )
    assert_dfo_role_variables(["DFO_Flow", "Shared_Var", "Unlisted"])
    with pytest.raises(UpstreamVariableRoutedToSdmx, match="Base_Var"):
        assert_dfo_role_variables(["DFO_Flow", "Base_Var"])
    with pytest.raises(UpstreamVariableRoutedToSdmx, match="undata/who/MORT"):
        assert_dfo_role_variables(["undata/who/MORT"])


def test_role_series_guard_passes_variables_the_permissive_catalogue_does_not_place():
    assert_dfo_role_variables(["Anything", "OECD_CRS_Flow"])


def _unreachable_sdmx() -> LiveSdmxClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"an upstream variable reached the SDMX wire: {request.url}")

    return LiveSdmxClient(base="http://sdmx.invalid", transport=httpx.MockTransport(handler))


def test_every_sdmx_entry_point_refuses_a_un_variable_before_the_wire():
    live = _unreachable_sdmx()
    cached = CachingSdmxClient(live)
    un = ["undata/who/MORT"]

    for client in (live, cached):
        with pytest.raises(UpstreamVariableRoutedToSdmx, match="undata/who/MORT"):
            client.role_series(
                variables=un,
                entities={"observationAbout": ["country/KEN"]},
                facet_ids={"F1"},
                max_rows=10,
                max_bytes=1000,
            )
        with pytest.raises(UpstreamVariableRoutedToSdmx, match="undata/who/MORT"):
            client.subject_observations(variables=un, places=["country/KEN"])
        with pytest.raises(UpstreamVariableRoutedToSdmx, match="undata/who/MORT"):
            client.availability_values(variables=un, component="observationAbout")


def test_role_series_refuses_a_base_only_variable_on_every_client(bind_catalogue):
    bind_catalogue(
        restrictive_catalogue(
            provenances=[provenance_row("dc/base/WHO", BASE)],
            base_variables={"Base_Var": (["dc/base/WHO"], [])},
        )
    )
    live = _unreachable_sdmx()

    for client in (live, CachingSdmxClient(live)):
        with pytest.raises(UpstreamVariableRoutedToSdmx, match="Base_Var"):
            client.role_series(
                variables=["Base_Var"],
                entities={"origin": ["country/USA"]},
                facet_ids={"F1"},
                max_rows=10,
                max_bytes=1000,
            )


# ---------------------------------------------------------------------------
# Source slot labels
# ---------------------------------------------------------------------------


def test_posted_source_slot_prefers_the_catalogue_label(bind_catalogue):
    row = dataclasses.replace(provenance_row(_WDI, BASE), name="World Development Indicators")
    bind_catalogue(FakeCatalogue(provenances={_WDI: row}))
    graph = FakeGraph(
        nodes={_WDI: {"label": "WDI graph name"}, "dc/base/Other": {"label": "Other"}}
    )

    (slot,) = _rebind_source_slot([_unbound_source_slot()], [_WDI], graph, deadline=None)
    (unlisted,) = _rebind_source_slot(
        [_unbound_source_slot()], ["dc/base/Other"], graph, deadline=None
    )
    (unnamed,) = _rebind_source_slot(
        [_unbound_source_slot()], ["dc/base/Unnamed"], graph, deadline=None
    )

    labels = [
        s.binding.value.ref.label
        for s in (slot, unlisted, unnamed)
        if isinstance(s.binding, BindingValue)
    ]
    assert labels == ["World Development Indicators", "Other", "dc/base/Unnamed"]


# ---------------------------------------------------------------------------
# un_system_unavailable on trailing streamed reads
# ---------------------------------------------------------------------------


def _refusing_read(result):
    """A read that meets a UN front-door refusal at `country/XYZ` and returns `result`."""

    def read(*_args, **_kwargs):
        note_un_blocked_places(["country/XYZ"])
        return result

    return read


def _refusal_warnings(warns) -> list[tuple[str, str | None]]:
    return [(w.message, w.spec_id) for w in warns if w.code == UN_SYSTEM_UNAVAILABLE]


def _failing_read(result):
    """A read whose UN part for `undata/who/MORT` at `country/KEN` fails, returning `result`."""

    def read(*_args, **_kwargs):
        record = un_unavailability_record()
        assert record is not None
        record.add_failure(
            UnReadFailure(variables=("undata/who/MORT",), places=("country/KEN",), detail="503")
        )
        return result

    return read


class _UnDown(FakeGraph):
    """A graph whose UN deployment is down: every label read fails, so a warning that read
    its variables' names would fail the answer it annotates."""

    def node_labels_batch(self, dcids):
        raise GraphInfraError("UN deployment unavailable")


def _variable_labels(warns) -> list[list[str]]:
    return [
        w.params["variable_labels"]
        for w in warns
        if w.code == UN_SYSTEM_UNAVAILABLE and w.params is not None
    ]


def test_trailing_observations_report_places_the_un_refused(monkeypatch):
    omitted = omitted_observations(OBSERVATIONS_NO_POINTS, "no points")
    monkeypatch.setattr(dispatch, "_build_observations", _refusing_read((omitted, None)))

    # The region's own record is already read when the trailing event runs.
    with recording_un_blocked_places():
        _events, warns = dispatch._observations_events(
            "spec-1",
            ["undata/who/MORT"],
            {"observationAbout": ["country/XYZ"]},
            graph=FakeGraph(),
            materialise_graph=None,
            materialise_sdmx=None,
            coverage=empty_coverage_snapshot(),
            variable_labels=NO_LABELS,
        )

    ((message, spec_id),) = _refusal_warnings(warns)
    assert "country/XYZ" in message
    assert spec_id == "spec-1"


def _slot_options_events(*, graph: FakeGraph, variable_labels):
    return option_coverage._stream_slot_options_events(
        "spec-2",
        ShapeDraft(
            shape_id="s",
            label="s",
            pop_type_dcid="Person",
            meas_prop_dcid="count",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            slot_keys=(),
            slot_taxonomy={},
        ),
        [],
        [],
        entities={},
        date_request=None,
        graph=graph,
        sdmx=FakeSdmxClient(),
        materialise_graph=None,
        coverage=empty_coverage_snapshot().index,
        deadline=None,
        facets_pinned=False,
        open_axes=frozenset(),
        variable_labels=variable_labels,
    )


def test_trailing_slot_options_report_places_the_un_refused(monkeypatch):
    monkeypatch.setattr(option_coverage, "_maybe_enrich_slot_options", _refusing_read(([], None)))

    with recording_un_blocked_places():
        _events, warns = _slot_options_events(graph=FakeGraph(), variable_labels=NO_LABELS)

    ((message, spec_id),) = _refusal_warnings(warns)
    assert "country/XYZ" in message
    assert spec_id == "spec-2"


def test_trailing_slot_options_name_a_failed_un_part_with_the_un_down(monkeypatch):
    """The warning names the failed variables from the Spec's members, with no label read
    of the deployment that just failed."""
    monkeypatch.setattr(option_coverage, "_maybe_enrich_slot_options", _failing_read(([], None)))

    _events, warns = _slot_options_events(
        graph=_UnDown(), variable_labels={"undata/who/MORT": "Mortality rate"}
    )

    assert _variable_labels(warns) == [["Mortality rate"]]


def test_trailing_reads_without_a_refusal_add_no_warning(monkeypatch):
    omitted = omitted_observations(OBSERVATIONS_NO_POINTS, "no points")
    monkeypatch.setattr(dispatch, "_build_observations", lambda *a, **k: (omitted, None))

    _events, warns = dispatch._observations_events(
        "spec-1",
        ["undata/who/MORT"],
        {"observationAbout": ["country/KEN"]},
        graph=FakeGraph(),
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=empty_coverage_snapshot(),
        variable_labels=NO_LABELS,
    )

    assert _refusal_warnings(warns) == []


def test_a_failed_un_part_of_a_mixed_read_is_reported_beside_refused_places():
    record = UnBlockedPlaces()
    record.add(["country/XYZ"])
    for variable in ("undata/who/MORT", "undata/who/UNNAMED"):
        record.add_failure(
            UnReadFailure(
                variables=(variable,),
                places=tuple(f"country/C{i}" for i in range(7)),
                detail="HTTP 503",
            )
        )

    warning = un_unavailable_warning(record, labels={"undata/who/MORT": "Mortality rate"})

    assert warning is not None
    assert warning.code == UN_SYSTEM_UNAVAILABLE
    assert "refused reads for country/XYZ" in warning.message
    assert "undata/who/MORT at country/C0" in warning.message
    assert "and 2 more" in warning.message
    # The params tell a refused place from a failed read without the message, and name each
    # variable from the labels the emitter holds, "" where it holds none.
    assert warning.params == {
        "places": ["country/XYZ"],
        "variables": ["undata/who/MORT", "undata/who/UNNAMED"],
        "variable_labels": ["Mortality rate", ""],
    }


def test_refused_places_alone_name_no_variable():
    record = UnBlockedPlaces()
    record.add(["country/XYZ"])

    warning = un_unavailable_warning(record, labels=NO_LABELS)

    assert warning is not None
    assert warning.params == {"places": ["country/XYZ"], "variables": [], "variable_labels": []}
    assert un_unavailable_warning(UnBlockedPlaces(), labels=NO_LABELS) is None


def test_trailing_observations_report_a_failed_un_part(monkeypatch):
    """The warning names the failed variable from the Spec's members, with no label read of
    the deployment that just failed."""
    omitted = omitted_observations(OBSERVATIONS_NO_POINTS, "no points")
    monkeypatch.setattr(dispatch, "_build_observations", _failing_read((omitted, None)))

    _events, warns = dispatch._observations_events(
        "spec-3",
        ["undata/who/MORT", "Count_Person"],
        {"observationAbout": ["country/KEN"]},
        graph=_UnDown(),
        materialise_graph=None,
        materialise_sdmx=None,
        coverage=empty_coverage_snapshot(),
        variable_labels={"undata/who/MORT": "Mortality rate", "Count_Person": "Population"},
    )

    ((message, spec_id),) = _refusal_warnings(warns)
    assert "undata/who/MORT at country/KEN failed" in message
    assert spec_id == "spec-3"
    assert _variable_labels(warns) == [["Mortality rate"]]


_RESUBMIT_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "Person",
    "measured_property": "count",
    "stat_type": "measuredValue",
}


def test_a_resubmit_promote_reports_places_the_un_refused(monkeypatch):
    promoted = dataclasses.replace(_no_data([_subject("country/XYZ")]), status="definite")
    monkeypatch.setattr(resubmit, "_resolve_standard_promote", _refusing_read(promoted))
    inp = ResubmitIdentityPayload(
        identity=make_identity(_RESUBMIT_SHAPE, [], entities=[subject_entity("country/XYZ")]),
    )

    region = resubmit.resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )

    ((message, _spec_id),) = _refusal_warnings(region.warnings)
    assert "country/XYZ" in message


def test_a_resubmit_names_a_failed_un_part_from_its_members_with_the_un_down(monkeypatch):
    """A resubmit's warning names the failed variable from the answer's members. The label
    read it once made went to the deployment that just failed, and failed the answer."""
    spec = conjoin_spec(shape_five_tuple("Person", "count"), "x", sv_dcid="undata/who/MORT")
    member = StatVar(
        ref=GraphRef(ref="undata/who/MORT", label="Mortality rate"),
        shape_id=spec.resolved.shape.shape_id,
        slot_values=[],
        constraint_properties=[],
    )
    spec = spec.model_copy(update={"resolved": spec.resolved.stating_members([member])})
    promoted = dataclasses.replace(_no_data([]), status="definite", specs=(spec,))
    monkeypatch.setattr(resubmit, "_resolve_standard_promote", _failing_read(promoted))
    inp = ResubmitIdentityPayload(
        identity=make_identity(_RESUBMIT_SHAPE, [], entities=[subject_entity("country/KEN")]),
    )

    region = resubmit.resolve_spec_resubmit(
        inp=inp,
        graph=_UnDown(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )

    assert _variable_labels(region.warnings) == [["Mortality rate"]]


@pytest.mark.parametrize(
    ("sv_dcid", "departs"),
    [
        pytest.param("undata/who/MORT", True, id="a_un_series_holds_no_values_there"),
        pytest.param("Count_Person", False, id="another_series_may_hold_values_there"),
    ],
)
def test_a_resubmit_states_the_refused_places_it_names_as_unavailable(
    monkeypatch, sv_dcid: str, departs: bool
):
    """The refused place departs as `entity_unavailable` on its role when every StatVar the
    answer reads is a UN one. A place it does not name departs nowhere."""
    spec = conjoin_spec(
        shape_five_tuple("Person", "count"),
        "x",
        sv_dcid=sv_dcid,
        entities=[
            Entity(ref=GraphRef(ref="country/XYZ", label="Xyzland"), role=EntityRoleSubject()),
            _subject("country/KEN"),
        ],
    )
    promoted = dataclasses.replace(_no_data([]), status="definite", specs=(spec,))
    monkeypatch.setattr(resubmit, "_resolve_standard_promote", _refusing_read(promoted))
    inp = ResubmitIdentityPayload(
        identity=make_identity(_RESUBMIT_SHAPE, [], entities=[subject_entity("country/XYZ")]),
    )

    region = resubmit.resolve_spec_resubmit(
        inp=inp,
        graph=FakeGraph(),
        sdmx=FakeSdmxClient(),
        coverage=empty_coverage_snapshot(),
        options=walk_options(FakeGraph()),
    )

    (warning,) = [w for w in region.warnings if w.code == UN_SYSTEM_UNAVAILABLE]
    expected = DepartureFact(
        code="entity_unavailable", binding_ref="where:", asked="Xyzland", entities=("Xyzland",)
    )
    assert region.departure_facts == ({warning_key(warning): expected} if departs else {})


@pytest.mark.parametrize(
    ("sv_dcid", "unserved", "late"),
    [
        pytest.param(
            "undata/who/MORT",
            (),
            ("Xyzland", "You asked for Xyzland. The data holds no values for it."),
            id="a_un_series_holds_no_values_there",
        ),
        pytest.param("Count_Person", (), None, id="another_series_may_hold_values_there"),
        pytest.param(
            "undata/who/MORT",
            (("observationAbout", "country/KEN"),),
            (
                "Kenya, Xyzland",
                "You asked for Kenya and Xyzland. The data holds no values for them.",
            ),
            id="the_role_states_one_line_naming_every_place",
        ),
        pytest.param(
            "undata/who/MORT",
            (("observationAbout", "country/XYZ"),),
            None,
            id="the_part_already_names_the_place",
        ),
    ],
)
def test_a_streamed_read_states_the_refused_places_on_done(
    sv_dcid: str, unserved: tuple[tuple[str, str], ...], late: tuple[str, str] | None
):
    """A streamed part ships before its observations are read, so when the UN front door
    refuses places in that read, `done` carries the part's complete departures
    (`late_departures`) as a buffered answer's part states them. That is one `entity_unavailable`
    for the role naming every place, and nothing when the list is the one the part
    shipped."""
    from search.engine.conjoin import assemble_region
    from search.engine.enrichment import late_departures, part_sources
    from search.engine.events import ObservationsEvent
    from search.models import Departure

    spec = conjoin_spec(
        shape_five_tuple("Person", "count"),
        "x",
        sv_dcid=sv_dcid,
        entities=[
            Entity(ref=GraphRef(ref="country/KEN", label="Kenya"), role=EntityRoleSubject()),
            Entity(ref=GraphRef(ref="country/XYZ", label="Xyzland"), role=EntityRoleSubject()),
        ],
    )
    spec = spec.model_copy(update={"resolved": spec.resolved.stating_unserved(unserved)})
    region = dataclasses.replace(_no_data([]), status="definite", specs=(spec,))
    record = UnBlockedPlaces()
    record.add(["country/XYZ"])
    warning = un_unavailable_warning(record, labels=NO_LABELS)
    assert warning is not None
    omitted = omitted_observations(OBSERVATIONS_NO_POINTS, "no points")

    response = assemble_region(
        region,
        extra_warnings=[],
        start_ms=0,
        engine_build="test",
        max_candidates=5,
    )
    (source,) = part_sources(response, answered=[region], streaming=[region])

    result = late_departures(
        source,
        events=(ObservationsEvent(spec_id=spec.spec_id, observations=omitted),),
        warnings=(warning.model_copy(update={"spec_id": spec.spec_id}),),
    )

    expected = (
        [Departure(code="entity_unavailable", binding_ref="where:", asked=late[0], text=late[1])]
        if late is not None
        else None
    )
    assert (result.departures if result is not None else None) == expected
