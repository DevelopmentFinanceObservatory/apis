"""Pin offline extraction to 2026-06-23 so prompt-based fixture keys stay stable."""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import math
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import aclosing, contextmanager
from dataclasses import dataclass, field, replace
from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import httpx
from dfo_frame_grammar import Frame

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import GraphReader, PreviewResult
from materialise.facets import FacetSelection, probe_facets
from materialise.graph import Facet as MaterialiseFacet
from materialise.graph import SubjectPage
from materialise.rate_tables import RateTableCache
from materialise.window import Coverage as MaterialisePreviewCoverage
from search.engine import config
from search.engine import deadline as deadline_mod
from search.engine import envelope as envelope_mod
from search.engine import fanout as fanout_mod
from search.engine.binders import Binder
from search.engine.clients.graph import EngineGraphClient
from search.engine.clients.llm import SupportsLLM
from search.engine.clients.sdmx import EngineSdmxClient, RoleFacetIndex
from search.engine.core import resolve_async, resolve_events
from search.engine.errors import GraphInfraError
from search.engine.events import Event
from search.engine.extract import DateRequest
from search.engine.region.bound_unit_gate import UnitEvidence, plan_unit_evidence
from search.engine.region.convertibility import elected_unit_refs
from search.engine.region.facets import FacetAdmission, facet_admitted
from search.engine.region.observations.bulk_role_series import _bulk_role_series
from search.engine.region.observations.bulk_scope import BulkScope, _bulk_scope_series
from search.engine.region.observations.fetch import _build_observations
from search.engine.region.place_options import PlaceOptionsPool
from search.engine.region.result import FacetAxisFilter
from search.engine.region.state import EngineOptions
from search.engine.response_cache import ResponseCache
from search.engine.response_cache_memo import PartMemo
from search.engine.retrieval.facet_election import merge_role_candidates, select_role_candidate
from search.engine.retrieval.member_facts import MemberFacts
from search.engine.retrieval.read_plan import (
    BulkPlan,
    PairPlan,
    PlanRoles,
    PlanSeries,
    PlanYears,
    ReadPlan,
    minimal_constraint_svs,
    plan_pairs,
    plan_standard,
    role_years,
)
from search.engine.retrieval.role_scope import role_index_for_named_entities
from search.engine.role_text import DoorRole
from search.models import (
    BindingValue,
    BreadthDim,
    Coverage,
    CoverageBreadth,
    CoverageExact,
    GraphRef,
    RawTextInput,
    ResolveRequest,
    ResolveResponse,
    RoleAxis,
    Slot,
    SpecObservations,
    TimeWindow,
    Warning,
)
from search.observations import observation_settings
from series_coverage.index import CoverageIndex
from series_coverage.shared import CoverageSnapshot
from tests.datasets_api._graph_doubles import FakeMintGraphReader
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import PINNED_DATE, FakeGraph, FakeLLM
from tests.fixtures.provenance import facet_provenance
from tests.materialise._scope_availability import NO_SCOPE_READ
from tests.materialise.conftest import FakeGraphReader

_TEST_PREPOSITION_RE = re.compile(r"\b(from|to)\b", re.IGNORECASE)


def door_roles_for_test_query(query: str, names: Sequence[str]) -> dict[str, list[DoorRole]]:
    """Assign each test name to the `from` or `to` segment containing it.

    Repeated names keep each distinct axis in mention order. Names before any
    preposition get no role.
    """
    matches = list(_TEST_PREPOSITION_RE.finditer(query))
    segments: list[tuple[int, int, RoleAxis]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(query)
        axis: RoleAxis = "origin" if m.group(1).lower() == "from" else "destination"
        segments.append((start, end, axis))
    roles: dict[str, list[DoorRole]] = {}
    for name in names:
        pos = query.find(name)
        while pos != -1:
            for start, end, axis in segments:
                if start <= pos < end:
                    occurrences = roles.setdefault(name, [])
                    role = DoorRole(axis=axis, span=name)
                    if role not in occurrences:
                        occurrences.append(role)
                    break
            pos = query.find(name, pos + len(name))
    return roles


MATERIALISE_BASE_URL = "https://example.org/core/api"

_SDMX_CSV_CONTENT_TYPE = "application/vnd.sdmx.data+csv;version=2.0.0"

_CSV_PREFIX = ("STRUCTURE", "STRUCTURE_ID", "ACTION", "variableMeasured")
_CSV_TRAILING = (
    "unit",
    "measurementMethod",
    "observationPeriod",
    "provenance",
    "TIME_PERIOD",
    "OBS_VALUE",
    "scalingFactor",
    "facetId",
)

PairKey = tuple[str, tuple[tuple[str, str], ...]]
"""A `(stat_var, role_tuple)` key; each item is `(role, entity_dcid)`.
A standard pair has one `observationAbout` entity.
"""


def sdmx_double(
    csv_body: str = "", *, base_url: str = MATERIALISE_BASE_URL
) -> MaterialiseSdmxClient:
    """Return an SDMX client that serves one fixed CSV body for every request.

    Use for one preview pair; use `build_materialise_double` for pair-specific bodies.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    return MaterialiseSdmxClient(base_url=base_url, transport=httpx.MockTransport(handler))


HARNESS_FACET = MaterialiseFacet(
    facet_id="f1",
    provenance_id="prov1",
    measurement_method=None,
    unit=None,
    obs_count=1,
    earliest_date=None,
    latest_date=None,
    scaling_factor=None,
)
"""The candidate `pair_plan` gives each series by default: the facet `stub_preview_result`
dispatches."""


def pair_plan(
    sv_dcids: Sequence[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    years: PlanYears = None,
    reader: GraphReader | None = None,
    coverage: CoverageIndex | None = None,
) -> PairPlan:
    """The per-series read plan a resolver records for every `(stat_var, role tuple)` of
    `sv_dcids` at the product of `obs_entities`, for tests that call `_build_observations`.

    A series' candidates come from `reader`'s facet probe for a standard pair, from
    materialise's coverage probe for a role-bearing pair `coverage` names, and otherwise are
    `HARNESS_FACET` alone. A pair with no candidate is no series.
    """
    roles = list(obs_entities)
    series: list[PlanSeries] = []
    for sv in sv_dcids:
        for combo in itertools.product(*(obs_entities[role] for role in roles)):
            role_tuple = tuple(zip(roles, combo, strict=True))
            if reader is not None and roles == ["observationAbout"]:
                candidates = reader.variable_facets_batch(stat_var=sv, entities=combo).get(
                    combo[0], []
                )
            elif coverage is not None and sv in coverage.role_names:
                candidates = probe_facets(
                    availability=NO_SCOPE_READ,
                    stat_var=sv,
                    bound_entities=combo,
                    settle=False,
                    graph=FakeGraphReader(),
                    declared_roles=tuple(roles),
                    role_bindings={role: [dcid] for role, dcid in role_tuple},
                    coverage=coverage,
                ).candidates
            else:
                candidates = [HARNESS_FACET]
            if candidates:
                series.append(
                    PlanSeries(stat_var=sv, role_tuple=role_tuple, candidates=tuple(candidates))
                )
    return plan_pairs(series, members=sv_dcids, years=years)


@dataclass
class HarnessScope(BulkScope):
    """A scope-wide read's inputs with each StatVar's probed facets per place, from which
    `build_observations_planned` plans the scope read a resolver would."""

    facets_by_sv_entity: Mapping[str, Mapping[str, Sequence[MaterialiseFacet]]] = field(
        default_factory=dict
    )


def harness_plan(
    sv_dcids: Sequence[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    coverage: CoverageSnapshot,
    window: PlanYears = None,
    materialise_graph: GraphReader | None = None,
    member_facts: Mapping[str, MemberFacts] | None = None,
    scope: BulkScope | None = None,
    discovered_role_index: RoleFacetIndex | None = None,
    discovered_time_period: Sequence[str] | None = None,
) -> ReadPlan:
    """The read plan a resolver records for these inputs: the antichain-minimal members, read
    over `window`. A `HarnessScope` plans one facet across its places; a role-bearing product
    above the pair cap plans a bulk read over `discovered_role_index`, or else over the
    coverage index at the named entities; any other call plans each pair (`pair_plan`)."""
    members = minimal_constraint_svs(
        sv_dcids, {sv: facts.read_constraints() for sv, facts in (member_facts or {}).items()}
    )
    if isinstance(scope, HarnessScope):
        places = set(obs_entities.get("observationAbout", ()))
        return plan_standard(
            {
                (sv, entity): facets
                for sv in members
                for entity, facets in scope.facets_by_sv_entity.get(sv, {}).items()
                if entity in places and facets
            },
            members=members,
            years=window,
            scope=True,
        )
    role_bearing = "observationAbout" not in obs_entities
    pairs = math.prod(len(dcids) for dcids in obs_entities.values()) * len(members)
    if role_bearing and pairs > observation_settings().obs_preview_max_pairs:
        return bulk_plan(
            members,
            obs_entities,
            coverage=coverage,
            window=window,
            discovered_role_index=discovered_role_index,
            read_years=discovered_time_period,
        )
    return pair_plan(
        members, obs_entities, years=window, reader=materialise_graph, coverage=coverage.index
    )


def bulk_plan(
    members: Sequence[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    coverage: CoverageSnapshot,
    window: PlanYears = None,
    discovered_role_index: RoleFacetIndex | None = None,
    read_years: Sequence[str] | None = None,
) -> BulkPlan:
    """The bulk role plan a resolver records: one facet elected over `discovered_role_index`,
    or else the coverage index at `obs_entities`, inside `read_years` (by default the year
    list of a range `window`)."""
    if read_years is None and isinstance(window, TimeWindow):
        read_years = role_years(
            DateRequest(window=window, latest=False), coverage.index.horizon
        ).listed
    index = (
        discovered_role_index
        if discovered_role_index is not None
        else role_index_for_named_entities(
            members, obs_entities, time_period=read_years, coverage_index=coverage.index
        )
    )
    merged = merge_role_candidates(index, members)
    best = select_role_candidate(merged, stat_vars=members, time_period=read_years)
    return BulkPlan(
        members=tuple(members),
        years=window,
        read_years=tuple(read_years) if read_years is not None else None,
        facet_id=best.facet_id if best is not None else None,
        candidates=merged,
    )


def bulk_role_series_planned(
    sv_dcids: list[str],
    obs_entities: Mapping[str, Sequence[str]],
    *,
    coverage: CoverageSnapshot,
    facet_filter: Mapping[str, FacetAxisFilter] | None = None,
    discovered_role_index: RoleFacetIndex | None = None,
    time_period: Sequence[str] | None = None,
    latest: bool = False,
    roles: PlanRoles | None = None,
    **kwargs,
) -> tuple[SpecObservations, Warning | None]:
    """`_bulk_role_series` over the bulk plan a resolver records for these inputs
    (`bulk_plan`): `latest`, a range over `time_period`, or every year, narrowed by
    `facet_filter`, with the plan's bound `roles` when given."""
    window: PlanYears = (
        "latest"
        if latest
        else TimeWindow(start_year=int(time_period[0]), end_year=int(time_period[-1]))
        if time_period
        else None
    )
    plan = bulk_plan(
        sv_dcids,
        obs_entities,
        coverage=coverage,
        window=window,
        discovered_role_index=discovered_role_index,
        read_years=time_period,
    )
    if roles is not None:
        plan = replace(plan, roles=roles)
    if facet_filter:
        plan = plan.narrowed(lambda _sv, candidate: facet_admitted(facet_filter, candidate))
    return _bulk_role_series(plan, sv_dcids, facet_filter=facet_filter, **kwargs)


def bulk_scope_series_planned(
    sv_dcids: list[str],
    entity_dcids: Sequence[str],
    *,
    scope: HarnessScope,
    **kwargs,
) -> tuple[SpecObservations, Warning | None]:
    """`_bulk_scope_series` over the scope plan a resolver records for `scope`'s probed
    facets at `entity_dcids` (`harness_plan`)."""
    plan = harness_plan(
        sv_dcids,
        {"observationAbout": list(entity_dcids)},
        coverage=empty_coverage_snapshot(),
        window="latest",
        scope=scope,
    )
    assert isinstance(plan, PairPlan)
    return _bulk_scope_series(plan, sv_dcids, entity_dcids, scope=scope, **kwargs)


def _planned_read(
    sv_dcids: list[str], obs_entities: Mapping[str, Sequence[str]], kwargs: dict
) -> ReadPlan:
    """The plan `build_observations_planned` reads, popping the plan-shaping keys from
    `kwargs`: the call's own `plan`, or the one a resolver records for these inputs
    (`harness_plan`), narrowed by its `facet_filter` and, for a unit request, its `elected`
    units, as `region.facets.filter_answer` narrows it."""
    plan = kwargs.pop("plan", None)
    window = kwargs.pop("window", None)
    discovered_role_index = kwargs.pop("discovered_role_index", None)
    discovered_time_period = kwargs.pop("discovered_time_period", None)
    if plan is not None:
        return plan
    plan = harness_plan(
        sv_dcids,
        obs_entities,
        coverage=kwargs["coverage"],
        window=window,
        materialise_graph=kwargs.get("materialise_graph"),
        member_facts=kwargs.get("member_facts"),
        scope=kwargs.get("scope"),
        discovered_role_index=discovered_role_index,
        discovered_time_period=discovered_time_period,
    )
    elected = kwargs.get("elected")
    admission = FacetAdmission(
        axes=kwargs.get("facet_filter"),
        elected_units=(
            elected_unit_refs(elected)
            if elected is not None and kwargs.get("unit_request") is not None
            else None
        ),
    )
    return plan.narrowed(admission.admits) if admission else plan


def build_observations_planned(
    sv_dcids: list[str], obs_entities: Mapping[str, Sequence[str]], **kwargs
) -> tuple[SpecObservations, Warning | None]:
    """`_build_observations` with the plan a resolver records for these inputs
    (`_planned_read`) when the call names none. `window`, `discovered_role_index` and
    `discovered_time_period` shape that plan."""
    plan = _planned_read(sv_dcids, obs_entities, kwargs)
    return _build_observations(plan, sv_dcids, obs_entities, **kwargs)


def preview_and_plan(
    sv_dcids: list[str], obs_entities: Mapping[str, Sequence[str]], **kwargs
) -> tuple[tuple[SpecObservations, Warning | None], ReadPlan]:
    """`build_observations_planned`'s preview and the plan it read, the plan the bound-unit
    gate decides from (`bound_unit_gate.bound_unit_gate_reason`)."""
    plan = _planned_read(sv_dcids, obs_entities, kwargs)
    return _build_observations(plan, sv_dcids, obs_entities, **kwargs), plan


def unit_check_planned(
    sv_dcids: list[str], obs_entities: Mapping[str, Sequence[str]], **kwargs
) -> tuple[tuple[SpecObservations, Warning | None], UnitEvidence]:
    """`build_observations_planned`'s preview and the bound-unit evidence the gate decides
    from over the same plan (`bound_unit_gate.plan_unit_evidence`)."""
    preview, plan = preview_and_plan(sv_dcids, obs_entities, **kwargs)
    return preview, plan_unit_evidence(plan)


def harness_reader(entities: Sequence[str], *, stat_var: str = "Count_Person") -> FakeGraphReader:
    """Return `HARNESS_FACET` as each entity's candidate, the facet `stub_preview_result`
    dispatches. Role-bearing pairs read from another source."""
    return FakeGraphReader(facets={(stat_var, entity): [HARNESS_FACET] for entity in entities})


@dataclass(frozen=True)
class ObsFacetFixture:
    """One graph facet and its SDMX points for a preview pair.

    Empty points still test facet selection; missing counts and dates derive from points.
    Explicit metadata supports summary-only facets with no observation rows.
    """

    facet_id: str
    points: tuple[tuple[str, float | None], ...] = ()
    provenance_id: str | None = None
    measurement_method: str | None = None
    unit: str | None = None
    scaling_factor: str | None = None
    obs_count: int | None = None
    earliest_date: str | None = None
    latest_date: str | None = None
    observation_period: str | None = None


class PointsGraphReader(FakeGraphReader):
    """`FakeGraphReader` over facets holding their fixtures' points (`_facet_from_fixture`),
    whose reads return each facet's dates and values only under `needs_dates`, as the graph's
    do."""

    def _facets_at(
        self, stat_var: str, entity: str, *, needs_dates: bool
    ) -> list[MaterialiseFacet]:
        facets = super()._facets_at(stat_var, entity, needs_dates=needs_dates)
        if needs_dates:
            return facets
        return [replace(facet, dates=(), values=(), dates_complete=True) for facet in facets]


def _facet_from_fixture(fixture: ObsFacetFixture) -> MaterialiseFacet:
    """The facet a dated graph read returns for `fixture`: its points' dates and values, short
    of its count (`dates_complete`) when `obs_count` names more points than it holds."""
    dates = [d for d, _ in fixture.points]
    obs_count = fixture.obs_count if fixture.obs_count is not None else len(fixture.points)
    return MaterialiseFacet(
        facet_id=fixture.facet_id,
        provenance_id=fixture.provenance_id,
        measurement_method=fixture.measurement_method,
        unit=fixture.unit,
        obs_count=obs_count,
        earliest_date=fixture.earliest_date
        if fixture.earliest_date is not None
        else (min(dates) if dates else None),
        latest_date=fixture.latest_date
        if fixture.latest_date is not None
        else (max(dates) if dates else None),
        scaling_factor=fixture.scaling_factor,
        dates=tuple(dates),
        values=tuple(value for _, value in fixture.points),
        dates_complete=len(dates) >= obs_count,
    )


def _csv_header(role_names: Sequence[str]) -> str:
    return ",".join((*_CSV_PREFIX, *role_names, *_CSV_TRAILING)) + "\r\n"


_NOT_APPLICABLE = "NotApplicable"
"""`NotApplicable` maps missing dimensions to `None`; empty dimensions stay `""`.
Empty `OBS_VALUE` stays `""`; empty scaling and facet cells map to `None`.
"""

_OPEN_ROLE_ENTITY = "dcid:OpenRoleAxisFixture/UnfilteredEntity"
"""Fixture value for an unbound but declared role; `Frame.build` requires a non-null entity.
`NotApplicable` would parse to null. Tests do not assert this dcid by name.
"""


def _csv_row(
    *,
    stat_var: str,
    role_dcids: Sequence[str],
    fixture: ObsFacetFixture,
    date_str: str,
    value: float | None,
) -> str:
    fields = [
        "dataflow",
        "DC:DF_OBS(1.0.0)",
        "I",
        f"dcid:{stat_var}",
        *role_dcids,
        fixture.unit or _NOT_APPLICABLE,
        fixture.measurement_method or _NOT_APPLICABLE,
        fixture.observation_period or _NOT_APPLICABLE,
        fixture.provenance_id or _NOT_APPLICABLE,
        date_str,
        "" if value is None else str(value),
        fixture.scaling_factor or "",
        fixture.facet_id or "",
    ]
    return ",".join(fields) + "\r\n"


def build_materialise_double(
    pairs: Mapping[PairKey, Sequence[ObsFacetFixture]],
    *,
    base_url: str = MATERIALISE_BASE_URL,
    declared_roles: Mapping[str, Sequence[str]] | None = None,
) -> tuple[FakeGraphReader, MaterialiseSdmxClient, list[httpx.Request]]:
    """Pair fixture-backed graph and SDMX clients for real materialise facet selection.

    Each `(stat_var, role_tuple)` gets its own facets and CSV body. A shared dcid must have one
    candidate set; use `build_materialise_double_union_candidates` for shared entities.

    `declared_roles` adds open-axis columns (#881/#627) carried by the SDMX header even when
    unfiltered. Fill them with `_OPEN_ROLE_ENTITY`; `Frame.build` rejects null role cells, and
    `NotApplicable` parses to null.

    Route responses by `variableMeasured` and bound role filters. `facetId` and `TIME_PERIOD`
    do not affect the pair key; open roles have no filter. A request binding only some of a
    pair's roles reads the rest open, as SDMX does: it gets the rows of every pair carrying
    its bound roles, under one header naming each role.
    """
    facets_map: dict[tuple[str, str], list[MaterialiseFacet]] = {}
    csv_by_key: dict[tuple[str, frozenset[tuple[str, str]]], str] = {}
    for (stat_var, role_tuple), fixtures in pairs.items():
        candidate_facets = [_facet_from_fixture(f) for f in fixtures]
        for _role, dcid in role_tuple:
            facets_map[(stat_var, dcid)] = candidate_facets
        role_names = [role for role, _ in role_tuple]
        role_dcids = [dcid for _, dcid in role_tuple]
        open_roles = [
            role for role in (declared_roles or {}).get(stat_var, ()) if role not in role_names
        ]
        body = _csv_header([*role_names, *open_roles])
        for fixture in fixtures:
            for date_str, value in fixture.points:
                body += _csv_row(
                    stat_var=stat_var,
                    role_dcids=[*role_dcids, *(_OPEN_ROLE_ENTITY for _ in open_roles)],
                    fixture=fixture,
                    date_str=date_str,
                    value=value,
                )
        csv_by_key[(stat_var, frozenset(role_tuple))] = body

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = request.url.params
        stat_var = params.get("c[variableMeasured]", "")
        role_tuple = tuple(
            (key[2:-1], params[key])
            for key in params
            if key.startswith("c[")
            and key not in ("c[variableMeasured]", "c[facetId]", "c[TIME_PERIOD]")
        )
        body = csv_by_key.get((stat_var, frozenset(role_tuple)))
        if body is None:
            body = _open_role_body(pairs, stat_var, role_tuple, declared_roles or {})
        return httpx.Response(200, text=body, headers={"content-type": _SDMX_CSV_CONTENT_TYPE})

    sdmx = MaterialiseSdmxClient(base_url=base_url, transport=httpx.MockTransport(handler))
    return PointsGraphReader(facets=facets_map), sdmx, requests


def _open_role_body(
    pairs: Mapping[PairKey, Sequence[ObsFacetFixture]],
    stat_var: str,
    bound: tuple[tuple[str, str], ...],
    declared_roles: Mapping[str, Sequence[str]],
) -> str:
    """The CSV for a request binding `bound` and leaving a pair's other roles open: every
    row of each `stat_var` pair that carries the bound roles, or a header alone."""
    matching = {
        role_tuple: fixtures
        for (sv, role_tuple), fixtures in pairs.items()
        if sv == stat_var and set(bound) < set(role_tuple)
    }
    if not matching:
        return _csv_header([role for role, _ in bound])
    roles = list(
        declared_roles.get(stat_var)
        or dict.fromkeys(role for role_tuple in matching for role, _dcid in role_tuple)
    )
    body = _csv_header(roles)
    for role_tuple, fixtures in matching.items():
        by_role = dict(role_tuple)
        for fixture in fixtures:
            for date_str, value in fixture.points:
                body += _csv_row(
                    stat_var=stat_var,
                    role_dcids=[by_role.get(role, _OPEN_ROLE_ENTITY) for role in roles],
                    fixture=fixture,
                    date_str=date_str,
                    value=value,
                )
    return body


def build_materialise_double_union_candidates(
    pairs: Mapping[PairKey, Sequence[ObsFacetFixture]],
    *,
    base_url: str = MATERIALISE_BASE_URL,
) -> tuple[FakeGraphReader, MaterialiseSdmxClient, list[httpx.Request]]:
    """Union graph candidate facets when one entity appears in several pairs.

    Real graph reads return every facet for that entity. Distinct pair facet IDs let materialise
    rank each pair's facet by its counts across bound entities; SDMX bodies remain keyed by role tuple.
    """
    _ignored_graph, materialise_sdmx, requests = build_materialise_double(pairs, base_url=base_url)
    candidates_by_dcid: dict[tuple[str, str], list[MaterialiseFacet]] = {}
    for (stat_var, role_tuple), fixtures in pairs.items():
        facets = [_facet_from_fixture(f) for f in fixtures]
        # A dcid can occur more than once in one pair's own role_tuple (a
        # self-flow pair binds the same dcid to two different roles) --
        # dedup to distinct dcids first, so that pair's facets are added to
        # its candidate list once per dcid, never once per role occurrence.
        for dcid in {dcid for _role, dcid in role_tuple}:
            candidates_by_dcid.setdefault((stat_var, dcid), []).extend(facets)
    return PointsGraphReader(facets=candidates_by_dcid), materialise_sdmx, requests


def _candidate_facets_by_dcid(
    pairs: Mapping[PairKey, Sequence[ObsFacetFixture]],
) -> dict[tuple[str, str], list[MaterialiseFacet]]:
    """Build `build_materialise_double`'s candidate map without its reader."""
    facets_map: dict[tuple[str, str], list[MaterialiseFacet]] = {}
    for (stat_var, role_tuple), fixtures in pairs.items():
        facets = [_facet_from_fixture(f) for f in fixtures]
        for _role, dcid in role_tuple:
            facets_map[(stat_var, dcid)] = facets
    return facets_map


class _ProbeFailingGraphReader(PointsGraphReader):
    """Fail batched facet-coverage reads while per-pair materialise reads succeed.

    This isolates option-coverage failures from preview dispatch; `message` names the error.
    """

    def __init__(self, *, message: str = "facet-probe boom", **kwargs) -> None:
        super().__init__(**kwargs)
        self._probe_failure_message = message

    def observation_facets_batch_many(
        self,
        *,
        stat_vars: Sequence[str],
        entities: Sequence[str],
        needs_dates: bool = False,
        deadline: float | None = None,
    ) -> dict[str, dict[str, list[MaterialiseFacet]]]:
        self._count("observation_facets_batch_many")
        raise GraphInfraError(self._probe_failure_message)


def materialise_double_from_obs(
    obs: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    base_url: str = MATERIALISE_BASE_URL,
    fail_option_coverage_probe: bool = False,
) -> tuple[FakeGraphReader, MaterialiseSdmxClient]:
    """Convert `FakeGraph(obs=...)` data keyed by `stat_var|entity` into graph and SDMX doubles.

    Graph facets retain metadata even without point rows, so dispatch can select a facet with empty
    output. Missing values are dropped before materialise row counts are computed.
    Reusing `obs` with `FakeGraph` lets tests compare coverage and observation results.
    Each key becomes a standard `observationAbout` pair, not a role-bearing pair.

    With `fail_option_coverage_probe`, batched facet reads raise `GraphInfraError` while per-pair
    reads work. This isolates the `OPTION_COVERAGE_PROBE_FAILED` path and absent coverage result.
    """
    pairs: dict[PairKey, list[ObsFacetFixture]] = {}
    for key, facet_dicts in obs.items():
        sv, entity = key.split("|", 1)
        fixtures = []
        for i, facet_dict in enumerate(facet_dicts):
            dates = facet_dict.get("dates")
            values = facet_dict.get("values")
            if dates is None:
                observations = facet_dict.get("observations", [])
                dates = [o["date"] for o in observations if "date" in o]
                values = [o.get("value") for o in observations if "date" in o]
            points = tuple((d, v) for d, v in zip(dates, values, strict=True) if v is not None)
            fixtures.append(
                ObsFacetFixture(
                    facet_id=facet_dict.get("facetId") or f"f{i}",
                    points=points,
                    provenance_id=facet_provenance(sv, facet_dict),
                    measurement_method=facet_dict.get("measurementMethod"),
                    unit=facet_dict.get("unit"),
                    scaling_factor=facet_dict.get("scalingFactor"),
                    obs_count=facet_dict.get("obsCount"),
                    earliest_date=facet_dict.get("earliestDate"),
                    latest_date=facet_dict.get("latestDate"),
                )
            )
        pairs[(sv, (("observationAbout", entity),))] = fixtures
    graph, sdmx, _requests = build_materialise_double(pairs, base_url=base_url)
    if fail_option_coverage_probe:
        return _ProbeFailingGraphReader(facets=_candidate_facets_by_dcid(pairs)), sdmx
    return graph, sdmx


def stub_preview_result(
    *,
    sv: str = "Count_Person",
    role_key: str = "observationAbout",
    entity: str = "country/IND",
    rows: tuple[tuple[str, float], ...] = (("2020", 5.0),),
    total_rows: int | None = None,
    provenance: str = "prov1",
) -> PreviewResult:
    """Build a pre-resolved preview for `_fetch_observations` tests that skip facet selection."""
    frame = Frame.build(
        role_keys=(role_key,),
        rows=[
            {
                "stat_var": f"dcid:{sv}",
                role_key: entity,
                "time_period": date_str,
                "value": value,
                "unit": "USD",
                "measurementMethod": "Test",
                "provenance": provenance,
            }
            for date_str, value in rows
        ],
    )
    return PreviewResult(
        frame=frame,
        coverage=MaterialisePreviewCoverage(rows=len(rows), dimensions=(), window=None),
        total_rows=total_rows if total_rows is not None else len(rows),
        truncated=total_rows is not None and total_rows > len(rows),
        conversion_record=None,
        conversion_recipe_id=None,
        selections={
            "provenance": FacetSelection(
                axis="provenance",
                ref=provenance,
                basis="deployment_fallback",
                default_policy="dfo/facet-policy/most-observations",
                default_policy_version="1",
                selection_reason=None,
                dispatch_facet_id="f1",
            ),
        },
        warnings=(),
    )


_FAKE_GRAPH = FakeGraph()
_FAKE_LLM = FakeLLM()
# This empty client prevents offline resolves from loading dev-finance fixtures or calling live SDMX.
# Share it only when tests do not inspect call records; those tests need their own instance.
EMPTY_SDMX = FakeSdmxClient(observations={}, availability={})
_FAKE_SDMX = EMPTY_SDMX

# The timeout tests advance the 30-second request budget with the fake clock.
DEADLINE_TIMEOUT_S = 30.0


class FakeClock:
    """A settable monotonic clock for tests that advance time without sleeping."""

    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def frozen_monotonic_clock(monkeypatch) -> FakeClock:
    """Patch monotonic time across envelope, deadline, and fan-out modules."""
    clock = FakeClock(time.monotonic())
    fake_time = SimpleNamespace(monotonic=clock)
    monkeypatch.setattr(envelope_mod, "time", fake_time)
    monkeypatch.setattr(deadline_mod, "time", fake_time)
    monkeypatch.setattr(fanout_mod, "time", fake_time)
    return clock


def run_pinned(coro):
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        return asyncio.run(coro)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    event_type: str | None = None
    for line in body.splitlines():
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:") and event_type is not None:
            frames.append((event_type, json.loads(line[len("data:") :].strip())))
            event_type = None
    return frames


def ref_dcid(ref: GraphRef | None) -> str:
    """Return a required graph ref's dcid; value-bound slots must be grounded."""
    assert ref is not None
    return ref.ref


def slot_value_dcid(slot: Slot) -> str:
    """Return a slot's dcid after checking its value binding and graph ref."""
    binding = slot.binding
    assert isinstance(binding, BindingValue)
    return ref_dcid(binding.value.ref)


def dimensions_of(coverage: Coverage) -> list[BreadthDim]:
    """Return populated dimensions from a breadth or exact coverage."""
    assert isinstance(coverage, (CoverageExact, CoverageBreadth))
    assert coverage.dimensions is not None
    return coverage.dimensions


def member_walk_reader(
    graph: FakeGraph | None = None, *, stat_vars: Sequence[str] | None = None
) -> FakeMintGraphReader:
    """The mint's member walk over every node `graph` holds (the shared fixture graph by
    default), as production walks the graph Search reads. `stat_vars`, when given, are the
    only StatVars it holds, so the walk settles a family of exactly them."""
    arcs = _walk_arcs(_FAKE_GRAPH if graph is None else graph)
    if stat_vars is not None:
        arcs = {sv: arcs[sv] for sv in stat_vars}
    return FakeMintGraphReader(arcs=arcs)


def _walk_arcs(graph: FakeGraph) -> dict[str, dict[str, tuple[str, ...]]]:
    """`graph`'s nodes as the member walk reads them, less stale recordings."""
    return {
        dcid: node_arcs
        for dcid, node_arcs in graph.member_arcs().items()
        if not _stale_recording(node_arcs)
    }


def _stale_recording(arcs: Mapping[str, tuple[str, ...]]) -> bool:
    """Whether a recorded node states two values on a five-tuple arc, which the live graph
    never does: `Count_BirthEvent_LiveBirth` was recorded with populationType `Person` beside
    `BirthEvent`, which the dev graph no longer holds, and a walk reading it would leave it
    undismissed."""
    return any(
        len(arcs.get(prop, ())) > 1 for prop in ("populationType", "measuredProperty", "statType")
    )


class MemberArcsOverlay:
    """`inner` with the StatVar arcs of `arcs` beneath it, so the mint's member walk reads
    a fixture graph's family through a materialise double built for facets alone. A reverse
    read returns `inner`'s subjects then those `arcs` adds; an arcs read takes each property
    from `inner` where it holds the node and the property, else from `arcs`; every other
    read is `inner`'s."""

    def __init__(self, inner: GraphReader, arcs: Mapping[str, Mapping[str, tuple[str, ...]]]):
        self._inner = inner
        self._arcs = arcs

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        inner = self._inner.node_subjects(
            dcid, property=property, filters=filters, max_results=max_results, deadline=deadline
        )
        matches = list(
            dict.fromkeys(
                [
                    *inner.dcids,
                    *(
                        subject
                        for subject, node in self._arcs.items()
                        if dcid in node.get(property, ())
                        and all(
                            value in node.get(key, ()) for key, value in (filters or {}).items()
                        )
                    ),
                ]
            )
        )
        return SubjectPage(
            dcids=tuple(matches[:max_results]),
            truncated=inner.truncated or len(matches) > max_results,
        )

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        found = self._inner.node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )
        result: dict[str, dict[str, tuple[str, ...]] | None] = {}
        for dcid in dcids:
            inner, below = found.get(dcid), self._arcs.get(dcid)
            if below is None:
                result[dcid] = inner
                continue
            result[dcid] = {
                prop: (inner or {}).get(prop) or below.get(prop, ()) for prop in properties
            }
        return result


def with_member_arcs(reader: GraphReader | None, graph: FakeGraph) -> GraphReader:
    """`reader` with `graph`'s StatVar arcs beneath it (`MemberArcsOverlay`), or the walk
    over `graph` alone (`member_walk_reader`) with no reader."""
    if reader is None:
        return member_walk_reader(graph)
    arcs = {
        dcid: node_arcs
        for dcid, node_arcs in _walk_arcs(graph).items()
        if "StatisticalVariable" in node_arcs.get("typeOf", ())
    }
    return cast(GraphReader, MemberArcsOverlay(reader, arcs))


def walk_options(graph: EngineGraphClient, options: EngineOptions | None = None) -> EngineOptions:
    """`options` for a resubmit over `graph`: a resubmit elects its members from the
    identity by the mint's member walk, so with no materialise graph of its own it walks
    `graph`'s nodes (`member_walk_reader`). A test whose materialise graph holds no StatVar
    arcs lays `graph`'s beneath it (`with_member_arcs`)."""
    options = options if options is not None else EngineOptions()
    if options.materialise_graph is not None or not isinstance(graph, FakeGraph):
        return options
    return dataclasses.replace(options, materialise_graph=member_walk_reader(graph))


def make_request(query: str) -> ResolveRequest:
    return ResolveRequest(input=RawTextInput(query=query))


def offline_resolve(
    request: ResolveRequest,
    *,
    graph: EngineGraphClient | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    pac: bool | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    part_memo: PartMemo | None = None,
    binder: Binder | None = None,
) -> ResolveResponse:
    """Run `resolve_async` with fake clients and a fixed extraction date.

    The 2026-06-23 pin keeps fixture keys stable across calendar days.

    `graph` and `llm` replace shared fakes for this call, and `binder` the configured
    binder.

    `sdmx` overrides the empty default; inject a fake for SDMX behavior without live calls.

    `pac` overrides `seam_default` for this call.

    Inject both materialise clients for observations; the default preview returns
    `(None, None)`
    without network access.

    Pass `rate_cache` for request-form unit conversion; that path raises before
    fetching without it. Pass `part_memo` to serve resubmitted identities from it.

    A dataset_resubmit over a fixture graph with no `materialise_graph` walks the mint's
    member enumeration over that graph's nodes (`member_walk_reader`), as a resubmit elects
    its members from the identity.
    """
    graph = _FAKE_GRAPH if graph is None else graph
    with _offline(pac):
        return asyncio.run(
            resolve_async(
                request,
                graph=graph,
                llm=_FAKE_LLM if llm is None else llm,
                sdmx=_FAKE_SDMX if sdmx is None else sdmx,
                materialise_graph=_materialise_graph(request, graph, materialise_graph),
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
                part_memo=part_memo,
                binder=binder,
            )
        )


def offline_events(
    request: ResolveRequest,
    *,
    graph: EngineGraphClient | None = None,
    llm: SupportsLLM | None = None,
    sdmx: EngineSdmxClient | None = None,
    materialise_graph: GraphReader | None = None,
    materialise_sdmx: MaterialiseSdmxClient | None = None,
    rate_cache: RateTableCache | None = None,
    stream_slot_options: bool = False,
    response_cache: ResponseCache | None = None,
    place_options_pool: PlaceOptionsPool | None = None,
    close_after: str | None = None,
    binder: Binder | None = None,
) -> list[Event]:
    """Every event `resolve_events` streams for `request`, in order, under the shared fakes
    and the pinned extraction date `offline_resolve` runs with. `graph` and `llm` replace
    the shared fakes for this call, and `binder` the configured binder. `stream_slot_options`
    streams the trailing events as the SSE route does; `response_cache` and
    `place_options_pool` are the app's shared ones.
    `close_after` names an event kind: the stream is closed after the first such event, as
    when a client disconnects."""
    graph = _FAKE_GRAPH if graph is None else graph

    async def collect() -> list[Event]:
        events: list[Event] = []
        stream = resolve_events(
            request,
            graph=graph,
            llm=_FAKE_LLM if llm is None else llm,
            sdmx=_FAKE_SDMX if sdmx is None else sdmx,
            materialise_graph=_materialise_graph(request, graph, materialise_graph),
            materialise_sdmx=materialise_sdmx,
            rate_cache=rate_cache,
            stream_slot_options=stream_slot_options,
            response_cache=response_cache,
            place_options_pool=place_options_pool,
            binder=binder,
        )
        async with aclosing(stream):
            async for event in stream:
                events.append(event)
                if event.kind == close_after:
                    break
        return events

    with _offline(None):
        return asyncio.run(collect())


def _materialise_graph(
    request: ResolveRequest, graph: EngineGraphClient, materialise_graph: GraphReader | None
) -> GraphReader | None:
    """`materialise_graph`, else for a dataset_resubmit over a fixture graph the mint's
    member enumeration over that graph's nodes (`member_walk_reader`)."""
    if (
        materialise_graph is None
        and request.input.kind == "dataset_resubmit"
        and isinstance(graph, FakeGraph)
    ):
        return member_walk_reader(graph)
    return materialise_graph


@contextmanager
def _offline(pac: bool | None) -> Iterator[None]:
    """Pin the extraction date, so fixture keys stay stable across calendar days, and set
    `seam_default` to `pac` when given."""
    with patch("search.engine.extract.date") as mock_date:
        mock_date.today.return_value = PINNED_DATE
        # Preserve real `date(*args)` construction during the patch.
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)
        if pac is None:
            yield
            return
        base = config.settings().model_dump(
            exclude={"capability_version_override", "jev_api_key", "max_entity_occurrences"}
        )
        overridden = config.SearchSettings.model_validate({**base, "seam_default": pac})
        with patch.object(config, "_SETTINGS", overridden):
            yield
