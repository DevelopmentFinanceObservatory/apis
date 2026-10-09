"""Fixture-backed `EngineSdmxClient` for offline tests.

Defaults load the observations and availability fixtures; explicit maps override them.
`role_series` filters by roles, facets, and period. `stub_keys` marks one-point `LATEST`
placeholders that lack history and trigger `StubSeriesReadError` on a matching read.
`subject_observations` has no fixture file and defaults to an empty mapping.
"""

from __future__ import annotations

import functools
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import replace
from pathlib import Path
from typing import Any

from dc_sdmx import ContainedIn, EntityFilter, SdmxObservation
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import (
    RoleTimePeriod,
    _validate_role_series_params,
    assert_no_un_variables,
)
from tests.materialise._scope_availability import ScopeAvailability

_ObsResult = dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]]

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _load_json(name: str) -> dict:
    with open(_FIXTURES_DIR / name) as f:
        return json.load(f)


def _facet_from_dict(raw: dict) -> Facet:
    """Build a Facet from a raw fixture dict. It is the same per-Facet shape
    tests/fixtures/record.py already writes for graph_obs.json entries."""
    return Facet(
        earliest_date=raw.get("earliestDate"),
        latest_date=raw.get("latestDate"),
        obs_count=raw.get("obsCount", 0),
        dates=tuple(
            raw.get("dates", [o["date"] for o in raw.get("observations", []) if o.get("date")])
        ),
        values=tuple(
            raw.get(
                "values", [o.get("value") for o in raw.get("observations", []) if o.get("date")]
            )
        ),
        provenance_id=raw.get("provenanceId"),
        measurement_method=raw.get("measurementMethod"),
        unit=raw.get("unit"),
        scaling_factor=raw.get("scalingFactor"),
        facet_id=raw.get("facetId"),
    )


def _parse_role_tuple(role_tuple_text: str) -> tuple[tuple[str, str], ...]:
    """Parse a "role=dcid,role=dcid" segment back into a sorted role tuple."""
    if not role_tuple_text:
        return ()
    pairs = (segment.split("=", 1) for segment in role_tuple_text.split(","))
    return tuple(sorted((role, dcid) for role, dcid in pairs))


_StubKey = tuple[str, tuple[tuple[str, str], ...], str]  # (variable, role_tuple, facet_id)


def _load_observations_fixture() -> tuple[
    dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]], set[_StubKey]
]:
    """Load observation facets and keys marked as one-point stubs."""
    raw = _load_json("sdmx_obs.json")
    result: dict[str, dict[tuple[tuple[str, str], ...], list[Facet]]] = {}
    stub_keys: set[_StubKey] = set()
    for key, facet_dicts in raw.items():
        if key == "_comment":
            continue
        variable, role_tuple_text = key.split("|", 1)
        role_tuple = _parse_role_tuple(role_tuple_text)
        result.setdefault(variable, {})[role_tuple] = [_facet_from_dict(f) for f in facet_dicts]
        for facet_dict in facet_dicts:
            if facet_dict.get("stub") and facet_dict.get("facetId"):
                stub_keys.add((variable, role_tuple, facet_dict["facetId"]))
    return result, stub_keys


class StubSeriesReadError(RuntimeError):
    """Raised when `role_series` reads a one-point `LATEST` placeholder without history."""


def _load_availability_fixture() -> dict[str, list[str]]:
    """Load sdmx_availability.json, keyed by _availability_key(component,
    variables, filters, time_period) -> [dcid, ...]."""
    raw = _load_json("sdmx_availability.json")
    return {k: list(v) for k, v in raw.items() if k != "_comment"}


def _availability_key(
    component: str,
    variables: Sequence[str],
    filters: Mapping[str, EntityFilter] | None,
    time_period: str | Sequence[str] | None = None,
) -> str:
    """Build the fixture key from component, variables, filters, and period.

    Variables are sorted and deduplicated; role names are sorted, with values in caller order.
    A containment filter renders as `containedIn:<parent>/<child type>`. Each filter and period
    can change members. Keep this format in sync with `record.py`.
    """
    key = f"{component}|variables={','.join(sorted(set(variables)))}"
    if filters:
        parts = ";".join(f"{role}={filter_text(filt)}" for role, filt in sorted(filters.items()))
        key = f"{key}|{parts}"
    if time_period is not None:
        period_text = time_period if isinstance(time_period, str) else ",".join(time_period)
        key = f"{key}@time_period={period_text}"
    return key


def scope_availability_of(obs: Mapping[str, Sequence[object]]) -> ScopeAvailability:
    """The default place scope's availability for graph observations in `graph_obs.json`'s
    shape: each `StatVar|place` key holding a recorded facet names that place as holding data,
    whatever its type. Live, availability and the graph name the same places."""
    pairs = []
    for key, facets in obs.items():
        variable, _, place = key.partition("|")
        if place and facets:
            pairs.append((variable, place))
    return ScopeAvailability(data=pairs)


@functools.cache
def _recorded_scope_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(scope_availability_of(_load_json("graph_obs.json")).pairs)


def recorded_scope_availability() -> ScopeAvailability:
    """The default place scope's availability for the recorded graph observations. Each call
    returns a fresh reader, so no test shares another's read log."""
    return ScopeAvailability(data=_recorded_scope_pairs())


def filter_text(filt: EntityFilter) -> str:
    """An entity filter as fixture keys write it."""
    if isinstance(filt, ContainedIn):
        return f"containedIn:{filt.parent}/{filt.child_type}"
    return ",".join(filt)


def _narrow_facet(facet: Facet, time_period: RoleTimePeriod) -> Facet:
    """Filter a fixture facet to `time_period`; `None` keeps its full history and `LATEST` its
    latest point, as the mixer computes it per series."""
    if time_period is None:
        return facet
    if time_period == "LATEST":
        periods = {max(facet.dates)} if facet.dates else set()
    else:
        periods = set(time_period)
    pairs = [
        (date, value)
        for date, value in zip(facet.dates, facet.values, strict=True)
        if date in periods
    ]
    dates = [date for date, _ in pairs]
    values = [value for _, value in pairs]
    return replace(
        facet,
        dates=dates,
        values=values,
        obs_count=len(dates),
        earliest_date=dates[0] if dates else None,
        latest_date=dates[-1] if dates else None,
    )


# A hook can raise a fixed exception or return one by zero-based call index; `None` succeeds.
_RaiseSpec = BaseException | Callable[[int], BaseException | None] | None

# Availability hooks key by component because role-scope probes run concurrently.
# Other hooks use the call index.
_AvailabilityRaiseSpec = BaseException | Callable[[str], BaseException | None] | None


def _maybe_raise(spec: _RaiseSpec, call_index: int) -> None:
    if spec is None:
        return
    exc = spec if isinstance(spec, BaseException) else spec(call_index)
    if exc is not None:
        raise exc


def _maybe_raise_for_component(spec: _AvailabilityRaiseSpec, component: str) -> None:
    if spec is None:
        return
    exc = spec if isinstance(spec, BaseException) else spec(component)
    if exc is not None:
        raise exc


def _role_tuple_matches(
    role_tuple: tuple[tuple[str, str], ...], entities: Mapping[str, Sequence[str]]
) -> bool:
    """Match every requested role to a dcid in the row's role tuple.

    Rows missing a requested role or carrying an unrequested dcid do not match.
    """
    role_map = dict(role_tuple)
    return all(role_map.get(role) in values for role, values in entities.items())


class FakeSdmxClient:
    """Record calls and replay fixtures or injected errors.

    Omitted `observations` and `availability` load recorded files; explicit mappings replace them.
    `observations` maps variables to role tuples and facets; `availability` uses `_availability_key`.
    `subject_observations` maps variables to entity rows and defaults empty.
    File markers identify one-point `LATEST` placeholders; explicit observations default to no stubs.
    Raise hooks accept exceptions or callables; availability callables receive the component.
    An unfiltered `observationAbout` read that the availability map lacks is the default place
    scope's: it names every place `scope_availability` holds data at, the recorded graph
    observations' unless given.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str, dict[tuple[tuple[str, str], ...], list[Facet]]] | None = None,
        availability: Mapping[str, Sequence[str]] | None = None,
        subject_observations: Mapping[str, dict[str, list[SdmxObservation]]] | None = None,
        stub_keys: set[_StubKey] | None = None,
        raise_series: _RaiseSpec = None,
        raise_availability: _AvailabilityRaiseSpec = None,
        raise_subject_observations: _RaiseSpec = None,
        scope_availability: ScopeAvailability | None = None,
    ) -> None:
        self._scope_availability = (
            scope_availability if scope_availability is not None else recorded_scope_availability()
        )
        if observations is not None:
            self._observations = dict(observations)
            self._stub_keys = set(stub_keys) if stub_keys is not None else set()
        else:
            loaded_observations, loaded_stub_keys = _load_observations_fixture()
            self._observations = loaded_observations
            self._stub_keys = set(stub_keys) if stub_keys is not None else loaded_stub_keys
        self._availability = {
            k: list(v)
            for k, v in (
                availability if availability is not None else _load_availability_fixture()
            ).items()
        }
        self._subject_observations = (
            {k: dict(v) for k, v in subject_observations.items()}
            if subject_observations is not None
            else {}
        )
        self._raise_series = raise_series
        self._raise_availability = raise_availability
        self._raise_subject_observations = raise_subject_observations
        self.series_calls: list[dict[str, Any]] = []
        self.availability_calls: list[dict[str, Any]] = []
        self.subject_observations_calls: list[dict[str, Any]] = []
        # Lock call indexing because role-scope availability probes run concurrently.
        self._calls_lock = threading.Lock()

    def role_series(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, Sequence[str]],
        facet_ids: AbstractSet[str],
        time_period: RoleTimePeriod = None,
        max_rows: int,
        max_bytes: int,
        deadline: float | None = None,
    ) -> _ObsResult:
        _validate_role_series_params(
            variables=variables,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
        )
        # UN variables only: raw test threads carry no bound catalogue for the base check.
        assert_no_un_variables(method="role_series", variables=variables)
        with self._calls_lock:
            call_index = len(self.series_calls)
            self.series_calls.append(
                {
                    "variables": list(variables),
                    "entities": {k: list(v) for k, v in entities.items()},
                    "facet_ids": set(facet_ids),
                    "time_period": (
                        time_period
                        if time_period is None or time_period == "LATEST"
                        else list(time_period)
                    ),
                    "max_rows": max_rows,
                    "max_bytes": max_bytes,
                    "deadline": deadline,
                }
            )
        _maybe_raise(self._raise_series, call_index)
        result: _ObsResult = {}
        for v in variables:
            if v not in self._observations:
                continue
            role_groups: dict[tuple[tuple[str, str], ...], list[Facet]] = {}
            for role_tuple, facets in self._observations[v].items():
                if not _role_tuple_matches(role_tuple, entities):
                    continue
                matching = [f for f in facets if f.facet_id in facet_ids]
                for f in matching:
                    if (v, role_tuple, f.facet_id) in self._stub_keys:
                        raise StubSeriesReadError(
                            f"role_series read a stub row for variable {v!r}, "
                            f"role_tuple {role_tuple!r}, facet_id {f.facet_id!r}: this "
                            "fixture entry is a one-point LATEST-read placeholder with no "
                            "recorded observation history; re-record it via "
                            "tests/fixtures/record.py"
                        )
                narrowed = [_narrow_facet(f, time_period) for f in matching]
                if narrowed:
                    role_groups[role_tuple] = narrowed
            if role_groups:
                result[v] = role_groups
        return result

    def source_key(self) -> str:
        return "fake-availability"

    def availability_values(
        self,
        *,
        variables: Sequence[str],
        component: str,
        filters: Mapping[str, EntityFilter] | None = None,
        time_period: str | Sequence[str] | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
        read_seconds: float | None = None,
    ) -> list[str]:
        assert_no_un_variables(method="availability_values", variables=variables)
        with self._calls_lock:
            self.availability_calls.append(
                {
                    "variables": list(variables),
                    "component": component,
                    "filters": (
                        {
                            k: v if isinstance(v, ContainedIn) else list(v)
                            for k, v in filters.items()
                        }
                        if filters
                        else None
                    ),
                    "time_period": (
                        time_period
                        if time_period is None or isinstance(time_period, str)
                        else list(time_period)
                    ),
                    "max_values": max_values,
                    "max_bytes": max_bytes,
                    "deadline": deadline,
                }
            )
        _maybe_raise_for_component(self._raise_availability, component)
        key = _availability_key(component, variables, filters, time_period)
        if key in self._availability:
            return list(self._availability[key])
        if component == "observationAbout" and not filters and time_period is None:
            return self._scope_availability.availability_values(
                variables=variables,
                component=component,
                max_values=max_values,
                max_bytes=max_bytes,
                deadline=deadline,
                read_seconds=read_seconds,
            )
        return []

    def subject_observations(
        self,
        *,
        variables: Sequence[str],
        places: Sequence[str],
        facet_ids: Sequence[str] | None = None,
        latest_only: bool = True,
        max_rows: int | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, dict[str, list[SdmxObservation]]]:
        assert_no_un_variables(method="subject_observations", variables=variables)
        with self._calls_lock:
            call_index = len(self.subject_observations_calls)
            self.subject_observations_calls.append(
                {
                    "variables": list(variables),
                    "places": list(places),
                    "facet_ids": list(facet_ids) if facet_ids else None,
                    "latest_only": latest_only,
                    "max_rows": max_rows,
                    "max_bytes": max_bytes,
                }
            )
        _maybe_raise(self._raise_subject_observations, call_index)
        named = set(places)
        # The service returns only the requested facets' rows. A row naming no facet passes,
        # as a malformed upstream row would, so the read's own check sees it.
        wanted = set(facet_ids) if facet_ids else None
        return {
            v: {
                entity: [
                    r for r in rows if wanted is None or r.facet_id is None or r.facet_id in wanted
                ]
                for entity, rows in self._subject_observations[v].items()
                if entity in named
            }
            for v in variables
            if v in self._subject_observations
        }
