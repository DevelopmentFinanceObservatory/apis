"""Observation rows for one fetch, from the Data Commons that holds its variables.

DFO and base variables stream from our mixer's SDMX service. UN System Data
Commons variables (`undata/`) are absent there, so their rows come from the
UN deployment through the origin graph client, whose dated facet read carries
each observation's date and value. Those rows are laid out as the SDMX service
lays out its own: `observation_schema(("observationAbout",))`, series sorted
by variable, place and facet id, newest period first, one row per series under
`LATEST`, narrowed to the requested facet ids and capped at `max_rows`. Every
later stage reads them as it reads SDMX rows, and the read fails as an SDMX
read does.

One fetch names variables of one origin: `generation.request_batch_key` keeps
UN variables out of chunks with the others.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Final, Literal

import pyarrow as pa

from dc_graph import Facet, GraphCapExceededError, GraphDeadlineExceededError, GraphReadError
from dc_sdmx import (
    ContainedIn,
    EntityFilter,
    SdmxCapExceededError,
    SdmxClient,
    SdmxHttpError,
    SdmxResponseError,
    SdmxTransportError,
    observation_columns,
    observation_schema,
    place_slices,
)
from materialise.errors import MaterialisationUsageError
from materialise.graph import GraphReader, UnObservationReader
from origin_catalogue.graph import is_un_dcid

_SUBJECT_ROLE: Final = "observationAbout"
UN_OBSERVATION_ROLES: Final[tuple[str, ...]] = (_SUBJECT_ROLE,)
"""The UN deployment's observations name one entity, the subject."""

_DEADLINE_EXCEEDED_DETAIL: Final = (
    "the UN System Data Commons observation read exceeded its absolute caller deadline"
)

UN_READ_MAX_BYTES: Final = 64 * 1024 * 1024
"""The response bytes one UN dated read may spend: 64 MiB, the cap search's SDMX
reads apply to one read (`SEARCH_SDMX_MAX_BYTES`). The SDMX service stops
reading once rows pass `max_rows`; the UN read is one JSON body parsed whole,
so this cap bounds its memory instead. A dated read of WHO's under-five
mortality by wealth quintile at 20 countries spends 61 bytes per observation,
so the cap holds about a million observations."""


@contextmanager
def stream_observation_batches(
    *,
    sdmx: SdmxClient,
    graph: GraphReader,
    variables: Sequence[str],
    entities: Mapping[str, EntityFilter],
    facet_ids: Sequence[str] | None = None,
    time_period: Literal["LATEST"] | Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
    max_rows: int | None = None,
    deadline: float | None = None,
) -> Iterator[tuple[tuple[str, ...], Iterator[pa.RecordBatch]]]:
    """Yields `(roles, batches)` for `variables`, as `SdmxClient.stream_data_batches` does.

    Variables with no UN dcid stream through `sdmx` with every argument
    passed on, `time_period` as `LATEST` or a list of exact periods. A UN read takes
    `LATEST` alone and returns every period for a list, which its caller filters. A subject
    place list too long for one request's URL budget is read in slices (`dc_sdmx.place_slices`) one after another, every
    slice under the first slice's roles and the rows of all of them counted
    against one `max_rows`. UN variables are read by `un_observation_batches` from
    `graph.un_observation_reader()`, which is `None` unless `graph` is or wraps
    the origin graph client. A `None` reader, or a mix of the two kinds of
    variable, raises `MaterialisationUsageError`.
    """
    un_variables = [variable for variable in variables if is_un_dcid(variable)]
    if not un_variables:
        with _stream_place_slices(
            sdmx,
            variables=variables,
            entities=entities,
            facet_ids=facet_ids,
            time_period=time_period,
            roles=roles,
            max_rows=max_rows,
            deadline=deadline,
        ) as streamed:
            yield streamed
        return
    if len(un_variables) != len(variables):
        raise MaterialisationUsageError(
            f"one observation fetch names UN System Data Commons variables {un_variables!r} "
            "beside others; the two are read from different Data Commons"
        )
    un_reader = graph.un_observation_reader()
    if un_reader is None:
        raise MaterialisationUsageError(
            f"UN System Data Commons variables {un_variables!r} need a graph reader over the "
            f"origin graph client to read their rows; {type(graph).__name__} reaches no UN "
            "deployment"
        )
    yield (
        UN_OBSERVATION_ROLES,
        un_observation_batches(
            un_reader,
            variables=variables,
            entities=entities,
            facet_ids=facet_ids,
            latest=time_period == "LATEST",
            max_rows=max_rows,
            deadline=deadline,
        ),
    )


@contextmanager
def _stream_place_slices(
    sdmx: SdmxClient,
    *,
    variables: Sequence[str],
    entities: Mapping[str, EntityFilter],
    facet_ids: Sequence[str] | None,
    time_period: Literal["LATEST"] | Sequence[str] | None,
    roles: Sequence[str] | None,
    max_rows: int | None,
    deadline: float | None,
) -> Iterator[tuple[tuple[str, ...], Iterator[pa.RecordBatch]]]:
    """`sdmx.stream_data_batches`, reading a plain subject place list in URL-budget slices.

    The first slice's stream supplies the roles. Each later slice opens once the one before it
    is exhausted, names those roles, and gets what is left of `max_rows`."""
    subject = entities.get(_SUBJECT_ROLE)
    slice_entities: list[Mapping[str, EntityFilter]] = (
        [
            {**entities, _SUBJECT_ROLE: place_slice}
            for place_slice in place_slices(
                variables,
                role=_SUBJECT_ROLE,
                places=subject,
                facet_ids=facet_ids,
                time_period=time_period,
                url_budget_bytes=sdmx.url_budget_bytes,
            )
        ]
        if subject and not isinstance(subject, ContainedIn)
        else [entities]
    )

    def _stream(
        slice_filter: Mapping[str, EntityFilter],
        slice_roles: Sequence[str] | None,
        slice_max_rows: int | None,
    ) -> AbstractContextManager[tuple[tuple[str, ...], Iterator[pa.RecordBatch]]]:
        return sdmx.stream_data_batches(
            variables=variables,
            entities=slice_filter,
            facet_ids=facet_ids,
            time_period=time_period,
            roles=slice_roles,
            max_rows=slice_max_rows,
            deadline=deadline,
        )

    with _stream(slice_entities[0], roles, max_rows) as (first_roles, first_batches):

        def _batches() -> Iterator[pa.RecordBatch]:
            read = 0
            for batch in first_batches:
                read += batch.num_rows
                yield batch
            for slice_filter in slice_entities[1:]:
                left = None if max_rows is None else max(max_rows - read, 0)
                with _stream(slice_filter, first_roles, left) as (_roles, batches):
                    for batch in batches:
                        read += batch.num_rows
                        yield batch

        yield first_roles, _batches()


def un_observation_batches(
    graph: UnObservationReader,
    *,
    variables: Sequence[str],
    entities: Mapping[str, EntityFilter],
    facet_ids: Sequence[str] | None,
    latest: bool,
    max_rows: int | None,
    deadline: float | None,
) -> Iterator[pa.RecordBatch]:
    """Reads UN rows in one dated facet read, at once, and returns one batch per variable,
    each built as the iterator reaches it.

    Only facets named in `facet_ids` are kept. An empty or absent
    `facet_ids` keeps every facet, as the SDMX `facetId` filter does.
    `latest` keeps each series' newest observation. The read passes through
    admission, and places the UN front door refuses come back empty and are
    reported by the origin client's `un_blocked_places()`.

    A failed read raises the error class a failed SDMX read raises:
    `SdmxHttpError` with the upstream status, or `SdmxTransportError` when
    no response arrived. Response bytes past `UN_READ_MAX_BYTES` raise
    `SdmxCapExceededError`, and so do rows past `max_rows`, counted over the
    kept series before any batch is built.
    A kept series whose dated read came back short of its observation count
    raises `SdmxResponseError`. `deadline` bounds the read, and an expired
    one raises `TimeoutError` before it, during it and between batches.
    """
    places = _subject_places(entities)
    _check_deadline(deadline)
    try:
        facets = graph.observation_facets_batch(
            variables, places, needs_dates=True, max_bytes=UN_READ_MAX_BYTES, deadline=deadline
        )
    except GraphCapExceededError as exc:
        raise SdmxCapExceededError(
            f"the UN System Data Commons observation read passed its cap of {exc.cap} bytes",
            cap_kind="bytes",
            cap=exc.cap,
            observed=exc.observed,
        ) from exc
    except GraphDeadlineExceededError as exc:
        raise TimeoutError(_DEADLINE_EXCEEDED_DETAIL) from exc
    except GraphReadError as exc:
        message = f"the UN System Data Commons observation read failed: {exc}"
        if exc.upstream_status is None:
            raise SdmxTransportError(message) from exc
        raise SdmxHttpError(message, upstream_status=exc.upstream_status) from exc
    wanted = frozenset(facet_ids) if facet_ids else None
    _refuse_truncated_series(facets, wanted=wanted)
    kept = {
        variable: _kept_series(facets.get(variable, {}), wanted=wanted)
        for variable in sorted(set(variables))
    }
    rows = sum(
        min(len(facet.dates), 1) if latest else len(facet.dates)
        for series in kept.values()
        for _place, facet in series
    )
    if max_rows is not None and rows > max_rows:
        raise SdmxCapExceededError(
            f"the UN System Data Commons observation read returned more than {max_rows} rows",
            cap_kind="rows",
            cap=max_rows,
            observed=rows,
        )
    return _variable_batches(kept, latest=latest, deadline=deadline)


def _refuse_truncated_series(
    facets: Mapping[str, Mapping[str, Sequence[Facet]]], *, wanted: frozenset[str] | None
) -> None:
    """Raises `SdmxResponseError` when a kept series carries fewer dated observations than its
    observation count, so a short read fails instead of yielding a short series."""
    for variable, by_place in facets.items():
        for place, place_facets in by_place.items():
            for facet in place_facets:
                if facet.dates_complete or (wanted is not None and facet.facet_id not in wanted):
                    continue
                raise SdmxResponseError(
                    f"the UN System Data Commons observation read returned "
                    f"{len(facet.dates)} of {facet.obs_count} dated observations for "
                    f"{variable!r} at {place!r} (facet {facet.facet_id!r})",
                    upstream_status=200,
                )


def _subject_places(entities: Mapping[str, EntityFilter]) -> list[str]:
    """The places a UN read names: the subject's plain dcids, deduplicated in order."""
    if set(entities) != {_SUBJECT_ROLE}:
        raise MaterialisationUsageError(
            f"a UN System Data Commons read binds only {_SUBJECT_ROLE!r}; got roles "
            f"{sorted(entities)!r}"
        )
    subject = entities[_SUBJECT_ROLE]
    if isinstance(subject, ContainedIn):
        raise MaterialisationUsageError(
            "a UN System Data Commons read names its places; a containment filter "
            f"({subject.parent!r}, {subject.child_type!r}) reached it"
        )
    return list(dict.fromkeys(subject))


def _kept_series(
    by_place: Mapping[str, Sequence[Facet]], *, wanted: frozenset[str] | None
) -> list[tuple[str, Facet]]:
    """One variable's kept series as `(place, facet)`, sorted by place and facet id."""
    return sorted(
        (
            (place, facet)
            for place, place_facets in by_place.items()
            for facet in place_facets
            if wanted is None or facet.facet_id in wanted
        ),
        key=lambda pair: (pair[0], pair[1].facet_id or ""),
    )


def _variable_batches(
    kept: Mapping[str, Sequence[tuple[str, Facet]]], *, latest: bool, deadline: float | None
) -> Iterator[pa.RecordBatch]:
    for variable, series in kept.items():
        _check_deadline(deadline)
        yield _variable_batch(variable, series, latest=latest)


def _variable_batch(
    variable: str, series: Sequence[tuple[str, Facet]], *, latest: bool
) -> pa.RecordBatch:
    """One variable's rows, in `series` order, newest period first."""
    columns: dict[str, list[str | None]] = {
        name: [] for name in observation_columns(UN_OBSERVATION_ROLES)
    }
    for place, facet in series:
        observations = sorted(
            zip(facet.dates, facet.values, strict=True), key=lambda obs: obs[0], reverse=True
        )
        for date, value in observations[:1] if latest else observations:
            columns["variableMeasured"].append(variable)
            columns[_SUBJECT_ROLE].append(place)
            columns["unit"].append(facet.unit)
            columns["measurementMethod"].append(facet.measurement_method)
            columns["observationPeriod"].append(facet.observation_period)
            columns["provenance"].append(facet.provenance_id)
            columns["TIME_PERIOD"].append(date)
            columns["OBS_VALUE"].append("" if value is None else repr(float(value)))
            columns["scalingFactor"].append(facet.scaling_factor)
            columns["facetId"].append(facet.facet_id)
    return pa.RecordBatch.from_pydict(columns, schema=observation_schema(UN_OBSERVATION_ROLES))


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(_DEADLINE_EXCEEDED_DETAIL)


__all__ = [
    "UN_OBSERVATION_ROLES",
    "UN_READ_MAX_BYTES",
    "stream_observation_batches",
    "un_observation_batches",
]
