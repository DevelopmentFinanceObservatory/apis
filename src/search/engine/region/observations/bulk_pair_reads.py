"""SDMX reads for a role-bearing per-series preview's many reads, batched into exact requests.

A per-series preview runs one `materialise_preview` per member and bound role tuple
(`fetch._Dispatch`), and each streams its rows from SDMX in a request of its own. Above
`obs_preview_concurrency` reads those requests run in more than one wave of the preview's
pool. `BulkPairSdmx` stands in for the materialise SDMX client those reads share: the first
read of a request fetches, in one `stream_data_batches` call, the rows of every planned read
the request covers, and each read is then served the rows its own request selects from them.
Materialise runs on each read over those rows, so election, collapse, conversion and the
preview's assembly see the rows a request of the read's own would have returned.

A read's rows are the bulk rows whose variable is the read's, whose dcid on each role the read
filters is one it names, and whose facet id is one it dispatches. SDMX filters variables,
entities and facets as a product, and `LATEST` keeps the newest row of each series (a variable,
its role values and its facet), so the rows of a narrower request are the rows of a wider one
that match its filters. Reads are grouped by the roles their variables declare and the roles
they filter, so one bulk request never mixes variables whose responses would name different
role columns.

A bulk request names the union of its reads' dcids on each role, so it asks for exactly the
reads' series only when its reads differ on one role at most. A time period's reads of a
group are therefore split into requests whose reads share every role's dcids but one, and
share the dcids of that one role per variable: the role that gives the fewest requests. The
request is split further by the client when it exceeds the URL budget.

Every case the selection cannot reproduce goes to the wrapped client, request by request: a
request no planned read matches, a bulk read that fails, crosses its row or byte cap or the
deadline, a response whose role columns differ from the group's declared roles, and a response
holding a row with no facet id. A bulk read that cannot serve costs each read of its request
one request of its own, with the failures and partial delivery of a per-series read.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

from dc_sdmx import (
    ContainedIn,
    EntityFilter,
    SdmxCapExceededError,
    SdmxClient,
    SdmxClientError,
)
from dc_sdmx.results import VARIABLE_COLUMN, observation_schema
from search.engine.config import settings

logger = logging.getLogger(__name__)

_FACET_COLUMN = "facetId"
_STAT_VAR_PREFIX = "dcid:"

type _Period = Literal["LATEST"] | tuple[str, ...] | None
type _Bound = tuple[tuple[str, tuple[str, ...]], ...]
type _Group = tuple[frozenset[str], frozenset[str]]


@dataclass(frozen=True)
class PlannedRead:
    """One per-series read the preview will make: its bare `variable`, the dcids it filters
    each bound role on (`bound`, in the read's role order), the facet ids it may dispatch
    (its planned group), the roles its variable declares, and under a `latest` window the
    year its member elected (`elected_year`)."""

    variable: str
    bound: _Bound
    facet_ids: frozenset[str]
    declared_roles: frozenset[str]
    latest: bool
    elected_year: int | None = None

    @property
    def group(self) -> _Group:
        """The reads one bulk request may serve together: the same declared roles and the
        same filtered roles."""
        return self.declared_roles, frozenset(role for role, _dcids in self.bound)

    def requests(self, period: _Period) -> bool:
        """Whether materialise's read may ask for `period`: no time filter outside `latest`;
        `LATEST`, or the one year its member elected, under it
        (`materialise.generation._time_period_for_window`)."""
        if not self.latest:
            return period is None
        if period == "LATEST":
            return True
        return self.elected_year is not None and period == (str(self.elected_year),)


@dataclass
class _Request:
    """One bulk request: the `reads` it serves and, once read, its rows (None when the bulk
    read cannot serve them). `lock` makes the first read to need it the only one to fetch."""

    reads: tuple[PlannedRead, ...]
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: bool = False
    rows: _BulkRows | None = None


@dataclass(frozen=True)
class _BulkRows:
    """A bulk response: its role columns and every row."""

    roles: tuple[str, ...]
    table: pa.Table


def _period_key(time_period: Literal["LATEST"] | Sequence[str] | None) -> _Period:
    if time_period is None:
        return None
    if time_period == "LATEST":
        return "LATEST"
    return tuple(time_period)


def _bound_key(entities: Mapping[str, EntityFilter]) -> _Bound | None:
    """`entities` as a planned read's `bound`, or None for a containment filter, which no
    planned read sends."""
    bound: list[tuple[str, tuple[str, ...]]] = []
    for role, dcids in entities.items():
        if isinstance(dcids, ContainedIn):
            return None
        bound.append((role, tuple(dcids)))
    return tuple(bound)


def _plan_requests(reads: Sequence[PlannedRead]) -> dict[tuple[str, _Bound], _Request]:
    """`reads`, which share a group, split into requests whose product of filters is the set of
    series their reads need: the reads of a request share every role's dcids but those of one
    role `varying`, and share the union of that role's dcids per variable. The role chosen is
    the one giving the fewest requests, the first of the group's filtered roles on a tie."""
    roles = [role for role, _dcids in reads[0].bound]
    best: dict[tuple[object, ...], list[PlannedRead]] | None = None
    for varying in roles or [None]:
        fixed = {
            r: tuple((role, dcids) for role, dcids in r.bound if role != varying) for r in reads
        }
        union: dict[tuple[object, str], set[str]] = {}
        for r in reads:
            dcids = dict(r.bound).get(varying, ())
            union.setdefault((fixed[r], r.variable), set()).update(dcids)
        split: dict[tuple[object, ...], list[PlannedRead]] = {}
        for r in reads:
            key = (fixed[r], frozenset(union[(fixed[r], r.variable)]))
            split.setdefault(key, []).append(r)
        if best is None or len(split) < len(best):
            best = split
    assert best is not None
    return {
        (r.variable, r.bound): request
        for members in best.values()
        for request in [_Request(tuple(members))]
        for r in members
    }


class BulkPairSdmx:
    """The materialise SDMX client for one preview's `reads`, serving each read's
    `stream_data_batches` from one bulk request per group and time period (module
    docstring), and every other call through `inner`. `deadline` bounds the bulk reads."""

    def __init__(
        self, inner: SdmxClient, reads: Sequence[PlannedRead], *, deadline: float | None
    ) -> None:
        self._inner = inner
        self._deadline = deadline
        self._reads = {(read.variable, read.bound): read for read in reads}
        self._requests: dict[tuple[_Group, _Period], dict[tuple[str, _Bound], _Request]] = {}
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    @contextmanager
    def stream_data_batches(
        self,
        *,
        variables: Sequence[str],
        entities: Mapping[str, EntityFilter],
        facet_ids: Sequence[str] | None = None,
        time_period: Literal["LATEST"] | Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        max_rows: int | None = None,
        max_bytes: int | None = None,
        deadline: float | None = None,
    ) -> Iterator[tuple[tuple[str, ...], Iterator[pa.RecordBatch]]]:
        """`SdmxClient.stream_data_batches`, served from the bulk rows when one planned read
        makes this request, else by the wrapped client. A served read past `max_rows` raises
        `SdmxCapExceededError`, as its own request would."""
        period = _period_key(time_period)
        bound = _bound_key(entities)
        read = (
            self._reads.get((variables[0], bound))
            if len(variables) == 1 and bound is not None
            else None
        )
        rows = (
            self._bulk_rows(read, period)
            if read is not None
            and roles is None
            and max_bytes is None
            and facet_ids
            and read.facet_ids.issuperset(facet_ids)
            and read.requests(period)
            else None
        )
        if rows is None:
            with self._inner.stream_data_batches(
                variables=variables,
                entities=entities,
                facet_ids=facet_ids,
                time_period=time_period,
                roles=roles,
                max_rows=max_rows,
                max_bytes=max_bytes,
                deadline=deadline,
            ) as streamed:
                yield streamed
            return
        assert read is not None and facet_ids  # `rows`
        selected = _select(rows, read.variable, bound=read.bound, facet_ids=facet_ids)
        if max_rows is not None and selected.num_rows > max_rows:
            raise SdmxCapExceededError(
                f"SDMX /data read exceeded the row cap of {max_rows} "
                f"(observed {selected.num_rows} rows)",
                cap_kind="rows",
                cap=max_rows,
                observed=selected.num_rows,
            )
        yield rows.roles, iter(selected.to_batches())

    def _bulk_rows(self, read: PlannedRead, period: _Period) -> _BulkRows | None:
        """The bulk rows of the request that serves `read` at `period`, read once for every
        planned read of the request, or None when the bulk read cannot serve them. Only the
        workers needing the same request wait on its read."""
        key = (read.group, period)
        with self._lock:
            if key not in self._requests:
                self._requests[key] = _plan_requests(
                    [
                        r
                        for r in self._reads.values()
                        if r.group == read.group and r.requests(period)
                    ]
                )
            request = self._requests[key][(read.variable, read.bound)]
        with request.lock:
            if not request.done:
                request.rows = self._read_bulk(request.reads, period)
                request.done = True
            return request.rows

    def _read_bulk(self, reads: Sequence[PlannedRead], period: _Period) -> _BulkRows | None:
        variables = list(dict.fromkeys(r.variable for r in reads))
        entities: dict[str, set[str]] = {}
        for r in reads:
            for role, dcids in r.bound:
                entities.setdefault(role, set()).update(dcids)
        declared, filtered = reads[0].group
        started = time.monotonic()
        try:
            with self._inner.stream_data_batches(
                variables=variables,
                entities={role: sorted(dcids) for role, dcids in entities.items()},
                facet_ids=sorted({f for r in reads for f in r.facet_ids}),
                time_period=list(period) if isinstance(period, tuple) else period,
                max_rows=settings().sdmx_max_rows,
                max_bytes=settings().sdmx_max_bytes,
                deadline=self._deadline,
            ) as (roles, batches):
                table = pa.Table.from_batches(list(batches), schema=observation_schema(roles))
        except (SdmxClientError, TimeoutError):
            logger.warning(
                "bulk pair read of %d variables at %r failed; each read requests its own rows",
                len(variables),
                period,
                exc_info=True,
            )
            return None
        if set(roles) != declared or not filtered <= declared:
            logger.warning(
                "bulk pair read of %d variables at %r named roles %r for the declared %r "
                "filtered on %r; each read requests its own rows",
                len(variables),
                period,
                sorted(roles),
                sorted(declared),
                sorted(filtered),
            )
            return None
        if table[_FACET_COLUMN].null_count:
            logger.warning(
                "bulk pair read of %d variables at %r holds rows with no facet id; each read "
                "requests its own rows",
                len(variables),
                period,
            )
            return None
        logger.info(
            "bulk pair read of %d variables for %d reads at %r: %d rows, %.3fs",
            len(variables),
            len(reads),
            period,
            table.num_rows,
            time.monotonic() - started,
        )
        return _BulkRows(roles=tuple(roles), table=table)


def _select(rows: _BulkRows, variable: str, *, bound: _Bound, facet_ids: Sequence[str]) -> pa.Table:
    """The rows of `rows` a request for `variable`, filtering each role of `bound` on its
    dcids and the facets on `facet_ids`, returns."""
    table = rows.table
    # A response states a variable bare or `dcid:`-prefixed.
    variables = pa.array([variable, f"{_STAT_VAR_PREFIX}{variable}"], type=pa.string())
    mask = pc.and_(
        pc.is_in(table[VARIABLE_COLUMN], value_set=variables),
        pc.is_in(table[_FACET_COLUMN], value_set=pa.array(list(facet_ids), type=pa.string())),
    )
    for role, dcids in bound:
        mask = pc.and_(
            mask, pc.is_in(table[role], value_set=pa.array(list(dcids), type=pa.string()))
        )
    return table.filter(mask)
