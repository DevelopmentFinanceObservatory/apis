"""Builds the base variable table, the withheld variable set and the topic
table from the graph.

No arc leads from a provenance to its variables, so the job finds them from
the observation data. It chooses scan places, then makes two passes over them.

1. Scan places are every country (`Earth <-containedInPlace+{typeOf:Country}`),
   Earth, every place of an aggregate type (`AGGREGATE_PLACE_TYPES`), and
   each registry entry's scan level. Admitted base provenances publish at
   continents, UN geographic regions, the European Union and the WHO regions
   typed `GeoRegion`. Countries and Earth are always kept. Any other place is
   kept when pass 1 finds a variable there, which drops most of the 1,236
   `undata-geo/*` regions.
2. Pass 1 reads each place's variable list with a place-only
   `/v2/observation` read, several places per request. The mixer answers it
   from its existence table. The answer lists variables only and ignores
   filters.
3. Pass 2 reads, per scan place, the variables pass 1 found there by name,
   selecting `facet` with no domain filter, and splits each variable's
   facets by provenance into admitted base, DFO's and the rest. A variable
   is kept under each admitted base provenance whose facet appears, and its
   bit for the place is set, as is the place's bit of each such provenance.
   A variable with a facet under one of the rest
   at some scan place and none under an admitted base provenance at any is
   withheld. DFO's facets count toward neither table, so a variable whose
   facets are all DFO's is in neither, as is a variable with no facet at
   any scan place, which covers the variables pass 1 lists that hold no data.

The topic table maps every Topic on our mixer to every leaf of its walk.
The walk is `topic_walk.walk_topics`, the one search runs at query time, with
its depth limit and no member cap. A topic with no leaves is left out. The
loaded catalogue applies admission and the member cap
(`CatalogueTables.admitted_topics`), as search does, so the table reads no
DFO data.

Cloud Run answers a response over 32 MiB with an empty HTTP 500 after the
mixer has done the work, so batches split in half on a 5xx, a request
timeout and the byte cap, down to one item. A single item is retried on
transient errors. A read that still fails, or the build deadline, raises
`EnumerationError`, and the job refuses to publish.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Final, Protocol

from artifact_publication.jobs import emit
from dc_graph import (
    GraphCapExceededError,
    GraphDeadlineExceededError,
    GraphReadError,
    SubjectPage,
)
from origin_catalogue.catalogue import BaseTables, BaseVariable, ProvenanceRow
from origin_catalogue.origins import BASE
from origin_catalogue.registry import Registry
from origin_catalogue.topic_walk import (
    TOPIC_EXPAND_MAX_DEPTH,
    ArcsReader,
    walk_topics,
)

OBSERVATION_PATH: Final[str] = "/v2/observation"

RESPONSE_CAP_BYTES: Final[int] = 32 * 1024 * 1024
"""Cloud Run's response limit. A larger response arrives as an empty 500."""

AGGREGATE_PLACE_TYPES: Final[tuple[str, ...]] = (
    "Continent",
    "UNGeoRegion",
    "ContinentalUnion",
    "GeoRegion",
)
"""Place types of the regional aggregates scanned beside the countries."""

_EARTH: Final[str] = "Earth"
_MAX_SUBJECTS: Final[int] = 100_000
"""Ceiling on one `<-typeOf` listing. A longer one fails the build."""


class EnumerationError(Exception):
    """A graph read failed after retries and splitting, a listing was
    truncated, or the build deadline passed."""


class EnumerationGraph(ArcsReader, Protocol):
    """The `dc_graph.GraphClient` reads the enumeration uses."""

    def child_places_batch(
        self, parent_dcids: Sequence[str], child_type: str, *, deadline: float | None = None
    ) -> dict[str, list[str]]: ...

    def node_subjects(
        self, dcid: str, *, property: str, max_results: int, deadline: float | None = None
    ) -> SubjectPage: ...

    def post_capped_bytes(
        self, path: str, payload: dict, *, max_bytes: int | None, deadline: float | None = None
    ) -> dict: ...


@dataclass(frozen=True, slots=True)
class EnumerationSettings:
    """`concurrency` bounds requests in flight in each pass. On dev, pass 2
    read about 20,000 pairs a second at 16. Pass 1 names up to
    `places_per_request` places per read, and the ten largest US states
    come to 20 MB. Pass 2 names up to `variables_per_request` variables per
    read, up to 11.4 MB for Germany. A single-item read is tried
    `max_attempts` times, waiting `retry_backoff_seconds[i]` after attempt
    `i + 1`."""

    concurrency: int = 16
    places_per_request: int = 10
    variables_per_request: int = 5_000
    max_attempts: int = 3
    retry_backoff_seconds: tuple[float, ...] = (1.0, 4.0)
    topic_max_depth: int = TOPIC_EXPAND_MAX_DEPTH

    def __post_init__(self) -> None:
        for name in (
            "concurrency",
            "places_per_request",
            "variables_per_request",
            "max_attempts",
            "topic_max_depth",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)!r}")
        if not self.retry_backoff_seconds or min(self.retry_backoff_seconds) < 0:
            raise ValueError(
                f"retry_backoff_seconds must be non-empty and non-negative, got "
                f"{self.retry_backoff_seconds!r}"
            )


@dataclass(frozen=True, slots=True)
class PassStats:
    """One pass: requests sent (retries and split halves included), batches
    split, (variable, place) pairs read, and wall time."""

    requests: int
    splits: int
    pairs: int
    seconds: float


@dataclass(frozen=True, slots=True)
class EnumerationStats:
    """What a build read, emitted as `origin_catalogue_enumerated`.
    `candidate_places` were read in pass 1; `scan_places` survived it."""

    candidate_places: int
    scan_places: int
    base_variables: int
    withheld_variables: int
    topics: int
    pass1: PassStats
    pass2: PassStats
    topic_roots: int
    topic_seconds: float


def build_base_tables(
    *,
    graph: EnumerationGraph,
    registry: Registry,
    admitted_base_provenances: Mapping[str, ProvenanceRow],
    dfo_provenances: frozenset[str],
    settings: EnumerationSettings | None = None,
    deadline: float | None = None,
) -> BaseTables:
    """Builds the base variable, withheld variable and topic tables, and
    emits `EnumerationStats` as the `origin_catalogue_enumerated` event.

    `admitted_base_provenances` are the provenance table's base rows, the
    provenances pass 2 admits facets under. `dfo_provenances` are DFO's
    provenance ids, whose facets neither admit nor withhold a variable. `scan_levels` is the
    registry's. `deadline` is an absolute `time.monotonic()` value bounding the build.

    Raises `EnumerationError` when a read fails for good or the deadline
    passes, and `ValueError` for an empty or non-base provenance set.
    """
    run = _Run(graph=graph, settings=settings or EnumerationSettings(), deadline=deadline)
    admitted = _admitted_ids(admitted_base_provenances)
    candidates, always = _candidate_places(run, registry)

    started = time.monotonic()
    variables_at = run.pass1(candidates)
    pass1 = run.close_pass(started, pairs=sum(len(v) for v in variables_at.values()))
    scan_places = tuple(p for p in candidates if p in always or variables_at.get(p))

    started = time.monotonic()
    base_variables, provenance_places, withheld_variables = run.pass2(
        scan_places, variables_at, admitted=admitted, dfo=dfo_provenances
    )
    pass2 = run.close_pass(started, pairs=sum(len(variables_at.get(p, ())) for p in scan_places))
    del variables_at

    started = time.monotonic()
    roots, topics = _topic_table(run)
    stats = EnumerationStats(
        candidate_places=len(candidates),
        scan_places=len(scan_places),
        base_variables=len(base_variables),
        withheld_variables=len(withheld_variables),
        topics=len(topics),
        pass1=pass1,
        pass2=pass2,
        topic_roots=roots,
        topic_seconds=time.monotonic() - started,
    )
    emit("origin_catalogue_enumerated", **asdict(stats))
    return BaseTables(
        base_variables=base_variables,
        provenance_places=provenance_places,
        withheld_variables=withheld_variables,
        topics=topics,
        scan_places=scan_places,
        scan_levels=registry.scan_levels(),
    )


def _admitted_ids(provenances: Mapping[str, ProvenanceRow]) -> frozenset[str]:
    """The admitted base provenance ids, refusing an empty set, under which
    every variable with data would be withheld, and a non-base row."""
    if not provenances:
        raise ValueError("no admitted base provenances; pass 2 would withhold every variable")
    for provenance_id, row in provenances.items():
        if row.data_commons != BASE:
            raise ValueError(f"provenance {provenance_id!r} is {row.data_commons.id}, not base")
    return frozenset(provenances)


def _candidate_places(run: _Run, registry: Registry) -> tuple[tuple[str, ...], frozenset[str]]:
    """Every place pass 1 reads, in bit order (Earth, countries, then the
    rest by dcid), and the places kept whatever pass 1 finds."""
    countries = run.child_places(_EARTH, "Country")
    always = frozenset({_EARTH, *countries})
    others: set[str] = set()
    for place_type in AGGREGATE_PLACE_TYPES:
        others.update(run.places_of_type(place_type))
    levels = {level for entry in registry.base if (level := entry.scan_level) is not None}
    for level in sorted(levels, key=lambda lv: (lv.place_type, lv.within or "")):
        if level.within is None:
            others.update(run.places_of_type(level.place_type))
        else:
            others.update(run.child_places(level.within, level.place_type))
    ordered = (_EARTH, *sorted(set(countries) - {_EARTH}), *sorted(others - always))
    return ordered, always


def _topic_table(run: _Run) -> tuple[int, dict[str, tuple[str, ...]]]:
    """Walks every Topic and keeps each with leaves. Returns the root count
    and the table."""
    roots = sorted(set(run.places_of_type("Topic")))
    run.check_deadline()
    try:
        walked = walk_topics(
            roots,
            graph=run.graph,
            max_depth=run.settings.topic_max_depth,
            max_members=None,
        )
    except GraphReadError as exc:
        raise EnumerationError(f"topic walk failed: {exc}") from exc
    run.check_deadline()
    return len(roots), {topic: tuple(leaves) for topic, leaves in walked.items() if leaves}


type _Facets = tuple[set[str], bool]
"""One variable's facets at one place: the admitted base provenances among
them, and whether any is under a provenance neither admitted nor DFO's."""


class _SplitBatch(Exception):
    """A multi-item read failed in a way a smaller read may not."""


@dataclass
class _Run:
    """One build's graph, settings, deadline and request counters."""

    graph: EnumerationGraph
    settings: EnumerationSettings
    deadline: float | None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _requests: int = 0
    _splits: int = 0

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise EnumerationError("the catalogue build passed its deadline")

    def close_pass(self, started: float, *, pairs: int) -> PassStats:
        """The counters since the last call, as one pass's stats."""
        with self._lock:
            stats = PassStats(
                requests=self._requests,
                splits=self._splits,
                pairs=pairs,
                seconds=time.monotonic() - started,
            )
            self._requests = self._splits = 0
        return stats

    def child_places(self, parent: str, place_type: str) -> list[str]:
        """`<-containedInPlace+{typeOf:<place_type>}` of `parent`."""
        try:
            return self.graph.child_places_batch([parent], place_type, deadline=self.deadline)[
                parent
            ]
        except GraphReadError as exc:
            raise EnumerationError(f"listing {place_type} in {parent!r} failed: {exc}") from exc

    def places_of_type(self, place_type: str) -> tuple[str, ...]:
        """Every `<-typeOf` subject of `place_type`, refusing a truncated list."""
        try:
            page = self.graph.node_subjects(
                place_type, property="typeOf", max_results=_MAX_SUBJECTS, deadline=self.deadline
            )
        except GraphReadError as exc:
            raise EnumerationError(f"listing {place_type!r} instances failed: {exc}") from exc
        if page.truncated:
            raise EnumerationError(f"{place_type!r} has more than {_MAX_SUBJECTS} instances")
        return page.dcids

    # --- Pass 1 --------------------------------------------------------------

    def pass1(self, places: Sequence[str]) -> dict[str, list[str]]:
        """Each place's variables, sorted. Strings are shared across places."""
        canonical: dict[str, str] = {}
        variables_at: dict[str, list[str]] = {}
        groups = _chunks(places, self.settings.places_per_request)
        for reads in self._run_all(groups, lambda group: self._split_reads(group, self._pass1)):
            for found in reads:
                for place, variables in found.items():
                    variables_at[place] = sorted(canonical.setdefault(v, v) for v in variables)
        return variables_at

    def _pass1(self, places: Sequence[str]) -> dict[str, list[str]]:
        body = self._read(
            {"select": ["variable", "entity"], "entity": {"dcids": list(places)}},
            splittable=len(places) > 1,
        )
        found: dict[str, list[str]] = {place: [] for place in places}
        for variable, entry in body.get("byVariable", {}).items():
            for place in entry.get("byEntity", {}):
                if place in found:
                    found[place].append(variable)
        return found

    # --- Pass 2 --------------------------------------------------------------

    def pass2(
        self,
        scan_places: Sequence[str],
        variables_at: Mapping[str, Sequence[str]],
        *,
        admitted: frozenset[str],
        dfo: frozenset[str],
    ) -> tuple[dict[str, BaseVariable], dict[str, int], frozenset[str]]:
        """The base variable table, admitted provenances and place bits per
        variable; the place bits of each admitted provenance holding one;
        and the withheld variables."""
        size = self.settings.variables_per_request
        tasks = [
            (bit, place, variables[i : i + size])
            for bit, place in enumerate(scan_places)
            if (variables := variables_at.get(place))
            for i in range(0, len(variables), size)
        ]
        provenances: dict[str, set[str]] = {}
        places: dict[str, int] = {}
        provenance_places: dict[str, int] = {}
        unadmitted: set[str] = set()

        def read(task: tuple[int, str, Sequence[str]]) -> tuple[int, dict[str, _Facets]]:
            bit, place, variables = task
            merged: dict[str, _Facets] = {}
            for found in self._split_reads(
                variables, lambda chunk: self._pass2(place, chunk, admitted=admitted, dfo=dfo)
            ):
                merged.update(found)
            return bit, merged

        for bit, found in self._run_all(tasks, read):
            for variable, (held, other) in found.items():
                if other:
                    unadmitted.add(variable)
                if held:
                    provenances.setdefault(variable, set()).update(held)
                    places[variable] = places.get(variable, 0) | (1 << bit)
                    for provenance in held:
                        provenance_places[provenance] = provenance_places.get(provenance, 0) | (
                            1 << bit
                        )
        canonical_lists: dict[tuple[str, ...], tuple[str, ...]] = {}
        table = {
            variable: BaseVariable(
                provenances=canonical_lists.setdefault(key := tuple(sorted(held)), key),
                places=places[variable],
            )
            for variable, held in sorted(provenances.items())
        }
        return (
            table,
            dict(sorted(provenance_places.items())),
            frozenset(unadmitted.difference(table)),
        )

    def _pass2(
        self,
        place: str,
        variables: Sequence[str],
        *,
        admitted: frozenset[str],
        dfo: frozenset[str],
    ) -> dict[str, _Facets]:
        """Each variable with a facet at `place`, mapped to the admitted
        provenances among its facets', which may be none, and whether one of
        its facets is under a provenance neither admitted nor DFO's."""
        body = self._read(
            {
                "select": ["variable", "entity", "facet"],
                "variable": {"dcids": list(variables)},
                "entity": {"dcids": [place]},
            },
            splittable=len(variables) > 1,
        )
        facets = body.get("facets", {})
        found: dict[str, _Facets] = {}
        for variable, entry in body.get("byVariable", {}).items():
            ordered = entry.get("byEntity", {}).get(place, {}).get("orderedFacets", [])
            present = {
                provenance
                for block in ordered
                if (provenance := facets.get(block.get("facetId"), {}).get("provenanceId"))
            }
            if present:
                found[variable] = (present & admitted, not present <= admitted | dfo)
        return found

    # --- Reads ---------------------------------------------------------------

    def _split_reads[T](self, items: Sequence[str], read: Callable[[Sequence[str]], T]) -> list[T]:
        """`read(items)`, or the reads of its halves, recursively, when it
        raises `_SplitBatch`."""
        try:
            return [read(items)]
        except _SplitBatch:
            with self._lock:
                self._splits += 1
            mid = len(items) // 2
            return self._split_reads(items[:mid], read) + self._split_reads(items[mid:], read)

    def _read(self, payload: dict, *, splittable: bool) -> dict:
        """One observation read. A 5xx, a request timeout or the byte cap on
        a splittable read raises `_SplitBatch`, since a smaller read may
        succeed. A transport error or 429 is retried, and so is a 5xx or
        timeout on a single item. Anything else, a read out of attempts, or
        the build deadline raises `EnumerationError`."""
        backoff = self.settings.retry_backoff_seconds
        attempt = 0
        while True:
            attempt += 1
            self.check_deadline()
            with self._lock:
                self._requests += 1
            try:
                return self.graph.post_capped_bytes(
                    OBSERVATION_PATH, payload, max_bytes=RESPONSE_CAP_BYTES, deadline=self.deadline
                )
            except GraphCapExceededError as exc:
                if splittable:
                    raise _SplitBatch from exc
                raise EnumerationError(f"one item's observation read exceeds {exc}") from exc
            except GraphReadError as exc:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    raise EnumerationError("the catalogue build passed its deadline") from exc
                status = exc.upstream_status
                server_error = status is not None and status >= 500
                timed_out = isinstance(exc, GraphDeadlineExceededError)
                if (server_error or timed_out) and splittable:
                    raise _SplitBatch from exc
                if not (status is None or server_error or status == 429):
                    raise EnumerationError(f"observation read refused: {exc}") from exc
                if attempt >= self.settings.max_attempts:
                    raise EnumerationError(
                        f"observation read failed {attempt} times: {exc}"
                    ) from exc
                pause = backoff[min(attempt - 1, len(backoff) - 1)]
                if self.deadline is not None and time.monotonic() + pause >= self.deadline:
                    raise EnumerationError("the catalogue build passed its deadline") from exc
                time.sleep(pause)

    def _run_all[T, R](self, tasks: Sequence[T], fn: Callable[[T], R]) -> Iterator[R]:
        """`fn` over `tasks` with `settings.concurrency` in flight, yielding
        results as they finish. The first failure cancels what has not
        started and is raised once the running tasks end."""
        executor = ThreadPoolExecutor(max_workers=self.settings.concurrency)
        try:
            for future in as_completed([executor.submit(fn, task) for task in tasks]):
                yield future.result()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def _chunks(items: Sequence[str], size: int) -> list[Sequence[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


__all__ = [
    "AGGREGATE_PLACE_TYPES",
    "OBSERVATION_PATH",
    "RESPONSE_CAP_BYTES",
    "EnumerationError",
    "EnumerationGraph",
    "EnumerationSettings",
    "EnumerationStats",
    "PassStats",
    "build_base_tables",
]
