"""Builds and publishes the search catalogue: `python -m search.catalogue.job`.

A run:

1. computes the build key from the build ids the origin catalogue's and
   the coverage index's pointers name, `DFO_BUILD_COMMIT` and a digest of
   `examples.toml`, and exits without loading either artifact when
   `current.json` already carries it (`search_catalogue_job_noop`) or
   `refused.json` refused it within `REFUSAL_RETRY_SECONDS`
   (`search_catalogue_job_still_refused`);
2. loads the origin catalogue build `ORIGIN_CATALOGUE_URI` names, under
   the serving surfaces' `ORIGIN_CATALOGUE_*` settings, and the coverage
   index `COVERAGE_ARTIFACT_URI` names, and checks the key again against
   the builds it loaded;
3. takes the create-only lease `lease.json` at the format root and exits
   while another run holds an unexpired one;
4. finds each example's places with data (`search.catalogue.presence`) and
   counts publishers, countries and aid (`search.catalogue.counts`);
5. runs each example through the search engine in-process, with the
   origin catalogue build bound, and the served graph release when
   `GRAPH_RELEASE_URI` is set, at its default place and at the test
   places where it has data (`search.catalogue.check`), trying a run that
   raised, timed out or had its preview read fail once more. When a run
   still fails in the engine, the job logs `search_catalogue_job_failed`
   at `ERROR` and exits 1 without publishing or recording a refusal, so
   the next scheduled run builds again. A line that fails presence or any
   other run, including one the engine refused as input, drops out and
   logs `search_catalogue_line_dropped` at `WARNING`;
6. refuses to publish when fewer than `FLOOR` lines pass, logging
   `search_catalogue_job_refused` at `ERROR`, recording the key in
   `refused.json` and exiting 1, so the last published build keeps
   serving;
7. discards the build if the coverage index moved while it ran, and
   otherwise publishes it with a conditional `current.json` write.

Every run prunes builds nothing can still be reading. Every run that exits
0 logs `search_catalogue_job_heartbeat` last, naming the published build
and its age.

With `GRAPH_RELEASE_COVERAGE` on, the served graph release takes the
coverage index's place, and `GRAPH_RELEASE_URI` is required. Step 1 keys on
the release id the release pointer names, and from step 2 to the end the run
binds the served release and reads its coverage view. The release id fills
the build inputs' `coverage_build_id`. The bound release never moves, so
step 7 publishes a build of the release the run bound. A run with no release
to bind, because `GRAPH_RELEASE_URI` is unset, no release is published or the
release's views do not build, logs `search_catalogue_job_failed` with the
`reason` and exits 1. The first two end the run before the origin catalogue
loads.

Environment: `SEARCH_CATALOGUE_URI` (`gs://` or `file://`),
`ORIGIN_CATALOGUE_URI` and `COVERAGE_ARTIFACT_URI` are required, with the
engine's own settings (`GEMINI_API_KEY`, `SEARCH_GRAPH_BASE`, ...).
`SEARCH_CATALOGUE_MAX_CONCURRENCY` bounds concurrent engine calls (default
4).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from artifact_publication.jobs import (
    AliasOnlySettings,
    acquire_lease,
    emit,
    new_build_id,
    release_lease,
)
from artifact_publication.uri import ArtifactUriError
from code_index.shared import get_index
from dc_graph import GraphReadError
from dfo_platform.build import build_commit
from graph_release.context import (
    URI_ENV_VAR,
    ReleaseNotBound,
    ReleaseViewsNotBuilt,
    bound_served_release,
    coverage_views_enabled,
    optional_release_service,
    release_uri,
)
from origin_catalogue import bound_catalogue
from origin_catalogue.artifact import CatalogueArtifactError as OriginArtifactError
from origin_catalogue.artifact import load_catalogue
from origin_catalogue.artifact import read_published_pointer as read_origin_pointer
from origin_catalogue.catalogue import CatalogueSnapshot
from origin_catalogue.service import CatalogueSettings
from search.catalogue.artifact import (
    BuildInputs,
    CatalogueArtifact,
    CatalogueArtifactError,
    CatalogueExample,
    CataloguePlace,
    CataloguePointer,
    CatalogueRefusal,
    build_key,
    format_root,
    load_build,
    prune,
    publish,
    read_pointer,
    read_refusal,
    record_refusal,
)
from search.catalogue.check import INFRA_FAILURES, Resolve, check_example, check_places
from search.catalogue.counts import catalogue_counts
from search.catalogue.examples import Example, ExamplesError, ExampleSet, load_examples
from search.catalogue.presence import PresenceRefusal, example_places
from search.models import ResolveRequest, ResolveResponse
from series_coverage.loader import CoverageArtifactError
from series_coverage.shared import (
    CoverageIncompatibleError,
    CoverageIndexCheckValueError,
    CoverageSnapshot,
    ServedCoverage,
    get_coverage_service,
)

if TYPE_CHECKING:
    from graph_release.service import ReleaseService

logger = logging.getLogger(__name__)

FLOOR: Final[int] = 5
"""The fewest passing lines a build publishes."""

_MAX_RUN_SECONDS: Final[float] = 1800.0
"""The lease's lifetime and the pruning grace period, well above the 80 to 95
seconds a live build takes for its counts and about 16 engine calls."""

REFUSAL_RETRY_SECONDS: Final[float] = 6 * 3600.0
"""How long a refused key stays refused. A run with that key inside the
window makes no engine calls; the first run after it builds again, since an
engine or graph outage can refuse a build whose inputs would pass."""


class CatalogueFloorError(Exception):
    """Fewer than `FLOOR` lines passed, so the run published nothing."""


class CatalogueEngineError(Exception):
    """A check run still failed in the engine after its retry, so the run
    published nothing and recorded no refusal."""


class ReleaseUnavailable(Exception):
    """The run has no graph release to bind; `reason` names why."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


class JobSettings(AliasOnlySettings):
    """Job settings from the environment. Each field's `alias` is the only
    variable read; construct by field name to override one."""

    model_config = SettingsConfigDict(frozen=True)

    artifact_uri: str = Field(min_length=1, alias="SEARCH_CATALOGUE_URI")
    origin_catalogue_uri: str = Field(min_length=1, alias="ORIGIN_CATALOGUE_URI")
    coverage_uri: str = Field(min_length=1, alias="COVERAGE_ARTIFACT_URI")
    max_concurrency: int = Field(default=4, ge=1, alias="SEARCH_CATALOGUE_MAX_CONCURRENCY")


class CoverageSource(Protocol):
    """Where the engine reads the coverage index, which
    `series_coverage.shared.CoverageService` provides."""

    def snapshot(self) -> CoverageSnapshot: ...


type PlaceNames = Callable[[Sequence[str]], Mapping[str, str]]
"""Maps place dcids to the name a template is filled with, leaving out a
place the graph names nothing for."""


async def run(
    settings: JobSettings,
    *,
    examples: ExampleSet,
    catalogue: CatalogueSnapshot,
    coverage: CoverageSource,
    resolve: Resolve,
    place_names: PlaceNames,
    commit: str,
    clock: Callable[[], float] = time.time,
) -> None:
    """One job run. `resolve` must read `catalogue`, which the caller binds.
    Raises `CatalogueFloorError` after logging and recording a refused
    build, and `CatalogueEngineError` after logging a check run the engine
    failed twice; a no-op, a still-refused key, a skip, a discard or a lost
    publish race returns, and a run that returns logs its heartbeat last."""
    await _build(
        settings,
        examples=examples,
        catalogue=catalogue,
        coverage=coverage,
        resolve=resolve,
        place_names=place_names,
        commit=commit,
        clock=clock,
    )
    _heartbeat(settings, clock=clock)


def nothing_to_build(
    settings: JobSettings, *, inputs: BuildInputs, clock: Callable[[], float] = time.time
) -> bool:
    """Whether a run with `inputs` has nothing to build: the published build
    carries their key, or a build with it was refused within
    `REFUSAL_RETRY_SECONDS`. When so, logs the no-op and the heartbeat, as
    `run` would. `main` asks this with the build ids the two artifacts'
    pointers name, so an unchanged run loads neither artifact."""
    published, _ = read_pointer(settings.artifact_uri)
    if not _settled(settings, build_key(inputs), published=published, clock=clock):
        return False
    _heartbeat(settings, clock=clock)
    return True


def _settled(
    settings: JobSettings,
    key: str,
    *,
    published: CataloguePointer | None,
    clock: Callable[[], float],
) -> bool:
    """Whether a run with `key` has nothing to build. Logs why and prunes
    when so."""
    if published is not None and published.key == key:
        emit(
            "search_catalogue_job_noop",
            key=key,
            catalogue_version=published.catalogue_version,
        )
    else:
        refusal = read_refusal(settings.artifact_uri)
        if (
            refusal is None
            or refusal.key != key
            or clock() - refusal.refused_at.timestamp() >= REFUSAL_RETRY_SECONDS
        ):
            return False
        emit(
            "search_catalogue_job_still_refused",
            severity="WARNING",
            key=key,
            refused_at=refusal.refused_at.isoformat(),
            passed=refusal.passed,
        )
    _prune(settings, clock)
    return True


async def _build(
    settings: JobSettings,
    *,
    examples: ExampleSet,
    catalogue: CatalogueSnapshot,
    coverage: CoverageSource,
    resolve: Resolve,
    place_names: PlaceNames,
    commit: str,
    clock: Callable[[], float],
) -> None:
    """`run` without the heartbeat."""
    covered = coverage.snapshot()
    inputs = BuildInputs(
        origin_catalogue_build_id=catalogue.build_id,
        coverage_build_id=covered.manifest.build_id,
        build_commit=commit,
        examples_digest=examples.digest,
    )
    key = build_key(inputs)
    published, generation = read_pointer(settings.artifact_uri)
    if _settled(settings, key, published=published, clock=clock):
        return

    root = format_root(settings.artifact_uri)
    version = new_build_id()
    lease = acquire_lease(root, build_id=version, ttl_seconds=_MAX_RUN_SECONDS, now=clock)
    if lease is None:
        emit("search_catalogue_job_skipped", reason="lease_held", key=key)
        return
    started = time.monotonic()
    try:
        artifact = await _assemble(
            settings,
            examples=examples,
            catalogue=catalogue,
            covered=covered,
            resolve=resolve,
            place_names=place_names,
            version=version,
            key=key,
            inputs=inputs,
            clock=clock,
        )
        moved_to = coverage.snapshot().manifest.build_id
        if moved_to != inputs.coverage_build_id:
            emit(
                "search_catalogue_job_discarded",
                reason="coverage_moved_during_build",
                key=key,
                coverage_build_id=moved_to,
            )
            return
        if not publish(settings.artifact_uri, artifact, expected_generation=generation):
            emit("search_catalogue_job_lost_publish_race", key=key, catalogue_version=version)
            return
        emit(
            "search_catalogue_job_published",
            key=key,
            catalogue_version=version,
            inputs=inputs.model_dump(),
            examples=[example.id for example in artifact.examples],
            counts=artifact.counts.model_dump(),
            build_seconds=round(time.monotonic() - started, 1),
        )
    finally:
        if not release_lease(root, lease):
            logger.warning("search catalogue: lease for %s was reclaimed before release", version)
    _prune(settings, clock)


async def _assemble(
    settings: JobSettings,
    *,
    examples: ExampleSet,
    catalogue: CatalogueSnapshot,
    covered: CoverageSnapshot,
    resolve: Resolve,
    place_names: PlaceNames,
    version: str,
    key: str,
    inputs: BuildInputs,
    clock: Callable[[], float],
) -> CatalogueArtifact:
    """The build, `CatalogueEngineError` when a check run failed in the
    engine, or `CatalogueFloorError` when fewer than `FLOOR` lines pass."""
    presence: dict[int, frozenset[str]] = {}
    for example in examples.examples:
        try:
            presence[example.id] = example_places(
                example, catalogue=catalogue, coverage=covered.index
            )
        except PresenceRefusal as refusal:
            _drop(example, reason=refusal.reason, detail=refusal.detail)

    index = get_index()
    counted = catalogue_counts(
        catalogue=catalogue,
        coverage=covered.index,
        datasets=index.datasets,
        totals=[dcid for dcid, party in index.parties.items() if party.kind == "total"],
    )
    names = place_names(sorted(counted.places))

    bounded = _bounded(resolve, limit=settings.max_concurrency)
    candidates = [example for example in examples.examples if example.id in presence]
    results = await asyncio.gather(
        *(
            check_example(
                example,
                places=check_places(example, presence[example.id]),
                names=names,
                resolve=bounded,
            )
            for example in candidates
        )
    )
    engine_failures = [
        {"example_id": example.id, **asdict(failure)}
        for example, failures in zip(candidates, results, strict=True)
        for failure in failures
        if failure.reason in INFRA_FAILURES
    ]
    if engine_failures:
        emit(
            "search_catalogue_job_failed",
            severity="ERROR",
            error="check runs failed in the engine after a retry",
            key=key,
            failures=engine_failures,
        )
        raise CatalogueEngineError(
            f"{len(engine_failures)} check run(s) failed in the engine after a retry"
        )

    passed: list[Example] = []
    for example, failures in zip(candidates, results, strict=True):
        if failures:
            _drop(
                example,
                reason="check_failed",
                failures=[asdict(failure) for failure in failures],
            )
        else:
            passed.append(example)

    if len(passed) < FLOOR:
        refusal = CatalogueRefusal(
            key=key,
            refused_at=datetime.fromtimestamp(clock(), UTC),
            passed=[example.id for example in passed],
        )
        emit(
            "search_catalogue_job_refused",
            severity="ERROR",
            key=key,
            passed=refusal.passed,
            floor=FLOOR,
        )
        record_refusal(settings.artifact_uri, refusal)
        raise CatalogueFloorError(f"{len(passed)} line(s) passed, under the floor of {FLOOR}")

    return CatalogueArtifact(
        catalogue_version=version,
        key=key,
        built_at=datetime.fromtimestamp(clock(), UTC),
        inputs=inputs,
        counts=counted.counts,
        examples=[
            CatalogueExample(
                id=example.id,
                text=example.text(names[example.place]),
                template=example.template,
                stat_var=example.stat_var,
                place=example.place,
            )
            for example in passed
        ],
        places={
            country: CataloguePlace(
                name=names.get(country),
                counts=place_counts,
                examples=[example.id for example in passed if country in presence[example.id]]
                if country in names
                else [],
            )
            for country, place_counts in counted.places.items()
        },
    )


def _drop(example: Example, *, reason: str, **fields: object) -> None:
    emit(
        "search_catalogue_line_dropped",
        severity="WARNING",
        example_id=example.id,
        template=example.template,
        stat_var=example.stat_var,
        reason=reason,
        **fields,
    )


def _bounded(resolve: Resolve, *, limit: int) -> Resolve:
    """`resolve` with at most `limit` calls in flight."""
    gate = asyncio.Semaphore(limit)

    async def bounded(request: ResolveRequest) -> ResolveResponse:
        async with gate:
            return await resolve(request)

    return bounded


def _prune(settings: JobSettings, clock: Callable[[], float]) -> None:
    pruned = prune(settings.artifact_uri, grace_seconds=_MAX_RUN_SECONDS, now=clock)
    if pruned:
        emit("search_catalogue_job_pruned", builds=list(pruned))


def _heartbeat(settings: JobSettings, *, clock: Callable[[], float]) -> None:
    """Logs `search_catalogue_job_heartbeat` with the published build and
    its age in seconds, all `None` when nothing is published, and the age
    `None` at `WARNING` when the build is unreadable."""
    published, _ = read_pointer(settings.artifact_uri)
    if published is None:
        emit(
            "search_catalogue_job_heartbeat",
            catalogue_version=None,
            built_at=None,
            age_seconds=None,
        )
        return
    try:
        built_at = load_build(settings.artifact_uri, published.catalogue_version).built_at
    except CatalogueArtifactError as exc:
        emit(
            "search_catalogue_job_heartbeat",
            severity="WARNING",
            catalogue_version=published.catalogue_version,
            built_at=None,
            age_seconds=None,
            error=str(exc),
        )
        return
    emit(
        "search_catalogue_job_heartbeat",
        catalogue_version=published.catalogue_version,
        built_at=built_at.isoformat(),
        age_seconds=round(clock() - built_at.timestamp(), 1),
    )


def _pointer_inputs(
    settings: JobSettings, *, coverage: ServedCoverage, examples: ExampleSet, commit: str
) -> BuildInputs | None:
    """The build inputs the origin pointer and the served coverage build name,
    or `None` when the origin pointer is missing or malformed. The service
    serves a fallback build while `current.json` names one its index rejects,
    so the served build keys the run. With `GRAPH_RELEASE_COVERAGE` on, the
    graph release's pointer names the coverage build, read first, so a run with
    no release raises `ReleaseUnavailable` whatever the origin pointer holds.
    The loaded catalogue decides the key a build carries; these only let an
    unchanged run skip loading it."""
    served = _pointed_release_id() if coverage_views_enabled() else coverage.manifest.build_id
    origin = read_origin_pointer(settings.origin_catalogue_uri)
    if origin is None:
        return None
    return BuildInputs(
        origin_catalogue_build_id=origin.build_id,
        coverage_build_id=served,
        build_commit=commit,
        examples_digest=examples.digest,
    )


def _pointed_release_id() -> str:
    """The release id the pointer under `GRAPH_RELEASE_URI` names. Reads the
    pointer alone, so an unchanged run opens no release. Raises
    `ReleaseUnavailable` when the variable is unset or blank
    (`release_uri_missing`) and when no release is published (`no_release`)."""
    uri = release_uri()
    if uri is None:
        raise ReleaseUnavailable("release_uri_missing", f"{URI_ENV_VAR} is not set")
    # The module imports NumPy and pyarrow, which only this path needs.
    from graph_release.format import read_pointer as read_release_pointer

    pointer = read_release_pointer(uri)
    if pointer is None:
        raise ReleaseUnavailable("no_release", f"no release pointer under {uri!r}")
    return pointer.release_id


def _release_service() -> ReleaseService | None:
    """`optional_release_service()`, raising `ReleaseUnavailable` when no release
    is published (`no_release`) or its views do not build (`views_refused`)."""
    if release_uri() is None:
        return None
    # The module imports NumPy and pyarrow, which only a configured release needs.
    from graph_release.views import ViewRefused

    try:
        return optional_release_service()
    except FileNotFoundError as exc:
        raise ReleaseUnavailable("no_release", str(exc)) from exc
    except ViewRefused as exc:
        raise ReleaseUnavailable("views_refused", str(exc)) from exc


def _load_origin_catalogue(settings: JobSettings) -> CatalogueSnapshot:
    """The published origin catalogue build, loaded under the settings the
    serving surfaces read, so the engine admits what users' requests admit."""
    env = CatalogueSettings()
    return load_catalogue(
        settings.origin_catalogue_uri,
        max_resident_bytes=env.max_resident_bytes,
        un_system_enabled=env.un_system_enabled,
    )


@contextmanager
def _live_engine() -> Iterator[tuple[Resolve, PlaceNames]]:
    """The live engine's buffered resolve with one set of clients for the
    run, and the graph's place names. Closes the clients on exit."""
    # The engine's imports are heavy and only a live run needs them.
    from entity_resolution import EntityResolver, acquire_shared, release_shared
    from materialise.rate_tables import RateTableCache
    from search.engine import resolve_async
    from search.engine.clients.graph_client import LiveGraphClient
    from search.engine.clients.materialise_bridge import build_materialise_sdmx_client
    from search.engine.clients.sdmx import LiveSdmxClient
    from search.engine.config import settings as engine_settings

    with ExitStack() as stack:
        shared = acquire_shared()
        stack.callback(release_shared, shared)
        graph = stack.enter_context(LiveGraphClient.from_search_settings(engine_settings()))
        sdmx = LiveSdmxClient()
        stack.callback(sdmx.close)
        materialise_sdmx = build_materialise_sdmx_client()
        stack.callback(materialise_sdmx.close)
        resolver = EntityResolver(graph=graph, shared=shared, code_index=get_index)
        rate_cache = RateTableCache()

        async def resolve(request: ResolveRequest) -> ResolveResponse:
            return await resolve_async(
                request,
                graph=graph,
                resolver=resolver,
                sdmx=sdmx,
                materialise_graph=graph,
                materialise_sdmx=materialise_sdmx,
                rate_cache=rate_cache,
            )

        def place_names(dcids: Sequence[str]) -> Mapping[str, str]:
            return {dcid: names[0] for dcid, names in graph.node_names_many(dcids).items() if names}

        yield resolve, place_names


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(prog="python -m search.catalogue.job").parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    settings = JobSettings()
    commit = build_commit()
    try:
        examples = load_examples()
        coverage = get_coverage_service()
        pointed = _pointer_inputs(settings, coverage=coverage, examples=examples, commit=commit)
        if pointed is not None and nothing_to_build(settings, inputs=pointed):
            return 0
        catalogue = _load_origin_catalogue(settings)
        # The release is bound before the engine is built, since its construction reads the
        # code index.
        with (
            bound_served_release(_release_service()),
            _live_engine() as (resolve, place_names),
            bound_catalogue(catalogue),
        ):
            asyncio.run(
                run(
                    settings,
                    examples=examples,
                    catalogue=catalogue,
                    coverage=coverage,
                    resolve=resolve,
                    place_names=place_names,
                    commit=commit,
                )
            )
    except CatalogueFloorError as exc:
        logger.error("search catalogue job refused: %s", exc)
        return 1
    except CatalogueEngineError as exc:
        logger.error("search catalogue job failed: %s", exc)
        return 1
    except ReleaseUnavailable as exc:
        emit("search_catalogue_job_failed", severity="ERROR", reason=exc.reason, error=str(exc))
        logger.error("search catalogue job failed: %s", exc)
        return 1
    except (
        ArtifactUriError,
        CatalogueArtifactError,
        CoverageArtifactError,
        CoverageIncompatibleError,
        CoverageIndexCheckValueError,
        ExamplesError,
        GraphReadError,
        OriginArtifactError,
        ReleaseNotBound,
        ReleaseViewsNotBuilt,
    ) as exc:
        emit("search_catalogue_job_failed", severity="ERROR", error=str(exc))
        logger.error("search catalogue job failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FLOOR",
    "REFUSAL_RETRY_SECONDS",
    "CatalogueEngineError",
    "CatalogueFloorError",
    "CoverageSource",
    "JobSettings",
    "PlaceNames",
    "ReleaseUnavailable",
    "main",
    "nothing_to_build",
    "run",
]
