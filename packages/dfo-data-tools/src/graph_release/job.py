"""The graph release builder, run as `python -m graph_release.job` every two minutes.

Each tick:

1. Reads the import key in a multi-use read-only snapshot and exits unless it is buildable.
2. Reads the DC service's serving revisions and exits unless each was created after the import
   completed, since the mixer reads Spanner as of the last ingestion it saw at start.
3. Forms the inputs (the success row's execution id, the serving revision, `FORMAT`, the
   builder code hash and the declarations hash) and exits when they match the last built or
   the last failed inputs in the state file, or when the pointer is held. With
   `GRAPH_RELEASE_SHADOW` on, it exits `graph_release_shadow_waiting` while the coverage build
   or the origin catalogue is behind the import, before building anything.
4. Takes the build lease, or exits.
5. Reads every input in one snapshot whose first statements re-read the import key, and
   discards the build when the key moved since step 1.
6. Builds the platform and DFO parts and gates them. A finding records the inputs as the last
   failed build; an unavailable dependency records nothing and the next tick retries.
7. Publishes, unless the release id is the serving one, in which case it records the inputs
   and publishes nothing. Prunes while still holding the lease, then releases it.

Every tick that completes logs `graph_release_heartbeat`, whatever its exit, and
`graph_release_stale` while the newest import of any status has waited more than
`STALE_SECONDS` for a release, less the time spent waiting for the shadow comparators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, Final, Protocol, Self

from google.api_core.exceptions import GoogleAPIError
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from artifact_publication.jobs import (
    AliasOnlySettings,
    ImportKey,
    acquire_lease,
    emit,
    new_build_id,
    read_import_key,
    release_lease,
)
from artifact_publication.uri import (
    ArtifactUriError,
    join_uri,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes,
)
from dc_graph import GraphClient
from dc_graph.errors import GraphCapExceededError, GraphReadError
from dc_sdmx.client import SdmxClient
from graph_release.declarations import DECLARATIONS, Declarations
from graph_release.dfo import base_party_ids, build_dfo_parts, read_base_parties, stage_dfo_parts
from graph_release.format import (
    FORMAT,
    LoadedRelease,
    Manifest,
    Pointer,
    ReleaseFormatError,
    StagedPart,
    manifest_uri,
    open_release,
    pointer_uri,
)
from graph_release.gate import (
    Dependency,
    GateResult,
    GateUnavailable,
    SeriesReader,
    merge,
    refused,
    run_gate,
)
from graph_release.platform import PlatformRefusal, build_platform_parts, stage_platform_parts
from graph_release.publish import PublishStatus, prune_releases, publish_release
from graph_release.reads import Snapshot, read_ingested_imports, read_inputs, snapshot_at
from graph_release.serving_revision import CloudRunAdmin, ServingRevision, ServingRevisionError
from graph_release.shadow import shadow_diff, shadow_ready

logger = logging.getLogger(__name__)

STATE_FILE: Final[str] = "builder_state.json"
"""At the release root: the last built and last failed inputs, and the shadow wait."""

LEASE_TTL_SECONDS: Final[float] = 1800.0
"""The job's task timeout. The lease has no renewal, so it lasts as long as a run can."""

STALE_SECONDS: Final[float] = 1800.0
"""How long the newest import may wait for a release before `graph_release_stale`. It sits
above the DC restart that follows each import."""

BUILDER_MODULES: Final[Mapping[str, tuple[str, ...]]] = {
    "graph_release": (
        "authored.py",
        "coverage.py",
        "declarations.py",
        "dfo.py",
        "format.py",
        "gate.py",
        "job.py",
        "platform.py",
        "publish.py",
        "reads.py",
        "serving_revision.py",
        "shadow.py",
    ),
    "code_index": ("export.py",),
    "dc_graph": ("visible_arcs.py",),
}
"""The modules whose source `builder_code_hash` covers: what reads, builds, gates and publishes
a release. A change elsewhere that changes a part still changes the release id, but builds only
at the next import or DC redeploy."""

_GRAPH_TIMEOUT_SECONDS: Final[float] = 120.0
_USER_AGENT: Final[str] = "dfo-graph-release"
_BUCKET_ERRORS: Final = (OSError, ArtifactUriError, GoogleAPIError, ReleaseFormatError)
"""What a read or write under the release root, or of a comparator, raises when it fails."""


@cache
def builder_code_hash() -> str:
    """SHA-256 over the source of `BUILDER_MODULES`, each keyed by package and file name."""
    digest = hashlib.sha256()
    for package, modules in sorted(BUILDER_MODULES.items()):
        files = resources.files(package)
        for name in sorted(modules):
            data = files.joinpath(name).read_bytes()
            digest.update(f"{package}/{name}\0{len(data)}\0".encode())
            digest.update(data)
    return digest.hexdigest()


class JobSettings(AliasOnlySettings):
    """Job settings from the environment, failing fast at construction. Each field's `alias` is
    the only environment variable name read; construct by field name to override one."""

    model_config = SettingsConfigDict(frozen=True)

    release_uri: str = Field(min_length=1, alias="GRAPH_RELEASE_URI")
    spanner_project: str = Field(min_length=1, alias="GRAPH_RELEASE_SPANNER_PROJECT")
    spanner_instance: str = Field(min_length=1, alias="GRAPH_RELEASE_SPANNER_INSTANCE")
    spanner_database: str = Field(min_length=1, alias="GRAPH_RELEASE_SPANNER_DATABASE")
    datacommons_service: str = Field(
        pattern=r"^projects/[^/]+/locations/[^/]+/services/[^/]+$",
        alias="DATACOMMONS_SERVICE_NAME",
    )
    graph_base_url: str = Field(min_length=1, alias="DATASETS_GRAPH_BASE_URL")
    sdmx_base_url: str = Field(min_length=1, alias="DATASETS_SDMX_BASE_URL")
    shadow: bool = Field(default=False, alias="GRAPH_RELEASE_SHADOW")
    coverage_uri: str = Field(default="", alias="COVERAGE_ARTIFACT_URI")
    origin_uri: str = Field(default="", alias="ORIGIN_CATALOGUE_URI")

    @model_validator(mode="after")
    def _shadow_needs_comparators(self) -> Self:
        if self.shadow and not (self.coverage_uri and self.origin_uri):
            raise ValueError(
                "GRAPH_RELEASE_SHADOW needs COVERAGE_ARTIFACT_URI and ORIGIN_CATALOGUE_URI"
            )
        return self


@dataclass(frozen=True)
class BuildInputs:
    """What a release is built from. Equal inputs build an equal release."""

    import_execution_id: str
    dc_serving_revision: str
    format: int
    builder_code_hash: str
    declarations_hash: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class Database(Protocol):
    """The part of a `google.cloud.spanner` `Database` the job uses."""

    def snapshot(self, **kwargs: Any) -> AbstractContextManager[Snapshot]: ...


class ServingRevisions(Protocol):
    """`CloudRunAdmin`."""

    def serving(self) -> ServingRevision: ...

    def created_at(self, revision: str) -> datetime: ...


class Mixer(Protocol):
    """`dc_graph.GraphClient.node_arcs_many`, for the base parties and the mixer differential."""

    def node_arcs_many(
        self, dcids: Sequence[str], *, properties: Sequence[str]
    ) -> dict[str, dict[str, tuple[str, ...]] | None]: ...


@dataclass(frozen=True)
class _Exit:
    outcome: str
    code: int = 0


class _Unavailable(Exception):
    """A dependency that could not answer: the tick records nothing and exits non-zero."""

    def __init__(self, *entries: GateUnavailable) -> None:
        super().__init__("; ".join(f"{entry.dependency}: {entry.detail}" for entry in entries))
        self.entries = entries


@contextmanager
def _needs(dependency: Dependency, *errors: type[BaseException]) -> Iterator[None]:
    """Turns `errors` raised inside the block into `_Unavailable` naming `dependency`."""
    try:
        yield
    except errors as exc:
        raise _Unavailable(GateUnavailable(dependency, str(exc))) from exc


def run(
    settings: JobSettings,
    *,
    database: Database,
    admin: ServingRevisions,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations = DECLARATIONS,
    clock: Callable[[], float] = time.time,
) -> int:
    """One tick. Returns the exit code: 1 for a gate finding or an unavailable dependency, 0
    otherwise. Logs `graph_release_heartbeat` last."""
    key: ImportKey | None = None
    try:
        with _needs(Dependency.SPANNER, GoogleAPIError), database.snapshot(multi_use=True) as snap:
            key = read_import_key(snap)
        if key is None:
            result = _skipped("no_import")
        else:
            result = _tick(
                settings,
                key,
                database=database,
                admin=admin,
                graph=graph,
                sdmx=sdmx,
                declarations=declarations,
                clock=clock,
            )
    except _Unavailable as exc:
        result = _unavailable(exc)
    if key is not None:
        try:
            _track_stale(settings.release_uri, key, now=clock())
        except _Unavailable as exc:
            if result.code == 0:
                result = _unavailable(exc)
    emit("graph_release_heartbeat", outcome=result.outcome)
    return result.code


def _tick(
    settings: JobSettings,
    key: ImportKey,
    *,
    database: Database,
    admin: ServingRevisions,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations,
    clock: Callable[[], float],
) -> _Exit:
    if key.lock_owner is not None:
        return _skipped("ingestion_lock_held")
    success = key.newest_success
    if not key.buildable or success is None:
        return _skipped("import_not_successful", status=key.newest.status)

    with _needs(Dependency.ADMIN_API, ServingRevisionError):
        serving = admin.serving()
        created = min(admin.created_at(revision) for revision in serving.revisions)
    completed = success.completed_at
    if completed is None or created <= datetime.fromisoformat(completed):
        return _skipped(
            "serving_revision_before_import",
            revision=serving.token,
            revision_created_at=created.isoformat(),
            import_completed_at=completed,
        )

    inputs = BuildInputs(
        import_execution_id=success.workflow_execution_id,
        dc_serving_revision=serving.token,
        format=FORMAT,
        builder_code_hash=builder_code_hash(),
        declarations_hash=declarations.declarations_hash(),
    )
    root = settings.release_uri
    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        state = _read_state(root)
        pointer, generation = _read_pointer(root)
    if state.get("last_built") == inputs.to_json():
        return _skipped("inputs_built")
    if state.get("last_failed") == inputs.to_json():
        return _skipped("inputs_failed")
    if pointer is not None and pointer.hold is not None:
        return _skipped("held", release_id=pointer.hold)
    if settings.shadow:
        ready = shadow_ready(
            key.stamp, coverage_uri=settings.coverage_uri, origin_uri=settings.origin_uri
        )
        if isinstance(ready, GateUnavailable):
            raise _Unavailable(ready)
        with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
            _note_shadow(root, state, stamp=key.stamp, ready=ready, now=clock())
        if not ready:
            emit("graph_release_shadow_waiting", stamp=key.stamp)
            return _Exit("shadow_waiting")

    build_id = new_build_id()
    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        lease = acquire_lease(root, build_id=build_id, ttl_seconds=LEASE_TTL_SECONDS, now=clock)
    if lease is None:
        return _skipped("lease_held")
    try:
        result = _build(
            settings,
            key,
            inputs,
            pointer=pointer,
            generation=generation,
            database=database,
            admin=admin,
            graph=graph,
            sdmx=sdmx,
            declarations=declarations,
            clock=clock,
        )
        # Pruning runs under the lease, so no publish can reuse a part it is deleting.
        with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
            pruned = prune_releases(root, now=clock())
        if pruned:
            emit("graph_release_pruned", deleted=list(pruned))
    finally:
        if not release_lease(root, lease):
            logger.warning("graph_release: lease for %s was reclaimed before release", build_id)
    return result


def _build(
    settings: JobSettings,
    key: ImportKey,
    inputs: BuildInputs,
    *,
    pointer: Pointer | None,
    generation: str | None,
    database: Database,
    admin: ServingRevisions,
    graph: Mixer,
    sdmx: SeriesReader,
    declarations: Declarations,
    clock: Callable[[], float],
) -> _Exit:
    """Steps 5 to 7, holding the lease."""
    root = settings.release_uri
    read_at = datetime.fromtimestamp(clock(), UTC)
    with (
        _needs(Dependency.SPANNER, GoogleAPIError),
        snapshot_at(database, read_timestamp=read_at) as snapshot,
    ):
        raw = read_inputs(snapshot)
        ingested = read_ingested_imports(snapshot, execution_id=inputs.import_execution_id)
    if raw.import_key != key:
        emit("graph_release_discarded", reason="import_moved", **inputs.to_json())
        return _Exit("import_moved")

    with tempfile.TemporaryDirectory(prefix="graph-release-") as scratch:
        staging = Path(scratch) / "staging"
        execution = inputs.import_execution_id
        try:
            platform = build_platform_parts(raw, authored=declarations.authored)
            staged = stage_platform_parts(platform, staging_dir=staging, execution_id=execution)
            party_ids = base_party_ids(platform, declarations)
            with _needs(Dependency.MIXER, GraphReadError, GraphCapExceededError):
                base_parties = read_base_parties(party_ids, graph=graph)
            dfo = build_dfo_parts(platform, declarations, base_parties)
        except PlatformRefusal as refusal:
            gate = refused(refusal)
        else:
            staged.extend(stage_dfo_parts(dfo, staging_dir=staging, execution_id=execution))
            gate = run_gate(
                staged,
                declarations=declarations,
                previous=_open_previous(root, pointer, local_dir=Path(scratch) / "previous"),
                graph=graph,
                sdmx=sdmx,
                seed=key.stamp,
            )
            if settings.shadow:
                gate = merge(
                    gate,
                    shadow_diff(
                        platform,
                        import_stamp=key.stamp,
                        role_axes=declarations.role_axes,
                        code_index_blocks=dfo.code_index_blocks,
                        coverage_uri=settings.coverage_uri,
                        origin_uri=settings.origin_uri,
                    ),
                )
        if gate.findings:
            with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
                _update_state(root, last_failed=inputs.to_json())
            emit("graph_release_gate_failed", **inputs.to_json(), **gate.summary())
            return _Exit("gate_failed", 1)
        if gate.unavailable:
            raise _Unavailable(*gate.unavailable)

        # The mixer reads above answered at the serving revision the inputs name.
        with _needs(Dependency.ADMIN_API, ServingRevisionError):
            serving = admin.serving()
        if serving.token != inputs.dc_serving_revision:
            emit(
                "graph_release_discarded",
                reason="serving_revision_moved",
                revision=serving.token,
                **inputs.to_json(),
            )
            return _Exit("serving_revision_moved")

        return _publish(
            root,
            inputs,
            _manifest(inputs, key, staged, ingested=ingested, read_at=read_at, gate=gate),
            staged=staged,
            pointer=pointer,
            generation=generation,
        )


def _manifest(
    inputs: BuildInputs,
    key: ImportKey,
    staged: Sequence[StagedPart],
    *,
    ingested: tuple[str, ...],
    read_at: datetime,
    gate: GateResult,
) -> Manifest:
    success = key.newest_success
    assert success is not None and success.completed_at is not None  # checked at step 2
    return Manifest(
        format=inputs.format,
        import_execution_id=inputs.import_execution_id,
        import_completed_at=success.completed_at,
        ingested_imports=ingested,
        read_timestamp=read_at.isoformat(timespec="microseconds"),
        dc_serving_revision=inputs.dc_serving_revision,
        builder_code_hash=inputs.builder_code_hash,
        declarations_hash=inputs.declarations_hash,
        parts=tuple(part.ref for part in staged),
        gates=gate.summary(),
    )


def _publish(
    root: str,
    inputs: BuildInputs,
    manifest: Manifest,
    *,
    staged: Sequence[StagedPart],
    pointer: Pointer | None,
    generation: str | None,
) -> _Exit:
    if pointer is not None and manifest.release_id == pointer.release_id:
        with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
            _update_state(root, last_built=inputs.to_json())
        emit("graph_release_unchanged", release_id=manifest.release_id, **inputs.to_json())
        return _Exit("release_unchanged")

    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        outcome = publish_release(
            root, staged=staged, manifest=manifest, expected_pointer_generation=generation
        )
        if outcome.status is PublishStatus.PUBLISHED:
            _update_state(root, last_built=inputs.to_json())
    match outcome.status:
        case PublishStatus.HELD:
            return _skipped("held")
        case PublishStatus.POINTER_MOVED:
            emit("graph_release_discarded", reason="pointer_moved", release_id=outcome.release_id)
            return _Exit("pointer_moved")
    rows: dict[str, int] = {}
    for ref in manifest.parts:
        if ref.rows is not None:
            rows[ref.name] = rows.get(ref.name, 0) + ref.rows
    emit(
        "graph_release_published",
        release_id=outcome.release_id,
        rows=rows,
        uploaded=len(outcome.uploaded),
        parts=len(manifest.parts),
        **inputs.to_json(),
    )
    return _Exit("published")


def _open_previous(root: str, pointer: Pointer | None, *, local_dir: Path) -> LoadedRelease | None:
    if pointer is None:
        return None
    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        return open_release(root, local_dir=local_dir, release_id=pointer.release_id)


def _read_pointer(root: str) -> tuple[Pointer | None, str | None]:
    """The format's pointer and its version token, both `None` when none is published."""
    found = read_uri_bytes_and_generation(pointer_uri(root))
    if found is None:
        return None, None
    return Pointer.from_json(found[0]), found[1]


# --- State file and stale tracking -------------------------------------------


def _read_state(root: str) -> dict[str, Any]:
    """The state file, empty when it is missing or does not parse."""
    try:
        state = json.loads(read_uri_bytes(join_uri(root, STATE_FILE)))
    except FileNotFoundError:
        return {}
    except ValueError:
        logger.warning("graph_release: %s does not parse; starting it afresh", STATE_FILE)
        return {}
    return state if isinstance(state, dict) else {}


def _update_state(root: str, **fields: object) -> None:
    """Rewrites `fields` in the state file, keeping the rest as it now stands."""
    state = _read_state(root) | fields
    write_uri_bytes(join_uri(root, STATE_FILE), json.dumps(state, sort_keys=True).encode("utf-8"))


def _note_shadow(
    root: str, state: Mapping[str, Any], *, stamp: str, ready: bool, now: float
) -> None:
    """Records when the comparators were first seen behind `stamp`, and when they caught up."""
    noted = state.get("shadow") or {}
    if noted.get("stamp") != stamp:
        if not ready:
            _update_state(root, shadow={"stamp": stamp, "waiting_since": now, "ready_at": None})
        return
    if ready and noted.get("waiting_since") is not None and noted.get("ready_at") is None:
        _update_state(root, shadow={**noted, "ready_at": now})


def _shadow_wait(state: Mapping[str, Any], stamp: str, now: float) -> float:
    """Seconds spent waiting for the shadow comparators at `stamp`."""
    noted = state.get("shadow") or {}
    if noted.get("stamp") != stamp or noted.get("waiting_since") is None:
        return 0.0
    ended = noted.get("ready_at")
    return (now if ended is None else float(ended)) - float(noted["waiting_since"])


def _track_stale(root: str, key: ImportKey, *, now: float) -> None:
    """Logs `graph_release_stale` while the newest import, of any status, has completed more
    than `STALE_SECONDS` ago, less its shadow wait, and the serving release is not built from
    it. A failed import never gets a release, so it raises the line until the next success."""
    newest = key.newest
    if newest.completed_at is None:
        return
    with _needs(Dependency.BUCKET, *_BUCKET_ERRORS):
        pointer, _ = _read_pointer(root)
        if pointer is not None:
            serving = Manifest.from_json(read_uri_bytes(manifest_uri(root, pointer.release_id)))
            if serving.import_execution_id == newest.workflow_execution_id:
                return
        state = _read_state(root)
    completed = datetime.fromisoformat(newest.completed_at).timestamp()
    waited = now - completed - _shadow_wait(state, key.stamp, now)
    if waited > STALE_SECONDS:
        emit(
            "graph_release_stale",
            import_execution_id=newest.workflow_execution_id,
            status=newest.status,
            failed=newest.failed,
            waited_seconds=round(waited, 1),
        )


# --- Exits -------------------------------------------------------------------


def _skipped(reason: str, **fields: object) -> _Exit:
    emit("graph_release_skipped", reason=reason, **fields)
    return _Exit(reason)


def _unavailable(exc: _Unavailable) -> _Exit:
    emit(
        "graph_release_unavailable",
        severity="WARNING",
        unavailable=[{"dependency": str(e.dependency), "detail": e.detail} for e in exc.entries],
    )
    return _Exit("unavailable", 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m graph_release.job")
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="publish under this local directory in place of GRAPH_RELEASE_URI",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    overrides = {"release_uri": args.local_dir.resolve().as_uri()} if args.local_dir else {}
    settings = JobSettings(**overrides)

    from google.cloud import spanner

    database = (
        spanner.Client(project=settings.spanner_project)
        .instance(settings.spanner_instance)
        .database(settings.spanner_database)
    )
    graph = GraphClient(
        base_url=settings.graph_base_url, timeout=_GRAPH_TIMEOUT_SECONDS, user_agent=_USER_AGENT
    )
    sdmx = SdmxClient(
        base_url=settings.sdmx_base_url, timeout=_GRAPH_TIMEOUT_SECONDS, user_agent=_USER_AGENT
    )
    admin = CloudRunAdmin(service=settings.datacommons_service)
    with graph, sdmx, closing(admin):
        return run(settings, database=database, admin=admin, graph=graph, sdmx=sdmx)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "BUILDER_MODULES",
    "LEASE_TTL_SECONDS",
    "STALE_SECONDS",
    "STATE_FILE",
    "BuildInputs",
    "Database",
    "JobSettings",
    "Mixer",
    "ServingRevisions",
    "builder_code_hash",
    "main",
    "run",
]
