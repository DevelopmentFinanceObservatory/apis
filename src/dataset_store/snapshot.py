"""Snapshot version, artifact-location and revocation records for dataset retrieval and pinning.

Working-tier artifacts expire 30 days after `stored_at`. Pinned artifacts are permanent. Artifact
links retain the rendering and media type recorded for each version.

`read_servable_version` is the one check every stored-version read makes: the version exists,
neither it nor its base version is revoked, and a named digest is linked to it. The snapshot,
pin and download doors, the definition head path and Presentation's frame reads call it.

A version stores its facts in columns of `datasets.version_record` (`VERSION_FACT_COLUMNS`): a
dataset version its `ManifestFacts`, from which every read assembles the wire manifest, and a
published derived frame its `FrameLineage`. `version_fact_values` encodes facts for those
columns and `LiveSnapshotReader.read_version` decodes them. Every row holds one of the two:
migration 0030's insert trigger fills a row written with only `immutable_body` from that body,
and migration 0031 fills the rows older than the trigger. A row holding neither, whose body
names neither `prepared_at` nor `input_content_versions`, refuses with `VersionFactsMissing`.
`frame_sha256` holds the hash of the version's frame, NULL on a row written before versions
stored it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from dataset_store.store import DatasetStore, Transaction
from dfo_platform.problem import Refusal

# The 30-day working-tier retention clock ignores downloads.
WORKING_TIER_RETENTION = timedelta(days=30)


def working_tier_expired(stored_at: datetime, *, now: datetime) -> bool:
    """Whether a working-tier artifact is more than 30 days old. Pinned locations are permanent."""
    return now - stored_at > WORKING_TIER_RETENTION


class SnapshotNotFound(Refusal):
    """The registry knows the id, and this content version has no record. It maps to
    `snapshot_not_found`, 404.
    """

    code = "snapshot_not_found"

    def __init__(self, *, id: str, content_version: str) -> None:
        self.id = id
        self.content_version = content_version
        super().__init__(f"{id!r} has no version at content_version {content_version!r}")


class SnapshotRevoked(Refusal):
    """A taken-down version, or a collapsed version whose base version was taken down. It maps
    to `snapshot_revoked`, 410, on snapshot retrieval, pin and download.
    """

    code = "snapshot_revoked"

    def __init__(self, *, id: str, content_version: str, reason: str | None) -> None:
        self.id = id
        self.content_version = content_version
        self.reason = reason
        detail = f"{id!r} at content_version {content_version!r} was taken down"
        if reason is not None:
            detail += f": {reason}"
        super().__init__(detail)


class ArtifactNotFound(Refusal):
    """The digest is absent from this version's artifacts.

    It maps to `artifact_not_found`, 404.
    """

    code = "artifact_not_found"

    def __init__(self, *, id: str, content_version: str, artifact_digest: str) -> None:
        self.id = id
        self.content_version = content_version
        self.artifact_digest = artifact_digest
        super().__init__(
            f"artifact_digest {artifact_digest!r} is not among the artifacts of "
            f"{id!r} at content_version {content_version!r}"
        )


@dataclass(frozen=True)
class ManifestFacts:
    """The facts a dataset version stores, from which a read assembles its manifest
    (`datasets_api.manifest.stored_manifest`).

    `definition`, `coverage`, `citation`, `trace` and each entry of `warnings` hold the
    manifest member of that name as the wire model dumps it. `series_count` and
    `series_coverage_stamp` are `None` on a version minted before the manifest carried them.
    `output_profile` is the version's `OutputProfile` JSON, `None` on a version minted before
    versions stored it. `series_conversions` holds one conversion record per converted series
    (`materialise.series_conversion.encode_series_conversions`), `None` on an unconverted
    version. `default_scope_plural` says whether the mint's default place scope held more than
    one place, `None` on a collapse and on a version minted before versions stored it.
    """

    prepared_at: datetime
    definition: Mapping[str, object]
    coverage: Mapping[str, object]
    citation: Mapping[str, object]
    trace: Mapping[str, object]
    warnings: Sequence[Mapping[str, object]]
    series_count: int | None = None
    series_coverage_stamp: str | None = None
    output_profile: Mapping[str, object] | None = None
    series_conversions: Sequence[Mapping[str, object]] | None = None
    default_scope_plural: bool | None = None


@dataclass(frozen=True)
class FrameLineage:
    """The facts a published derived frame stores: the versions it was derived from, and the
    composed StatVars its rows name (`datasets_api.publish_models`), `None` when the publish
    described none."""

    input_content_versions: Sequence[str]
    composed_stat_vars: Sequence[Mapping[str, object]] | None = None


type VersionFacts = ManifestFacts | FrameLineage

# The fact columns of `datasets.version_record`, in the order `version_fact_values` gives
# their values.
VERSION_FACT_COLUMNS = (
    "prepared_at",
    "definition",
    "coverage",
    "citation",
    "trace",
    "warnings",
    "series_count",
    "series_coverage_stamp",
    "output_profile",
    "series_conversions",
    "default_scope_plural",
    "input_content_versions",
    "composed_stat_vars",
)


def _json(value: object) -> str | None:
    # A JSON parameter goes as text: the driver would send a list as a Postgres array.
    return None if value is None else json.dumps(value)


def version_fact_values(facts: VersionFacts) -> tuple[object, ...]:
    """The values of `VERSION_FACT_COLUMNS` for `facts`, NULL for the other kind's columns."""
    if isinstance(facts, FrameLineage):
        return (
            *(None,) * 11,
            _json(list(facts.input_content_versions)),
            _json(None if facts.composed_stat_vars is None else list(facts.composed_stat_vars)),
        )
    return (
        facts.prepared_at,
        _json(dict(facts.definition)),
        _json(dict(facts.coverage)),
        _json(dict(facts.citation)),
        _json(dict(facts.trace)),
        _json(list(facts.warnings)),
        facts.series_count,
        facts.series_coverage_stamp,
        _json(None if facts.output_profile is None else dict(facts.output_profile)),
        _json(None if facts.series_conversions is None else list(facts.series_conversions)),
        facts.default_scope_plural,
        None,
        None,
    )


class VersionFactsMissing(Exception):
    """A version row holding neither a dataset version's facts nor a published frame's
    lineage: one whose `immutable_body` names neither `prepared_at` nor
    `input_content_versions`. Migration 0030's insert trigger and migration 0031's backfill fill
    every other row, and 0031 fails while such a row exists."""

    def __init__(self, *, subject_id: str, content_version: str) -> None:
        self.subject_id = subject_id
        self.content_version = content_version
        super().__init__(
            f"version record subject_id {subject_id!r} content_version {content_version!r} "
            "stores no facts; its immutable_body names neither prepared_at nor "
            "input_content_versions"
        )


def _version_facts(
    row: Mapping[str, object], *, subject_id: str, content_version: str
) -> VersionFacts:
    """The facts a `VERSION_FACT_COLUMNS` row holds. `prepared_at` reads in UTC, whatever the
    session's time zone."""
    prepared_at = cast("datetime | None", row["prepared_at"])
    if prepared_at is not None:
        series_count = row["series_count"]
        default_scope_plural = row["default_scope_plural"]
        return ManifestFacts(
            prepared_at=prepared_at.astimezone(UTC),
            definition=cast("Mapping[str, object]", row["definition"]),
            coverage=cast("Mapping[str, object]", row["coverage"]),
            citation=cast("Mapping[str, object]", row["citation"]),
            trace=cast("Mapping[str, object]", row["trace"]),
            warnings=cast("list[Mapping[str, object]] | None", row["warnings"]) or [],
            series_count=None if series_count is None else int(cast(int, series_count)),
            series_coverage_stamp=cast("str | None", row["series_coverage_stamp"]),
            output_profile=cast("Mapping[str, object] | None", row["output_profile"]),
            series_conversions=cast("list[Mapping[str, object]] | None", row["series_conversions"]),
            default_scope_plural=(
                None if default_scope_plural is None else bool(default_scope_plural)
            ),
        )
    input_content_versions = cast("list[str] | None", row["input_content_versions"])
    if input_content_versions is not None:
        return FrameLineage(
            input_content_versions=input_content_versions,
            composed_stat_vars=cast("list[Mapping[str, object]] | None", row["composed_stat_vars"]),
        )
    raise VersionFactsMissing(subject_id=subject_id, content_version=content_version)


@dataclass(frozen=True)
class StoredVersion:
    subject_id: str
    subject_kind: str
    content_version: str
    schema_version: int
    facts: VersionFacts
    # Collapsed versions record the raw materialisation's content version. Other versions use None.
    base_content_version: str | None = None
    # The hash of the version's frame (`dfo_frame_grammar.frame_sha256`), the same for every
    # encoding of the same rows. `None` on a version recorded before versions stored it.
    frame_sha256: str | None = None


@dataclass(frozen=True)
class StoredArtifactLocation:
    """One version-linked artifact location, keyed by digest, rendering and tier. A digest can have
    multiple renderings or both working and pinned locations. `created_at` records the digest's
    first sighting. Promotion leaves it unchanged. `stored_at` starts this tier's retention clock.
    """

    artifact_digest: str
    rendering_key: str
    rendering: Mapping[str, object]
    media_type: str
    size_bytes: int
    created_at: datetime
    stored_at: datetime
    tier: str
    bucket: str
    object_name: str


@dataclass(frozen=True)
class Revocation:
    reason: str | None
    revoked_at: datetime


@dataclass(frozen=True)
class VersionState:
    """A version record, the revocation that withdraws it, and every location linked to it, as
    one transaction reads them."""

    version: StoredVersion
    # The version's own revocation, else its base version's. `None` when neither is revoked.
    revocation: Revocation | None
    # Every linked location, unfiltered by expiry, in `read_artifact_locations` order.
    locations: list[StoredArtifactLocation]


class SnapshotReader(Protocol):
    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        """Return `None` when the subject/version pair has no record. Raise
        `VersionFactsMissing` for a record holding no facts."""
        ...

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        """Return all linked locations, unfiltered by expiry. Data renderings precede
        `series-statistics/1`. Each group is ordered by `(artifact_digest, rendering_key, tier)`.
        """
        ...

    def read_revocation(self, subject_id: str, content_version: str) -> Revocation | None:
        """Return the revocation record, or `None` when the version is live."""
        ...

    def artifact_inventoried(self, artifact_digest: str) -> bool:
        """Whether `datasets.artifact_inventory` holds `artifact_digest`, bytes this deployment
        stored at some point."""
        ...

    def read_version_state(
        self, subject_id: str, content_version: str, *, revoked_under: Sequence[str] = ()
    ) -> VersionState | None:
        """Return the version, its or its base version's revocation, and its linked locations
        from one transaction. `None` when the subject/version pair has no record.

        A revocation of the version or its base version under `subject_id` or any id in
        `revoked_under` withdraws it, so a door that answers one version under several ids
        honours a revocation recorded under any of them."""
        ...


_LOCATIONS_SQL = (
    "SELECT va.artifact_digest AS artifact_digest, "
    "va.rendering_key AS rendering_key, va.rendering AS rendering, "
    "va.media_type AS media_type, ai.size_bytes AS size_bytes, "
    "ai.created_at AS created_at, al.stored_at AS stored_at, al.tier AS tier, "
    "al.bucket AS bucket, al.object_name AS object_name "
    "FROM datasets.version_artifact va "
    "JOIN datasets.artifact_inventory ai ON ai.artifact_digest = va.artifact_digest "
    "JOIN datasets.artifact_location al ON al.artifact_digest = va.artifact_digest "
    "WHERE va.subject_id = %s AND va.content_version = %s "
    # Put data renderings before the auxiliary statistics file, even when `statistics` is
    # absent. Remaining keys make multiple renderings and tiers deterministic.
    "ORDER BY COALESCE((va.rendering ->> 'statistics') = 'series', false), "
    "va.artifact_digest, va.rendering_key, al.tier"
)

_VERSION_SQL = (
    "SELECT subject_kind, schema_version, base_content_version, frame_sha256, "
    f"{', '.join(VERSION_FACT_COLUMNS)} "
    "FROM datasets.version_record WHERE subject_id = %s AND content_version = %s"
)


def revoking_subjects_sql(count: int) -> str:
    """`subject_id IN (...)` over `count` placeholders, the subjects whose revocation of a
    version withdraws it."""
    return f"subject_id IN ({', '.join(['%s'] * count)})"


def _stored_version(
    row: Mapping[str, object], *, subject_id: str, content_version: str
) -> StoredVersion:
    return StoredVersion(
        subject_id=subject_id,
        subject_kind=cast(str, row["subject_kind"]),
        content_version=content_version,
        schema_version=cast(int, row["schema_version"]),
        facts=_version_facts(row, subject_id=subject_id, content_version=content_version),
        base_content_version=cast("str | None", row["base_content_version"]),
        frame_sha256=cast("str | None", row["frame_sha256"]),
    )


def _stored_location(row: Mapping[str, object]) -> StoredArtifactLocation:
    return StoredArtifactLocation(
        artifact_digest=cast(str, row["artifact_digest"]),
        rendering_key=cast(str, row["rendering_key"]),
        rendering=cast(Mapping[str, object], row["rendering"]),
        media_type=cast(str, row["media_type"]),
        size_bytes=cast(int, row["size_bytes"]),
        created_at=cast(datetime, row["created_at"]),
        stored_at=cast(datetime, row["stored_at"]),
        tier=cast(str, row["tier"]),
        bucket=cast(str, row["bucket"]),
        object_name=cast(str, row["object_name"]),
    )


class LiveSnapshotReader:
    """`SnapshotReader` backed by one read-only store transaction per public call."""

    def __init__(self, *, store: DatasetStore) -> None:
        self._store = store

    def read_version(self, subject_id: str, content_version: str) -> StoredVersion | None:
        with self._store.transaction() as tx:
            row = tx.fetch_one(_VERSION_SQL, (subject_id, content_version))
        if row is None:
            return None
        return _stored_version(row, subject_id=subject_id, content_version=content_version)

    def read_artifact_locations(
        self, subject_id: str, content_version: str
    ) -> list[StoredArtifactLocation]:
        with self._store.transaction() as tx:
            rows = tx.fetch_all(_LOCATIONS_SQL, (subject_id, content_version))
        return [_stored_location(row) for row in rows]

    def read_revocation(self, subject_id: str, content_version: str) -> Revocation | None:
        with self._store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT reason, revoked_at FROM datasets.version_revocation "
                "WHERE subject_id = %s AND content_version = %s",
                (subject_id, content_version),
            )
        if row is None:
            return None
        return Revocation(
            reason=cast("str | None", row["reason"]), revoked_at=cast(datetime, row["revoked_at"])
        )

    def artifact_inventoried(self, artifact_digest: str) -> bool:
        with self._store.transaction() as tx:
            row = tx.fetch_one(
                "SELECT 1 AS inventoried FROM datasets.artifact_inventory "
                "WHERE artifact_digest = %s",
                (artifact_digest,),
            )
        return row is not None

    def read_version_state(
        self, subject_id: str, content_version: str, *, revoked_under: Sequence[str] = ()
    ) -> VersionState | None:
        with self._store.transaction() as tx:
            return version_state_in(
                tx,
                subject_id=subject_id,
                content_version=content_version,
                revoked_under=revoked_under,
            )


def version_state_in(
    tx: Transaction, *, subject_id: str, content_version: str, revoked_under: Sequence[str] = ()
) -> VersionState | None:
    """`SnapshotReader.read_version_state`, read inside the caller's transaction `tx`. The
    version's own revocation wins over its base version's, and a revocation under
    `subject_id` over one under another id."""
    row = tx.fetch_one(_VERSION_SQL, (subject_id, content_version))
    if row is None:
        return None
    subjects = list(dict.fromkeys((subject_id, *revoked_under)))
    revocations = tx.fetch_all(
        "SELECT subject_id, content_version, reason, revoked_at "
        f"FROM datasets.version_revocation WHERE {revoking_subjects_sql(len(subjects))} "
        "AND content_version IN (%s, %s)",
        (*subjects, content_version, row["base_content_version"]),
    )
    rows = tx.fetch_all(_LOCATIONS_SQL, (subject_id, content_version))
    revoked = min(
        revocations,
        key=lambda r: (r["content_version"] != content_version, r["subject_id"] != subject_id),
        default=None,
    )
    revocation = (
        Revocation(
            reason=cast("str | None", revoked["reason"]),
            revoked_at=cast(datetime, revoked["revoked_at"]),
        )
        if revoked is not None
        else None
    )
    return VersionState(
        version=_stored_version(row, subject_id=subject_id, content_version=content_version),
        revocation=revocation,
        locations=[_stored_location(location) for location in rows],
    )


def select_servable_locations(
    locations: Sequence[StoredArtifactLocation], *, now: datetime
) -> list[StoredArtifactLocation]:
    """Choose one servable location per `(artifact_digest, rendering_key)`. Prefer `pinned`. Return
    an unexpired `working` location as fallback. Return pairs with a servable location. Group by
    rendering as well as digest because distinct renderings can share stored bytes.
    """
    by_rendering: dict[tuple[str, str], list[StoredArtifactLocation]] = {}
    for location in locations:
        by_rendering.setdefault((location.artifact_digest, location.rendering_key), []).append(
            location
        )

    servable: list[StoredArtifactLocation] = []
    for rendering_locations in by_rendering.values():
        pinned = next((loc for loc in rendering_locations if loc.tier == "pinned"), None)
        if pinned is not None:
            servable.append(pinned)
            continue
        working = next((loc for loc in rendering_locations if loc.tier == "working"), None)
        if working is not None and not working_tier_expired(working.stored_at, now=now):
            servable.append(working)
    return servable


@dataclass(frozen=True)
class ServableVersion:
    """A stored version cleared to serve, and its linked locations split by servability."""

    version: StoredVersion
    # One location per linked `(artifact_digest, rendering_key)` pair, per
    # `select_servable_locations`, in `read_artifact_locations` order.
    servable: list[StoredArtifactLocation]
    # The linked locations of every pair with no servable location: working locations past
    # retention.
    aged_out: list[StoredArtifactLocation]

    def servable_digest(self, artifact_digest: str) -> StoredArtifactLocation | None:
        """The first servable location holding `artifact_digest`, or `None`."""
        return next((loc for loc in self.servable if loc.artifact_digest == artifact_digest), None)


def read_servable_version(
    reader: SnapshotReader,
    *,
    subject_id: str,
    content_version: str,
    now: datetime,
    artifact_digest: str | None = None,
    revoked_under: Sequence[str] = (),
) -> ServableVersion:
    """The version at `content_version`, cleared to serve, from one `read_version_state`
    transaction.

    Raises `SnapshotNotFound` for an unknown version, `SnapshotRevoked` when the version or its
    base version is revoked under `subject_id` or any id in `revoked_under`, and, when
    `artifact_digest` is given, `ArtifactNotFound` when no location of that digest is linked to
    the version. Servability is judged at `now`.
    """
    state = reader.read_version_state(subject_id, content_version, revoked_under=revoked_under)
    if state is None:
        raise SnapshotNotFound(id=subject_id, content_version=content_version)
    if state.revocation is not None:
        raise SnapshotRevoked(
            id=subject_id, content_version=content_version, reason=state.revocation.reason
        )
    if artifact_digest is not None and not any(
        location.artifact_digest == artifact_digest for location in state.locations
    ):
        raise ArtifactNotFound(
            id=subject_id, content_version=content_version, artifact_digest=artifact_digest
        )
    servable = select_servable_locations(state.locations, now=now)
    served = {(location.artifact_digest, location.rendering_key) for location in servable}
    return ServableVersion(
        version=state.version,
        servable=servable,
        aged_out=[
            location
            for location in state.locations
            if (location.artifact_digest, location.rendering_key) not in served
        ],
    )


class ArtifactReferenceWriter(Protocol):
    def record_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> datetime:
        """Insert or reactivate the reference at `now`. A live row keeps its `created_at` timestamp.
        Return the resulting timestamp.
        """
        ...

    def record_reference_unless_revoked(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
        revoked_under: Sequence[str] = (),
    ) -> datetime:
        """Same write as `record_reference`, made in one statement that also checks
        `datasets.version_revocation` for the version and its base version, under `subject_id`
        and every id in `revoked_under`. A revocation visible in the statement's starting snapshot
        raises `SnapshotRevoked` and writes nothing. A revocation committed later goes unseen,
        and nothing here locks against a concurrent revocation write.

        Before the write it takes the version's link lock, the one every artifact link write
        takes, and requires a `datasets.version_artifact` row linking `artifact_digest` to the
        version. A digest no longer linked, because a re-render replaced its expired link, raises
        `ArtifactNotFound` and writes nothing. A link write after
        this one commits sees the pinned location its caller recorded first, and keeps the link.
        """
        ...

    def remove_reference(
        self,
        *,
        principal: str,
        artifact_digest: str,
        subject_id: str,
        content_version: str,
        reference_kind: str,
        now: datetime,
    ) -> None:
        """Set `removed_at` on a matching live row. Missing or removed rows are a no-op."""
        ...
