"""Version-record and subject-head interfaces used by the Datasets API.

`VersionRecordWriter` stores immutable manifest data. `HeadWriter` and
`HeadReader` manage the current content version. The default null writers
record or advance nothing. `datasets_surface` injects live implementations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


@dataclass(frozen=True)
class ArtifactRecord:
    """Rendering metadata for one artifact in a version record.

    `artifact_digest` joins to stored bytes and location.
    `rendering` and `media_type` preserve how the manifest entry was produced.
    """

    artifact_digest: str
    rendering: Mapping[str, object]
    media_type: str


@dataclass(frozen=True)
class VersionRecord:
    subject_id: str
    subject_kind: str  # "spec" | "collection"
    content_version: str
    schema_version: int
    immutable_body: Mapping[str, object]
    member_versions: tuple[tuple[str, str], ...] = ()
    # Signed URLs expire. Writers store artifact metadata separately without them.
    artifacts: tuple[ArtifactRecord, ...] = ()
    # Full collapses store their source version in a column because older Manifest
    # readers reject unknown body fields. Collections and other records use `None`.
    base_content_version: str | None = None


class VersionRecordWriter(Protocol):
    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        """Insert the immutable record once. Return `True` only when this call creates the
        row, including under concurrent first writes.
        """
        ...

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        """Add one rendering, idempotent on `(subject_id, content_version, rendering_key)`."""
        ...


class NullVersionRecordWriter:
    """Records nothing: the injection point / interface other `VersionRecordWriter`
    implementations plug into (see `datasets_surface`'s `version_writer`
    parameter).
    """

    def record_version(self, record: VersionRecord, *, trace_id: str | None = None) -> bool:
        # No stored row can win a concurrent insert race.
        return True

    def record_artifact(
        self,
        *,
        subject_id: str,
        content_version: str,
        artifact: ArtifactRecord,
        trace_id: str | None = None,
    ) -> None:
        return None


NULL_VERSION_RECORD_WRITER = NullVersionRecordWriter()


@dataclass(frozen=True)
class MemberSignal:
    """Currency-check signal captured for one materialised member.

    A frame head carries one signal. A collection carries one per member in
    `member_versions` order. `obs_count` is measured on the ordinary path,
    but stored role-bearing signals may have `None`. Two `None` counts match.
    checks then compare the facet id and dates. `None` stays distinct from
    zero, which would falsely assert an observation count.

    On the role path, the dates cover only the latest period per bound entity.


    `observation_period` and provenance refresh dates come from the
    dispatched facet. Refresh dates catch value-only source updates. When
    cadence or refresh dates are unavailable, currency checks fall back to
    maximum age.

    Converted members carry sorted `(provenance_id, refresh_date)` pairs
    for their rate or deflator tables, plus `(recipe_id, digest)` for the
    conversion recipe body. Only materialisation runs recipes, so it captures
    both. Currency checks compare table refresh dates and recompute
    the live recipe digest, catching changed inputs under an unchanged id.
    Unconverted members use `()` and `None`, respectively.

    `clock_window` is the start and end year a range read when its definition
    leaves the end unstated: the current year ends it, or for a start after
    the current year it stays open (`materialise.window.election_window`).
    A read in a later year can take another window, so currency checks
    compare it with the window a read now takes (`currency.clock_moved`). It
    is `None` for a window stating its end, `all` and `latest`, which read
    no clock.
    """

    dispatch_facet_id: str | None
    obs_count: int | None
    earliest_date: str | None
    latest_date: str | None
    observation_period: str | None = None
    provenance_last_refresh: str | None = None
    provenance_next_refresh: str | None = None
    rate_provenance_refresh: tuple[tuple[str, str | None], ...] = ()
    recipe_dependency_fingerprint: tuple[str, str] | None = None
    clock_window: tuple[int | None, int | None] | None = None


@dataclass(frozen=True)
class SubjectHead:
    """`confirmed_at` advances on a current probe or materialisation.
    `materialised_at` advances only when materialisation writes a version.
    Maximum-age checks use it.
    """

    subject_id: str
    content_version: str
    confirmed_at: datetime
    materialised_at: datetime
    signal: tuple[MemberSignal, ...]


_SIGNAL_MEMBERS_KEY = "members"


def encode_signal(signal: Sequence[MemberSignal]) -> dict[str, object]:
    """Encode as `{"members": [...]}` because pg8000 maps a dict to JSONB and a bare
    list to a Postgres array. Optional conversion fields and `clock_window` are omitted when
    empty. `decode_signal` reads them back as `()` or `None`.
    """
    members: list[dict[str, object]] = []
    for member in signal:
        entry: dict[str, object] = {
            "dispatch_facet_id": member.dispatch_facet_id,
            "obs_count": member.obs_count,
            "earliest_date": member.earliest_date,
            "latest_date": member.latest_date,
            "observation_period": member.observation_period,
            "provenance_last_refresh": member.provenance_last_refresh,
            "provenance_next_refresh": member.provenance_next_refresh,
        }
        if member.rate_provenance_refresh:
            entry["rate_provenance_refresh"] = [
                [provenance_id, refresh_date]
                for provenance_id, refresh_date in member.rate_provenance_refresh
            ]
        if member.recipe_dependency_fingerprint is not None:
            recipe_id, digest = member.recipe_dependency_fingerprint
            entry["recipe_dependency_fingerprint"] = [recipe_id, digest]
        if member.clock_window is not None:
            entry["clock_window"] = list(member.clock_window)
        members.append(entry)
    return {_SIGNAL_MEMBERS_KEY: members}


class MalformedSignalError(ValueError):
    """`decode_signal` raises this for malformed data. `obs_count` is required.
    Refresh pairs are `[str, str | None]`; recipe fingerprints are `[str, str]`; clock
    windows are `[int | None, int | None]`.
    A bad row never escapes as `KeyError` or `TypeError`.
    `LiveHeadStore.read_head` maps this to store-unavailable.
    """


def _decode_member_signal(member: object) -> MemberSignal:
    if not isinstance(member, Mapping):
        raise MalformedSignalError(f"signal member is not an object: {member!r}")

    def _optional_str(key: str) -> str | None:
        value = member.get(key)
        if value is not None and not isinstance(value, str):
            raise MalformedSignalError(f"signal member {key!r} is not a str or None: {value!r}")
        return value

    def _rate_provenance_refresh() -> tuple[tuple[str, str | None], ...]:
        """Decode refresh pairs.

        Missing means `()`. Entries are `[str, str | None]`.
        """
        value = member.get("rate_provenance_refresh", [])
        if not isinstance(value, list):
            raise MalformedSignalError(
                f"signal member 'rate_provenance_refresh' is not a list: {value!r}"
            )
        pairs: list[tuple[str, str | None]] = []
        for entry in value:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or (entry[1] is not None and not isinstance(entry[1], str))
            ):
                raise MalformedSignalError(
                    f"signal member 'rate_provenance_refresh' entry is malformed: {entry!r}"
                )
            pairs.append((entry[0], entry[1]))
        return tuple(pairs)

    def _recipe_dependency_fingerprint() -> tuple[str, str] | None:
        """Decode the optional `[recipe_id, digest]` pair. Missing means `None`."""
        value = member.get("recipe_dependency_fingerprint")
        if value is None:
            return None
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], str)
        ):
            raise MalformedSignalError(
                f"signal member 'recipe_dependency_fingerprint' is malformed: {value!r}"
            )
        return (value[0], value[1])

    def _clock_window() -> tuple[int | None, int | None] | None:
        """Decode the optional `[start_year, end_year]` pair. Missing means `None`."""
        value = member.get("clock_window")
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise MalformedSignalError(f"signal member 'clock_window' is malformed: {value!r}")

        def _year(year: object) -> int | None:
            if year is None or (isinstance(year, int) and not isinstance(year, bool)):
                return year
            raise MalformedSignalError(f"signal member 'clock_window' is malformed: {value!r}")

        return (_year(value[0]), _year(value[1]))

    _missing = object()
    obs_count = member.get("obs_count", _missing)
    if obs_count is _missing:
        raise MalformedSignalError(f"signal member is missing 'obs_count': {member!r}")
    if obs_count is not None and not isinstance(obs_count, int):
        raise MalformedSignalError(
            f"signal member 'obs_count' is not an int or None: {obs_count!r}"
        )
    return MemberSignal(
        dispatch_facet_id=_optional_str("dispatch_facet_id"),
        obs_count=obs_count,
        earliest_date=_optional_str("earliest_date"),
        latest_date=_optional_str("latest_date"),
        observation_period=_optional_str("observation_period"),
        provenance_last_refresh=_optional_str("provenance_last_refresh"),
        provenance_next_refresh=_optional_str("provenance_next_refresh"),
        rate_provenance_refresh=_rate_provenance_refresh(),
        recipe_dependency_fingerprint=_recipe_dependency_fingerprint(),
        clock_window=_clock_window(),
    )


def decode_signal(encoded: Mapping[str, object]) -> tuple[MemberSignal, ...]:
    """Decode `encode_signal`'s shape, raising `MalformedSignalError` for malformed data."""
    if not isinstance(encoded, Mapping):
        raise MalformedSignalError(f"signal is not an object: {encoded!r}")
    members = encoded.get(_SIGNAL_MEMBERS_KEY, [])
    if not isinstance(members, list):
        raise MalformedSignalError(f"signal {_SIGNAL_MEMBERS_KEY!r} is not a list: {members!r}")
    return tuple(_decode_member_signal(member) for member in members)


class HeadWriter(Protocol):
    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal: Sequence[MemberSignal],
        expected_content_version: str | None,
    ) -> SubjectHead | None:
        """Write the new head only if its stored version matches `expected_content_version`.

        The caller passes the version read before materialisation. `None`
        matches an empty row. An applied write sets both timestamps and
        returns the stored head. A mismatch returns `None` without changing
        the row. The caller rereads the head to find the winner.
        """
        ...

    def confirm_head(
        self,
        *,
        subject_id: str,
        content_version: str,
    ) -> SubjectHead | None:
        """Advance `confirmed_at` only if the stored version matches. Leave the version,
        `materialised_at`, and signal unchanged. Return `None` on a mismatch or missing row.
        """
        ...


class HeadReader(Protocol):
    def read_head(self, subject_id: str) -> SubjectHead | None:
        """`None` when no head has ever been written for this subject."""
        ...


class HeadStore(HeadReader, HeadWriter, Protocol):
    """Before deciding a new version, writers read the head and pass it to `advance_head`.
    The definition door also reads heads. `LiveHeadStore` implements both protocols.
    """


class NullHeadWriter:
    """No-op `HeadStore` for surfaces without persistence. `advance_head` accepts any
    expected version and returns a head stamped with the current time. `confirm_head` and
    `read_head` return `None`.
    """

    def advance_head(
        self,
        *,
        subject_id: str,
        content_version: str,
        signal: Sequence[MemberSignal],
        expected_content_version: str | None,
    ) -> SubjectHead:
        now = datetime.now(UTC)
        return SubjectHead(
            subject_id=subject_id,
            content_version=content_version,
            confirmed_at=now,
            materialised_at=now,
            signal=tuple(signal),
        )

    def confirm_head(self, *, subject_id: str, content_version: str) -> SubjectHead | None:
        return None

    def read_head(self, subject_id: str) -> SubjectHead | None:
        return None
