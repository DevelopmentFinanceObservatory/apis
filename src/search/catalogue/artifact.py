"""The search catalogue artifact: its schema, build key and publication.

`SEARCH_CATALOGUE_URI` (`gs://` or `file://`) is the bucket root. The job
reads and writes only its format root, `<SEARCH_CATALOGUE_URI>/format-<N>/`,
which holds `current.json` (the pointer, written last), one
`versions/<catalogue_version>/catalogue.json` per build, the job's
`lease.json`, and `refused.json`, the key of the last build that fell under
the floor. A build is one JSON document, so a place-scoped answer is a
lookup in `places`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from artifact_publication import POINTER_FILE, VERSIONS_PREFIX
from artifact_publication.jobs import prune_versions
from artifact_publication.uri import (
    join_uri,
    read_uri_bytes,
    read_uri_bytes_and_generation,
    write_uri_bytes,
    write_uri_bytes_if_unchanged,
)

logger = logging.getLogger(__name__)

FORMAT_VERSION: Final[int] = 2
"""Bumped when a build's JSON changes shape, so an image reads only the
builds it understands."""

FORMAT_ROOT: Final[str] = f"format-{FORMAT_VERSION}"
BUILD_FILE: Final[str] = "catalogue.json"
REFUSAL_FILE: Final[str] = "refused.json"

_VERSION_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

CatalogueVersion = Annotated[str, StringConstraints(pattern=_VERSION_PATTERN)]
"""A build's directory name under `versions/`: one path segment of letters,
digits, `.`, `_` and `-`, starting with a letter or digit."""


class CatalogueArtifactError(Exception):
    """A published pointer or build is missing or does not parse."""


class _Model(BaseModel):
    """Refuses unknown fields, so a build from another shape fails to load."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class YearRange(_Model):
    first: int
    last: int


class AidCounts(_Model):
    """Counts over the coverage index's rows of the declared datasets."""

    donors: int = Field(
        description=(
            "distinct donors: countries and organisations, leaving out the parties that total "
            "other donors (DAC countries, the G7)"
        )
    )
    recipients: int = Field(
        description=(
            "distinct recipients: countries, organisations and the residuals no country holds "
            "(Africa, unallocated by country), leaving out the parties that total other "
            "recipients (Africa, Total)"
        )
    )
    years: YearRange | None = Field(description="the years with aid data; null when none")


class Counts(_Model):
    """What DFO holds, counted from one origin catalogue build and one
    coverage index build."""

    indicators: int = Field(
        description="distinct StatVars search can serve: base, DFO and coverage variables"
    )
    datasets: int = Field(description="distinct source-index entries of the admitted provenances")
    publishers: int = Field(description="distinct sources of the admitted provenances")
    countries: int = Field(
        description="countries with base data in the origin catalogue or coverage rows"
    )
    aid: AidCounts


class PlaceCounts(_Model):
    """What DFO holds at one country. For aid, `donors` counts the donors to
    the country and `recipients` the recipients of its own aid."""

    publishers: int = Field(description="distinct sources with data at the country")
    variables: int = Field(description="distinct StatVars with data at the country")
    aid: AidCounts


class CatalogueExample(_Model):
    """An example that passed the build's check, filled at its default place."""

    id: int = Field(description="the line's position in examples.toml")
    text: str
    template: str
    stat_var: str
    place: str


class CataloguePlace(_Model):
    """One country with data. `examples` holds the ids of the examples whose
    StatVar has data there, empty when the graph names no `name` to fill
    them with."""

    name: str | None
    counts: PlaceCounts
    examples: list[int]


class BuildInputs(_Model):
    """What a build read. `build_key` hashes these. `coverage_build_id` is the
    id of the graph release whose coverage view the build read."""

    origin_catalogue_build_id: str
    coverage_build_id: str
    build_commit: str
    examples_digest: str


class CatalogueArtifact(_Model):
    """One published build."""

    catalogue_version: CatalogueVersion
    key: str
    built_at: datetime
    inputs: BuildInputs
    counts: Counts
    examples: list[CatalogueExample]
    places: dict[str, CataloguePlace] = Field(description="every counted country, by dcid")


class CataloguePointer(_Model):
    """`current.json`: the live build and its key."""

    catalogue_version: CatalogueVersion
    key: str


class CatalogueRefusal(_Model):
    """`refused.json`: the last build that fell under the floor, so a run
    with the same key can skip the engine calls that refused it."""

    key: str
    refused_at: datetime
    passed: list[int] = Field(description="the ids of the lines that passed")


def build_key(inputs: BuildInputs) -> str:
    """The SHA-256 of the format version and `inputs`. A run whose key
    equals the published one has nothing to build."""
    payload = {"format_version": FORMAT_VERSION, **inputs.model_dump()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def format_root(base_uri: str) -> str:
    return join_uri(base_uri, FORMAT_ROOT)


def read_pointer(base_uri: str) -> tuple[CataloguePointer | None, str | None]:
    """The published pointer and its version token. A missing pointer gives
    `(None, None)`. A malformed one gives `(None, token)`, so a publish can
    replace it. Other read failures propagate."""
    uri = join_uri(format_root(base_uri), POINTER_FILE)
    found = read_uri_bytes_and_generation(uri)
    if found is None:
        return None, None
    raw, generation = found
    try:
        return CataloguePointer.model_validate_json(raw), generation
    except ValidationError:
        logger.warning("search catalogue: current.json at %r is malformed; rebuilding", uri)
        return None, generation


def read_refusal(base_uri: str) -> CatalogueRefusal | None:
    """The recorded refusal, or `None` when there is none or it does not
    parse. Other read failures propagate."""
    uri = join_uri(format_root(base_uri), REFUSAL_FILE)
    found = read_uri_bytes_and_generation(uri)
    if found is None:
        return None
    try:
        return CatalogueRefusal.model_validate_json(found[0])
    except ValidationError:
        logger.warning("search catalogue: refused.json at %r is malformed; ignoring it", uri)
        return None


def record_refusal(base_uri: str, refusal: CatalogueRefusal) -> None:
    """Writes `refused.json` over any earlier refusal. The run writing it
    holds the lease."""
    write_uri_bytes(
        join_uri(format_root(base_uri), REFUSAL_FILE), refusal.model_dump_json().encode("utf-8")
    )


def publish(base_uri: str, artifact: CatalogueArtifact, *, expected_generation: str | None) -> bool:
    """Writes the build create-only, then `current.json` conditional on
    `expected_generation` (`None` requires no pointer). Returns whether the
    pointer write landed; `False` means another run published first."""
    root = format_root(base_uri)
    build_uri = _build_uri(base_uri, artifact.catalogue_version)
    if not write_uri_bytes_if_unchanged(
        build_uri, artifact.model_dump_json().encode("utf-8"), expected_generation=None
    ):
        raise CatalogueArtifactError(f"{build_uri!r} already exists")
    pointer = CataloguePointer(catalogue_version=artifact.catalogue_version, key=artifact.key)
    return write_uri_bytes_if_unchanged(
        join_uri(root, POINTER_FILE),
        pointer.model_dump_json().encode("utf-8"),
        expected_generation=expected_generation,
    )


def load_build(base_uri: str, catalogue_version: str) -> CatalogueArtifact:
    """Reads one build. Raises `CatalogueArtifactError` when
    `catalogue_version` is not one path segment, or the build is missing,
    does not parse, or carries another version."""
    uri = _build_uri(base_uri, catalogue_version)
    try:
        artifact = CatalogueArtifact.model_validate_json(read_uri_bytes(uri))
    except (FileNotFoundError, ValidationError) as exc:
        raise CatalogueArtifactError(f"catalogue build at {uri!r} is unreadable: {exc}") from exc
    if artifact.catalogue_version != catalogue_version:
        raise CatalogueArtifactError(
            f"catalogue build at {uri!r} carries version {artifact.catalogue_version!r}"
        )
    return artifact


def _build_uri(base_uri: str, catalogue_version: str) -> str:
    """Where build `catalogue_version` lives, refusing a version that would
    name a path outside `versions/`."""
    if not re.fullmatch(_VERSION_PATTERN, catalogue_version):
        raise CatalogueArtifactError(
            f"catalogue version {catalogue_version!r} is not one path segment"
        )
    return join_uri(format_root(base_uri), VERSIONS_PREFIX, catalogue_version, BUILD_FILE)


def load_published(base_uri: str) -> CatalogueArtifact:
    """Reads the build `current.json` names. Raises `CatalogueArtifactError`
    when nothing is published, or the build is unreadable or carries
    another version or key than the pointer."""
    pointer, _ = read_pointer(base_uri)
    if pointer is None:
        raise CatalogueArtifactError(f"no readable current.json under {format_root(base_uri)!r}")
    artifact = load_build(base_uri, pointer.catalogue_version)
    if artifact.key != pointer.key:
        raise CatalogueArtifactError(
            f"current.json names key {pointer.key!r}, but build "
            f"{pointer.catalogue_version!r} carries {artifact.key!r}"
        )
    return artifact


def prune(
    base_uri: str, *, grace_seconds: float, now: Callable[[], float] = time.time
) -> tuple[str, ...]:
    """Deletes and returns the builds other than the one `current.json` names, once
    they and the pointer are `grace_seconds` old."""
    return prune_versions(
        format_root(base_uri),
        pointer_build_id=lambda raw: CataloguePointer.model_validate_json(raw).catalogue_version,
        grace_seconds=grace_seconds,
        now=now,
    )


__all__ = [
    "BUILD_FILE",
    "FORMAT_ROOT",
    "FORMAT_VERSION",
    "REFUSAL_FILE",
    "AidCounts",
    "BuildInputs",
    "CatalogueArtifact",
    "CatalogueArtifactError",
    "CatalogueExample",
    "CataloguePlace",
    "CataloguePointer",
    "CatalogueRefusal",
    "CatalogueVersion",
    "Counts",
    "PlaceCounts",
    "YearRange",
    "build_key",
    "format_root",
    "load_build",
    "load_published",
    "prune",
    "publish",
    "read_pointer",
    "read_refusal",
    "record_refusal",
]
