"""Layout, pointer and manifest of a published origin catalogue, without I/O.

A catalogue lives under `<ORIGIN_CATALOGUE_URI>/format-<N>/`, laid out as
the coverage index is: `current.json` names one immutable build directory,
`versions/<build_id>/`, which holds the manifest and the tables it names.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from artifact_publication import POINTER_FILE, VERSIONS_PREFIX

CATALOGUE_FORMAT_VERSION: Final[int] = 4
"""The on-disk shape: file names, Parquet schemas and manifest fields. Every
artifact lives under its format's own root, so a change to this number
starts a new, empty root that the next job run fills, and the loader refuses
a manifest of any other format."""

FORMAT_ROOT: Final[str] = f"format-{CATALOGUE_FORMAT_VERSION}"
MANIFEST_FILE: Final[str] = "manifest.json"

PROVENANCES_FILE: Final[str] = "provenances.json"
DFO_VARIABLES_FILE: Final[str] = "dfo_variables.parquet"
BASE_VARIABLES_FILE: Final[str] = "base_variables.parquet"
PROVENANCE_PLACES_FILE: Final[str] = "provenance_places.parquet"
WITHHELD_VARIABLES_FILE: Final[str] = "withheld_variables.parquet"
TOPICS_FILE: Final[str] = "topics.parquet"
SCAN_FILE: Final[str] = "scan.json"
UN_FILE: Final[str] = "un.json"
DATASET_FACTS_FILE: Final[str] = "dataset_facts.json"
SOURCE_DOCUMENTATION_FILE: Final[str] = "source_documentation.json"
SUPERSEDED_VARIABLES_FILE: Final[str] = "superseded_variables.parquet"
UN_VARIABLES_FILE: Final[str] = "un_variables.parquet"
UN_ARCS_FILE: Final[str] = "un_arcs.parquet"
UN_EXCLUDED_VARIABLES_FILE: Final[str] = "un_excluded_variables.parquet"
UN_TOPICS_FILE: Final[str] = "un_topics.parquet"
UN_MIRROR_FILE: Final[str] = "un_mirror.json"

_BUILD_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def identity_key(inputs: Mapping[str, str]) -> str:
    """SHA-256 over the sorted identity inputs. Two builds with equal
    inputs describe the same catalogue, so the job skips the second."""
    encoded = json.dumps(dict(inputs), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ScanLevel:
    """Places of `place_type` contained, at any depth, in `within`
    (`<-containedInPlace+{typeOf:<place_type>}`), or every place of the type
    when `within` is `None`."""

    place_type: str
    within: str | None = None


@dataclass(frozen=True, slots=True)
class CataloguePointer:
    """`current.json`: the published build and the identity it was built from."""

    build_id: str
    identity: str

    def to_json(self) -> str:
        return json.dumps({"build_id": self.build_id, "identity": self.identity}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: bytes | str) -> CataloguePointer:
        """Raises `ValueError` unless `raw` is a JSON object with a non-empty
        string `identity` and a `build_id` that is one path segment of
        letters, digits, `.`, `_` and `-` starting with a letter or digit."""
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"pointer is not a JSON object: {payload!r}")
        build_id = payload.get("build_id")
        identity = payload.get("identity")
        if not isinstance(build_id, str) or not _BUILD_ID_PATTERN.fullmatch(build_id):
            raise ValueError(f"pointer has no valid 'build_id': {payload!r}")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"pointer has no string 'identity': {payload!r}")
        return cls(build_id=build_id, identity=identity)


@dataclass(frozen=True, slots=True)
class UnCarryForward:
    """Why a build serves an earlier build's UN rows, refused places and UN
    tables: `from_build` is the build that last read them from the UN
    deployment, and `reason` is the UN stage failure that kept this build
    from reading them."""

    from_build: str
    reason: str


@dataclass(frozen=True, slots=True)
class UnAgencyCounts:
    """One UN agency's part of the UN tables. `variables` counts every
    member variable of the agency, the excluded ones included, and equals
    the UN deployment's own count for the agency's root group.
    `excluded_series` counts the series its registry entry excludes and
    `excluded_variables` their member variables."""

    series: int
    variables: int
    excluded_series: int
    excluded_variables: int


@dataclass(frozen=True, slots=True)
class UnMirrorManifest:
    """The UN tables' part of a manifest: each UN table's row count, the
    counts of each agency by id segment, and the tables' size in memory once
    decoded (`origin_catalogue.artifact.decode_un_mirror`)."""

    row_counts: Mapping[str, int] = field(default_factory=dict)
    agencies: Mapping[str, UnAgencyCounts] = field(default_factory=dict)
    resident_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CatalogueManifest:
    """One build's manifest.

    `identity_inputs` are what the build's identity hashes, as the job
    chooses them (`origin_catalogue.job.identity_inputs`).
    `resident_bytes` is the loaded tables' size in memory, which the loader
    checks against its budget before reading them. `overlaps` lists every
    variable the DFO variable set and an admitted base provenance both hold.
    `files` maps each table's logical name to its path in the build directory.
    `un_carried_forward` is set when the build carries its UN rows, refused
    places and UN tables from an earlier build, and `None` when it read them
    from the UN. `un_mirror` describes the UN tables, which `row_counts` and
    `resident_bytes` leave out.
    """

    build_id: str
    built_at: datetime
    build_seconds: float
    identity_inputs: Mapping[str, str]
    registry_hash: str
    row_counts: Mapping[str, int]
    resident_bytes: int
    overlaps: tuple[str, ...]
    files: Mapping[str, str]
    un_carried_forward: UnCarryForward | None = None
    un_mirror: UnMirrorManifest = field(default_factory=UnMirrorManifest)
    format_version: int = CATALOGUE_FORMAT_VERSION

    @property
    def identity(self) -> str:
        return identity_key(self.identity_inputs)

    def to_json(self) -> str:
        return json.dumps(
            {
                "build_id": self.build_id,
                "built_at": self.built_at.isoformat(),
                "build_seconds": self.build_seconds,
                "identity_inputs": dict(self.identity_inputs),
                "registry_hash": self.registry_hash,
                "row_counts": dict(self.row_counts),
                "resident_bytes": self.resident_bytes,
                "overlaps": list(self.overlaps),
                "files": dict(self.files),
                "un_carried_forward": (
                    None
                    if self.un_carried_forward is None
                    else {
                        "from_build": self.un_carried_forward.from_build,
                        "reason": self.un_carried_forward.reason,
                    }
                ),
                "un_mirror": {
                    "row_counts": dict(self.un_mirror.row_counts),
                    "agencies": {
                        segment: {
                            "series": counts.series,
                            "variables": counts.variables,
                            "excluded_series": counts.excluded_series,
                            "excluded_variables": counts.excluded_variables,
                        }
                        for segment, counts in self.un_mirror.agencies.items()
                    },
                    "resident_bytes": self.un_mirror.resident_bytes,
                },
                "format_version": self.format_version,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: bytes | str) -> CatalogueManifest:
        """Raises `ValueError` for a malformed manifest, `KeyError` naming a
        missing field. `built_at` must carry a UTC offset."""
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"manifest is not a JSON object: {payload!r}")
        built_at = datetime.fromisoformat(payload["built_at"])
        if built_at.tzinfo is None:
            raise ValueError(f"manifest built_at {payload['built_at']!r} has no UTC offset")
        return cls(
            build_id=str(payload["build_id"]),
            built_at=built_at,
            build_seconds=float(payload["build_seconds"]),
            identity_inputs={str(k): str(v) for k, v in dict(payload["identity_inputs"]).items()},
            registry_hash=str(payload["registry_hash"]),
            row_counts={str(k): int(v) for k, v in dict(payload["row_counts"]).items()},
            resident_bytes=int(payload["resident_bytes"]),
            overlaps=tuple(str(v) for v in payload["overlaps"]),
            files={str(k): str(v) for k, v in dict(payload["files"]).items()},
            un_carried_forward=(
                None
                if (carried := payload["un_carried_forward"]) is None
                else UnCarryForward(
                    from_build=str(carried["from_build"]), reason=str(carried["reason"])
                )
            ),
            un_mirror=UnMirrorManifest(
                row_counts={
                    str(k): int(v) for k, v in dict(payload["un_mirror"]["row_counts"]).items()
                },
                agencies={
                    str(segment): agency_counts_from_json(counts)
                    for segment, counts in dict(payload["un_mirror"]["agencies"]).items()
                },
                resident_bytes=int(payload["un_mirror"]["resident_bytes"]),
            ),
            format_version=int(payload["format_version"]),
        )


def agency_counts_from_json(payload: Mapping[str, Any]) -> UnAgencyCounts:
    """Raises `KeyError` naming a missing count, `ValueError` for a non-integer one."""
    return UnAgencyCounts(
        series=int(payload["series"]),
        variables=int(payload["variables"]),
        excluded_series=int(payload["excluded_series"]),
        excluded_variables=int(payload["excluded_variables"]),
    )


__all__ = [
    "BASE_VARIABLES_FILE",
    "CATALOGUE_FORMAT_VERSION",
    "DATASET_FACTS_FILE",
    "DFO_VARIABLES_FILE",
    "FORMAT_ROOT",
    "MANIFEST_FILE",
    "POINTER_FILE",
    "PROVENANCES_FILE",
    "PROVENANCE_PLACES_FILE",
    "SCAN_FILE",
    "SOURCE_DOCUMENTATION_FILE",
    "SUPERSEDED_VARIABLES_FILE",
    "TOPICS_FILE",
    "UN_ARCS_FILE",
    "UN_EXCLUDED_VARIABLES_FILE",
    "UN_FILE",
    "UN_MIRROR_FILE",
    "UN_TOPICS_FILE",
    "UN_VARIABLES_FILE",
    "VERSIONS_PREFIX",
    "WITHHELD_VARIABLES_FILE",
    "CatalogueManifest",
    "CataloguePointer",
    "ScanLevel",
    "UnAgencyCounts",
    "UnCarryForward",
    "UnMirrorManifest",
    "agency_counts_from_json",
    "identity_key",
]
