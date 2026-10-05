"""Curated clusters of StatVars that encode the same indicator across vocabularies.

The graph has no reliable link for these twins, and a five-tuple cannot identify
members whose measured-property node lacks a dcid. The JSON schema is tooling-only
because ``jsonschema`` is a dev/eval dependency. The package-data registry is
validated by its loader at import. Missing or malformed data raises an error.
This leaf module reads no data beyond its own package resource.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import NoReturn, cast

from search.engine.unit.lexicon import ISO_4217_CODES

_PACKAGE = "search.engine.region"
_DATA_FILENAME = "equivalent_series.json"
_EXPECTED_FORMAT_VERSION = 1
_ALLOWED_DOCUMENT_KEYS = {"format_version", "entries"}
_ALLOWED_ENTRY_KEYS = {"members", "note", "preferred", "currency"}


@dataclass(frozen=True)
class EquivalentCluster:
    """A registered set of StatVars encoding the same indicator.

    `preferred` is an authority choice between members that cover a question equally: the
    member an answer serves first when it is shown to hold data in every requested year the
    elected twin holds, at every place (`candidate_collapse._preferred_twin`). A question
    that names no dates compares twins by place only, so the preferred member serves
    whenever it has data at the question's places, whatever its history. Where it covers
    less, the elected twin answers and lists it. `None` leaves the elected twin first. Membership alone states semantic equivalence. `currency` is the ISO 4217
    code the members are equivalent in, so a member is a twin only through its facets
    in that currency. It is `None` when the indicator carries no currency.
    """

    members: frozenset[str]
    note: str
    preferred: str | None = None
    currency: str | None = None


class EquivalentSeriesRegistryError(Exception):
    """A missing or invalid registry raises on import. There is no empty fallback."""


def _fail(*, source: str, problem: str) -> NoReturn:
    raise EquivalentSeriesRegistryError(f"{source}: {problem}")


def _parse_entry(
    *, source: str, index: int, entry: object, seen: dict[str, int]
) -> EquivalentCluster:
    if not isinstance(entry, dict):
        _fail(source=source, problem=f"entries[{index}] must be an object")
    entry = cast("dict[str, object]", entry)

    unknown_keys = sorted(set(entry) - _ALLOWED_ENTRY_KEYS)
    if unknown_keys:
        _fail(
            source=source,
            problem=(
                f"entries[{index}] has unknown key(s) {unknown_keys!r} -- "
                "only 'members', 'note', 'preferred' and 'currency' are allowed"
            ),
        )

    members_raw = entry.get("members")
    if (
        not isinstance(members_raw, list)
        or len(members_raw) < 2
        or not all(isinstance(member, str) and member for member in members_raw)
        or len(set(members_raw)) != len(members_raw)
    ):
        _fail(
            source=source,
            problem=(
                f"entries[{index}].members must be an array of at least two "
                "distinct, non-empty strings"
            ),
        )
    members = cast("list[str]", members_raw)

    note = entry.get("note")
    if not isinstance(note, str) or not note:
        _fail(source=source, problem=f"entries[{index}].note must be a non-empty string")

    preferred = entry.get("preferred")
    if preferred is not None and (not isinstance(preferred, str) or preferred not in members):
        _fail(
            source=source,
            problem=f"entries[{index}].preferred must be one of the entry's members",
        )

    currency = entry.get("currency")
    if currency is not None and (not isinstance(currency, str) or currency not in ISO_4217_CODES):
        _fail(
            source=source,
            problem=f"entries[{index}].currency must be an ISO 4217 code, got {currency!r}",
        )

    for member in members:
        if member in seen:
            _fail(
                source=source,
                problem=(
                    f"dcid {member!r} appears in both entries[{seen[member]}] and "
                    f"entries[{index}] -- a StatVar can belong to at most one twin cluster"
                ),
            )
        seen[member] = index

    return EquivalentCluster(
        members=frozenset(members),
        note=note,
        preferred=preferred,
        currency=currency,
    )


def _parse_document(*, source: str, text: str) -> tuple[EquivalentCluster, ...]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        _fail(source=source, problem=f"invalid JSON ({exc})")

    if not isinstance(document, dict):
        _fail(source=source, problem="top-level document must be an object")
    document = cast("dict[str, object]", document)

    unknown_keys = sorted(set(document) - _ALLOWED_DOCUMENT_KEYS)
    if unknown_keys:
        _fail(
            source=source,
            problem=(
                f"unknown top-level key(s) {unknown_keys!r} -- "
                "only 'format_version' and 'entries' are allowed"
            ),
        )

    format_version = document.get("format_version")
    # isinstance(True, int) is True in Python, so an explicit bool check is
    # needed to reject `format_version: true` -- it would otherwise pass the
    # `!= 1` comparison below by comparing equal to the int 1.
    if isinstance(format_version, bool) or format_version != _EXPECTED_FORMAT_VERSION:
        _fail(
            source=source,
            problem=f"format_version must be {_EXPECTED_FORMAT_VERSION}, got {format_version!r}",
        )

    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail(source=source, problem="entries must be a non-empty array")

    seen: dict[str, int] = {}
    return tuple(
        _parse_entry(source=source, index=index, entry=entry, seen=seen)
        for index, entry in enumerate(entries)
    )


def _load_clusters(*, directory: Traversable | None = None) -> tuple[EquivalentCluster, ...]:
    """Load the registry from ``directory`` or this package's resources.

    The resource reader supports installed wheels and zip files. Tests can
    supply a temporary directory to check invalid or missing data.
    """
    resource = (directory if directory is not None else resources.files(_PACKAGE)) / _DATA_FILENAME
    source = str(resource)
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EquivalentSeriesRegistryError(f"{source}: file not found") from exc
    return _parse_document(source=source, text=text)


_CLUSTERS: tuple[EquivalentCluster, ...] = _load_clusters()
# The cluster of every registered StatVar. A StatVar belongs to at most one, and a member
# set intersected with the keys holds only registered StatVars.
CLUSTER_BY_MEMBER: Mapping[str, EquivalentCluster] = MappingProxyType(
    {member: cluster for cluster in _CLUSTERS for member in cluster.members}
)


def shared_cluster(
    elected_dcids: Iterable[str], candidate_dcids: Iterable[str]
) -> EquivalentCluster | None:
    """Return the first cluster containing both dcid sets, or ``None``.

    Both sets must be non-empty, distinct, and fully contained in one cluster.
    Extra dcids disqualify a side. If multiple clusters qualify, registration
    order decides.
    """
    elected = set(elected_dcids)
    candidate = set(candidate_dcids)
    for cluster in _CLUSTERS:
        if (
            elected
            and candidate
            and elected != candidate
            and elected <= cluster.members
            and candidate <= cluster.members
        ):
            return cluster
    return None
