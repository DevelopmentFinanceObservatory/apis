"""The dependency fingerprint of a materialisation: what the run read that can change under an
unchanged definition, each member a token that exists today.

- `members`: a digest of the sorted, distinct bare member dcids (`members_digest`).
- `sources`: each dispatched provenance with its `lastDataRefreshDate`.
- `conversion_sources`: each rate table provenance with the refresh date its tables were
  admitted under (`ConversionRecord.table_refresh_date`).
- `recipes`: each conversion recipe with the digest of its deployed body
  (`content_version.recipe_dependency_fingerprint`), `None` for a recipe with no body.
- `window`: the start and end year a range read when its definition left the end unstated,
  the end `window.election_window` gave it from the clock, or `None` for a window that reads
  no clock.

The graph serving revision is not a member: it moves on every graph restart. Materialisation
builds the fingerprint from what the run dispatched (`build_dependencies`), so every entry
point that runs the same inputs records the same fingerprint. A freshness check rebuilds the
fingerprint a run would record now against a stored one's refs (`refreshed_dependencies`), and
`dependency_changes` compares the two member by member.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from materialise.content_version import recipe_dependency_fingerprint
from materialise.convert import ConversionRecord
from materialise.facets import ProvenanceRefresh, _probe_provenance_refresh
from materialise.graph import GraphReader
from materialise.window import RequestedWindow

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest

DEPENDENCIES_SCHEMA = 1
"""The version of `Dependencies.as_json`'s shape. A stored fingerprint of another schema
compares as changed (`dependency_changes`)."""

_STAT_VAR_REF_PREFIX = "dcid:"

# The JSON list fields of `Dependencies.as_json`, each with its entry's key and value names.
_PAIR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("sources", "ref", "token"),
    ("conversion_sources", "ref", "token"),
    ("recipes", "id", "digest"),
)


@dataclass(frozen=True)
class Dependencies:
    """A run's dependency fingerprint. Each pair tuple is sorted by its first element and
    holds each ref or recipe id once."""

    members: str
    sources: tuple[tuple[str, str | None], ...]
    conversion_sources: tuple[tuple[str, str | None], ...]
    recipes: tuple[tuple[str, str | None], ...]
    window: tuple[int | None, int | None] | None

    def as_json(self) -> dict[str, object]:
        """The stored shape, `{"schema", "members", "sources", "conversion_sources",
        "recipes", "window"}`, with a JSON object per pair and `window` as
        `{"start_year", "end_year"}` or null."""
        encoded: dict[str, object] = {"schema": DEPENDENCIES_SCHEMA, "members": self.members}
        for name, key, value in _PAIR_FIELDS:
            encoded[name] = [{key: ref, value: token} for ref, token in getattr(self, name)]
        encoded["window"] = (
            None
            if self.window is None
            else {"start_year": self.window[0], "end_year": self.window[1]}
        )
        return encoded


def members_digest(stat_vars: Iterable[str]) -> str:
    """`sha256:` and the hex digest of the JSON list of `stat_vars` as sorted, distinct bare
    dcids, so the digest is independent of resolution order and the `dcid:` prefix."""
    bare = sorted({ref.removeprefix(_STAT_VAR_REF_PREFIX) for ref in stat_vars})
    encoded = json.dumps(bare, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _recipe_digest(recipe_id: str) -> str | None:
    fingerprint = recipe_dependency_fingerprint(recipe_id)
    return fingerprint[1] if fingerprint is not None else None


def _clock_window(window: RequestedWindow) -> tuple[int | None, int | None] | None:
    if window.kind != "range" or not window.end_unstated:
        return None
    return window.start_year, window.end_year


def build_dependencies(
    requests: Sequence[MaterialisationRequest],
    *,
    source_refs: Iterable[str],
    provenance_refresh: Mapping[str, ProvenanceRefresh],
    conversion_records: Iterable[ConversionRecord | None],
) -> Dependencies:
    """The fingerprint of a run over `requests` that dispatched the provenances `source_refs`
    and converted with `conversion_records`. `provenance_refresh` holds each dispatched
    provenance's refresh dates; a ref it lacks records no date. The requests of one run share
    one window, so the first request's gives `window`."""
    records = [record for record in conversion_records if record is not None]
    sources: dict[str, str | None] = {}
    for ref in source_refs:
        refresh = provenance_refresh.get(ref)
        sources[ref] = refresh.last_data_refresh_date if refresh is not None else None
    conversion_sources = {
        record.table_provenance_id: record.table_refresh_date for record in records
    }
    recipes = {record.recipe_id: _recipe_digest(record.recipe_id) for record in records}
    return Dependencies(
        members=members_digest(request.stat_var for request in requests),
        sources=tuple(sorted(sources.items())),
        conversion_sources=tuple(sorted(conversion_sources.items())),
        recipes=tuple(sorted(recipes.items())),
        window=_clock_window(requests[0].window),
    )


def _decode_pairs(
    entries: object, *, key: str, value: str
) -> tuple[tuple[str, str | None], ...] | None:
    """A stored pair list as sorted `(ref, token)` tuples, or `None` when malformed."""
    if not isinstance(entries, list):
        return None
    pairs: dict[str, str | None] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        pair = cast("Mapping[str, object]", entry)
        ref, token = pair.get(key), pair.get(value)
        if key not in pair or value not in pair or not isinstance(ref, str):
            return None
        if token is not None and not isinstance(token, str):
            return None
        pairs[ref] = token
    return tuple(sorted(pairs.items()))


def _is_year(year: object) -> bool:
    return year is None or (isinstance(year, int) and not isinstance(year, bool))


def decode_dependencies(encoded: Mapping[str, object]) -> Dependencies | None:
    """The fingerprint `Dependencies.as_json` stored as `encoded`, or `None` when its schema is
    unknown or its shape malformed."""
    members = encoded.get("members")
    if encoded.get("schema") != DEPENDENCIES_SCHEMA or not isinstance(members, str):
        return None
    if "window" not in encoded:
        return None
    window: tuple[int | None, int | None] | None = None
    if encoded["window"] is not None:
        if not isinstance(encoded["window"], Mapping):
            return None
        years = cast("Mapping[str, object]", encoded["window"])
        start, end = years.get("start_year"), years.get("end_year")
        if not (_is_year(start) and _is_year(end)):
            return None
        window = (cast("int | None", start), cast("int | None", end))
    pairs: dict[str, tuple[tuple[str, str | None], ...]] = {}
    for name, key, value in _PAIR_FIELDS:
        decoded = _decode_pairs(encoded.get(name), key=key, value=value)
        if decoded is None:
            return None
        pairs[name] = decoded
    return Dependencies(
        members=members,
        sources=pairs["sources"],
        conversion_sources=pairs["conversion_sources"],
        recipes=pairs["recipes"],
        window=window,
    )


def refreshed_dependencies(
    stored: Dependencies, requests: Sequence[MaterialisationRequest], *, graph: GraphReader
) -> Dependencies:
    """The fingerprint a run over `requests` would record now against the refs `stored` names,
    without materialising.

    The source and conversion-source refs keep `stored`'s, each with the `lastDataRefreshDate`
    the graph holds now, read for all of them in one `node_arcs_many`, which `GraphReader`
    serves uncached. Each stored recipe takes the digest of its deployed body. `members` and
    `window` come from `requests` as `build_dependencies` takes them, so an open window ends
    at the year the requests were resolved in.
    """
    refs = sorted(
        {ref for ref, _ in stored.sources} | {ref for ref, _ in stored.conversion_sources}
    )
    refresh = _probe_provenance_refresh(refs, graph=graph)

    def _token(ref: str) -> str | None:
        return refresh[ref].last_data_refresh_date

    return Dependencies(
        members=members_digest(request.stat_var for request in requests),
        sources=tuple((ref, _token(ref)) for ref, _ in stored.sources),
        conversion_sources=tuple((ref, _token(ref)) for ref, _ in stored.conversion_sources),
        recipes=tuple((recipe_id, _recipe_digest(recipe_id)) for recipe_id, _ in stored.recipes),
        window=_clock_window(requests[0].window),
    )


# The pair fields whose tokens are refresh dates. A current `None` date says nothing about the
# source, so it never differs from a stored one (`dependency_changes`).
_DATE_FIELDS = ("sources", "conversion_sources")


def _pairs_by_name(dependencies: Dependencies) -> dict[str, object]:
    members: dict[str, object] = {
        "members": dependencies.members,
        "window": dependencies.window,
    }
    for name, _key, _value in _PAIR_FIELDS:
        for ref, token in getattr(dependencies, name):
            members[f"{name}:{ref}"] = token
    return members


def dependency_changes(
    stored: Mapping[str, object] | None, current: Dependencies
) -> tuple[str, ...]:
    """The members in which `stored`, a fingerprint as `Dependencies.as_json` stored it, differs
    from `current`, sorted; empty when they agree.

    A member is `members`, `window`, or a pair named `<field>:<ref>`, such as
    `sources:provenance/OECDDAC2A`. A ref present on one side only differs. A source or
    conversion source whose current refresh date is `None` is unknown and never differs, so a
    source with no refresh date, or one whose date the graph stops serving, never rebuilds on
    its token. A missing, malformed, or other-schema `stored` returns `("schema",)`.
    """
    decoded = decode_dependencies(stored) if stored is not None else None
    if decoded is None:
        return ("schema",)
    stored_members = _pairs_by_name(decoded)
    current_members = _pairs_by_name(current)

    def _differs(name: str) -> bool:
        if name not in stored_members or name not in current_members:
            return True
        if current_members[name] is None and name.partition(":")[0] in _DATE_FIELDS:
            return False
        return stored_members[name] != current_members[name]

    return tuple(sorted(filter(_differs, stored_members.keys() | current_members.keys())))


__all__ = [
    "DEPENDENCIES_SCHEMA",
    "Dependencies",
    "build_dependencies",
    "decode_dependencies",
    "dependency_changes",
    "members_digest",
    "refreshed_dependencies",
]
