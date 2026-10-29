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
- `latest_years`: for each member of a `latest` run, the year the coverage index elected for
  it and the year its rows hold, with the facets and omitted entities the election counted
  (`window.LatestYear`).
- `imported_data`: the bound graph release's digest of the import it copied and the serving
  views a mint reads (`graph_release.service.Release.imported_data`) when only DFO holds each
  member StatVar and every provenance the run read, rate tables included, is one the bound
  origin catalogue lists as DFO's, or `None` otherwise (`run_imported_data`). A DFO
  provenance's rows reach the mixer only through an import, and every import that lands moves
  the digest, so an unchanged digest means unchanged rows. Base Data Commons and UN rows move
  without one, and a base or UN holder of a member could win its facet election.

The graph serving revision is not a member: it moves on every graph restart. Materialisation
builds the fingerprint from what the run dispatched (`build_dependencies`), so every entry
point that runs the same inputs records the same fingerprint. A freshness check rebuilds the
fingerprint a run would record now against a stored one's refs and election arguments
(`refreshed_dependencies`), and `dependency_changes` compares the two member by member.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from graph_release.context import bound_release_or_none
from materialise.content_version import recipe_dependency_fingerprint
from materialise.convert import ConversionRecord
from materialise.facets import ProvenanceRefresh, _probe_provenance_refresh
from materialise.generation import bare_stat_var_ref, elected_latest_year
from materialise.graph import GraphReader
from materialise.window import LatestYear, RequestedWindow
from origin_catalogue import current_catalogue
from origin_catalogue.origins import DFO

if TYPE_CHECKING:
    from materialise.pipeline import MaterialisationRequest
    from series_coverage.index import CoverageIndex

DEPENDENCIES_SCHEMA = 1
"""The version of `Dependencies.as_json`'s shape. A stored fingerprint of another schema
compares as changed (`dependency_changes`)."""

# The JSON list fields of `Dependencies.as_json`, each with its entry's key and value names.
_PAIR_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("sources", "ref", "token"),
    ("conversion_sources", "ref", "token"),
    ("recipes", "id", "digest"),
)

# A `latest` member that elected no year, over no facet.
_NO_ELECTION = LatestYear(elected=None, served=None)


@dataclass(frozen=True)
class Dependencies:
    """A run's dependency fingerprint. Each pair tuple is sorted by its first element and
    holds each ref, recipe id or StatVar once. `latest_years` holds one pair per member of a
    `latest` run, keyed by the request's `stat_var`, and is empty for any other window."""

    members: str
    sources: tuple[tuple[str, str | None], ...]
    conversion_sources: tuple[tuple[str, str | None], ...]
    recipes: tuple[tuple[str, str | None], ...]
    window: tuple[int | None, int | None] | None
    latest_years: tuple[tuple[str, LatestYear], ...] = ()
    imported_data: str | None = None

    def as_json(self) -> dict[str, object]:
        """The stored shape, `{"schema", "members", "sources", "conversion_sources",
        "recipes", "window"}`, with a JSON object per pair and `window` as
        `{"start_year", "end_year"}` or null. A `latest` run adds `latest_years`, one
        `{"stat_var", "elected", "served", "facet_ids", "omitted"}` per member, `omitted` a
        list of `{"role", "entity"}`. A run keyed on an import adds `imported_data`."""
        encoded: dict[str, object] = {"schema": DEPENDENCIES_SCHEMA, "members": self.members}
        for name, key, value in _PAIR_FIELDS:
            encoded[name] = [{key: ref, value: token} for ref, token in getattr(self, name)]
        encoded["window"] = (
            None
            if self.window is None
            else {"start_year": self.window[0], "end_year": self.window[1]}
        )
        if self.latest_years:
            encoded["latest_years"] = [
                {
                    "stat_var": stat_var,
                    "elected": latest.elected,
                    "served": latest.served,
                    "facet_ids": list(latest.facet_ids),
                    "omitted": [
                        {"role": role, "entity": entity} for role, entity in latest.omitted
                    ],
                }
                for stat_var, latest in self.latest_years
            ]
        if self.imported_data is not None:
            encoded["imported_data"] = self.imported_data
        return encoded


def members_digest(stat_vars: Iterable[str]) -> str:
    """`sha256:` and the hex digest of the JSON list of `stat_vars` as sorted, distinct bare
    dcids, so the digest is independent of resolution order and the `dcid:` prefix."""
    bare = sorted({bare_stat_var_ref(ref) for ref in stat_vars})
    encoded = json.dumps(bare, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _recipe_digest(recipe_id: str) -> str | None:
    fingerprint = recipe_dependency_fingerprint(recipe_id)
    return fingerprint[1] if fingerprint is not None else None


def _clock_window(window: RequestedWindow) -> tuple[int | None, int | None] | None:
    if window.kind != "range" or not window.end_unstated:
        return None
    return window.start_year, window.end_year


def _latest_members(
    requests: Sequence[MaterialisationRequest], latest_years: Mapping[str, LatestYear]
) -> dict[str, LatestYear]:
    """Each `latest` request's entry of `latest_years` by its `stat_var`, `_NO_ELECTION`
    for one it lacks, which a check elects for over no facet and so matches."""
    return {
        request.stat_var: latest_years.get(request.stat_var, _NO_ELECTION)
        for request in requests
        if request.window.kind == "latest"
    }


def run_imported_data(*, stat_vars: Iterable[str], provenances: Iterable[str]) -> str | None:
    """The bound graph release's `imported_data` digest when the bound origin catalogue names
    DFO alone as holding each of `stat_vars` and lists each of `provenances`, which must not be
    empty, as DFO's. Otherwise `None`, and also `None` with no release bound.

    Holding is the catalogue's `data_commons`, every origin with admitted data for the
    StatVar at any place, so a base or UN facet that facet election could rank above DFO's
    (`facets.rank_candidates`, `facets.dfo_precedence`) unkeys the run whatever it
    dispatched."""
    release = bound_release_or_none()
    refs = set(provenances)
    if release is None or not refs:
        return None
    catalogue = current_catalogue()
    if any(catalogue.data_commons(bare_stat_var_ref(ref)) != (DFO,) for ref in set(stat_vars)):
        return None
    for ref in refs:
        row = catalogue.provenance(ref)
        if row is None or row.data_commons != DFO:
            return None
    return release.imported_data


def build_dependencies(
    requests: Sequence[MaterialisationRequest],
    *,
    source_refs: Iterable[str],
    provenance_refresh: Mapping[str, ProvenanceRefresh],
    conversion_records: Iterable[ConversionRecord | None],
    latest_years: Mapping[str, LatestYear],
) -> Dependencies:
    """The fingerprint of a run over `requests` that dispatched the provenances `source_refs`,
    converted with `conversion_records`, and served each `latest` member at `latest_years`
    (`MaterialisationResult.latest_years`). `provenance_refresh` holds each dispatched
    provenance's refresh dates; a ref it lacks records no date. The requests of one run share
    one window, so the first request's gives `window`. `imported_data` is `run_imported_data`
    over the requests' StatVars and the sources and rate table provenances."""
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
        latest_years=tuple(sorted(_latest_members(requests, latest_years).items())),
        imported_data=run_imported_data(
            stat_vars=(request.stat_var for request in requests),
            provenances=sources.keys() | conversion_sources.keys(),
        ),
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


def _is_text_list(values: object) -> bool:
    return isinstance(values, list) and all(isinstance(value, str) for value in values)


def _decode_latest_years(entries: object) -> tuple[tuple[str, LatestYear], ...] | None:
    """A stored `latest_years` list as sorted `(stat_var, LatestYear)` pairs, or `None` when
    malformed."""
    if not isinstance(entries, list):
        return None
    years: dict[str, LatestYear] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        member = cast("Mapping[str, object]", entry)
        stat_var, elected, served = (
            member.get("stat_var"),
            member.get("elected"),
            member.get("served"),
        )
        facet_ids, omitted = member.get("facet_ids"), member.get("omitted")
        if not (isinstance(stat_var, str) and _is_year(elected) and _is_year(served)):
            return None
        if not (_is_text_list(facet_ids) and isinstance(omitted, list)):
            return None
        rows: list[tuple[str, str]] = []
        for row in omitted:
            if not isinstance(row, Mapping):
                return None
            role, entity = row.get("role"), row.get("entity")
            if not (isinstance(role, str) and isinstance(entity, str)):
                return None
            rows.append((role, entity))
        years[stat_var] = LatestYear(
            elected=cast("int | None", elected),
            served=cast("int | None", served),
            facet_ids=tuple(cast("list[str]", facet_ids)),
            omitted=tuple(rows),
        )
    return tuple(sorted(years.items()))


def decode_dependencies(encoded: Mapping[str, object]) -> Dependencies | None:
    """The fingerprint `Dependencies.as_json` stored as `encoded`, or `None` when its schema is
    unknown or its shape malformed. One stored with no `latest_years` holds none, and one
    with no `imported_data` is keyed on no import."""
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
    latest_years = _decode_latest_years(encoded.get("latest_years", []))
    if latest_years is None:
        return None
    imported_data = encoded.get("imported_data")
    if imported_data is not None and not isinstance(imported_data, str):
        return None
    return Dependencies(
        members=members,
        sources=pairs["sources"],
        conversion_sources=pairs["conversion_sources"],
        recipes=pairs["recipes"],
        window=window,
        latest_years=latest_years,
        imported_data=imported_data,
    )


def refreshed_dependencies(
    stored: Dependencies,
    requests: Sequence[MaterialisationRequest],
    *,
    graph: GraphReader,
    coverage: CoverageIndex | None,
) -> Dependencies:
    """The fingerprint a run over `requests` would record now against the refs and election
    arguments `stored` names, without materialising.

    The source and conversion-source refs keep `stored`'s, each with the `lastDataRefreshDate`
    the graph holds now, read for all of them in one `node_arcs_many`, which `GraphReader`
    serves uncached. Each stored recipe takes the digest of its deployed body. `members` and
    `window` come from `requests` as `build_dependencies` takes them, so an open window ends
    at the year the requests were resolved in.

    Each `latest` request keeps the entry `stored` holds for it with the year `coverage`
    elects now over the entry's facets and omitted entities (`elected_latest_year`), read
    under the source scope in context, as the run's was. The year the run served is kept, so
    a run that served another year than it elected differs only once the index elects
    another. A `latest` request `stored` holds no entry for elects over every facet the
    index holds of its StatVar under that scope. If it elects a year, it gains an entry
    `stored` lacks and so differs in `latest_years`. If it elects none, as a StatVar outside
    the index or a subject-only request does, it gains none and matches.

    `imported_data` keeps `stored`'s. The head rule compares it with the bound release's
    (`datasets_api.current`), and `dependency_changes` never does.
    """
    refs = sorted(
        {ref for ref, _ in stored.sources} | {ref for ref, _ in stored.conversion_sources}
    )
    refresh = _probe_provenance_refresh(refs, graph=graph)

    def _token(ref: str) -> str | None:
        return refresh[ref].last_data_refresh_date

    recorded = dict(stored.latest_years)

    def _elected_now(request: MaterialisationRequest) -> LatestYear | None:
        latest = recorded.get(request.stat_var)
        if latest is not None:
            year = elected_latest_year(
                request, latest.omitted, coverage=coverage, facet_ids=latest.facet_ids
            )
            return replace(latest, elected=year)
        held = (
            coverage.facet_spans(variables=[bare_stat_var_ref(request.stat_var)])
            if coverage is not None
            else {}
        )
        year = elected_latest_year(
            request, (), coverage=coverage, facet_ids=sorted(facet for _, facet in held)
        )
        return replace(_NO_ELECTION, elected=year) if year is not None else None

    return Dependencies(
        members=members_digest(request.stat_var for request in requests),
        sources=tuple((ref, _token(ref)) for ref, _ in stored.sources),
        conversion_sources=tuple((ref, _token(ref)) for ref, _ in stored.conversion_sources),
        recipes=tuple((recipe_id, _recipe_digest(recipe_id)) for recipe_id, _ in stored.recipes),
        window=_clock_window(requests[0].window),
        latest_years=tuple(
            sorted(
                (request.stat_var, latest)
                for request in requests
                if request.window.kind == "latest" and (latest := _elected_now(request)) is not None
            )
        ),
        imported_data=stored.imported_data,
    )


# The pair fields whose tokens are refresh dates. A current `None` date says nothing about the
# source, so it never differs from a stored one (`dependency_changes`).
_DATE_FIELDS = ("sources", "conversion_sources")


def _pairs_by_name(dependencies: Dependencies) -> dict[str, object]:
    members: dict[str, object] = {
        "members": dependencies.members,
        "window": dependencies.window,
        "latest_years": dependencies.latest_years,
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

    A member is `members`, `window`, `latest_years`, or a pair named `<field>:<ref>`, such as
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
    "run_imported_data",
]
