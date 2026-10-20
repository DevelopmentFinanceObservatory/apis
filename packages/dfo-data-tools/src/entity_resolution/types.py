"""The public contract: what a name resolves to, and the graph protocol
the component reads through. Free of any `resolvekit` or `httpx` import,
so a consumer can type against this module without pulling in either.
"""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, get_args, runtime_checkable

Kind = Literal[
    "country",
    "continent",
    "subregion",
    "continental_union",
    "place_other",
    "aggregate",
    "organisation",
    "group",
    "stat_var",
    "code_system",
    "code",
    "unit",
    "source",
    "dataset",
    "provenance",
    "other",
]
"""The closed list of kinds a node takes (`kinds.classify`). `typeOf` values stay open
upstream; a reader switches on 16 values instead of the graph's types."""

KINDS: tuple[Kind, ...] = get_args(Kind)
"""Every kind, in the order the Graph API's `/capabilities` publishes them."""

PlaceKind = Literal["country", "continent", "subregion", "continental_union", "place_other"]
PLACE_KINDS: frozenset[PlaceKind] = frozenset(get_args(PlaceKind))

# The kinds a resolve request may name, a subset of `Kind`.
EntityKind = Literal[
    "country",
    "continent",
    "subregion",
    "continental_union",
    "organisation",
    "aggregate",
    "place_other",
    "group",
    "code",
    "other",
]

# The kinds a resolve admits when the caller names none: every place and organisation kind and
# `group`. `code` and `other` are left out.
DEFAULT_KINDS: frozenset[EntityKind] = frozenset(
    {
        "country",
        "continent",
        "subregion",
        "continental_union",
        "place_other",
        "aggregate",
        "organisation",
        "group",
    }
)

MatchedAs = Literal["name", "alias", "code"]

# `members` when the text names a group's or region's members ("EU countries"), else `entity`.
Reading = Literal["members", "entity"]


@dataclass(frozen=True, slots=True)
class Matched:
    """The listed text a result matched, and whether it is the entity's name, an alias, or a
    code. A query alias ("EU countries") matches as `alias` with the text as asked."""

    text: str
    matched_as: MatchedAs


@dataclass(frozen=True, slots=True)
class Entity:
    """A graph-confirmed entity. Resolvekit hits use the canonical name.
    Fallback hits use the graph name or input text.

    Display metadata. Rendered sentences use the query's wording."""

    dcid: str
    name: str
    kind: Kind
    graph_type: str


@dataclass(frozen=True, slots=True)
class Resolved:
    """One graph-confirmed entity, or a confirmed code. `matched` is the listed text the
    entity matched; `reading` is `members` when the text resolved through a query alias that
    declares it names the entity's members."""

    entity: Entity
    matched: Matched | None = None
    reading: Reading = "entity"


@dataclass(frozen=True, slots=True)
class Group:
    """A group the membership registry holds (`groups.GroupRegistry`); its kind is `group`.

    The registry confirms the group and its members, so no graph read does. `dcid` is the
    group's id (`regions.identity_to_dcid`), such as `GroupOfSeven` or `groups/UN.LDC`, whether
    or not the graph holds a node for it, and `label` the registry's label. `members` is the
    authority's member list under the `countries` projection (`membership.DEFAULT_PROJECTION`)
    today, ordered by dcid, typed `Country` and named by resolvekit. It is the resolver's
    answer: Search binds it, and the resolve door serves it on request. `entity` is the group's
    graph node, a confirmed `org.organisation` survivor, or `None`; Search chooses between
    `entity` and `members` by role admission. `matched` and `reading` follow `Resolved`."""

    members: tuple[Entity, ...]
    entity: Entity | None
    dcid: str | None = None
    label: str | None = None
    matched: Matched | None = None
    reading: Reading = "entity"

    @property
    def member_count(self) -> int:
        """The registry's count of members under the default projection."""
        return len(self.members)


@dataclass(frozen=True, slots=True)
class Ambiguous:
    """Multiple listed entities match the text.

    Resolvekit candidates sort by confidence then dcid. Fallback candidates sort by dcid."""

    candidates: tuple[Entity, ...]


@dataclass(frozen=True, slots=True)
class NotHeld:
    """resolvekit (or the region id map) named a real entity, but this
    deployment's graph holds no node for its dcid."""

    dcid: str
    name: str


@dataclass(frozen=True, slots=True)
class NotFound:
    """No resolution was found. `near_matches` lists up to 3 gate-rejected candidates, ordered
    by confidence and deduplicated by dcid. They are hints the graph has not confirmed, so their
    `graph_type` is empty and their kind comes from the candidate's resolvekit type
    (`candidate_kind`). It is empty when resolvekit returns no candidates or the text is empty.
    `removed` holds the candidates a `kinds` filter left out, with their kinds, in the order
    the resolver ranked them; a caller shows them as near matches. A hit the graph does not
    hold is among them with its resolvekit kind and an empty `graph_type`."""

    near_matches: tuple[Entity, ...] = ()
    removed: tuple[Entity, ...] = ()


Resolution = Resolved | Group | Ambiguous | NotHeld | NotFound


@runtime_checkable
class GraphLookup(Protocol):
    """Graph operations required by the resolver."""

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        """Return distinct property values in first-occurrence order across provenances.

        `deadline` is an absolute `time.monotonic()` value. Live reads raise
        `GraphDeadlineExceededError` after expiry and discard partial results. `None` leaves the
        read unbounded."""
        ...

    def node_names_many(
        self, dcids: Sequence[str], *, deadline: float | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Return names from one `/v2/node` read, with `name` before `alternateName`.

        Values are deduplicated in first-occurrence order. Each requested dcid maps to a tuple,
        including `()` for missing or unnamed nodes. Repeated dcids collapse to one key. Empty
        input returns `{}` without a network call. `deadline` follows `node_arcs_many`."""
        ...

    def resolve_by_arc(
        self, names: Sequence[str], *, arc: str, deadline: float | None = None
    ) -> dict[str, list[str]]:
        """Return distinct candidate dcids in first-occurrence order across provenances.

        `deadline` follows `node_arcs_many`."""
        ...


@dataclass(frozen=True, slots=True)
class CodeHit:
    """A code whose label or synonym lists the text. `entity.kind` is `code` and
    `entity.graph_type` the code system's graph id. `text` is the label or synonym that
    matched, as the code index writes it, so every casing of a query gives one hit."""

    entity: Entity
    text: str
    matched_as: MatchedAs


@runtime_checkable
class CodeLookup(Protocol):
    """The code index's text search, which the `code` kind reads."""

    def codes_by_text(self, text: str, *, code_system: str | None) -> Sequence[CodeHit]:
        """Every code whose label (`name`) or synonym (`alias`) lists `text` casefolded with
        its whitespace collapsed, narrowed to `code_system` (a graph id such as
        `CRSPurposeCodeEnum`) when set. The resolver passes the text its cache keys on, so
        the answer depends on nothing the cache key leaves out."""
        ...


class IndexStampLike(Protocol):
    """The part of a code index's stamp the resolver keys its cache on."""

    @property
    def content_hash(self) -> str:
        """The hash of the index's exported rows."""
        ...


class KindIndex(Protocol):
    """The code index's facts the kind rule reads (`kinds.classify`), and its stamp, which the
    resolver keys its cache on. `code_index.CodeIndex` serves them."""

    @property
    def stamp(self) -> IndexStampLike:
        """The index's export stamp."""
        ...

    def is_code_system(self, dcid: str) -> bool:
        """Whether `dcid` is a code system's graph id."""
        ...

    def code_by_dcid(self, dcid: str) -> object | None:
        """The code row for `dcid`, or `None`."""
        ...

    def party_kind_of(self, dcid: str, types: Iterable[str]) -> str | None:
        """The party kind of `dcid` whose graph types are `types`: its party row's kind, else
        the kind its types give, else `None`."""
        ...


ADMITTED_TYPES: frozenset[str] = frozenset(
    {
        "geo.country",
        "geo.subregion",
        "geo.continent",
        "geo.continental_union",
        "org.organisation",
        "org.dac_aggregate",
    }
)

_KIND_BY_ENTITY_TYPE: dict[str, EntityKind] = {
    "geo.country": "country",
    "geo.subregion": "subregion",
    "geo.continent": "continent",
    "geo.continental_union": "continental_union",
    "org.organisation": "organisation",
    "org.dac_aggregate": "aggregate",
    # Resolvekit's membership groups (G20, OPEC) are intergovernmental organisations. One the
    # group registry holds resolves as `group` before this table is read.
    "geo.organization": "organisation",
}


def candidate_kind(entity_type: str | None) -> EntityKind:
    """The kind of a resolvekit candidate the graph has not confirmed, by its pack type. Each
    of `ADMITTED_TYPES` and `geo.organization` has one; any other type, or none, maps to
    `other`."""
    return _KIND_BY_ENTITY_TYPE.get(entity_type or "", "other")


# The graph types with a finer kind than `place_other`, and the organisation and aggregate types.
_KIND_BY_GRAPH_TYPE: dict[str, EntityKind] = {
    "Country": "country",
    "Continent": "continent",
    "UNGeoRegion": "subregion",
    "ContinentalUnion": "continental_union",
    "Organization": "organisation",
    "DACRecipientUnallocatedByCountry": "aggregate",
    "DACUnspecifiedParty": "aggregate",
}


def _load_place_types() -> frozenset[str]:
    """`Place` and every class below it by `subClassOf` on the graph, from the packaged
    `data/place_types.json` (`scripts/export_graph_entities.py place-types`)."""
    path = importlib.resources.files("entity_resolution") / "data" / "place_types.json"
    return frozenset(json.loads(path.read_text(encoding="utf-8")))


PLACE_TYPES: frozenset[str] = _load_place_types()
"""The graph's place types. `Place` alone types the World (`Earth`); `EurostatNUTS2`,
`GeoRegion`, `Town` and the rest type the places a name reaches on the resolver's last rung."""

# A node typed both a country and something broader (India is `Country` and `Place`) takes the
# most specific kind its types give.
_KIND_PRECEDENCE: tuple[EntityKind, ...] = (
    "country",
    "continent",
    "subregion",
    "continental_union",
    "organisation",
    "aggregate",
    "place_other",
)


def _type_kind(graph_type: str) -> EntityKind:
    kind = _KIND_BY_GRAPH_TYPE.get(graph_type)
    if kind is not None:
        return kind
    return "place_other" if graph_type in PLACE_TYPES else "other"


def entity_kind(graph_types: Iterable[str]) -> EntityKind:
    """The place or organisation kind of a graph node whose `typeOf` values are
    `graph_types`, the last step of the kind rule (`kinds.classify`).

    Each type maps through `_KIND_BY_GRAPH_TYPE`, and any other of `PLACE_TYPES` to
    `place_other`. The node takes the first kind in `_KIND_PRECEDENCE` any of its types gives,
    else `other`."""
    found = {_type_kind(graph_type) for graph_type in graph_types}
    return next((kind for kind in _KIND_PRECEDENCE if kind in found), "other")
