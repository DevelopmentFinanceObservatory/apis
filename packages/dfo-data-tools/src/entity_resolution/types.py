"""The public contract: what a name resolves to, and the graph protocol
the component reads through. Free of any `resolvekit` or `httpx` import,
so a consumer can type against this module without pulling in either.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

EntityKind = Literal[
    "country",
    "continent",
    "subregion",
    "continental_union",
    "organisation",
    "aggregate",
    "place_other",
    "other",
]


@dataclass(frozen=True, slots=True)
class Entity:
    """A graph-confirmed entity. Resolvekit hits use the canonical name.
    Fallback hits use the graph name or input text.

    Display metadata. Rendered sentences use the query's wording."""

    dcid: str
    name: str
    kind: EntityKind
    graph_type: str


@dataclass(frozen=True, slots=True)
class Resolved:
    """One graph-confirmed entity. `names_members` is true when the text resolved through a
    query alias that declares it names the entity's members ("EU countries")."""

    entity: Entity
    names_members: bool = False


@dataclass(frozen=True, slots=True)
class Group:
    """Group result with graph-confirmed members and an optional entity.

    `members` are ordered by dcid. An empty held-member set returns `NotHeld`, even if `entity`
    exists. `entity` is the group's node, a confirmed `org.organisation` survivor, or `None`.
    Callers choose between `entity` and `members` based on role admission. Display text uses the
    query wording, so this type has no separate `name` field. `names_members` is true when the
    text resolved through a query alias that declares it names the group's members
    ("EU countries"). `dcid` is the group's own id as a dcid (`regions.identity_to_dcid`),
    whether or not the graph holds a node for it (`groups/G7`)."""

    members: tuple[Entity, ...]
    entity: Entity | None
    names_members: bool = False
    dcid: str | None = None


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
    """No resolution was found. `near_matches` lists up to 3 gate-rejected canonical names,
    ordered by confidence and deduplicated. It is empty when resolvekit returns no candidates or
    the text is empty."""

    near_matches: tuple[str, ...] = ()


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

_KIND_BY_GRAPH_TYPE: dict[str, EntityKind] = {
    "Country": "country",
    "Continent": "continent",
    "UNGeoRegion": "subregion",
    "ContinentalUnion": "continental_union",
    "Organization": "organisation",
    "DACRecipientUnallocatedByCountry": "aggregate",
    "DACUnspecifiedParty": "aggregate",
    "State": "place_other",
    "City": "place_other",
    "AdministrativeArea": "place_other",
    "AdministrativeArea1": "place_other",
    "AdministrativeArea2": "place_other",
    "AdministrativeArea3": "place_other",
    "AdministrativeArea4": "place_other",
    "AdministrativeArea5": "place_other",
    "CensusZipCodeTabulationArea": "place_other",
}


def entity_kind(graph_type: str) -> EntityKind:
    """Map the first confirmed `typeOf` value through `_KIND_BY_GRAPH_TYPE`.

    Unlisted `AdministrativeArea*` types map to `place_other`. Other unlisted types map to
    `other`."""
    kind = _KIND_BY_GRAPH_TYPE.get(graph_type)
    if kind is not None:
        return kind
    if graph_type.startswith("AdministrativeArea"):
        return "place_other"
    return "other"
