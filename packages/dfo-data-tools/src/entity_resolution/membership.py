"""Declared membership policy for groups and regions.

Every membership names its authority, whether that authority dates its rows, the entity classes
it admits, and the projections it records. A projection splits the members by class:

- `countries`, the default: countries that exist today and belong to the United Nations at the
  date asked. Resolvekit's curated, dated UN membership is the authority for this class.
- `other_states`: states outside the UN, from the declared list in `data/membership_policy.json`,
  which names the source of the class.
- `territories`: every other country that exists today, which leaves dependent territories.
- `former_states`: countries resolvekit records as dissolved (`valid_until` set).
- `institutions`: members that are bodies, not countries (G20's EU and African Union). No
  authority DFO holds records them, so every membership answers `NotRecorded` for it.

Groups take their rows from resolvekit's curated membership (`groups.py`). Regions take theirs
from resolvekit's UN M49 containment, which holds no dates, so a region refuses `as_of`. The
policy file composes a region from several resolvekit containers where M49 and the continent a
reader means differ (North America), and names the region projections resolvekit leaves
incomplete, whose answers carry a `membership_incomplete` gap. Raw graph containment answers a
different question (places with data) and stays with `materialise.place_scope`.

The builders read resolvekit's stores, which are thread-affine: run them on the thread that
opened the resolver they read. The objects they return are immutable snapshots that any thread
can read."""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING, Literal, get_args

import resolvekit
from resolvekit.core.model import EntityRecord

from entity_resolution.regions import identity_to_dcid

if TYPE_CHECKING:
    from resolvekit import Resolver

    from entity_resolution.groups import GroupRegistry

MembershipKind = Literal["group", "region"]
# The kind a ref the policy holds takes on every route (`MembershipPolicy.kind_of`).
PolicyKind = Literal["group", "continent", "subregion"]
# The resolvekit classes a region's authority gives it.
RegionClass = Literal["geo.continent", "geo.subregion"]

# The resolvekit group whose dated rows decide which countries are sovereign states.
SOVEREIGNTY_GROUP = "groups/UN"
COUNTRY_CLASS = "geo.country"
CONTINENT_CLASS: RegionClass = "geo.continent"
REGION_CLASSES: tuple[RegionClass, ...] = get_args(RegionClass)
GEO_DOMAIN = "geo"
MEMBERSHIP_INCOMPLETE = "membership_incomplete"
_POLICY_FILE = "membership_policy.json"


class Projection(StrEnum):
    """A named cut of a membership by member class."""

    COUNTRIES = "countries"
    OTHER_STATES = "other_states"
    TERRITORIES = "territories"
    FORMER_STATES = "former_states"
    INSTITUTIONS = "institutions"


DEFAULT_PROJECTION = Projection.COUNTRIES
COUNTRY_PROJECTIONS = frozenset(
    {
        Projection.COUNTRIES,
        Projection.OTHER_STATES,
        Projection.TERRITORIES,
        Projection.FORMER_STATES,
    }
)


class UnknownMembership(LookupError):
    """No group or region in the policy has this id."""

    def __init__(self, ref: str) -> None:
        super().__init__(f"no membership for {ref!r}")
        self.ref = ref


class AsOfNotRecorded(ValueError):
    """`as_of` was asked of a membership whose authority records no dates."""

    def __init__(self, ref: str) -> None:
        super().__init__(f"{ref!r} has an undated membership; as_of cannot select from it")
        self.ref = ref


@dataclass(frozen=True, slots=True)
class NotRecorded:
    """The authority does not record this projection, so it has no member list to give."""

    ref: str
    projection: Projection


@dataclass(frozen=True, slots=True)
class Authority:
    """Where a membership's rows come from, with the data version that reproduces them."""

    name: str
    version: str


@dataclass(frozen=True, slots=True)
class Gap:
    """A projection the authority is known to hold only in part. `code` is machine-readable."""

    code: str
    projection: Projection
    authority: Authority


@dataclass(frozen=True, slots=True)
class PolicyData:
    """The declared policy file: the root, region compositions, other states, and incomplete
    projections.

    `root` is the resolvekit id of the World, which every region sits under and which is no
    region itself. `compositions` maps a region's resolvekit id to the containers whose
    countries it holds."""

    root: str
    compositions: Mapping[str, tuple[str, ...]]
    other_states: frozenset[str]
    incomplete_region_projections: frozenset[Projection]


@cache
def policy_data() -> PolicyData:
    """The packaged `data/membership_policy.json`, read once."""
    path = importlib.resources.files("entity_resolution") / "data" / _POLICY_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PolicyData(
        root=raw["root"]["region"],
        compositions={
            row["region"]: tuple(row["members_from"]) for row in raw["region_compositions"]
        },
        other_states=frozenset(raw["other_states"]["members"]),
        incomplete_region_projections=frozenset(
            Projection(name) for name in raw["region_gaps"]["projections"]
        ),
    )


@dataclass(frozen=True, slots=True)
class Row:
    """One member's row. Dates are half-open, `[valid_from, valid_until)`; `None` is unbounded."""

    member: str
    member_class: str
    valid_from: date | None = None
    valid_until: date | None = None

    def active(self, day: date) -> bool:
        """Whether the row holds on `day`."""
        return (self.valid_from is None or self.valid_from <= day) and (
            self.valid_until is None or day < self.valid_until
        )


@dataclass(frozen=True, slots=True)
class Sovereignty:
    """Classes a country member into exactly one of the four country projections.

    `states` holds the UN rows, `dissolved` resolvekit's dissolved countries, and `other_states`
    the declared states outside the UN."""

    states: Mapping[str, tuple[Row, ...]]
    dissolved: frozenset[str]
    other_states: frozenset[str]

    def classify(self, member: str, day: date) -> Projection:
        """The country projection `member` falls in on `day`."""
        if member in self.dissolved:
            return Projection.FORMER_STATES
        if any(row.active(day) for row in self.states.get(member, ())):
            return Projection.COUNTRIES
        if member in self.other_states:
            return Projection.OTHER_STATES
        return Projection.TERRITORIES


@dataclass(frozen=True, slots=True)
class Membership:
    """One group's or region's membership under its declared authority.

    `ref` is the dcid callers use; `entity_id` is the authority's own id for it. Counts follow
    the projection: the African Union's `member_count()` is 54 under `countries`, since Western
    Sahara falls under `territories`. `incomplete` names the projections whose answers carry a
    `membership_incomplete` gap. `member_labels` maps each member's dcid to the authority's
    canonical name for it. `region_class` is the class the authority gives a region, a
    continent or a UN M49 subregion, and `None` for a group.

    Raises:
        ValueError: a region names no class, or a group names one.
    """

    ref: str
    entity_id: str
    label: str
    kind: MembershipKind
    authority: Authority
    dated: bool
    rows: tuple[Row, ...]
    sovereignty: Sovereignty
    incomplete: frozenset[Projection] = field(default_factory=frozenset)
    member_labels: Mapping[str, str] = field(default_factory=dict)
    region_class: RegionClass | None = None

    def __post_init__(self) -> None:
        if (self.kind == "region") != (self.region_class is not None):
            raise ValueError(f"{self.ref!r}: a region names its class and a group none")

    @property
    def eligible_classes(self) -> frozenset[str]:
        """The entity classes the authority admits as members."""
        return frozenset(row.member_class for row in self.rows)

    @property
    def projections(self) -> frozenset[Projection]:
        """The projections the authority records."""
        recorded = set(COUNTRY_PROJECTIONS) if COUNTRY_CLASS in self.eligible_classes else set()
        if self.eligible_classes - {COUNTRY_CLASS}:
            recorded.add(Projection.INSTITUTIONS)
        return frozenset(recorded)

    def members(
        self, projection: Projection = DEFAULT_PROJECTION, as_of: date | None = None
    ) -> tuple[str, ...] | NotRecorded:
        """Member dcids under `projection` at `as_of` (today when `None`), sorted.

        Raises:
            AsOfNotRecorded: `as_of` is set and the authority records no dates.
        """
        day = self._day(as_of)
        if projection not in self.projections:
            return NotRecorded(ref=self.ref, projection=projection)
        active = {row.member: row.member_class for row in self.rows if row.active(day)}
        if projection is Projection.INSTITUTIONS:
            return tuple(sorted(m for m, cls in active.items() if cls != COUNTRY_CLASS))
        return tuple(
            sorted(
                m
                for m, cls in active.items()
                if cls == COUNTRY_CLASS and self.sovereignty.classify(m, day) is projection
            )
        )

    def gap(self, projection: Projection) -> Gap | None:
        """The gap an answer under `projection` carries, or `None` when the authority is whole."""
        if projection not in self.incomplete:
            return None
        return Gap(code=MEMBERSHIP_INCOMPLETE, projection=projection, authority=self.authority)

    def member_count(
        self, projection: Projection = DEFAULT_PROJECTION, as_of: date | None = None
    ) -> int | NotRecorded:
        """The authority's count of members under `projection` at `as_of`."""
        members = self.members(projection, as_of)
        return members if isinstance(members, NotRecorded) else len(members)

    def effective_date(self, as_of: date | None = None) -> date | None:
        """The latest membership change on or before `as_of` (today when `None`).

        `None` for an undated membership, whose composition carries no date."""
        day = self._day(as_of)
        if not self.dated:
            return None
        bounds = (
            bound
            for row in self.rows
            for bound in (row.valid_from, row.valid_until)
            if bound is not None and bound <= day
        )
        return max(bounds, default=None)

    def _day(self, as_of: date | None) -> date:
        if as_of is None:
            return date.today()
        if not self.dated:
            raise AsOfNotRecorded(self.ref)
        return as_of


@dataclass(frozen=True, slots=True)
class Evidence:
    """What a dataset records about the membership a total used, so it can be reproduced."""

    ref: str
    kind: MembershipKind
    projection: Projection
    as_of: date | None
    authority: Authority


@dataclass(frozen=True, slots=True)
class MembershipAnswer:
    """Members under one projection, with the membership's effective date and the evidence.

    `gap` is set when the authority holds this projection only in part."""

    members: tuple[str, ...] | NotRecorded
    effective_date: date | None
    evidence: Evidence
    gap: Gap | None = None


class MembershipPolicy:
    """Every group and region membership, keyed by dcid, and the regions it leaves out by rule.

    `omitted_regions` holds the continents and subregions `build_regions` leaves out because
    their containment holds no country (Antarctica). A lookup refuses them as it refuses an id
    the policy never heard of; a reader that must tell the two apart checks `holds_region`."""

    def __init__(
        self, memberships: Iterable[Membership], *, omitted_regions: Iterable[str] = ()
    ) -> None:
        self.omitted_regions: frozenset[str] = frozenset(omitted_regions)
        self._by_ref: dict[str, Membership] = {}
        for membership in memberships:
            if membership.ref in self._by_ref:
                raise ValueError(f"two memberships share the id {membership.ref!r}")
            self._by_ref[membership.ref] = membership

    def get(self, ref: str) -> Membership:
        """The membership for `ref`.

        Raises:
            UnknownMembership: no group or region has this id.
        """
        try:
            return self._by_ref[ref]
        except KeyError:
            raise UnknownMembership(ref) from None

    def __contains__(self, ref: object) -> bool:
        """Whether the policy holds `ref` as a group or region. The deployment holds such a ref
        with or without a graph node: resolve, the node route and the members route all read
        this rule."""
        return ref in self._by_ref

    def kind_of(self, ref: str) -> PolicyKind | None:
        """The kind of `ref` when the policy holds it: `group` for a group, and for a region
        `continent` when its authority classes it a continent, else `subregion`. The graph's
        types for the node play no part. `None` for any other ref, an omitted region included.
        The resolver, on every path, and the Graph API's node route both class by this rule,
        so they give a held ref one kind."""
        membership = self._by_ref.get(ref)
        if membership is None:
            return None
        if membership.kind == "group":
            return "group"
        return "continent" if membership.region_class == CONTINENT_CLASS else "subregion"

    def is_group(self, ref: str) -> bool:
        """Whether the policy holds `ref` as a group."""
        membership = self._by_ref.get(ref)
        return membership is not None and membership.kind == "group"

    def holds_region(self, ref: str) -> bool:
        """Whether `ref` is a continent or subregion by the policy: one it holds, which answers
        its countries, or one it omits (`omitted_regions`), which answers no members. Search
        keys every region path on this, never on the node's graph types. A node the policy
        neither holds nor omits is no region, whatever its type."""
        if ref in self.omitted_regions:
            return True
        membership = self._by_ref.get(ref)
        return membership is not None and membership.kind == "region"

    def __iter__(self) -> Iterator[Membership]:
        return iter(self._by_ref.values())

    def members(
        self,
        ref: str,
        *,
        projection: Projection = DEFAULT_PROJECTION,
        as_of: date | None = None,
    ) -> MembershipAnswer:
        """Members of a group or region under `projection` at `as_of`, with evidence.

        Raises:
            UnknownMembership: no group or region has this id.
            AsOfNotRecorded: `as_of` is set and the authority records no dates.
        """
        membership = self.get(ref)
        return MembershipAnswer(
            members=membership.members(projection, as_of),
            effective_date=membership.effective_date(as_of),
            evidence=Evidence(
                ref=ref,
                kind=membership.kind,
                projection=projection,
                as_of=as_of,
                authority=membership.authority,
            ),
            gap=membership.gap(projection),
        )


def authority(resolver: Resolver, name: str) -> Authority:
    """An authority read from the resolver's bundled data."""
    return Authority(
        name=name,
        version=f"resolvekit {resolvekit.__version__}, data {resolver.data_version_summary()}",
    )


def to_dcid(entity_id: str) -> str:
    """The dcid for a resolvekit group or member id."""
    dcid = identity_to_dcid(entity_id)
    if dcid is None:
        raise ValueError(f"resolvekit id {entity_id!r} maps to no dcid")
    return dcid


def parse_date(value: str | None) -> date | None:
    """A resolvekit relation bound as a date."""
    return None if value is None else date.fromisoformat(value)


def build_regions(
    resolver: Resolver, *, sovereignty: Sovereignty
) -> tuple[list[Membership], frozenset[str]]:
    """One membership per continent and UN M49 subregion that maps to a dcid and holds a country,
    and the dcids of the regions left out.

    The World (the policy file's `root`) is no region and is skipped outright. A region the
    policy file composes holds the countries of each container it names; any other region holds
    its own containment. A region whose containment holds no country (Antarctica) is left out,
    so a lookup refuses it instead of answering an empty list, and its dcid is returned as
    omitted (`MembershipPolicy.omitted_regions`).

    Raises:
        ValueError: the policy file composes, or declares as root, a region resolvekit does
            not hold.
    """
    declared = policy_data()
    region_authority = authority(resolver, "resolvekit UN M49 containment")
    store = resolver.store_for(GEO_DOMAIN)
    regions = [
        (region, region_class)
        for region_class in REGION_CLASSES
        for region in store.list_entities_by_type(region_class)
    ]
    unknown = {*declared.compositions, declared.root} - {region.entity_id for region, _ in regions}
    if unknown:
        raise ValueError(f"membership policy names unknown regions: {sorted(unknown)}")
    memberships: list[Membership] = []
    omitted: set[str] = set()
    for region, region_class in regions:
        ref = identity_to_dcid(region.entity_id)
        if ref is None or region.entity_id == declared.root:
            continue
        containers = declared.compositions.get(region.entity_id, (region.entity_id,))
        labels = {
            to_dcid(e.entity_id): e.canonical_name
            for c in containers
            for e in _countries_within(resolver, c)
        }
        if not labels:
            omitted.add(ref)
            continue
        memberships.append(
            Membership(
                ref=ref,
                entity_id=region.entity_id,
                label=region.canonical_name,
                kind="region",
                authority=region_authority,
                dated=False,
                rows=tuple(Row(member=m, member_class=COUNTRY_CLASS) for m in sorted(labels)),
                sovereignty=sovereignty,
                incomplete=declared.incomplete_region_projections,
                member_labels=labels,
                region_class=region_class,
            )
        )
    return memberships, frozenset(omitted)


def build_policy(resolver: Resolver, *, groups: GroupRegistry) -> MembershipPolicy:
    """The policy over `groups` and resolvekit's regions, sharing the groups' sovereignty."""
    regions, omitted = build_regions(resolver, sovereignty=groups.sovereignty)
    return MembershipPolicy([*groups, *regions], omitted_regions=omitted)


def _countries_within(resolver: Resolver, container: str) -> list[EntityRecord]:
    found = resolver.within(container, entity_type=COUNTRY_CLASS)
    return [entity for entity in found if isinstance(entity, EntityRecord)]
