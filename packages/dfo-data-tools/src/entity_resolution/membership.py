"""Declared membership policy for groups and regions.

Every membership names its authority, whether that authority dates its rows, the entity classes
it admits, and the projections it records. A projection splits the members by class:

- `countries`, the default: countries that exist today and belong to the United Nations. For the
  UN group itself the UN membership is read at the date asked; for every other group a member
  is classed by its standing today, so the dated DAC lists Switzerland under `countries` in 1995
  though it joined the UN in 2002. Resolvekit's curated, dated UN membership is the authority
  for this class.
- `other_states`: states outside the UN, from the declared list in `data/membership_policy.json`,
  which names the source of the class.
- `territories`: every other country that exists today, which leaves dependent territories.
- `former_states`: countries resolvekit records as dissolved (`valid_until` set).
- `institutions`: members that are bodies, not countries (G20's EU and African Union). No
  authority DFO holds records them, so every membership answers `NotRecorded` for it.

Groups take their rows from resolvekit's curated membership (`groups.py`). The policy file's
`dated_group_rows` replaces the undated rows of a group resolvekit lists without dates (the OECD
DAC, 14 of the 55 groups being dated); such a group names the policy file as its authority, with
a hash of its own entry in its version. Regions take theirs
from resolvekit's UN M49 containment, which holds no dates, so a region refuses `as_of`. The
policy file composes a region from several resolvekit containers where M49 and the continent a
reader means differ (North America), and names the region projections resolvekit leaves
incomplete, whose answers carry a `membership_incomplete` gap. Raw graph containment answers a
different question (places with data) and stays with `materialise.place_scope`.

The builders read resolvekit's stores, which are thread-affine: run them on the thread that
opened the resolver they read. The objects they return are immutable snapshots that any thread
can read."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from functools import cache
from itertools import pairwise
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


class PolicyFileError(ValueError):
    """The policy file declares dated rows that cannot hold."""


class DuplicateDatedGroup(PolicyFileError):
    """The policy file dates one group in two entries."""


class EmptySpan(PolicyFileError):
    """A dated span starts at or after its end."""


class OverlappingSpans(PolicyFileError):
    """One member's dated spans for a group overlap."""


class DatedGroupMismatch(RuntimeError):
    """The policy file's dated rows do not fit the group resolvekit holds."""


class DatedGroupNotHeld(DatedGroupMismatch):
    """The policy file dates a group resolvekit does not hold."""


class DatedGroupAlreadyDated(DatedGroupMismatch):
    """Resolvekit already dates the group the policy file dates."""


class DatedGroupMembersDiffer(DatedGroupMismatch):
    """The policy file dates different members than resolvekit lists for the group."""


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
    """The declared policy file: the root, region compositions, other states, incomplete
    projections, and dated rows for groups resolvekit holds undated.

    `root` is the resolvekit id of the World, which every region sits under and which is no
    region itself. `compositions` maps a region's resolvekit id to the containers whose
    countries it holds. `dated_group_rows` maps a group's resolvekit id to each member's
    `(valid_from, valid_until)` spans, replacing the group's undated rows; `digests` maps it to a
    hash of that group's entry alone."""

    root: str
    compositions: Mapping[str, tuple[str, ...]]
    other_states: frozenset[str]
    incomplete_region_projections: frozenset[Projection]
    dated_group_rows: Mapping[str, Mapping[str, tuple[tuple[date | None, date | None], ...]]]
    digests: Mapping[str, str]


@cache
def policy_data() -> PolicyData:
    """The packaged `data/membership_policy.json`, read once."""
    path = importlib.resources.files("entity_resolution") / "data" / _POLICY_FILE
    return parse_policy(path.read_text(encoding="utf-8"))


def parse_policy(text: str) -> PolicyData:
    """The policy file's `text` as declared data, with a `digests` entry per dated group.

    Raises:
        DuplicateDatedGroup: a group's dated rows are declared twice.
        EmptySpan: a span starts at or after its end.
        OverlappingSpans: a member's spans for one group overlap.
    """
    raw = json.loads(text)
    groups = [entry["group"] for entry in raw["dated_group_rows"]]
    if len(set(groups)) != len(groups):
        raise DuplicateDatedGroup(f"policy file dates a group twice: {sorted(groups)}")
    return PolicyData(
        root=raw["root"]["region"],
        compositions={
            row["region"]: tuple(row["members_from"]) for row in raw["region_compositions"]
        },
        other_states=frozenset(raw["other_states"]["members"]),
        incomplete_region_projections=frozenset(
            Projection(name) for name in raw["region_gaps"]["projections"]
        ),
        dated_group_rows={
            entry["group"]: _spans_by_member(entry["group"], entry["rows"])
            for entry in raw["dated_group_rows"]
        },
        digests={
            entry["group"]: hashlib.sha256(
                json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
            for entry in raw["dated_group_rows"]
        },
    )


def _spans_by_member(
    group: str, rows: list[dict[str, str | None]]
) -> dict[str, tuple[tuple[date | None, date | None], ...]]:
    spans: dict[str, list[tuple[date | None, date | None]]] = {}
    for row in rows:
        spans.setdefault(str(row["member"]), []).append(
            (parse_date(row["valid_from"]), parse_date(row["valid_until"]))
        )
    for member, held in spans.items():
        ordered = sorted(held, key=lambda span: span[0] or date.min)
        for start, until in ordered:
            if start is not None and until is not None and start >= until:
                raise EmptySpan(f"{group}: {member} has a span that starts at or after its end")
        for (_, until), (start, _) in pairwise(ordered):
            if until is None or (start or date.min) < until:
                raise OverlappingSpans(f"{group}: {member} has overlapping spans")
    return {member: tuple(held) for member, held in spans.items()}


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
        standing = day if self.entity_id == SOVEREIGNTY_GROUP else date.today()
        if projection is Projection.INSTITUTIONS:
            return tuple(sorted(m for m, cls in active.items() if cls != COUNTRY_CLASS))
        return tuple(
            sorted(
                m
                for m, cls in active.items()
                if cls == COUNTRY_CLASS and self.sovereignty.classify(m, standing) is projection
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


def policy_authority(base: Authority, *, digest: str) -> Authority:
    """`base`, the authority of a group's members, with the policy file that dates its rows
    named and `digest` of the group's dated entry in the version."""
    return Authority(
        name=f"{_POLICY_FILE} over {base.name}",
        version=f"{_POLICY_FILE} sha256 {digest}; {base.version}",
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
    """The policy over `groups` and resolvekit's regions, sharing the groups' sovereignty.

    A group whose id `group_ids.json` maps to a region the policy holds ("Africa (DAC)" to
    `africa`) is answered by that region, so the policy holds the region alone under the id.

    Raises:
        ValueError: such a group's members under the default projection differ from the
            region's.
    """
    regions, omitted = build_regions(resolver, sovereignty=groups.sovereignty)
    regions_by_ref = {region.ref: region for region in regions}
    kept: list[Membership] = []
    for group in groups:
        region = regions_by_ref.get(group.ref)
        if region is None:
            kept.append(group)
        elif group.members() != region.members():
            raise ValueError(
                f"group {group.entity_id!r} maps to region {group.ref!r} but their members differ"
            )
    return MembershipPolicy([*kept, *regions], omitted_regions=omitted)


def _countries_within(resolver: Resolver, container: str) -> list[EntityRecord]:
    found = resolver.within(container, entity_type=COUNTRY_CLASS)
    return [entity for entity in found if isinstance(entity, EntityRecord)]
