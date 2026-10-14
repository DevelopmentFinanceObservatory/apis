"""The registry of groups resolvekit's curated membership knows, built once and shared.

Each group's rows are read from the geo store's `member_of` relations, dated where the curated
data dates them, so `members(projection, as_of)` answers any date without calling resolvekit
again. 13 of the 55 groups date their rows; the rest are undated lists and refuse `as_of`.
A group's id is its resolvekit entity id through `regions.identity_to_dcid`.

Build with `build_group_registry` on the thread that opened its resolver, since resolvekit's
stores are thread-affine. The registry it returns is immutable."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from entity_resolution.membership import (
    COUNTRY_CLASS,
    GEO_DOMAIN,
    SOVEREIGNTY_GROUP,
    Membership,
    Row,
    Sovereignty,
    UnknownMembership,
    authority,
    parse_date,
    policy_data,
    to_dcid,
)

if TYPE_CHECKING:
    from resolvekit import Resolver
    from resolvekit.core.model import EntityRecord
    from resolvekit.core.store import EntityStore

_MEMBER_OF = "member_of"


class GroupRegistry:
    """Group memberships keyed by dcid and by resolvekit entity id."""

    def __init__(self, groups: list[Membership], *, sovereignty: Sovereignty) -> None:
        self._by_ref = {group.ref: group for group in groups}
        self._by_entity_id = {group.entity_id: group for group in groups}
        if len(self._by_ref) != len(groups):
            raise ValueError("two groups share one dcid")
        self.sovereignty = sovereignty

    def __iter__(self) -> Iterator[Membership]:
        return iter(self._by_ref.values())

    def __len__(self) -> int:
        return len(self._by_ref)

    def __contains__(self, ref: object) -> bool:
        return ref in self._by_ref

    def get(self, ref: str) -> Membership:
        """The group with dcid `ref`.

        Raises:
            UnknownMembership: no group has this dcid.
        """
        try:
            return self._by_ref[ref]
        except KeyError:
            raise UnknownMembership(ref) from None

    def by_entity_id(self, entity_id: str) -> Membership:
        """The group with resolvekit entity id `entity_id` (a resolve candidate's id).

        Raises:
            UnknownMembership: no group has this entity id.
        """
        try:
            return self._by_entity_id[entity_id]
        except KeyError:
            raise UnknownMembership(entity_id) from None


def build_group_registry(resolver: Resolver) -> GroupRegistry:
    """Read every group `resolver.known_groups()` names, with all its rows.

    Each name resolves the way `members_of` resolves it. A name that does not resolve to one
    entity raises, so the registry never drops a group silently."""
    store = resolver.store_for(GEO_DOMAIN)
    group_authority = authority(resolver, "resolvekit curated membership")
    entities: list[EntityRecord] = []
    for name in resolver.known_groups():
        entity_id = resolver.resolve_id(name)
        group = None if entity_id is None else store.get_entity(entity_id)
        if group is None:
            raise RuntimeError(f"resolvekit group {name!r} does not resolve to a stored entity")
        entities.append(group)

    rows_by_group: dict[str, tuple[Row, ...]] = {}
    labels_by_group: dict[str, dict[str, str]] = {}
    for group in entities:
        rows_by_group[group.entity_id], labels_by_group[group.entity_id] = _rows(
            store, group.entity_id
        )
    sovereign_rows = rows_by_group.get(SOVEREIGNTY_GROUP)
    if not sovereign_rows:
        raise RuntimeError(f"resolvekit holds no {SOVEREIGNTY_GROUP} rows to class states by")
    countries = store.list_entities_by_type(COUNTRY_CLASS)
    sovereignty = Sovereignty(
        states=_by_member(sovereign_rows),
        dissolved=frozenset(
            to_dcid(entity.entity_id) for entity in countries if entity.valid_until is not None
        ),
        other_states=policy_data().other_states,
    )
    groups = [
        Membership(
            ref=to_dcid(group.entity_id),
            entity_id=group.entity_id,
            label=group.canonical_name,
            kind="group",
            authority=group_authority,
            dated=any(
                row.valid_from is not None or row.valid_until is not None
                for row in rows_by_group[group.entity_id]
            ),
            rows=rows_by_group[group.entity_id],
            sovereignty=sovereignty,
            member_labels=labels_by_group[group.entity_id],
        )
        for group in entities
    ]
    return GroupRegistry(groups, sovereignty=sovereignty)


def _rows(store: EntityStore, group_id: str) -> tuple[tuple[Row, ...], dict[str, str]]:
    """Every `member_of` row that names `group_id`, whatever its dates, and each member's
    canonical name by dcid."""
    member_ids = sorted(set(store.get_reverse_relations(group_id, _MEMBER_OF)))
    members = store.bulk_get_entities(member_ids)
    labels = {to_dcid(member_id): members[member_id].canonical_name for member_id in member_ids}
    rows: list[Row] = []
    for member_id in member_ids:
        member = members[member_id]
        rows.extend(
            Row(
                member=to_dcid(member_id),
                member_class=member.entity_type,
                valid_from=parse_date(relation.valid_from),
                valid_until=parse_date(relation.valid_until),
            )
            for relation in member.relations
            if relation.relation_type == _MEMBER_OF and relation.target_id == group_id
        )
    return tuple(rows), labels


def _by_member(rows: tuple[Row, ...]) -> dict[str, tuple[Row, ...]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.member, []).append(row)
    return {member: tuple(member_rows) for member, member_rows in grouped.items()}
