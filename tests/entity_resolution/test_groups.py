"""The group registry against the bundled resolvekit membership it is built from."""

from __future__ import annotations

from datetime import date

import pytest

from entity_resolution.groups import GroupRegistry, build_group_registry
from entity_resolution.membership import (
    COUNTRY_PROJECTIONS,
    AsOfNotRecorded,
    NotRecorded,
    Projection,
    UnknownMembership,
)
from entity_resolution.regions import identity_to_dcid, replacing_group_id
from entity_resolution.shared import SharedResolver

_DATES = (date(1960, 1, 1), date(1995, 6, 1), date(2015, 6, 1), date(2021, 6, 1))


@pytest.fixture(scope="module")
def registry(_shared_resolvekit_pack: SharedResolver) -> GroupRegistry:
    shared = _shared_resolvekit_pack
    return shared.run(build_group_registry, shared.resolver).result()


def _members_of(shared: SharedResolver, label: str, as_of: date) -> set[str]:
    return set(shared.run(lambda: shared.resolver.members_of(label, as_of=as_of)).result())


def _listed(members: tuple[str, ...] | NotRecorded) -> tuple[str, ...]:
    assert isinstance(members, tuple), members
    return members


def test_registry_holds_every_known_group_under_its_dcid(
    registry: GroupRegistry, _shared_resolvekit_pack: SharedResolver
) -> None:
    shared = _shared_resolvekit_pack
    known = shared.run(shared.resolver.known_groups).result()
    assert sorted(group.label for group in registry) == sorted(known)
    for group in registry:
        assert group.ref == identity_to_dcid(group.entity_id)
        assert registry.by_entity_id(group.entity_id) is registry.get(group.ref)
    assert registry.get("EuropeanUnion").entity_id == "EuropeanUnion"


@pytest.mark.parametrize(
    ("entity_id", "ref"),
    [
        ("groups/G7", "GroupOfSeven"),
        ("groups/OECD", "undata-geo/G00406000"),
        ("EuropeanUnion", "EuropeanUnion"),
        ("groups/EU27_2020", "groups/EU27_2020"),
        ("groups/UN.SIDS", "groups/UN.SIDS"),
        ("groups/Commonwealth", "groups/Commonwealth"),
    ],
)
def test_a_group_takes_its_graph_node_id_where_the_node_holds_its_membership(
    registry: GroupRegistry, entity_id: str, ref: str
) -> None:
    assert registry.by_entity_id(entity_id).ref == ref
    if ref != entity_id:
        assert entity_id not in registry
        assert replacing_group_id(entity_id) == ref
    else:
        assert replacing_group_id(entity_id) is None


def test_dated_groups_answer_as_resolvekit_does_at_every_date(
    registry: GroupRegistry, _shared_resolvekit_pack: SharedResolver
) -> None:
    dated = [group for group in registry if group.dated]
    assert dated
    for group in dated:
        for as_of in _DATES:
            members = {
                member
                for projection in COUNTRY_PROJECTIONS
                for member in _listed(group.members(projection, as_of))
            }
            assert members == _members_of(_shared_resolvekit_pack, group.label, as_of), (
                group.ref,
                as_of,
            )


@pytest.mark.parametrize(
    ("ref", "as_of", "count"),
    [
        ("EuropeanUnion", date(2015, 6, 1), 28),
        ("EuropeanUnion", date(2021, 6, 1), 27),
        ("groups/UN", date(1960, 1, 1), 81),
    ],
)
def test_as_of_selects_the_dated_composition(
    registry: GroupRegistry, ref: str, as_of: date, count: int
) -> None:
    assert registry.get(ref).member_count(as_of=as_of) == count


def test_an_undated_group_refuses_as_of_and_has_no_effective_date(
    registry: GroupRegistry,
) -> None:
    undated = [group for group in registry if not group.dated]
    assert undated
    for group in undated:
        assert group.effective_date() is None
        with pytest.raises(AsOfNotRecorded) as caught:
            group.members(as_of=date(2015, 6, 1))
        assert caught.value.ref == group.ref


def test_a_dated_group_dates_its_composition(registry: GroupRegistry) -> None:
    eu = registry.get("EuropeanUnion")
    # GBR left on 2020-02-01 and no row changes after it.
    assert eu.effective_date() == date(2020, 2, 1)
    # Croatia joined on 2013-07-01, the last change before mid-2015.
    assert eu.effective_date(date(2015, 6, 1)) == date(2013, 7, 1)


def test_every_current_member_falls_in_exactly_one_country_projection(
    registry: GroupRegistry,
) -> None:
    today = date.today()
    for group in registry:
        current = {row.member for row in group.rows if row.active(today)}
        projected = [m for p in COUNTRY_PROJECTIONS for m in _listed(group.members(p))]
        assert sorted(projected) == sorted(current), group.ref


def test_a_territory_member_is_kept_under_its_own_projection(registry: GroupRegistry) -> None:
    au = registry.get("groups/AU")
    assert "country/ESH" in _listed(au.members(Projection.TERRITORIES))
    assert "country/ESH" not in _listed(au.members())


def test_institutional_members_are_not_recorded(registry: GroupRegistry) -> None:
    g20 = registry.get("groups/G20")
    assert g20.member_count() == 19
    assert g20.members(Projection.INSTITUTIONS) == NotRecorded(
        ref="groups/G20", projection=Projection.INSTITUTIONS
    )
    assert isinstance(g20.member_count(Projection.INSTITUTIONS), NotRecorded)


def test_an_unknown_group_is_refused(registry: GroupRegistry) -> None:
    with pytest.raises(UnknownMembership):
        registry.get("groups/NoSuchGroup")
    with pytest.raises(UnknownMembership):
        registry.by_entity_id("groups/NoSuchGroup")
