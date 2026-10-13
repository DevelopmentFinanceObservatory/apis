"""The membership policy: projections, dates, evidence, and resolvekit's regions."""

from __future__ import annotations

from datetime import date

import pytest

from entity_resolution.groups import GroupRegistry, build_group_registry
from entity_resolution.membership import (
    COUNTRY_PROJECTIONS,
    MEMBERSHIP_INCOMPLETE,
    AsOfNotRecorded,
    Authority,
    Evidence,
    Gap,
    Membership,
    MembershipPolicy,
    NotRecorded,
    Projection,
    Row,
    Sovereignty,
    UnknownMembership,
    build_policy,
)
from entity_resolution.shared import SharedResolver

_AUTHORITY = Authority(name="test authority", version="1")
_SOVEREIGNTY = Sovereignty(
    states={
        "country/AAA": (Row(member="country/AAA", member_class="geo.country"),),
        "country/LFT": (
            Row(
                member="country/LFT",
                member_class="geo.country",
                valid_until=date(2000, 1, 1),
            ),
        ),
    },
    dissolved=frozenset({"country/OLD"}),
    other_states=frozenset({"country/OTH", "country/LFT"}),
)
_CONTINENTS = ("africa", "asia", "europe", "northamerica", "southamerica", "oceania")


def _membership(
    *rows: Row,
    dated: bool = True,
    ref: str = "groups/T",
    incomplete: frozenset[Projection] = frozenset(),
) -> Membership:
    return Membership(
        ref=ref,
        entity_id=ref,
        label="Test group",
        kind="group",
        authority=_AUTHORITY,
        dated=dated,
        rows=rows,
        sovereignty=_SOVEREIGNTY,
        incomplete=incomplete,
    )


def _listed(members: tuple[str, ...] | NotRecorded) -> tuple[str, ...]:
    assert isinstance(members, tuple), members
    return members


@pytest.mark.parametrize(
    ("valid_from", "valid_until", "day", "active"),
    [
        (None, None, date(1900, 1, 1), True),
        (date(2000, 1, 1), None, date(1999, 12, 31), False),
        (date(2000, 1, 1), None, date(2000, 1, 1), True),
        (None, date(2020, 2, 1), date(2020, 1, 31), True),
        (None, date(2020, 2, 1), date(2020, 2, 1), False),
    ],
)
def test_a_row_holds_over_a_half_open_window(
    valid_from: date | None, valid_until: date | None, day: date, active: bool
) -> None:
    row = Row(
        member="country/AAA",
        member_class="geo.country",
        valid_from=valid_from,
        valid_until=valid_until,
    )
    assert row.active(day) is active


@pytest.mark.parametrize(
    ("member", "day", "projection"),
    [
        ("country/AAA", date(2020, 1, 1), Projection.COUNTRIES),
        ("country/LFT", date(1999, 1, 1), Projection.COUNTRIES),
        # UN membership outranks the declared list of states outside it.
        ("country/LFT", date(2000, 1, 1), Projection.OTHER_STATES),
        ("country/OTH", date(2020, 1, 1), Projection.OTHER_STATES),
        ("country/TER", date(2020, 1, 1), Projection.TERRITORIES),
        ("country/OLD", date(1980, 1, 1), Projection.FORMER_STATES),
    ],
)
def test_sovereignty_classes_a_country_on_a_day(
    member: str, day: date, projection: Projection
) -> None:
    assert _SOVEREIGNTY.classify(member, day) is projection


def test_projections_split_the_active_members_by_class() -> None:
    membership = _membership(
        Row(member="country/AAA", member_class="geo.country"),
        Row(member="country/TER", member_class="geo.country"),
        Row(member="country/OLD", member_class="geo.country", valid_from=date(1950, 1, 1)),
        Row(member="country/GON", member_class="geo.country", valid_until=date(1990, 1, 1)),
    )
    assert membership.members() == ("country/AAA",)
    assert membership.members(Projection.TERRITORIES) == ("country/TER",)
    assert membership.members(Projection.FORMER_STATES) == ("country/OLD",)
    assert membership.members(Projection.TERRITORIES, date(1980, 1, 1)) == (
        "country/GON",
        "country/TER",
    )
    assert membership.member_count(Projection.TERRITORIES) == 1


def test_institutions_are_recorded_only_when_the_authority_holds_non_country_members() -> None:
    countries_only = _membership(Row(member="country/AAA", member_class="geo.country"))
    assert countries_only.members(Projection.INSTITUTIONS) == NotRecorded(
        ref="groups/T", projection=Projection.INSTITUTIONS
    )
    with_body = _membership(
        Row(member="country/AAA", member_class="geo.country"),
        Row(member="EuropeanUnion", member_class="geo.continental_union"),
    )
    assert with_body.members(Projection.INSTITUTIONS) == ("EuropeanUnion",)
    assert with_body.members() == ("country/AAA",)


def test_an_undated_membership_refuses_as_of() -> None:
    membership = _membership(Row(member="country/AAA", member_class="geo.country"), dated=False)
    assert membership.effective_date() is None
    with pytest.raises(AsOfNotRecorded):
        membership.members(as_of=date(2000, 1, 1))
    with pytest.raises(AsOfNotRecorded):
        membership.effective_date(date(2000, 1, 1))


def test_the_effective_date_is_the_last_change_on_or_before_the_date() -> None:
    membership = _membership(
        Row(member="country/AAA", member_class="geo.country", valid_from=date(1958, 1, 1)),
        Row(
            member="country/TER",
            member_class="geo.country",
            valid_from=date(1973, 1, 1),
            valid_until=date(2020, 2, 1),
        ),
    )
    assert membership.effective_date(date(1960, 1, 1)) == date(1958, 1, 1)
    assert membership.effective_date(date(2019, 1, 1)) == date(1973, 1, 1)
    assert membership.effective_date() == date(2020, 2, 1)
    assert membership.effective_date(date(1900, 1, 1)) is None


def test_the_policy_answers_with_evidence() -> None:
    policy = MembershipPolicy([_membership(Row(member="country/AAA", member_class="geo.country"))])
    answer = policy.members("groups/T", as_of=date(2001, 1, 1))
    assert answer.members == ("country/AAA",)
    assert answer.evidence == Evidence(
        ref="groups/T",
        kind="group",
        projection=Projection.COUNTRIES,
        as_of=date(2001, 1, 1),
        authority=_AUTHORITY,
    )


def test_the_policy_refuses_an_unknown_id_and_a_shared_one() -> None:
    policy = MembershipPolicy([_membership()])
    with pytest.raises(UnknownMembership) as caught:
        policy.members("groups/Other")
    assert caught.value.ref == "groups/Other"
    with pytest.raises(ValueError, match="share"):
        MembershipPolicy([_membership(), _membership()])


@pytest.fixture(scope="module")
def policy(_shared_resolvekit_pack: SharedResolver) -> MembershipPolicy:
    shared = _shared_resolvekit_pack
    registry: GroupRegistry = shared.run(build_group_registry, shared.resolver).result()
    return shared.run(lambda: build_policy(shared.resolver, groups=registry)).result()


@pytest.mark.parametrize(
    ("ref", "inside", "outside"),
    [
        # UN M49 places the Republic of the Congo in Middle Africa; graph containment does not.
        ("MiddleAfrica", "country/COG", None),
        ("WesternAfrica", None, "country/COG"),
        ("NorthernEurope", "country/DNK", None),
        # M49 Central America holds Mexico; the graph's `CentralAmericaExceptMexico` does not.
        ("CentralAmerica", "country/MEX", None),
        # Transcontinental members sit in one M49 region only.
        ("europe", None, "country/TUR"),
        ("asia", "country/TUR", "country/EGY"),
        ("africa", "country/EGY", "country/ESH"),
    ],
)
def test_regions_follow_m49_containment_under_the_default_projection(
    policy: MembershipPolicy, ref: str, inside: str | None, outside: str | None
) -> None:
    members = _listed(policy.members(ref).members)
    if inside is not None:
        assert inside in members
    if outside is not None:
        assert outside not in members


def test_a_region_keeps_territories_and_former_states_under_their_own_projections(
    policy: MembershipPolicy,
) -> None:
    caribbean = policy.get("Caribbean")
    assert "country/PRI" in _listed(caribbean.members(Projection.TERRITORIES))
    assert "country/ANT" in _listed(caribbean.members(Projection.FORMER_STATES))
    for membership in (policy.get(ref) for ref in ("africa", "europe", "Caribbean", "asia")):
        projected = [m for p in COUNTRY_PROJECTIONS for m in _listed(membership.members(p))]
        assert sorted(projected) == sorted(row.member for row in membership.rows)


def test_a_region_refuses_as_of(policy: MembershipPolicy) -> None:
    with pytest.raises(AsOfNotRecorded):
        policy.members("africa", as_of=date(2000, 1, 1))
    assert policy.members("africa").evidence.kind == "region"


def test_a_region_whose_containment_holds_no_country_is_refused_and_omitted_by_rule(
    policy: MembershipPolicy,
) -> None:
    with pytest.raises(UnknownMembership):
        policy.members("antarctica")
    assert policy.omitted_regions == {"antarctica"}
    assert policy.holds_region("antarctica")


def test_the_world_is_the_root_and_no_region(policy: MembershipPolicy) -> None:
    """The World is declared the root, so the policy neither holds nor omits it."""
    assert "Earth" not in policy
    assert not policy.holds_region("Earth")


def test_an_incomplete_projection_answers_with_a_gap() -> None:
    row = Row(member="country/TER", member_class="geo.country")
    policy = MembershipPolicy(
        [_membership(row, dated=False, incomplete=frozenset({Projection.TERRITORIES}))]
    )
    answer = policy.members("groups/T", projection=Projection.TERRITORIES)
    assert answer.members == ("country/TER",)
    assert answer.gap == Gap(
        code=MEMBERSHIP_INCOMPLETE, projection=Projection.TERRITORIES, authority=_AUTHORITY
    )
    assert policy.members("groups/T").gap is None


@pytest.mark.parametrize(
    ("ref", "projection", "gap"),
    [
        ("Caribbean", Projection.TERRITORIES, True),
        ("europe", Projection.FORMER_STATES, True),
        ("africa", Projection.COUNTRIES, False),
        ("asia", Projection.OTHER_STATES, True),
        ("europe", Projection.OTHER_STATES, True),
        ("groups/AU", Projection.TERRITORIES, False),
        ("groups/UN", Projection.OTHER_STATES, False),
    ],
)
def test_region_territories_former_and_other_states_carry_the_incomplete_gap(
    policy: MembershipPolicy, ref: str, projection: Projection, gap: bool
) -> None:
    answer = policy.members(ref, projection=projection)
    assert (answer.gap is not None) is gap
    if answer.gap is not None:
        assert answer.gap.code == MEMBERSHIP_INCOMPLETE
        assert answer.gap.authority == answer.evidence.authority


def test_north_america_is_composed_from_three_m49_regions(policy: MembershipPolicy) -> None:
    members = _listed(policy.members("northamerica").members)
    assert len(members) == 23
    assert {"country/USA", "country/MEX", "country/TTO", "country/PAN"} <= set(members)


@pytest.mark.parametrize("member", ["country/TTO", "country/PAN"])
def test_a_country_sits_in_exactly_one_continent(policy: MembershipPolicy, member: str) -> None:
    holding = [ref for ref in _CONTINENTS if member in _listed(policy.members(ref).members)]
    assert len(holding) == 1, holding


def test_states_outside_the_un_have_their_own_projection(policy: MembershipPolicy) -> None:
    asia = policy.get("asia")
    assert {"country/PSE", "country/TWN"} <= set(_listed(asia.members(Projection.OTHER_STATES)))
    assert "country/PSE" not in _listed(asia.members(Projection.TERRITORIES))
    assert "country/HKG" in _listed(asia.members(Projection.TERRITORIES))


@pytest.mark.parametrize(
    ("ref", "member", "label"),
    [("groups/G20", "country/DEU", "Germany"), ("africa", "country/KEN", "Kenya")],
)
def test_a_membership_names_each_member_by_its_canonical_name(
    policy: MembershipPolicy, ref: str, member: str, label: str
) -> None:
    membership = policy.get(ref)
    assert membership.member_labels[member] == label
    assert set(_listed(membership.members())) <= set(membership.member_labels)
