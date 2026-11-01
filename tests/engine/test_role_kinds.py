"""The kinds each flow role takes, derived from the fixture code index's parties
(`search.role_kinds`)."""

from __future__ import annotations

import pytest

from code_index.query import CodeIndex
from code_index.shared import get_index
from code_index.types import IndexStamp, PartyRow
from entity_resolution import Entity
from search.engine.region.place_roles import PlaceAssignment, set_choices
from search.engine.region.place_set import PlaceNode, PlaceSet
from search.role_kinds import RoleKindRefused, check_role_kinds, entity_kind_of, role_kinds


def test_each_axis_takes_the_kinds_of_the_parties_listed_on_it():
    """Recipients take the continents and subregions DAC regional totals hold; providers,
    whose totals hold only groups, take neither."""
    kinds = role_kinds(get_index())

    assert kinds["origin"] == {"country", "organisation", "group", "aggregate"}
    assert kinds["destination"] == kinds["origin"] | {"continent", "subregion"}


@pytest.mark.parametrize(
    ("entities", "read", "refused"),
    [
        pytest.param(
            {"origin": ["country/FRA"], "destination": ["country/KEN", "africa", "DACAfrica"]},
            [],
            None,
            id="listed-and-held",
        ),
        pytest.param(
            {"origin": ["africa"]}, [], ("africa", "continent"), id="continent-as-provider"
        ),
        pytest.param(
            {"destination": ["country/KEN"], "recipient": ["country/XXX"]},
            ["country/XXX"],
            ("country/XXX", None),
            id="absent",
        ),
        pytest.param({"observationAbout": ["Count_Person"]}, [], None, id="subject-unchecked"),
    ],
)
def test_only_entities_no_party_row_or_membership_classes_are_read(
    entities: dict[str, list[str]], read: list[str], refused: tuple[str, str | None] | None
):
    """A party listed on the role's axis and a region the membership policy holds take
    their kind without a graph read."""
    reads: list[list[str]] = []

    def read_types(dcids: list[str]) -> dict[str, None]:
        reads.append(dcids)
        return dict.fromkeys(dcids)

    if refused is None:
        check_role_kinds(entities, read_types=read_types, code_index=get_index(), family=None)
    else:
        with pytest.raises(RoleKindRefused) as exc_info:
            check_role_kinds(entities, read_types=read_types, code_index=get_index(), family=None)
        assert (exc_info.value.dcid, exc_info.value.kind) == refused
    assert reads == ([read] if read else [])


def test_an_axis_the_index_lists_no_party_on_is_not_judged():
    """With no party on an axis the rule has no kinds to judge by, so it reads and refuses
    nothing there."""
    index = CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test",
        ),
        declarations=(),
        codes=(),
        statvars=(),
        party_types={"Country": "party"},
        parties=(
            PartyRow(dcid="country/FRA", kind="party", party_class="official", axes=("origin",)),
        ),
    )
    reads: list[list[str]] = []

    def read_types(dcids: list[str]) -> dict[str, tuple[str, ...]]:
        reads.append(dcids)
        return dict.fromkeys(dcids, ("Place",))

    assert role_kinds(index)["destination"] == frozenset()
    check_role_kinds(
        {"destination": ["country/GHA"]}, read_types=read_types, code_index=index, family=None
    )
    assert reads == []
    with pytest.raises(RoleKindRefused):
        check_role_kinds(
            {"origin": ["country/GHA"]}, read_types=read_types, code_index=index, family=None
        )


def test_both_doors_judge_a_node_by_the_kind_of_all_its_types():
    """A node typed `("Place", "Country")` that the index lists on no axis is a place other
    than a country by its first type alone and a country by all its types, the kind the
    resolver gives it (`Entity.kind`). The resubmit judges all its types, and a first answer
    the kind its place node carries from resolution (`PlaceNode.of_entity`), so both take it
    as a provider."""
    index = get_index()
    dcid, types = "country/ZZZ", ("Place", "Country")
    assert dcid not in index.parties
    assert entity_kind_of(dcid, types[:1], code_index=index) == "place_other"
    kind = entity_kind_of(dcid, types, code_index=index)
    assert kind == "country"
    node = PlaceNode.of_entity(Entity(dcid=dcid, name="Zedland", kind=kind, graph_types=types))
    assignment = PlaceAssignment(
        sets={"origin": (PlaceSet(name="Zedland", kind="place", node=node),)}
    )

    check_role_kinds(
        {"origin": [dcid]},
        read_types=lambda dcids: dict.fromkeys(dcids, types),
        code_index=index,
        family=None,
    )
    assert set_choices(assignment, admits=lambda _key, _type: True, index=index) == {dcid: "node"}
