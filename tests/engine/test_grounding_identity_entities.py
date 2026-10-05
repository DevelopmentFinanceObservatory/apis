"""Unit tests for grounding helpers that build identity entities, reject reversed windows,
and describe defaulted role scopes and their bound opposite-axis entities.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from search.engine.errors import EngineInputError
from search.engine.extract import DateRequest
from search.engine.region.candidate_grounding import scopes_on_axis
from search.engine.region.grounding import (
    bound_opposite_role_dcids,
    build_identity_entities,
    memoised_labels,
    reject_reversed_window,
    requested_window_from_date_request,
    restate_role_scope_warnings,
    role_scope_defaulted_message,
    role_scope_defaulted_warnings,
)
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.role_scope import DiscoveredScope
from search.models import (
    CoverageBare,
    EntityRoleKey,
    EntityRoleSubject,
    GraphRef,
    TimeWindow,
)
from search.models import Warning as ResponseWarning


def _entity_data(dcid: str) -> tuple[EntityRoleKey, GraphRef, GraphRef | None]:
    """A grounded (role, entity_ref, entity_type_ref) triple with a
    presentation "observationAbout" role, mirroring what _ground_answer builds."""
    return "observationAbout", GraphRef(dcid=dcid, label=dcid), None


def test_build_identity_entities_mirrors_presented_entities_as_subjects():
    """Each grounded triple becomes a subject Entity without another graph call."""
    entity_data = [_entity_data("country/USA"), _entity_data("country/ETH")]
    result = build_identity_entities(entity_data, role_refs={})
    by_dcid = {e.ref.dcid: e for e in result}
    assert isinstance(by_dcid["country/USA"].role, EntityRoleSubject)
    assert isinstance(by_dcid["country/ETH"].role, EntityRoleSubject)


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(None, id="none"),
        pytest.param("latest", id="latest"),
        pytest.param(TimeWindow(start_year=2015, end_year=2020), id="closed_range"),
        pytest.param(TimeWindow(start_year=2015), id="start_only"),
        pytest.param(TimeWindow(end_year=2020), id="end_only"),
    ],
)
def test_reject_reversed_window_accepts_non_reversed_inputs(window):
    reject_reversed_window(window)


def test_reject_reversed_window_raises_for_reversed_range():
    with pytest.raises(EngineInputError) as exc_info:
        reject_reversed_window(TimeWindow(start_year=2020, end_year=2015))
    assert exc_info.value.code == "request_invalid"


def test_requested_window_from_date_request_raises_for_reversed_window():
    """A reversed extracted window raises EngineInputError before spec ID construction."""
    date_request = DateRequest(window=TimeWindow(start_year=2020, end_year=2015), latest=False)
    with pytest.raises(EngineInputError) as exc_info:
        requested_window_from_date_request(date_request)
    assert exc_info.value.code == "request_invalid"


def test_requested_window_from_date_request_still_passes_through_well_formed():
    date_request = DateRequest(window=TimeWindow(start_year=2015, end_year=2020), latest=False)
    assert requested_window_from_date_request(date_request) == TimeWindow(
        start_year=2015, end_year=2020
    )


def _discovered_scope(axis: EntityRoleKey, member_count: int) -> DiscoveredScope:
    return DiscoveredScope(
        axis=axis,
        role_name="debtor" if axis == "destination" else "creditor",
        member_dcids=[f"country/{i}" for i in range(member_count)],
        filters={},
    )


class _RecordingLabels:
    """A label_of that answers from a fixed labels list and records every
    dcid list it is asked to label."""

    def __init__(self, labels: Sequence[str] = ()) -> None:
        self._labels = list(labels)
        self.calls: list[list[str]] = []

    def __call__(self, dcids: Sequence[str]) -> list[str]:
        self.calls.append(list(dcids))
        return self._labels[: len(dcids)]


def _dcids(count: int) -> list[str]:
    return [f"country/B{i}" for i in range(count)]


def _unqualified(discovered: DiscoveredScope) -> str:
    """The message for a query with no bound opposite entity."""
    return role_scope_defaulted_message(
        discovered,
        member_count=len(discovered.member_dcids),
        bound_dcids=[],
        label_of=_RecordingLabels(),
    )


def test_role_scope_defaulted_message_names_the_singular_member_count():
    message = _unqualified(_discovered_scope("destination", 1))
    assert (
        message
        == "no recipient was named; the answer was resolved across the 1 recipient with data."
    )


def test_role_scope_defaulted_message_names_the_plural_member_count():
    message = _unqualified(_discovered_scope("destination", 3))
    assert (
        message
        == "no recipient was named; the answer was resolved across the 3 recipients with data."
    )


def test_role_scope_defaulted_message_distinguishes_small_and_large_scopes():
    """A 3-member scope and a 200-member scope produce distinct counts."""
    small = _unqualified(_discovered_scope("destination", 3))
    large = _unqualified(_discovered_scope("destination", 200))
    assert small != large
    assert "3 recipients" in small
    assert "200 recipients" in large


def test_role_scope_defaulted_message_uses_the_origin_axis_noun_and_count():
    message = _unqualified(_discovered_scope("origin", 2))
    assert (
        message
        == "no provider was named; the answer was resolved across the 2 providers with data."
    )


_DESTINATION_BASE = (
    "no recipient was named; the answer was resolved across the 2 recipients with data"
)
_ORIGIN_BASE = "no provider was named; the answer was resolved across the 2 providers with data"


@pytest.mark.parametrize(
    ("labels", "named"),
    [
        (["France"], "from France"),
        (["Canada", "Germany"], "from Canada or Germany"),
        (["Canada", "Germany", "Japan"], "from Canada, Germany, or Japan"),
        (["Canada", "Germany", "Japan", "France"], "from any of the 4 providers"),
        ([f"Origin {i}" for i in range(7)], "from any of the 7 providers"),
    ],
)
def test_role_scope_defaulted_message_names_every_bound_origin(labels: list[str], named: str):
    """A discovered destination scope names bound origins, with counts above three."""
    message = role_scope_defaulted_message(
        _discovered_scope("destination", 2),
        member_count=2,
        bound_dcids=_dcids(len(labels)),
        label_of=_RecordingLabels(labels),
    )
    assert message == f"{_DESTINATION_BASE} {named}."


@pytest.mark.parametrize(
    ("labels", "named"),
    [
        (["Kenya"], "to Kenya"),
        (["Kenya", "Norway"], "to Kenya or Norway"),
        (["Kenya", "Norway", "Peru"], "to Kenya, Norway, or Peru"),
        (["Kenya", "Norway", "Peru", "Chad"], "to any of the 4 recipients"),
        ([f"Recipient {i}" for i in range(9)], "to any of the 9 recipients"),
    ],
)
def test_role_scope_defaulted_message_names_every_bound_destination(labels: list[str], named: str):
    """A discovered origin scope names bound destinations."""
    message = role_scope_defaulted_message(
        _discovered_scope("origin", 2),
        member_count=2,
        bound_dcids=_dcids(len(labels)),
        label_of=_RecordingLabels(labels),
    )
    assert message == f"{_ORIGIN_BASE} {named}."


def test_role_scope_defaulted_message_keeps_the_single_label_text():
    """The one-label sentence is the text corpus/standard.yaml quotes."""
    message = role_scope_defaulted_message(
        _discovered_scope("destination", 2),
        member_count=2,
        bound_dcids=["country/FRA"],
        label_of=_RecordingLabels(["France"]),
    )
    assert message == (
        "no recipient was named; the answer was resolved across the 2 recipients "
        "with data from France."
    )


def test_role_scope_defaulted_message_looks_up_no_label_without_a_bound_dcid():
    label_of = _RecordingLabels()
    role_scope_defaulted_message(
        _discovered_scope("destination", 2), member_count=2, bound_dcids=[], label_of=label_of
    )
    assert label_of.calls == []


@pytest.mark.parametrize("count", [4, 7, 27])
def test_role_scope_defaulted_message_looks_up_no_label_for_four_or_more_dcids(count: int):
    """Counts above three need no label lookup, so graph failures cannot break the warning."""
    label_of = _RecordingLabels()
    message = role_scope_defaulted_message(
        _discovered_scope("destination", 2),
        member_count=2,
        bound_dcids=_dcids(count),
        label_of=label_of,
    )
    assert message == f"{_DESTINATION_BASE} from any of the {count} providers."
    assert label_of.calls == []


@pytest.mark.parametrize("count", [1, 2, 3])
def test_role_scope_defaulted_message_looks_up_labels_once_for_the_full_dcid_list(count: int):
    label_of = _RecordingLabels(["A", "B", "C"])
    dcids = _dcids(count)
    role_scope_defaulted_message(
        _discovered_scope("destination", 2), member_count=2, bound_dcids=dcids, label_of=label_of
    )
    assert label_of.calls == [dcids]


def test_bound_opposite_role_dcids_returns_every_opposite_dcid_in_binding_order():
    roles: list[tuple[str, EntityRoleKey]] = [
        ("country/JPN", "origin"),
        ("country/CAN", "origin"),
        ("country/KEN", "destination"),
        ("country/DEU", "origin"),
    ]
    assert bound_opposite_role_dcids(_discovered_scope("destination", 2), roles) == [
        "country/JPN",
        "country/CAN",
        "country/DEU",
    ]
    assert bound_opposite_role_dcids(_discovered_scope("origin", 2), roles) == ["country/KEN"]


def test_bound_opposite_role_dcids_deduplicates_a_repeated_dcid_keeping_first_position():
    """Overlapping groups ("G7 and EU countries") bind one dcid twice."""
    roles: list[tuple[str, EntityRoleKey]] = [
        ("country/DEU", "origin"),
        ("country/FRA", "origin"),
        ("country/DEU", "origin"),
        ("country/ITA", "origin"),
        ("country/FRA", "origin"),
    ]
    assert bound_opposite_role_dcids(_discovered_scope("destination", 2), roles) == [
        "country/DEU",
        "country/FRA",
        "country/ITA",
    ]


def test_bound_opposite_role_dcids_keeps_a_dcid_bound_to_both_roles_for_the_opposite_only():
    """A dcid on both flow axes counts through its opposite-role pair alone.
    The discovered axis's own pair adds nothing."""
    roles: list[tuple[str, EntityRoleKey]] = [
        ("country/FRA", "destination"),
        ("country/DEU", "origin"),
        ("country/FRA", "origin"),
    ]
    assert bound_opposite_role_dcids(_discovered_scope("destination", 2), roles) == [
        "country/DEU",
        "country/FRA",
    ]
    assert bound_opposite_role_dcids(_discovered_scope("origin", 2), roles) == ["country/FRA"]


def test_bound_opposite_role_dcids_is_empty_without_an_opposite_entity():
    assert bound_opposite_role_dcids(_discovered_scope("destination", 2), []) == []
    only_own_axis: list[tuple[str, EntityRoleKey]] = [("country/KEN", "destination")]
    assert bound_opposite_role_dcids(_discovered_scope("destination", 2), only_own_axis) == []


def test_bound_opposite_role_dcids_reads_a_legacy_role_map():
    assert bound_opposite_role_dcids(
        _discovered_scope("destination", 2), {"country/FRA": "origin", "country/KEN": "destination"}
    ) == ["country/FRA"]


def _materialised(**overrides: object) -> Materialised:
    """A Materialised built by hand, so this unit test pins the builder's own
    rule rather than a resolve that happens to produce a carrier."""
    fields: dict = {
        "sv_dcids": ["TESTFIXTURE/GroundingIdentityEntities/Flow"],
        "has_data": True,
        "coverage": CoverageBare(has_data=True),
    }
    fields.update(overrides)
    return Materialised(**fields)


class _EchoLabels:
    def __call__(self, dcids: Sequence[str]) -> list[str]:
        return list(dcids)


def test_role_scope_defaulted_warnings_names_the_discovered_axis_and_its_bound_opposite():
    """The discovered-scope carrier keeps the message and the member count it
    had before the two doors were collapsed into one builder."""
    discovered = _discovered_scope("destination", 2)
    warnings = role_scope_defaulted_warnings(
        _materialised(discovered_scope=discovered, scope_member_dcids=discovered.member_dcids),
        boundary_roles=[("country/FRA", "origin")],
        label_of=_EchoLabels(),
    )

    assert [(w.code, w.severity) for w in warnings] == [("role_scope_defaulted", "warn")]
    assert (
        warnings[0].message
        == "no recipient was named; the answer was resolved across the 2 recipients with data "
        "from country/FRA."
    )


def test_role_scope_defaulted_warnings_adds_the_companion_axis_when_both_are_open():
    discovered = _discovered_scope("destination", 2)
    warnings = role_scope_defaulted_warnings(
        _materialised(
            discovered_scope=discovered,
            scope_member_dcids=discovered.member_dcids,
            also_defaulted_axis="origin",
        ),
        boundary_roles=[],
        label_of=_EchoLabels(),
    )

    assert [w.code for w in warnings] == ["role_scope_defaulted", "role_scope_defaulted"]
    assert [w.message for w in warnings] == [
        "no recipient was named; the answer was resolved across the 2 recipients with data.",
        "no provider was named; the answer was resolved across every provider with data.",
    ]


def test_a_filtered_answer_restates_the_scope_warning_in_place_without_a_label_request():
    """The materialised plan serves three recipients from France; a facet filter leaves one.
    The warning is restated in its own position, counting one, and France's label is read
    once, when the warning was first built."""
    discovered = _discovered_scope("destination", 3)
    mat = _materialised(discovered_scope=discovered, scope_member_dcids=discovered.member_dcids)
    roles: list[tuple[str, EntityRoleKey]] = [("country/FRA", "origin")]
    recording = _RecordingLabels(["France"])
    label_of = memoised_labels(recording)
    before = ResponseWarning(code="other_first", severity="info", message="first")
    after = ResponseWarning(code="other_last", severity="info", message="last")
    warnings = [
        before,
        *role_scope_defaulted_warnings(mat, boundary_roles=roles, label_of=label_of),
        after,
    ]

    restate_role_scope_warnings(
        warnings,
        mat,
        members=discovered.member_dcids[:1],
        boundary_roles=roles,
        label_of=label_of,
    )

    assert [w.message for w in warnings] == [
        "first",
        "no recipient was named; the answer was resolved across the 1 recipient with data "
        "from France.",
        "last",
    ]
    assert recording.calls == [["country/FRA"]]


def test_a_scope_warning_naming_every_member_is_restated_to_count_the_answers_scope():
    """A response whose candidates list different scopes names every recipient with data.
    Once one candidate is left, the same warning is restated to count that answer's scope,
    and restated back when asked to name every member."""
    discovered = _discovered_scope("destination", 3)
    mat = _materialised(discovered_scope=discovered, scope_member_dcids=discovered.member_dcids)
    warnings = role_scope_defaulted_warnings(
        mat, boundary_roles=[], label_of=_EchoLabels(), state_count=False
    )
    every = "no recipient was named; the answer was resolved across every recipient with data."
    assert [w.message for w in warnings] == [every]

    for members, state_count, restated in (
        (discovered.member_dcids[:1], True, every.replace("every recipient", "the 1 recipient")),
        (discovered.member_dcids, True, every.replace("every recipient", "the 3 recipients")),
        (discovered.member_dcids, False, every),
    ):
        restate_role_scope_warnings(
            warnings,
            mat,
            members=members,
            boundary_roles=[],
            label_of=_EchoLabels(),
            state_count=state_count,
        )
        assert [w.message for w in warnings] == [restated]


def test_a_scope_warning_is_removed_when_the_answer_lists_no_member():
    """Both flow axes are open and the facet filter leaves no recipient served: the answer
    resolved across no recipient, so the recipient warning goes and the provider warning,
    which states no count, stays where it was."""
    discovered = _discovered_scope("destination", 3)
    mat = _materialised(
        discovered_scope=discovered,
        scope_member_dcids=discovered.member_dcids,
        also_defaulted_axis="origin",
    )
    before = ResponseWarning(code="other_first", severity="info", message="first")
    warnings = [
        before,
        *role_scope_defaulted_warnings(mat, boundary_roles=[], label_of=_EchoLabels()),
    ]
    assert [w.message for w in warnings] == [
        "first",
        "no recipient was named; the answer was resolved across the 3 recipients with data.",
        "no provider was named; the answer was resolved across every provider with data.",
    ]

    restate_role_scope_warnings(
        warnings, mat, members=[], boundary_roles=[], label_of=_EchoLabels()
    )

    assert [w.message for w in warnings] == [
        "first",
        "no provider was named; the answer was resolved across every provider with data.",
    ]


def test_role_scope_defaulted_warnings_counts_the_members_the_plan_serves():
    """Three recipients hold data, but a bulk plan serves two of them: the warning names
    the two, as coverage and the scope list do."""
    discovered = _discovered_scope("destination", 3)
    warnings = role_scope_defaulted_warnings(
        _materialised(discovered_scope=discovered, scope_member_dcids=discovered.member_dcids[:2]),
        boundary_roles=[],
        label_of=_EchoLabels(),
    )

    assert [w.message for w in warnings] == [
        "no recipient was named; the answer was resolved across the 2 recipients with data."
    ]


def test_role_scope_defaulted_warnings_names_the_open_origin_on_the_bound_destination_carrier():
    """The bound-destination arm (#664): open_origin_dcids set, no discovered
    scope. The unqualified axis message is the one the builder returns, since
    this path carries no DiscoveredScope and so no member count."""
    warnings = role_scope_defaulted_warnings(
        _materialised(open_origin_dcids=frozenset({"country/FRA", "country/DEU"})),
        boundary_roles=[("country/KEN", "destination")],
        label_of=_EchoLabels(),
    )

    assert [w.code for w in warnings] == ["role_scope_defaulted"]
    assert (
        warnings[0].message
        == "no provider was named; the answer was resolved across every provider with data."
    )


def test_role_scope_defaulted_warnings_still_warns_on_an_empty_open_origin():
    """An empty open-origin set triggers the warning even when the index names no population."""
    warnings = role_scope_defaulted_warnings(
        _materialised(open_origin_dcids=frozenset()),
        boundary_roles=[],
        label_of=_EchoLabels(),
    )

    assert len(warnings) == 1
    assert warnings[0].code == "role_scope_defaulted"
    assert "no provider was named" in warnings[0].message


def test_role_scope_defaulted_warnings_is_empty_with_every_axis_bound():
    assert (
        role_scope_defaulted_warnings(
            _materialised(), boundary_roles=[("country/FRA", "origin")], label_of=_EchoLabels()
        )
        == []
    )


@pytest.mark.parametrize("warned_axis", ["destination", "origin"])
def test_only_scopes_on_the_warnings_axis_are_compared(warned_axis: EntityRoleKey) -> None:
    """A discovered-scope warning counts its members only when every candidate with a scope
    on its axis lists the same ones. Whatever a candidate lists on another axis, a
    subject-only alternate's places or the opposite role's members, and whatever a candidate
    with no scope lists, the comparison holds one scope: the role-bearing candidates' own."""
    other_axis: EntityRoleKey = "origin" if warned_axis == "destination" else "destination"
    members = ["country/0", "country/1"]
    warned = _materialised(
        discovered_scope=_discovered_scope(warned_axis, 2), scope_member_dcids=members
    )
    same_scope = _materialised(
        discovered_scope=_discovered_scope(warned_axis, 2), scope_member_dcids=members
    )
    subject_only = _materialised(scope_member_dcids=["country/KEN"])
    opposite = _materialised(
        discovered_scope=_discovered_scope(other_axis, 3),
        scope_member_dcids=["country/7", "country/8", "country/9"],
    )
    unscoped = _materialised()
    candidates = [warned, same_scope, subject_only, opposite, unscoped]
    scope_members = [members, members, ["country/KEN"], ["country/7"], None]

    assert scopes_on_axis(candidates, scope_members, axis=warned_axis) == {frozenset(members)}

    differing = [members, members[:1], ["country/KEN"], ["country/7"], None]
    assert scopes_on_axis(candidates, differing, axis=warned_axis) == {
        frozenset(members),
        frozenset(members[:1]),
    }
