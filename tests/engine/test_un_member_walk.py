"""The member walk of a UN shape reads the catalogue's UN tables at both doors.

Search's walk (`RoleClassWalks.family`) and the Datasets mint's walk
(`MemberEnumerator.enumerate` under a subject ceiling) run through one
`OriginGraphClient` whose transport refuses every request to the UN deployment.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping

import httpx
import pytest

from code_index.members import FamilyQuery, absent_readings
from code_index.shared import get_index
from origin_catalogue import UN_SYSTEM, CatalogueSnapshot
from origin_catalogue.graph import OriginGraphClient, build_graph_client
from search.engine.retrieval.bound_values import MembershipAuthority, RoleClassWalks
from statvar_members import DEFAULT_SUBJECT_CEILING, FiveTuple, MemberEnumerator, SlotBinding
from tests.origin_catalogue.fake_catalogue import (
    provenance_row,
    restrictive_catalogue,
    un_mirror,
    un_variable,
)

_OURS = "https://graph.example.org/core/api"
_UN = "https://un.example.org/core/api"

_SERIES = "undata/who/SERIES-MORT_U5_RATE"
_OTHER_SERIES = "undata/who/SERIES-MORT_INFANT_RATE"
_SEX = "undataSex"
_AGE = "undataAge"
_FEMALE = ("Sex", "undata/SEX-F", "Female")
_MALE = ("Sex", "undata/SEX-M", "Male")
_UNDER_ONE = ("Age", "undata/AGE-Y0", "Under 1")

_EMPLOYMENT_SERIES = "undata/ilo/SERIES-EMP_TEMP_NB"
_OCCUPATION = "undataOccupation"
_TEACHERS = "undata/ilo/EMP_TEMP_NB.OCCUPATION--ISCO08_23"
_NURSES = "undata/ilo/EMP_TEMP_NB.OCCUPATION--ISCO08_222"
# More siblings lacking the occupation constraint than the walk re-reads for a dropped arc
# (`statvar_members.enumerate._REPROBE_BUDGET`).
_AGE_SIBLINGS = 40

_TOTAL = "undata/who/MORT_U5_RATE"
_FEMALE_SV = "undata/who/MORT_U5_RATE.SEX--F"
_MALE_SV = "undata/who/MORT_U5_RATE.SEX--M"
_FEMALE_UNDER_ONE = "undata/who/MORT_U5_RATE.AGE--Y0__SEX--F"
_OTHER = "undata/who/MORT_INFANT_RATE"


def _catalogue() -> CatalogueSnapshot:
    return restrictive_catalogue(
        provenances=[provenance_row("undata/p/WHO", UN_SYSTEM, segment="who")],
        un_mirror=un_mirror(
            {
                _TOTAL: un_variable(_SERIES, name="Under-five mortality"),
                _FEMALE_SV: un_variable(_SERIES, name="Female", constraints={_SEX: _FEMALE}),
                _MALE_SV: un_variable(_SERIES, name="Male", constraints={_SEX: _MALE}),
                _FEMALE_UNDER_ONE: un_variable(
                    _SERIES, name="Female under 1", constraints={_SEX: _FEMALE, _AGE: _UNDER_ONE}
                ),
                _OTHER: un_variable(_OTHER_SERIES, name="Infant mortality"),
                **{
                    dcid: un_variable(
                        _EMPLOYMENT_SERIES,
                        name=dcid,
                        constraints={_OCCUPATION: ("Occupation", f"undata/{code}", code)},
                    )
                    for dcid, code in ((_TEACHERS, "ISCO08_23"), (_NURSES, "ISCO08_222"))
                },
                **{
                    f"undata/ilo/EMP_TEMP_NB.AGE--Y{age:02d}": un_variable(
                        _EMPLOYMENT_SERIES,
                        name=f"Aged {age}",
                        constraints={_AGE: ("Age", f"undata/AGE-Y{age:02d}", f"{age}")},
                    )
                    for age in range(_AGE_SIBLINGS)
                },
            }
        ),
    )


@pytest.fixture
def requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def graph(requests: list[httpx.Request], bind_catalogue) -> Iterator[OriginGraphClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {}})

    bind_catalogue(_catalogue())
    with build_graph_client(
        base_url=_OURS, token=None, un_base_url=_UN, transport=httpx.MockTransport(handler)
    ) as client:
        yield client


def _query(slots: Mapping[str, SlotBinding], *, series: str = _SERIES) -> FamilyQuery:
    return FamilyQuery(
        five_tuple=FiveTuple(
            population_type=series,
            measured_property="value",
            stat_type="measuredValue",
            measurement_qualifier=None,
            measurement_denominator=None,
        ),
        slots=slots,
        source=None,
        role_keys=frozenset({"observationAbout"}),
    )


def _mint_family(graph: OriginGraphClient, query: FamilyQuery) -> tuple[set[str], set[str]]:
    """The members and family the mint's walk settles, as `resolve_definition_members`
    asks for them."""
    group = MemberEnumerator().enumerate(
        graph=graph,
        stamp_key=graph.generation_stamp().key,
        five_tuple=query.five_tuple,
        slots=query.slots,
        readings=absent_readings(get_index()),
        role_keys=query.role_keys,
        open_role_keys=query.open_role_keys,
        subject_ceiling=DEFAULT_SUBJECT_CEILING,
        deadline=time.monotonic() + 30.0,
    )
    return {member.stat_var for member in group.members}, set(group.family)


@pytest.mark.parametrize(
    ("slots", "seeds", "members", "family"),
    [
        pytest.param(
            {_SEX: ("unbound",), _AGE: ("unbound",)},
            (_FEMALE_SV,),
            {_TOTAL},
            {_TOTAL, _FEMALE_SV, _MALE_SV, _FEMALE_UNDER_ONE},
            id="headline_from_a_recalled_breakdown",
        ),
        pytest.param(
            {_SEX: ("value", _FEMALE[1]), _AGE: ("unbound",)},
            (_TOTAL,),
            {_FEMALE_SV},
            {_FEMALE_SV, _FEMALE_UNDER_ONE},
            id="bound_sex",
        ),
        pytest.param(
            {_SEX: ("value", _FEMALE[1]), _AGE: ("value", _UNDER_ONE[1])},
            (),
            {_FEMALE_UNDER_ONE},
            {_FEMALE_UNDER_ONE},
            id="bound_sex_and_age",
        ),
    ],
)
def test_both_doors_settle_a_un_shapes_members_from_the_tables(
    graph: OriginGraphClient,
    requests: list[httpx.Request],
    slots: dict[str, SlotBinding],
    seeds: tuple[str, ...],
    members: set[str],
    family: set[str],
) -> None:
    query = _query(slots)

    walked = RoleClassWalks(graph, enumerator=MemberEnumerator()).family(
        query, seeds, deadline=time.monotonic() + 30.0
    )
    minted = _mint_family(graph, query)

    assert walked.authority is MembershipAuthority.AUTHORITATIVE
    assert walked.value is not None
    group = walked.value.group
    assert ({member.stat_var for member in group.members}, set(group.family)) == minted
    assert minted == (members, family)
    assert requests == []


def test_both_doors_settle_a_set_bound_slot_whose_siblings_lack_it(
    graph: OriginGraphClient, requests: list[httpx.Request]
) -> None:
    """The series' age breakdowns lack the set-bound occupation constraint, more of them than
    the walk re-reads for a dropped arc. The tables hold every arc, so each absence is a
    verdict and the walk settles the occupation members at both doors."""
    slots: dict[str, SlotBinding] = {
        _OCCUPATION: ("set", ("undata/ISCO08_23", "undata/ISCO08_222")),
        _AGE: ("unbound",),
    }
    query = _query(slots, series=_EMPLOYMENT_SERIES)

    walked = RoleClassWalks(graph, enumerator=MemberEnumerator()).family(
        query, (), deadline=time.monotonic() + 30.0
    )
    minted = _mint_family(graph, query)

    assert walked.authority is MembershipAuthority.AUTHORITATIVE
    assert walked.value is not None
    assert {member.stat_var for member in walked.value.group.members} == minted[0]
    assert minted[0] == {_TEACHERS, _NURSES}
    assert requests == []


@pytest.mark.parametrize(
    ("dcid", "complete"),
    [
        (_TEACHERS, True),
        ("undata/who/MORT_NEONATAL_RATE", False),
        ("Count_Person", False),
    ],
    ids=["un_variable_held", "un_dcid_not_held", "ours_not_held"],
)
def test_the_client_reads_in_full_only_what_the_tables_hold(
    graph: OriginGraphClient, dcid: str, complete: bool
) -> None:
    """A node the tables do not hold is read live, where a batch read can drop an arc."""
    assert graph.arcs_complete(dcid) is complete
