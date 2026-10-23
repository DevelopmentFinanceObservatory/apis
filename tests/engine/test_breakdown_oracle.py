"""A mint of each answer's definition serves the members the answer serves. For every
family the no-total tests build, the Datasets mint's member walk (`statvar_members`) runs
over the Spec's identity against a graph holding the same member facts, apart from the
engine, and states the answer's served series and its served count."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest

from statvar_members import (
    enumerate_members,
    five_tuple_from_identity,
    member_digest,
    slot_bindings_from_identity,
)
from tests.engine.test_standard_resolve import (
    _UN_FEMALE_U5,
    _UN_MALE_U5,
    _UN_U5,
    _resolve_child_mortality_in_chad,
    _un_arcs,
)
from tests.engine.test_un_tables_serving import (
    _FEMALE,
    _MALE,
    _QUINTILES,
    _RESIDENCE,
    _SERIES,
    _SEX,
    _TWO_BREAKDOWNS,
    _URBAN,
    _WEALTH,
    _hold_wealth_series,
    _resolve_by_wealth_quintile,
    _resolve_two_breakdowns,
)
from tests.statvar_members.conftest import FakeMemberArcReader

type _Constraints = Mapping[str, str]


def _arcs(population_type: str, constraints: _Constraints) -> dict[str, tuple[str, ...]]:
    """The out-arcs a member StatVar of `population_type` asserting `constraints` carries."""
    return {
        "typeOf": ("StatisticalVariable",),
        "populationType": (population_type,),
        "measuredProperty": ("value",),
        "statType": ("measuredValue",),
        "constraintProperties": tuple(constraints),
        **{prop: (value,) for prop, value in constraints.items()},
    }


def _who_members(*, urban: bool = False) -> dict[str, _Constraints]:
    """The WHO series `_hold_wealth_series` binds: five quintiles, each in the urban
    residence area too with `urban`, and two by-sex members."""
    residence = {_RESIDENCE: _URBAN[0]} if urban else {}
    return {
        **{sv: {_WEALTH: value, **residence} for sv, (value, _label) in _QUINTILES.items()},
        _FEMALE: {_SEX: "undata/SEX-F"},
        _MALE: {_SEX: "undata/SEX-M"},
    }


_TWO_BREAKDOWN_MEMBERS = {
    sv: {prop: f"undata/{prop}-{value}"} for sv, (_name, prop, value) in _TWO_BREAKDOWNS.items()
}


def _chad_constraints(members: Mapping[str, Mapping]) -> dict[str, _Constraints]:
    """The constraints each child-mortality member's graph arcs assert."""
    return {
        sv: {
            prop: arcs[prop]["nodes"][0]["dcid"]
            for prop in (node["dcid"] for node in arcs["constraintProperties"]["nodes"])
        }
        for sv, arcs in members.items()
    }


_SEX_PAIR = {
    _UN_MALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-M"),
    _UN_FEMALE_U5: _un_arcs(age="undata/AGE-Y0T4", sex="undata/SEX-F"),
}
_SEX_PAIR_WITH_TOTAL = {**_SEX_PAIR, _UN_U5: _un_arcs(age="undata/AGE-Y0T4", sex=None)}


def _by_wealth_quintile(bind_catalogue, *, un_pass: bool, urban: bool = False):
    return _resolve_by_wealth_quintile(
        _hold_wealth_series(bind_catalogue, un_pass=un_pass, urban=urban)
    )


@dataclass(frozen=True)
class _Case:
    resolve: Callable
    """Resolves the fixture's question, given the `bind_catalogue` fixture."""
    population_type: str
    members: dict[str, _Constraints]
    """Every member of the family the fixture builds, with the constraints it asserts."""
    counted: bool
    """Whether the answer states a served count: a family the catalogue holds whole does,
    a recall sample whose walk is unsettled states none."""


_CASES = {
    "two_breakdowns_candidates": _Case(
        resolve=lambda bind: _resolve_two_breakdowns(bind, with_data=frozenset(_TWO_BREAKDOWNS)),
        population_type=_SERIES,
        members=_TWO_BREAKDOWN_MEMBERS,
        counted=True,
    ),
    "lone_breakdown_with_data": _Case(
        resolve=lambda bind: _resolve_two_breakdowns(bind, with_data=frozenset({_FEMALE, _MALE})),
        population_type=_SERIES,
        members=_TWO_BREAKDOWN_MEMBERS,
        counted=True,
    ),
    "by_wealth_quintile_our_pass": _Case(
        resolve=lambda bind: _by_wealth_quintile(bind, un_pass=False),
        population_type=_SERIES,
        members=_who_members(),
        counted=True,
    ),
    "by_wealth_quintile_un_pass": _Case(
        resolve=lambda bind: _by_wealth_quintile(bind, un_pass=True),
        population_type=_SERIES,
        members=_who_members(),
        counted=True,
    ),
    "by_wealth_quintile_no_member_alone": _Case(
        resolve=lambda bind: _by_wealth_quintile(bind, un_pass=False, urban=True),
        population_type=_SERIES,
        members=_who_members(urban=True),
        counted=True,
    ),
    "sex_pair_walked": _Case(
        resolve=lambda _bind: _resolve_child_mortality_in_chad(
            _SEX_PAIR, with_data=tuple(_SEX_PAIR)
        ),
        population_type="undata/unicef/SERIES-CME_MR",
        members=_chad_constraints(_SEX_PAIR),
        counted=False,
    ),
    "control_with_total": _Case(
        resolve=lambda _bind: _resolve_child_mortality_in_chad(
            _SEX_PAIR_WITH_TOTAL, with_data=tuple(_SEX_PAIR_WITH_TOTAL)
        ),
        population_type="undata/unicef/SERIES-CME_MR",
        members=_chad_constraints(_SEX_PAIR_WITH_TOTAL),
        counted=False,
    ),
}


def _minted(identity: Mapping[str, object], graph: FakeMemberArcReader) -> list[str]:
    """The members a mint of `identity` states, walked over `graph`."""
    group = enumerate_members(
        five_tuple=five_tuple_from_identity(identity["shape"]),
        slots=slot_bindings_from_identity(identity["slots"], widen=lambda ref: (ref,)),
        graph=graph,
        role_keys=frozenset(identity.get("entities", {})),
    )
    return [member.stat_var for member in group.members]


@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_a_mint_of_each_spec_serves_the_members_the_answer_serves(
    bind_catalogue, case: _Case
) -> None:
    graph = FakeMemberArcReader(
        arcs={
            sv: _arcs(case.population_type, constraints) for sv, constraints in case.members.items()
        }
    )

    result = case.resolve(bind_catalogue)

    assert result.specs
    for spec in result.specs:
        identity = spec.definition.identity.model_dump(mode="json", exclude_none=True)
        minted = _minted(identity, graph)
        resolved = spec.resolved
        assert sorted(sv.ref.ref for sv in resolved.members) == sorted(minted), spec.spec_id
        assert resolved.stat_vars.served_count == (len(minted) if case.counted else None), (
            spec.spec_id
        )
        if resolved.shape.member_digest is not None:
            assert resolved.shape.member_digest == member_digest(minted), spec.spec_id
