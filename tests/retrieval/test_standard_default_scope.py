"""`standard_resolve` defaults a placeless shape's scope from a presence read and settles it
on the StatVars the answer serves.

The probe reads the members that can answer under the facet read plan with fewer POSTs
(`facets_at_own_places`), and each member keeps its facets at its own default scope alone.
The answer is then read over the default scope of the members it elects, rescanned from
those reads, so it equals what a resubmit of those members resolves: a resubmit posts them
as its membership and derives its scope from them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from materialise.place_scope import (
    DefaultPlaceScope,
    DefaultPresence,
    scope_of_places,
)
from search.engine.bind import SlotBindingDraft
from search.engine.extract import DateRequest
from search.engine.retrieval import standard
from search.engine.retrieval.retrieve import Materialised
from search.engine.retrieval.standard import facets_at_own_places, standard_resolve
from search.engine.shape import ShapeDraft
from search.models import TimeWindow
from tests.fixtures import FakeGraph

_KEN, _TZA, _GHA = "country/KEN", "country/TZA", "country/GHA"
_EARTH = "Earth"
_AFRICA = "africa"
_FEMALE = "Count_Death_FemaleDefaultScope"
_MALE = "Count_Death_MaleDefaultScope"
_TOTAL = "Count_Death_TotalDefaultScope"
_OTHER = "Count_Death_OtherDefaultScope"
_SOURCE = "dc/base/DefaultScopeFixture"


def _arcs(gender: str | None) -> dict:
    arcs = {
        "populationType": {"nodes": [{"dcid": "MortalityEvent"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if gender is not None:
        arcs["constraintProperties"] = {"nodes": [{"dcid": "gender"}]}
        arcs["gender"] = {"nodes": [{"dcid": gender}]}
    return arcs


_ARCS = {
    _FEMALE: _arcs("Female"),
    _MALE: _arcs("Male"),
    _TOTAL: _arcs(None),
    _OTHER: _arcs("Other"),
}


def _shape(*members: str, complete: bool = False) -> ShapeDraft:
    return ShapeDraft(
        shape_id="death_count_default_scope",
        label="Deaths",
        pop_type_dcid="MortalityEvent",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={sv: _ARCS[sv] for sv in members},
        sv_arc_facts_complete=complete,
        representative_score=0.9,
    )


def _gender(kind: str, *values: str) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="what", property_dcid="gender", kind=kind, value_dcids=list(values)
    )


def _facets(start: str = "2015", end: str = "2022") -> list[dict]:
    return [{"earliestDate": start, "latestDate": end, "obsCount": 5, "provenanceId": _SOURCE}]


class _BatchRecordingGraph(FakeGraph):
    """Records the StatVars and entities of every facet batch. With `request_size` it sends a
    read apart in groups of that many StatVars, as a deployment reading by GET does."""

    def __init__(self, obs: dict[str, list[dict]], *, request_size: int | None = None) -> None:
        super().__init__(
            nodes={},
            obs=obs,
            detect={},
            resolve={},
            children={_EARTH: {"Country": [_GHA, _KEN, _TZA]}},
        )
        self.batches: list[tuple[list[str], list[str]]] = []
        self._request_size = request_size

    def observation_facets_batch(self, stat_vars, entities, *args, **kwargs):
        self.batches.append((list(stat_vars), list(entities)))
        return super().observation_facets_batch(stat_vars, entities, *args, **kwargs)

    def observation_read_groups(self, stat_vars, entities, needs_dates=False):
        size = self._request_size or len(stat_vars)
        return [list(stat_vars[i : i + size]) for i in range(0, len(stat_vars), size)]


def _at(stat_var: str, *places: str, start: str = "2015", end: str = "2022") -> dict:
    return {f"{stat_var}|{place}": _facets(start, end) for place in places}


def _presence(scopes: Mapping[str, Sequence[str]], *, every_place: bool = False) -> DefaultPresence:
    """A presence read giving each StatVar the default scope of its `scopes` places: every
    place with data, or the countries fallback."""
    return DefaultPresence(
        countries=(_GHA, _KEN, _TZA),
        scopes={
            sv: scope_of_places(places, every_place=every_place) for sv, places in scopes.items()
        },
    )


def _countries(*places: str) -> DefaultPlaceScope:
    return scope_of_places(places, every_place=False)


def _resolve(
    shape: ShapeDraft,
    graph: FakeGraph,
    presence: DefaultPresence,
    bindings: list[SlotBindingDraft] | None = None,
    date_request: DateRequest | None = None,
) -> Materialised:
    result = standard_resolve(
        shape=shape,
        bindings=bindings or [],
        entities={},
        graph=graph,
        date_request=date_request,
        default_presence=presence,
    )
    assert isinstance(result, Materialised), result
    return result


def _place_facts(mat: Materialised) -> dict:
    """What an answer states about its places and data, as both doors build it."""
    return {
        "sv_dcids": mat.sv_dcids,
        "has_data": mat.has_data,
        "available_at_place": mat.available_at_place,
        "coverage": mat.coverage,
        "scope_member_dcids": mat.scope_member_dcids,
        "default_scope": mat.default_scope,
        "facets_by_entity": mat.facets_by_entity,
        "facets_by_sv": mat.facets_by_sv,
    }


def _resubmitted(
    mat: Materialised,
    obs: dict,
    presence: DefaultPresence,
    bindings: list[SlotBindingDraft] | None = None,
    date_request: DateRequest | None = None,
) -> Materialised:
    """The same graph resolved for a resubmit that posts `mat`'s StatVars as its exact
    membership."""
    return _resolve(
        _shape(*mat.sv_dcids, complete=True),
        _BatchRecordingGraph(obs),
        presence,
        bindings,
        date_request,
    )


def _labels(mat: Materialised) -> list[str]:
    return [dim.label for dim in mat.coverage.dimensions or []]


def test_m1_a_member_is_read_over_its_own_countries() -> None:
    """Female has data in Kenya, male in Tanzania. One read over both countries costs fewer
    POSTs than one per country; neither member rolls up the other, so the answer serves
    both, as a mint does, and settles on each member's own country from that read."""
    obs = {**_at(_FEMALE, _KEN), **_at(_MALE, _TZA)}
    graph = _BatchRecordingGraph(obs)
    presence = _presence({_FEMALE: [_KEN], _MALE: [_TZA]})

    mat = _resolve(_shape(_FEMALE, _MALE), graph, presence)

    assert graph.batches == [([_FEMALE, _MALE], [_KEN, _TZA])]
    assert mat.sv_dcids == [_FEMALE, _MALE]
    assert mat.default_scope == _countries(_KEN, _TZA)
    assert mat.scope_member_dcids == [_KEN, _TZA]
    assert "countries" in _labels(mat)
    assert _place_facts(mat) == _place_facts(_resubmitted(mat, obs, presence))


def test_m2_a_worldwide_total_answers_before_a_member_that_extends_it() -> None:
    """Total has data at Earth only; female, which extends the total's constraints, has data
    in Kenya. The family election keeps the total alone, as the mint does, so the probe takes
    the total's scope, Earth, and the total answers worldwide."""
    obs = {**_at(_TOTAL, _EARTH), **_at(_FEMALE, _KEN)}
    graph = _BatchRecordingGraph(obs)
    presence = _presence({_TOTAL: [_EARTH], _FEMALE: [_KEN]})

    mat = _resolve(_shape(_TOTAL, _FEMALE), graph, presence)

    # Female counts only toward the confirmed population, read at the total's scope.
    assert graph.batches == [([_TOTAL, _FEMALE], [_EARTH])]
    assert mat.sv_dcids == [_TOTAL]
    assert mat.default_scope == _countries(_EARTH)
    assert _place_facts(mat) == _place_facts(_resubmitted(mat, obs, presence))


def test_m3_a_bound_member_with_worldwide_data_answers_worldwide() -> None:
    """Gender is bound to female, which has data at Earth only, while male has data in
    Tanzania. The probe and its scope take female alone, so the answer is a worldwide total
    where the shape's own scope is Tanzania."""
    obs = {**_at(_FEMALE, _EARTH), **_at(_MALE, _TZA)}
    graph = _BatchRecordingGraph(obs)
    presence = _presence({_FEMALE: [_EARTH], _MALE: [_TZA]})
    bindings = [_gender("value", "Female")]

    mat = _resolve(_shape(_FEMALE, _MALE), graph, presence, bindings)

    assert graph.batches == [([_FEMALE], [_EARTH])]
    assert mat.has_data
    assert mat.sv_dcids == [_FEMALE]
    assert mat.default_scope == _countries(_EARTH)
    assert "worldwide" in _labels(mat)
    assert _place_facts(mat) == _place_facts(_resubmitted(mat, obs, presence, bindings))


def test_a_set_binding_probes_only_the_members_it_collects() -> None:
    """A set binding collects female and male; the total, which asserts no gender, holds
    the only data in Ghana and stays out of the probe scope. Both collected members serve,
    so the scope stands at their two countries, and coverage counts countries."""
    obs = {**_at(_FEMALE, _KEN), **_at(_MALE, _TZA), **_at(_TOTAL, _GHA)}
    graph = _BatchRecordingGraph(obs)
    presence = _presence({_FEMALE: [_KEN], _MALE: [_TZA], _TOTAL: [_GHA]})
    bindings = [_gender("set", "Female", "Male")]

    mat = _resolve(_shape(_FEMALE, _MALE, _TOTAL), graph, presence, bindings)

    # The total counts only toward the confirmed population, read at the collected scopes.
    assert graph.batches == [([_FEMALE, _MALE, _TOTAL], [_KEN, _TZA])]
    assert mat.sv_dcids == [_FEMALE, _MALE]
    assert mat.default_scope == _countries(_KEN, _TZA)
    assert "countries" in _labels(mat)
    assert sorted(mat.facets_by_entity or {}) == [_KEN, _TZA]
    assert _place_facts(mat) == _place_facts(_resubmitted(mat, obs, presence, bindings))


def test_a_set_binding_settles_on_the_members_that_serve() -> None:
    """A set binding collects female and male, but only female has data, in Kenya; male's
    presence sits outside the years read. The answer settles on Kenya."""
    obs = {**_at(_FEMALE, _KEN), **_at(_MALE, _TZA, start="1990", end="1995")}
    graph = _BatchRecordingGraph(obs)
    presence = _presence({_FEMALE: [_KEN], _MALE: [_TZA]})
    bindings = [_gender("set", "Female", "Male")]
    window = DateRequest(window=TimeWindow(start_year=2015, end_year=2022), latest=False)

    mat = _resolve(_shape(_FEMALE, _MALE), graph, presence, bindings, window)

    assert graph.batches == [([_FEMALE, _MALE], [_KEN, _TZA])]
    assert mat.sv_dcids == [_FEMALE]
    assert mat.default_scope == _countries(_KEN)
    assert _place_facts(mat) == _place_facts(_resubmitted(mat, obs, presence, bindings, window))


def test_no_member_with_presence_reads_no_facets() -> None:
    """Neither member has data anywhere: the scope is empty and the probe is skipped."""
    graph = _BatchRecordingGraph({})
    presence = _presence({_FEMALE: [], _MALE: []})

    mat = _resolve(_shape(_FEMALE, _MALE), graph, presence)

    assert graph.batches == []
    assert mat.available_at_place is False
    assert mat.default_scope == _countries()


@pytest.mark.parametrize(
    ("scopes", "obs", "served", "settled"),
    [
        pytest.param(
            {_FEMALE: [_KEN], _MALE: [_TZA]},
            {**_at(_FEMALE, _TZA), **_at(_MALE, _TZA)},
            [_MALE],
            _countries(_TZA),
            id="presence-claims-a-place-the-facets-lack",
        ),
        pytest.param(
            {_FEMALE: [_TZA], _MALE: [_KEN]},
            {**_at(_FEMALE, _KEN, _TZA), **_at(_MALE, _KEN)},
            [_FEMALE, _MALE],
            _countries(_KEN, _TZA),
            id="presence-misses-a-place-the-facets-hold",
        ),
    ],
)
def test_a_presence_read_that_disagrees_with_the_facets_settles_as_a_resubmit_does(
    scopes: dict[str, list[str]], obs: dict, served: list[str], settled: DefaultPlaceScope
) -> None:
    """Presence and facets disagree on where female has data. Each member is read and
    scanned at its own presence countries: where female's presence country holds none of
    her facets, male alone answers. The members that answer are settled on their presence
    countries, which is what a resubmit of them resolves under the same presence."""
    presence = _presence(scopes)
    graph = _BatchRecordingGraph(obs)

    mat = _resolve(_shape(_FEMALE, _MALE), graph, presence)

    assert graph.batches == [([_FEMALE, _MALE], [*scopes[_FEMALE], *scopes[_MALE]])]
    assert mat.sv_dcids == served
    assert mat.default_scope == settled
    resubmit = _resubmitted(
        Materialised(sv_dcids=served, has_data=False, coverage=mat.coverage), obs, presence
    )
    assert _place_facts(mat) == _place_facts(resubmit)


def test_a_member_with_facets_beyond_its_scope_keeps_its_scope_alone() -> None:
    """Female has facets at Earth and in Kenya, but the presence read reports her at Earth
    only; male has data in Kenya. The read over both places fetches female's Kenya facets,
    which she does not keep, so the answer serves female at Earth and male in Kenya, and
    Kenya's facets are male's alone, as a resubmit of both resolves them under the same
    presence."""
    presence = _presence({_FEMALE: [_EARTH], _MALE: [_KEN]})
    obs = {**_at(_FEMALE, _EARTH, _KEN), **_at(_MALE, _KEN)}
    graph = _BatchRecordingGraph(obs)

    mat = _resolve(_shape(_FEMALE, _MALE), graph, presence)

    assert graph.batches == [([_FEMALE, _MALE], [_EARTH, _KEN])]
    assert mat.has_data
    assert mat.sv_dcids == [_FEMALE, _MALE]
    assert mat.default_scope == presence.union_scope([_FEMALE, _MALE])
    assert mat.facets_by_entity is not None
    assert len(mat.facets_by_entity[_KEN]) == 1
    assert _place_facts(mat) == _place_facts(_resubmitted(mat, obs, presence))


def test_a_presence_read_without_a_probe_member_raises() -> None:
    presence = _presence({_FEMALE: [_KEN]})

    with pytest.raises(AssertionError, match=_MALE):
        _resolve(_shape(_FEMALE, _MALE), _BatchRecordingGraph({}), presence)


def test_a_presence_read_beside_a_subject_entity_raises() -> None:
    presence = _presence({_FEMALE: [_KEN]})

    with pytest.raises(AssertionError, match=_KEN):
        standard_resolve(
            shape=_shape(_FEMALE),
            bindings=[],
            entities={"observationAbout": [_KEN]},
            graph=_BatchRecordingGraph({}),
            default_presence=presence,
        )


def test_a_scope_settling_on_a_place_the_batch_did_not_read_raises(monkeypatch) -> None:
    """Female holds data in Kenya, and the batch reads Kenya. A scope that settled on
    Tanzania as well would state a place whose rows were never read, so it raises."""
    presence = _presence({_FEMALE: [_KEN]})
    graph = _BatchRecordingGraph(_at(_FEMALE, _KEN))
    assert _resolve(_shape(_FEMALE), graph, presence).default_scope == _countries(_KEN)
    monkeypatch.setattr(
        DefaultPresence, "union_scope", lambda self, stat_vars: _countries(_KEN, _TZA)
    )

    with pytest.raises(AssertionError, match="did not cover"):
        _resolve(_shape(_FEMALE), graph, presence)


def test_an_every_place_scope_counts_places_of_any_level() -> None:
    """Female's default scope is every place with data, a region and a country, so the
    answer counts places rather than countries."""
    obs = _at(_FEMALE, _AFRICA, _KEN)
    graph = _BatchRecordingGraph(obs)
    presence = _presence({_FEMALE: [_AFRICA, _KEN]}, every_place=True)

    mat = _resolve(_shape(_FEMALE), graph, presence)

    assert graph.batches == [([_FEMALE], [_AFRICA, _KEN])]
    assert mat.default_scope == DefaultPlaceScope(places=(_AFRICA, _KEN), every_place=True)
    assert "places" in _labels(mat)
    assert "countries" not in _labels(mat)
    assert _place_facts(mat) == _place_facts(_resubmitted(mat, obs, presence))


def test_the_per_scope_plan_runs_when_it_costs_fewer_posts(monkeypatch) -> None:
    """Under a bound of 4 pairs the per-scope plan costs 3 POSTs and the union plan 4, one per
    StatVar over five places. StatVars sharing a place set share reads of at most the bound,
    in first-StatVar order; one whose places alone exceed it is read alone, and one with no
    places is not read."""
    monkeypatch.setattr(standard, "_FACET_PROBE_MAX_PAIRS_PER_POST", 4)
    graph = _BatchRecordingGraph({})

    facets = facets_at_own_places(
        graph,
        {
            "a": (_KEN, _TZA),
            "b": (_GHA, _KEN, _TZA, _EARTH, _AFRICA),
            "c": (_KEN, _TZA),
            "d": (_KEN, _TZA),
            "e": (),
        },
        needs_dates=False,
    )

    assert graph.batches == [
        (["a", "c"], [_KEN, _TZA]),
        (["d"], [_KEN, _TZA]),
        (["b"], [_GHA, _KEN, _TZA, _EARTH, _AFRICA]),
    ]
    assert facets["e"] == {}
    assert sorted(facets["b"]) == sorted([_GHA, _KEN, _TZA, _EARTH, _AFRICA])


# Two StatVars over disjoint pairs of places: 2 POSTs per scope at any bound of 4 or more,
# and 2 POSTs over the union of four places at a bound of 4 to 7, 1 from 8.
_DISJOINT = {"a": (_KEN, _TZA), "b": (_GHA, _EARTH)}


@pytest.mark.parametrize(
    ("bound", "batches"),
    [
        pytest.param(4, [(["a"], [_KEN, _TZA]), (["b"], [_GHA, _EARTH])], id="tie-per-scope"),
        pytest.param(7, [(["a"], [_KEN, _TZA]), (["b"], [_GHA, _EARTH])], id="below-boundary"),
        pytest.param(8, [(["a", "b"], [_KEN, _TZA, _GHA, _EARTH])], id="union-fewer-posts"),
    ],
)
def test_the_plan_with_fewer_posts_runs_and_a_tie_reads_per_scope(
    monkeypatch, bound: int, batches: list
) -> None:
    monkeypatch.setattr(standard, "_FACET_PROBE_MAX_PAIRS_PER_POST", bound)
    graph = _BatchRecordingGraph({})

    facets_at_own_places(graph, _DISJOINT, needs_dates=False)

    assert graph.batches == batches


def test_both_plans_give_the_same_answer(monkeypatch) -> None:
    """Female has facets in Kenya and Tanzania but a scope of Kenya; male in Tanzania and
    Ghana with a scope of both. Under the default bound one union read runs, under a bound
    of 1, which ties the plans, one read per scope; each member keeps its own places alone either way, so the
    facets and the answer are the same."""
    obs = {**_at(_FEMALE, _KEN, _TZA), **_at(_MALE, _TZA, _GHA)}
    presence = _presence({_FEMALE: [_KEN], _MALE: [_GHA, _TZA]})
    places_of = {_FEMALE: (_KEN,), _MALE: (_GHA, _TZA)}

    union_graph = _BatchRecordingGraph(obs)
    union_facets = facets_at_own_places(union_graph, places_of, needs_dates=False)
    union_mat = _resolve(_shape(_FEMALE, _MALE), union_graph, presence)
    monkeypatch.setattr(standard, "_FACET_PROBE_MAX_PAIRS_PER_POST", 1)
    scope_graph = _BatchRecordingGraph(obs)
    scope_facets = facets_at_own_places(scope_graph, places_of, needs_dates=False)
    scope_mat = _resolve(_shape(_FEMALE, _MALE), scope_graph, presence)

    assert union_graph.batches[0] == ([_FEMALE, _MALE], [_KEN, _GHA, _TZA])
    assert scope_graph.batches[:2] == [([_FEMALE], [_KEN]), ([_MALE], [_GHA, _TZA])]
    assert sorted(union_facets[_FEMALE]) == [_KEN]
    assert union_facets == scope_facets
    assert _place_facts(union_mat) == _place_facts(scope_mat)


# A family of a total and six breakdowns, each with data in three countries. Under a bound
# of 6 pairs one read holds two StatVars, so reading the whole family takes 4 POSTs.
_BREAKDOWNS = [f"Count_Death_Breakdown{i}DefaultScope" for i in range(6)]
_THREE = (_GHA, _KEN, _TZA)


@pytest.mark.parametrize(
    ("total_has_data", "request_size", "reads"),
    [
        pytest.param(True, None, [[_TOTAL, _BREAKDOWNS[0]]], id="the-total-answers"),
        pytest.param(False, None, [[_TOTAL, _BREAKDOWNS[0]]], id="the-total-has-no-data"),
        pytest.param(True, 1, [[_TOTAL]], id="a-read-sent-in-several-requests-stops-between-them"),
    ],
)
def test_an_answer_naming_one_member_reads_the_family_only_until_its_election_settles(
    monkeypatch, total_has_data: bool, request_size: int | None, reads: list[list[str]]
) -> None:
    """ "Latest life expectancy data" leaves every slot of the 351-member family open, so the
    total answers alone. The reads run in election order, the total first, and stop once the
    total is read, with or without data: no breakdown can be elected over it. The confirmed
    count is then a floor. A graph that sends one read as several requests is asked no
    request past the one that settles the election: "refugees from Somalia" on the UN System
    Data Commons reads 1 of the 40 GETs its 2,896-member family takes."""
    monkeypatch.setattr(standard, "_FACET_PROBE_MAX_PAIRS_PER_POST", 6)
    members = [_TOTAL, *_BREAKDOWNS]
    obs = {
        key: facets
        for sv in members
        if sv != _TOTAL or total_has_data
        for key, facets in _at(sv, *_THREE).items()
    }
    graph = _BatchRecordingGraph(obs, request_size=request_size)
    shape = ShapeDraft(
        shape_id="death_count_default_scope",
        label="Deaths",
        pop_type_dcid="MortalityEvent",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts={
            _TOTAL: _arcs(None),
            **{sv: _arcs(f"Breakdown{i}") for i, sv in enumerate(_BREAKDOWNS)},
        },
        sv_arc_facts_complete=True,
        representative_score=0.9,
    )

    result = standard_resolve(
        shape=shape,
        bindings=[],
        entities={},
        graph=graph,
        default_presence=_presence(dict.fromkeys(members, _THREE)),
    )

    assert [stat_vars for stat_vars, _places in graph.batches] == reads
    assert result.sv_dcids == [_TOTAL]
    assert result.population is not None
    assert (result.population.confirmed, result.population.member_count_exact) == (
        sum(sv != _TOTAL or total_has_data for sv in reads[0]),
        False,
    )
