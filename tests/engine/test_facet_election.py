"""The bulk scope and bulk role previews rank with `materialise.facets.rank_pool`.

Outside the origin catalogue's overlap set the order is the one each bulk path used before:
series covered, then observations, then recency, then provenance, method, unit and facet id.
Inside it, DFO's facet leads when it covers at least as many series as the best base facet,
as it does for a single member; a broader base facet leads a narrower DFO one (#1271).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pytest

from materialise.window import RequestedWindow
from origin_catalogue import BASE, DFO, Catalogue
from search.engine.clients.graph import Facet
from search.engine.retrieval import facet_election
from search.engine.retrieval.facet_election import (
    merge_role_candidates,
    merge_scope_candidates,
    rank_role_candidates,
    rank_scope_candidates,
    role_continuation,
)
from tests.engine._role_candidates import ListedRolePairs, role_candidate
from tests.origin_catalogue.fake_catalogue import provenance_row, restrictive_catalogue

_SV = "Count_Person"
_DFO_PROVENANCE = "provenance/DfoImport"
_BASE_PROVENANCE = "dc/base/WorldDevelopmentIndicators"
_PLACES = ["country/KEN", "country/NOR", "country/FRA"]


@pytest.fixture
def overlap(bind_catalogue: Callable[[Catalogue], None]) -> None:
    """`_SV` is held by DFO and by an admitted base provenance."""
    bind_catalogue(
        restrictive_catalogue(
            provenances=[
                provenance_row(_DFO_PROVENANCE, DFO),
                provenance_row(_BASE_PROVENANCE, BASE),
            ],
            dfo_variables=[_SV],
            base_variables={_SV: ([_BASE_PROVENANCE], [])},
        )
    )


def _facet(facet_id: str, *, obs: int, latest: str, provenance: str = _BASE_PROVENANCE) -> Facet:
    return Facet(
        facet_id=facet_id,
        provenance_id=provenance,
        obs_count=obs,
        earliest_date="1990",
        latest_date=latest,
    )


def _scope_order(facets_by_place: dict[str, list[Facet]]) -> list[str]:
    merged = merge_scope_candidates(facets_by_place, list(facets_by_place))
    return [c.facet_id for c in rank_scope_candidates(merged, stat_vars=[_SV])]


def test_scope_ranking_leads_with_places_then_observations_then_recency():
    """`wide` covers three places with the fewest observations; `deep` and `late` cover two,
    tied on observations, and `late` reaches further."""
    order = _scope_order(
        {
            _PLACES[0]: [
                _facet("wide", obs=1, latest="2020"),
                _facet("deep", obs=50, latest="2020"),
                _facet("late", obs=50, latest="2024"),
            ],
            _PLACES[1]: [
                _facet("wide", obs=1, latest="2020"),
                _facet("deep", obs=50, latest="2020"),
                _facet("late", obs=50, latest="2024"),
            ],
            _PLACES[2]: [_facet("wide", obs=1, latest="2020")],
        }
    )

    assert order == ["wide", "late", "deep"]


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize(
    ("dfo_places", "expected"),
    [
        pytest.param(1, ["base", "dfo"], id="dfo-narrower-base-leads"),
        pytest.param(2, ["dfo", "base"], id="dfo-as-broad-dfo-leads"),
    ],
)
def test_scope_ranking_puts_dfo_first_only_when_it_covers_as_many_places(
    dfo_places: int, expected: list[str]
):
    base = _facet("base", obs=60, latest="2023")
    dfo = _facet("dfo", obs=5, latest="2023", provenance=_DFO_PROVENANCE)
    order = _scope_order(
        {place: [base, dfo] if i < dfo_places else [base] for i, place in enumerate(_PLACES[:2])}
    )

    assert order == expected


def test_scope_election_merges_each_place_once(monkeypatch: pytest.MonkeyPatch):
    """Electing across 40 places and two StatVars passes each probed point to a merge at most
    twice, once at its place and once across places, so the cost follows the points probed.
    The merged facet sums the counts and holds every point in place order."""
    places = [f"country/{i:03d}" for i in range(40)]
    dates = ("2019", "2020", "2021")

    def dated(facet_id: str, *, years: tuple[str, ...] = dates) -> Facet:
        return Facet(
            facet_id=facet_id,
            provenance_id=_BASE_PROVENANCE,
            obs_count=len(years),
            earliest_date=years[0],
            latest_date=years[-1],
            dates=years,
            values=tuple(float(y) for y in years),
        )

    # Two StatVars hold `wide` at every place; `deep` holds more points at half of them.
    probe = {
        place: [dated("wide"), dated("wide")]
        + ([dated("deep", years=("2017", "2018", *dates))] if i % 2 else [])
        for i, place in enumerate(places)
    }
    points_merged: list[int] = []
    merge_facets = facet_election.merge_facets

    def recording(facets):
        points_merged.append(sum(len(f.dates) for f in facets))
        return merge_facets(facets)

    monkeypatch.setattr(facet_election, "merge_facets", recording)

    candidates = merge_scope_candidates(probe, places)
    window = RequestedWindow(kind="range", start_year=2020, end_year=2021)
    ranked = rank_scope_candidates(candidates, stat_vars=[_SV], window=window)

    assert [c.facet_id for c in ranked] == ["wide", "deep"]
    probed = sum(len(f.dates) for facets in probe.values() for f in facets)
    assert sum(points_merged) <= 2 * probed
    wide = candidates["wide"].facet
    assert (wide.obs_count, wide.earliest_date, wide.latest_date) == (240, "2019", "2021")
    assert wide.dates == dates * 80
    assert wide.dates_complete
    assert wide.values == tuple(float(y) for y in dates) * 80
    assert candidates["wide"].breadth(window) == 40
    assert candidates["deep"].breadth(window) == 20


def _role_order(candidates: list) -> list[str]:
    merged = merge_role_candidates({_SV: candidates}, [_SV])
    return [c.facet_id for c in rank_role_candidates(merged, stat_vars=[_SV], window=None)]


def _pairs(*recipients: str, n_obs: int = 1) -> dict:
    return {(("creditor", "creditor/X"), ("debtor", recipient)): n_obs for recipient in recipients}


def test_role_ranking_leads_with_pairs_then_observations_then_recency():
    order = _role_order(
        [
            role_candidate(
                facet_id="wide",
                role_tuples=_pairs(*_PLACES),
                latest_date="2020",
                provenance_id=_BASE_PROVENANCE,
            ),
            role_candidate(
                facet_id="deep",
                role_tuples=_pairs(*_PLACES[:2], n_obs=40),
                latest_date="2020",
                provenance_id=_BASE_PROVENANCE,
            ),
            role_candidate(
                facet_id="late",
                role_tuples=_pairs(*_PLACES[:2], n_obs=40),
                latest_date="2024",
                provenance_id=_BASE_PROVENANCE,
            ),
        ]
    )

    assert order == ["wide", "late", "deep"]


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize(
    ("dfo_pairs", "expected"),
    [
        pytest.param(1, ["base", "dfo"], id="dfo-narrower-base-leads"),
        pytest.param(3, ["dfo", "base"], id="dfo-as-broad-dfo-leads"),
    ],
)
def test_role_ranking_puts_dfo_first_only_when_it_covers_as_many_pairs(
    dfo_pairs: int, expected: list[str]
):
    order = _role_order(
        [
            role_candidate(
                facet_id="base",
                role_tuples=_pairs(*_PLACES, n_obs=60),
                latest_date="2023",
                provenance_id=_BASE_PROVENANCE,
            ),
            role_candidate(
                facet_id="dfo",
                role_tuples=_pairs(*_PLACES[:dfo_pairs], n_obs=5),
                latest_date="2023",
                provenance_id=_DFO_PROVENANCE,
            ),
        ]
    )

    assert order == expected


# A recorded series at two places and the `Estimated` series continuing it at two others,
# under one provenance, beside another source's series at three of the four places.
_JOIN_PLACES = ["country/KEN", "country/GHA", "country/TGO", "country/NGA"]
_JOIN_WINDOW = RequestedWindow(kind="range", start_year=2015, end_year=2028)
_JOIN = {
    "rec": (None, "2010", "2020", _JOIN_PLACES[:2]),
    "est": ("Estimated", "2021", "2030", _JOIN_PLACES[2:]),
    "other": (None, "2010", "2025", _JOIN_PLACES[:3]),
}


def _join_provenance(facet_id: str) -> str:
    return "dc/base/Other" if facet_id == "other" else _BASE_PROVENANCE


def test_scope_ranking_counts_a_joined_candidate_at_every_place_any_of_its_facets_holds():
    """Under a 2015 to 2028 window `rec` and the estimates continuing it rank as one
    candidate. Each holds two places and `other` three, but joined they serve all four."""
    by_place = {
        place: [
            Facet(
                facet_id=facet_id,
                provenance_id=_join_provenance(facet_id),
                measurement_method=method,
                obs_count=1,
                earliest_date=earliest,
                latest_date=latest,
            )
            for facet_id, (method, earliest, latest, places) in _JOIN.items()
            if place in places
        ]
        for place in _JOIN_PLACES
    }
    merged = merge_scope_candidates(by_place, _JOIN_PLACES)
    ranked = rank_scope_candidates(merged, stat_vars=[_SV], window=_JOIN_WINDOW)

    assert [c.facet_id for c in ranked] == ["rec", "est", "other"]


@pytest.mark.parametrize(
    ("estimates_period", "continuing"),
    [
        pytest.param("P1Y", ("est",), id="same-cadence-continues"),
        pytest.param("P1M", (), id="another-cadence-does-not"),
    ],
)
def test_a_role_bulk_read_continues_its_series_only_at_its_cadence(
    estimates_period: str, continuing: tuple[str, ...]
):
    """Coverage states each facet's `observationPeriod`. Monthly estimates neither continue
    the annual `rec` nor join it in the ranking, as a mint of the same request reads them."""
    candidates = [
        role_candidate(
            facet_id=facet_id,
            role_tuples=_pairs(*places),
            latest_date=latest,
            provenance_id=_join_provenance(facet_id),
            measurement_method=method,
            observation_period=estimates_period if facet_id == "est" else "P1Y",
        )
        for facet_id, (method, _earliest, latest, places) in _JOIN.items()
    ]
    merged = merge_role_candidates({_SV: candidates}, [_SV])

    ranked = rank_role_candidates(merged, stat_vars=[_SV], window=_JOIN_WINDOW)

    assert role_continuation(merged["rec"], merged, window=_JOIN_WINDOW) == continuing
    assert (ranked[0].facet_id == "rec") == bool(continuing)


@dataclass(frozen=True)
class _CountingPairs(ListedRolePairs):
    """Listed pairs recording `facet_id` at each read of their tuples."""

    facet_id: str = ""
    reads: list[str] = field(default_factory=list)

    def head(self, limit, *, first_year, last_year, with_data=False):
        self.reads.append(self.facet_id)
        return super().head(limit, first_year=first_year, last_year=last_year, with_data=with_data)


@pytest.mark.parametrize(
    ("window", "order", "reads"),
    [
        pytest.param(
            _JOIN_WINDOW, ["rec", "est", "other"], ["est", "rec"], id="join-reads-its-pairs-once"
        ),
        pytest.param(None, ["other", "est", "rec"], [], id="no-join-reads-no-pairs"),
    ],
)
def test_role_ranking_reads_pairs_only_to_count_a_joined_candidate(
    window: RequestedWindow | None, order: list[str], reads: list[str]
):
    """Joined, `rec` and its estimates hold all four debtors against `other`'s three, so
    they lead; counting that reads each of their pairs once, however often the ranking asks.
    A ranking that forms no join reads no pairs."""
    read: list[str] = []
    candidates = []
    for facet_id, (method, _earliest, latest, places) in _JOIN.items():
        candidate = role_candidate(
            facet_id=facet_id,
            role_tuples=_pairs(*places),
            latest_date=latest,
            provenance_id=_join_provenance(facet_id),
            measurement_method=method,
            observation_period="P1Y",
        )
        assert isinstance(candidate.pairs, ListedRolePairs)
        candidates.append(
            replace(
                candidate,
                pairs=_CountingPairs(candidate.pairs.n_obs_by_role_tuple, facet_id, read),
            )
        )
    merged = merge_role_candidates({_SV: candidates}, [_SV])

    ranked = rank_role_candidates(merged, stat_vars=[_SV], window=window)

    assert [c.facet_id for c in ranked] == order
    assert sorted(read) == reads
