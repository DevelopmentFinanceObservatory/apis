"""The bulk scope and bulk role previews rank with `materialise.facets.rank_pool`.

Outside the origin catalogue's overlap set the order is the one each bulk path used before:
series covered, then observations, then recency, then provenance, method, unit and facet id.
Inside it, DFO's facet leads, ahead of breadth, as it does for a single member.
"""

from __future__ import annotations

from collections.abc import Callable

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
)
from tests.engine._role_candidates import role_candidate
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
def test_scope_ranking_puts_dfo_first_ahead_of_places_covered():
    order = _scope_order(
        {
            _PLACES[0]: [
                _facet("base", obs=60, latest="2023"),
                _facet("dfo", obs=5, latest="2023", provenance=_DFO_PROVENANCE),
            ],
            _PLACES[1]: [_facet("base", obs=60, latest="2023")],
        }
    )

    assert order == ["dfo", "base"]


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
def test_role_ranking_puts_dfo_first_ahead_of_pairs_covered():
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
                role_tuples=_pairs(_PLACES[0], n_obs=5),
                latest_date="2023",
                provenance_id=_DFO_PROVENANCE,
            ),
        ]
    )

    assert order == ["dfo", "base"]
