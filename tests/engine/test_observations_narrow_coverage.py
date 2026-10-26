"""Tests that a facet filter narrows the read plan and coverage counts what it serves."""

from __future__ import annotations

from search.engine.clients.graph import Facet
from search.engine.retrieval.coverage import CoverageForm, answer_coverage
from search.engine.retrieval.read_plan import plan_standard, served_provenances
from search.models import (
    BreadthDim,
    CoverageExact,
    TimeWindow,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._harness import build_observations_planned, dimensions_of
from tests.engine._observations_doubles import _MULTI_AXIS_SV, _facet_axis_materialise_double
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# A facet filter narrows the read plan; place-counted coverage counts the places the
# narrowed plan serves, and the scope's members are the places with an admitted facet.
# ---------------------------------------------------------------------------


def _entity_facet(
    *,
    measurement_method: str,
    obs_count: int = 1,
    provenance_id: str | None = "prov-a",
    facet_id: str = "f1",
) -> Facet:
    return Facet(
        facet_id=facet_id,
        earliest_date="2020",
        latest_date="2020",
        obs_count=obs_count,
        dates=["2020"] * obs_count,
        values=[1.0] * obs_count,
        measurement_method=measurement_method,
        provenance_id=provenance_id,
    )


def _wanted(_sv: str, facet) -> bool:
    return facet.measurement_method == "Wanted"


def test_a_filtered_place_count_counts_the_places_the_narrowed_plan_serves():
    """Unfiltered, the variable serves `f2`, the facet covering all three places. A filter
    admitting `Wanted` leaves `f1` at two places; Country C, whose only facet it removes,
    drops out."""
    unwanted = _entity_facet(measurement_method="Unwanted", provenance_id="prov-b", facet_id="f2")
    series = {
        ("sv", "country/A"): [_entity_facet(measurement_method="Wanted"), unwanted],
        ("sv", "country/B"): [_entity_facet(measurement_method="Wanted"), unwanted],
        ("sv", "country/C"): [unwanted],
    }
    plan = plan_standard(series, members=["sv"], years=None)
    form = CoverageForm(facet_label="countries", counts_places=True)

    assert dimensions_of(answer_coverage(plan, form))[0] == BreadthDim(label="countries", count=3)
    assert dimensions_of(answer_coverage(plan.narrowed(_wanted), form))[0] == BreadthDim(
        label="countries", count=2
    )


def test_a_filtered_source_count_recounts_the_served_group():
    """Unfiltered, the three variables serve `f3`, the facet every one of them holds. Once
    the filter drops it, `sv-a` and `sv-b` serve `f1` from another source, and `sv-c`,
    whose only facet it removes, serves nothing: the count reads the group served."""
    unwanted = _entity_facet(
        measurement_method="Unwanted", obs_count=3, provenance_id="prov-c", facet_id="f3"
    )
    wanted = _entity_facet(measurement_method="Wanted", provenance_id="prov-a")
    series = {
        ("sv-a", "country/A"): [wanted, unwanted],
        ("sv-b", "country/A"): [wanted, unwanted],
        ("sv-c", "country/A"): [unwanted],
    }
    plan = plan_standard(series, members=["sv-a", "sv-b", "sv-c"], years=None)
    narrowed = plan.narrowed(_wanted)
    form = CoverageForm(variables=True)

    assert (served_provenances(plan), served_provenances(narrowed)) == (("prov-c",), ("prov-a",))
    assert dimensions_of(answer_coverage(plan, form)) == [
        BreadthDim(label="sources", count=1),
        BreadthDim(label="observations", count=1),
        BreadthDim(label="variables", count=3),
    ]
    assert dimensions_of(answer_coverage(narrowed, form)) == [
        BreadthDim(label="sources", count=1),
        BreadthDim(label="observations", count=1),
        BreadthDim(label="variables", count=2),
    ]


def test_a_filtered_window_count_sets_has_data_from_the_recount():
    """A narrowed exact count answers the plan's window, not lifetime availability."""
    plan = plan_standard(
        {("sv", "country/A"): [_entity_facet(measurement_method="Wanted")]},
        members=["sv"],
        years=TimeWindow(start_year=2024, end_year=2024),
    )

    narrowed = answer_coverage(plan.narrowed(_wanted), CoverageForm())

    assert isinstance(narrowed, CoverageExact)
    assert (narrowed.observation_count, narrowed.has_data) == (0, False)


def test_facet_map_labels_share_the_existing_label_batch_zero_new_round_trips():
    """Method and unit labels share the existing `node_labels_batch` call."""
    graph = FakeGraph(
        nodes={
            "GrossDisbursement": {"label": "gross disbursements"},
            "USD": {"label": "US dollars"},
        },
        detect={},
        resolve={},
    )
    materialise_graph, materialise_sdmx = _facet_axis_materialise_double()
    calls: list[list[str]] = []
    original = graph.node_labels_batch

    def _counting(dcids: list[str]) -> dict[str, str]:
        calls.append(list(dcids))
        return original(dcids)

    graph.node_labels_batch = _counting  # type: ignore[method-assign]

    observations, warn = build_observations_planned(
        [_MULTI_AXIS_SV],
        {"observationAbout": ["country/USA"]},
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    assert len(calls) == 1
    series = observations.series[0]
    assert series.facet["measurementMethod"].ref == "GrossDisbursement"
    assert series.facet["measurementMethod"].label == "gross disbursements"
    assert series.facet["unit"].ref == "USD"
    assert series.facet["unit"].label == "US dollars"
