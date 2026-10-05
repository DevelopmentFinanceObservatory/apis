"""Tests observation row labels and descending latest-value ordering across preview paths."""

from __future__ import annotations

import statvar_members
from dc_sdmx import SdmxObservation
from search.engine.clients.graph import Facet
from search.engine.retrieval.member_facts import ArcsMemberFacts
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._harness import (
    HarnessScope,
    build_observations_planned,
    bulk_scope_series_planned,
    materialise_double_from_obs,
)
from tests.fixtures import FakeGraph

# ---------------------------------------------------------------------------
# Rows use member facts, rank by latest value, and keep points chronological.
# ---------------------------------------------------------------------------


def test_group_shaped_preview_labels_only_from_member_facts_never_enumerate_members(
    monkeypatch,
):
    """Group labels use `member_facts` without enumerating members.
    Uncovered labels fall back to the graph."""

    def _unreachable(*args: object, **kwargs: object) -> None:
        raise AssertionError("enumerate_members must not be called on the standard preview path")

    monkeypatch.setattr(statvar_members, "enumerate_members", _unreachable)

    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Flow_Male|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 60.0}],
                }
            ],
            "Flow_Female|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 70.0}],
                }
            ],
        }
    )
    graph = FakeGraph(nodes={"gender-value/Male": {"label": "Male"}})

    observations, warn = build_observations_planned(
        ["Flow_Male", "Flow_Female"],
        {"observationAbout": ["country/KEN"]},
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        # member_facts covers ONLY Flow_Male -- Flow_Female sits outside it.
        member_facts={
            "Flow_Male": ArcsMemberFacts(
                arcs={
                    "constraintProperties": {"nodes": [{"dcid": "gender"}]},
                    "gender": {"nodes": [{"dcid": "gender-value/Male"}]},
                }
            )
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    labels_by_sv = {s.stat_var.ref: s.stat_var.label for s in observations.series}
    assert labels_by_sv == {"Flow_Male": "Male", "Flow_Female": None}
    # Female's higher latest value ranks it first, independent of its label.
    assert [s.stat_var.ref for s in observations.series] == ["Flow_Female", "Flow_Male"]


def test_confirmed_non_group_shape_labels_every_row_from_member_facts():
    """A non-group (single confirmed StatVar) spec labels every row from
    `member_facts` directly."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Flow_Male|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 60.0}],
                }
            ],
            "Flow_Female|country/KEN": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 70.0}],
                }
            ],
        }
    )
    graph = FakeGraph(
        nodes={"gender-value/Male": {"label": "Male"}, "gender-value/Female": {"label": "Female"}}
    )

    def _facts(value_dcid: str) -> ArcsMemberFacts:
        return ArcsMemberFacts(
            arcs={
                "constraintProperties": {"nodes": [{"dcid": "gender"}]},
                "gender": {"nodes": [{"dcid": value_dcid}]},
            }
        )

    observations, warn = build_observations_planned(
        ["Flow_Male", "Flow_Female"],
        {"observationAbout": ["country/KEN"]},
        graph=graph,
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        member_facts={
            "Flow_Male": _facts("gender-value/Male"),
            "Flow_Female": _facts("gender-value/Female"),
        },
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    labels_by_sv = {s.stat_var.ref: s.stat_var.label for s in observations.series}
    assert labels_by_sv == {"Flow_Male": "Male", "Flow_Female": "Female"}


def test_per_pair_rows_order_by_latest_value_descending_no_data_pinned_last():
    """Rows rank by latest value, with no-data last and points chronological within each series."""
    materialise_graph, materialise_sdmx = materialise_double_from_obs(
        {
            "Count_Person|country/AAA": [
                {
                    "obsCount": 2,
                    "provenanceId": "dc/base/Src",
                    "observations": [
                        {"date": "2020", "value": 5.0},
                        {"date": "2015", "value": 1.0},
                    ],
                }
            ],
            # The facet's metadata counts a point the read returns no row for.
            "Count_Person|country/BBB": [{"obsCount": 1, "provenanceId": "dc/base/Src"}],
            "Count_Person|country/CCC": [
                {
                    "obsCount": 1,
                    "provenanceId": "dc/base/Src",
                    "observations": [{"date": "2020", "value": 10.0}],
                }
            ],
        }
    )
    observations, warn = build_observations_planned(
        ["Count_Person"],
        {"observationAbout": ["country/AAA", "country/BBB", "country/CCC"]},
        graph=FakeGraph(),
        materialise_graph=materialise_graph,
        materialise_sdmx=materialise_sdmx,
        coverage=empty_coverage_snapshot(),
    )
    assert warn is None
    assert observations is not None
    ordered_entities = [s.entities[0].ref for s in observations.series]
    # CCC (10.0) > AAA (latest point 2020 -> 5.0) > BBB (no data, pinned last).
    assert ordered_entities == ["country/CCC", "country/AAA", "country/BBB"]
    aaa_series = next(s for s in observations.series if s.entities[0].ref == "country/AAA")
    assert [p.date for p in aaa_series.points] == ["2015", "2020"]


# ---------------------------------------------------------------------------
# _bulk_scope_series: the same label composition and rank key, direct.
# ---------------------------------------------------------------------------

_SCOPE_SV = "Count_Person"
_SCOPE_GRAPH = FakeGraph(
    nodes={
        "gender-value/Male": {"label": "Male"},
        "gender-value/Female": {"label": "Female"},
        "country/AAA": {"label": "Country AAA"},
        "country/BBB": {"label": "Country BBB"},
    }
)


def test_bulk_scope_rows_order_by_latest_value_descending_and_labels_from_constraints():
    def _obs(entity: str, value: str) -> SdmxObservation:
        return SdmxObservation(
            variable=_SCOPE_SV,
            role_values={"observationAbout": entity},
            unit=None,
            measurement_method=None,
            observation_period=None,
            provenance=None,
            time_period="2020",
            value=value,
            scaling_factor=None,
            facet_id="F1",
        )

    facets_by_entity = {
        e: [Facet(earliest_date="2020", latest_date="2020", obs_count=1, facet_id="F1")]
        for e in ("country/AAA", "country/BBB")
    }
    rows = {
        _SCOPE_SV: {
            "country/AAA": [_obs("country/AAA", "3.0")],
            "country/BBB": [_obs("country/BBB", "9.0")],
        }
    }
    sdmx = FakeSdmxClient(subject_observations=rows)
    scope = HarnessScope(
        places=("country/AAA", "country/BBB"),
        sdmx=sdmx,
        facets_by_sv_entity={_SCOPE_SV: facets_by_entity},
    )

    observations, warn = bulk_scope_series_planned(
        [_SCOPE_SV],
        ["country/AAA", "country/BBB"],  # Enumeration order differs from value order.
        scope=scope,
        graph=_SCOPE_GRAPH,
        facet_filter=None,
        member_constraints={_SCOPE_SV: {"gender": "gender-value/Male"}},
    )
    assert warn is None
    assert observations is not None
    # BBB (9.0) outranks AAA (3.0), even though AAA was enumerated first.
    assert [s.entities[0].ref for s in observations.series] == ["country/BBB", "country/AAA"]
    assert observations.series[0].stat_var.label == "Male"
