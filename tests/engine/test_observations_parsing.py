"""Pins facet parsing and entity selection against a synthetic graph."""

from __future__ import annotations

import pytest

from dc_graph.client import _parse_ordered_facets
from search.engine.region import _observation_entities
from search.engine.shape import ShapeDraft


def test_parse_ordered_facets_aligns_dates_and_values_missing_value_is_none():
    """A dated observation without a value contributes `None` at the matching index."""
    facets_raw = [
        {
            "facetId": "f1",
            "earliestDate": "2020",
            "latestDate": "2022",
            "obsCount": 3,
            "observations": [
                {"date": "2020", "value": 10.0},
                {"date": "2021"},
                {"date": "2022", "value": 12.0},
            ],
        }
    ]
    result = _parse_ordered_facets(facets_raw, {}, with_dates=True)
    assert len(result) == 1
    assert result[0].dates == ("2020", "2021", "2022")
    assert result[0].values == (10.0, None, 12.0)


def test_parse_ordered_facets_skips_observations_without_a_date():
    """An observation without a date is dropped from both aligned lists."""
    facets_raw = [
        {
            "obsCount": 1,
            "observations": [
                {"value": 5.0},
                {"date": "2020", "value": 10.0},
            ],
        }
    ]
    result = _parse_ordered_facets(facets_raw, {}, with_dates=True)
    assert result[0].dates == ("2020",)
    assert result[0].values == (10.0,)


def test_parse_ordered_facets_empty_observations_yields_empty_dates_and_values():
    facets_raw = [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 0}]
    result = _parse_ordered_facets(facets_raw, {}, with_dates=True)
    assert result[0].dates == ()
    assert result[0].values == ()


def test_parse_ordered_facets_reads_measurement_method_and_unit_from_facet_meta():
    """measurementMethod/unit come from the top-level facets map, keyed by facetId,
    the same lookup provenanceId already uses."""
    facets_raw = [{"facetId": "f1", "obsCount": 1}]
    top_facets_map = {
        "f1": {
            "provenanceId": "c/p/2",
            "measurementMethod": "GrantEquivalent",
            "unit": "USD",
        }
    }
    result = _parse_ordered_facets(facets_raw, top_facets_map, with_dates=True)
    assert result[0].measurement_method == "GrantEquivalent"
    assert result[0].unit == "USD"


@pytest.mark.parametrize(
    ("facets_raw", "top_facets_map", "expected_fields"),
    [
        pytest.param(
            [{"facetId": "f1", "obsCount": 1}],
            {"f1": {"provenanceId": "c/p/2"}},
            {"measurement_method": None, "unit": None},
            id="method-and-unit-absent",
        ),
        pytest.param(
            [{"obsCount": 1}],
            {},
            {
                "measurement_method": None,
                "unit": None,
                "provenance_id": None,
                "scaling_factor": None,
            },
            id="all-facet-metadata-absent",
        ),
        pytest.param(
            [{"facetId": "f1", "obsCount": 1}],
            {"f1": {"unit": "USD"}},
            {"scaling_factor": None},
            id="scaling-factor-absent",
        ),
    ],
)
def test_parse_ordered_facets_missing_facet_metadata(facets_raw, top_facets_map, expected_fields):
    result = _parse_ordered_facets(facets_raw, top_facets_map, with_dates=True)
    parsed = result[0]
    assert {field: getattr(parsed, field) for field in expected_fields} == expected_fields


def test_parse_ordered_facets_reads_scaling_factor_from_facet_meta():
    """scalingFactor comes from the top-level facets map, the same lookup
    measurementMethod/unit already use."""
    facets_raw = [{"facetId": "f1", "obsCount": 1}]
    top_facets_map = {"f1": {"unit": "USD", "scalingFactor": "100"}}
    result = _parse_ordered_facets(facets_raw, top_facets_map, with_dates=True)
    assert result[0].scaling_factor == "100"


def test_parse_ordered_facets_distinguishes_measurement_method_per_facet_id():
    """Each facet ID resolves to its own metadata entry."""
    facets_raw = [
        {"facetId": "f1", "obsCount": 29},
        {"facetId": "f2", "obsCount": 23},
    ]
    top_facets_map = {
        "f1": {"measurementMethod": "GrossDisbursement", "unit": "USD"},
        "f2": {"measurementMethod": "Commitment", "unit": "EUR"},
    }
    result = _parse_ordered_facets(facets_raw, top_facets_map, with_dates=True)
    assert result[0].measurement_method == "GrossDisbursement"
    assert result[0].unit == "USD"
    assert result[1].measurement_method == "Commitment"
    assert result[1].unit == "EUR"


def test_parse_ordered_facets_merges_blocks_sharing_a_facet_id():
    """Blocks sharing an ID sum counts, preserve observation order, and widen the date span."""
    facets_raw = [
        {
            "facetId": "A",
            "obsCount": 1,
            "earliestDate": "2021",
            "latestDate": "2021",
            "observations": [{"date": "2021", "value": 30.0}],
        },
        {
            "facetId": "A",
            "obsCount": 2,
            "earliestDate": "2019",
            "latestDate": "2020",
            "observations": [
                {"date": "2019", "value": 10.0},
                {"date": "2020", "value": 20.0},
            ],
        },
        {
            "facetId": "B",
            "obsCount": 1,
            "earliestDate": "2018",
            "latestDate": "2018",
            "observations": [{"date": "2018", "value": 9.0}],
        },
    ]
    top_facets_map = {
        "A": {"provenanceId": "prov/A", "measurementMethod": "Method", "unit": "USD"},
        "B": {"provenanceId": "prov/B", "measurementMethod": "Method", "unit": "EUR"},
    }

    result = _parse_ordered_facets(facets_raw, top_facets_map, with_dates=True)

    assert [f.facet_id for f in result] == ["A", "B"]
    merged_a = result[0]
    assert merged_a.obs_count == 3
    assert merged_a.earliest_date == "2019"
    assert merged_a.latest_date == "2021"
    # Input block order is preserved, so "2021" precedes "2019" and "2020".
    assert merged_a.dates == ("2021", "2019", "2020")
    assert merged_a.values == (30.0, 10.0, 20.0)
    assert len(merged_a.dates) == len(merged_a.values) == merged_a.obs_count
    assert result[1].obs_count == 1


def test_parse_ordered_facets_no_repeated_facet_id_in_result():
    """A facet list from `_parse_ordered_facets` carries each facetId at most once."""
    facets_raw = [
        {"facetId": "A", "obsCount": 1},
        {"facetId": "A", "obsCount": 1},
        {"facetId": "A", "obsCount": 1},
        {"facetId": "B", "obsCount": 1},
    ]
    result = _parse_ordered_facets(facets_raw, {}, with_dates=True)
    facet_ids = [f.facet_id for f in result]
    assert len(facet_ids) == len(set(facet_ids))
    assert result[0].obs_count == 3


def test_parse_ordered_facets_id_less_blocks_never_merge_with_each_other():
    """Blocks without `facetId` stay separate; `probe_facets` merges them across entities."""
    facets_raw = [
        {"obsCount": 1, "earliestDate": "2020", "latestDate": "2020"},
        {"obsCount": 1, "earliestDate": "2021", "latestDate": "2021"},
        {"facetId": "A", "obsCount": 1},
    ]
    result = _parse_ordered_facets(facets_raw, {}, with_dates=True)
    assert len(result) == 3
    assert [f.facet_id for f in result] == [None, None, "A"]
    assert result[0].obs_count == 1
    assert result[1].obs_count == 1


def test_parse_ordered_facets_merge_is_a_no_op_when_facet_ids_already_distinct():
    facets_raw = [
        {"facetId": "A", "obsCount": 5, "earliestDate": "2020", "latestDate": "2020"},
        {"facetId": "B", "obsCount": 7, "earliestDate": "2021", "latestDate": "2021"},
    ]
    result = _parse_ordered_facets(facets_raw, {}, with_dates=True)
    assert [f.facet_id for f in result] == ["A", "B"]
    assert [f.obs_count for f in result] == [5, 7]


# ---------------------------------------------------------------------------
# region/observations/entities.py: _observation_entities (the chosen entity-set selection)
# ---------------------------------------------------------------------------

_SUBJECT_SHAPE = ShapeDraft(
    shape_id="test_subject_shape",
    label="test flow",
    pop_type_dcid="Person",
    meas_prop_dcid="count",
    stat_type_dcid="measuredValue",
    meas_qual_dcid=None,
    meas_denom_dcid=None,
    slot_keys=(),
)


def test_observation_entities_standard_prefers_destination_dcids():
    """Region expansion (e.g. 'Africa' -> member countries) wins over the
    single destination_dcid when both are present."""
    result, warn = _observation_entities(
        _SUBJECT_SHAPE,
        destination_dcids=["a", "b"],
        destination_dcid="c",
    )
    assert result == {"observationAbout": ["a", "b"]}
    assert warn is None


def test_observation_entities_standard_falls_back_to_destination_dcid():
    result, warn = _observation_entities(
        _SUBJECT_SHAPE,
        destination_dcids=None,
        destination_dcid="country/KEN",
    )
    assert result == {"observationAbout": ["country/KEN"]}
    assert warn is None


def test_observation_entities_standard_empty_when_no_entity():
    """Variable disambiguation returns no entities without probing observations."""
    result, warn = _observation_entities(
        _SUBJECT_SHAPE,
        destination_dcids=None,
        destination_dcid=None,
    )
    assert result == {}
    assert warn is None
