"""Tests the preview-mode rule and the assembly from series in hand to `SpecObservations`."""

from __future__ import annotations

import pytest

from materialise.series_conversion import SeriesConversion
from search.observations import (
    PreviewMode,
    SeriesRead,
    assemble_per_series,
    assemble_role_bulk,
    assemble_scope_bulk,
    observations_too_large,
    preview_mode,
)

_POP = "Count_Person"
_AID = "DFO/ODA"
_KEN = (("observationAbout", "country/KEN"),)
_GHA = (("observationAbout", "country/GHA"),)
_NOR = (("observationAbout", "country/NOR"),)
_PROV = "dc/base/WorldBank"
_LABELS = {
    _POP: "Population",
    "country/KEN": "Kenya",
    "country/GHA": "Ghana",
    "country/NOR": "Norway",
    "country/USA": "United States",
    _PROV: "World Bank",
    "CurrencyEUR": "Euro",
    "USDollar": "US dollar",
    "gender-value/Female": "Female",
}


class _Labels:
    """A label source answering from `_LABELS` and recording each batch."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
        self.batches.append(list(dcids))
        return {dcid: _LABELS[dcid] for dcid in dcids if dcid in _LABELS}


def _read(*values: float | None, total: int | None = None, **kwargs) -> SeriesRead:
    points = [(str(2000 + i), value) for i, value in enumerate(values)]
    return SeriesRead(points=points, total_rows=len(points) if total is None else total, **kwargs)


def _conversion() -> SeriesConversion:
    return SeriesConversion(
        recipe_id="usd-to-eur",
        table_provenance_id=_PROV,
        exchange_rate_table_version="v1",
        deflator_table_version=None,
        fallback_count=1,
        fallback_examples=(("country/KEN", "Earth"),),
        projected_count=0,
        projected_examples=(),
    )


# ---------------------------------------------------------------------------
# preview_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role_bearing", "places_defaulted", "bound_pairs", "series", "expected", "counted"),
    [
        # A defaulted standard scope reads in bulk at any size.
        (False, True, 1, 999, PreviewMode.SCOPE_BULK, False),
        (False, False, 999, 2, PreviewMode.PER_SERIES, True),
        (False, False, 1, 3, PreviewMode.TOO_LARGE, True),
        # Defaulted places say nothing about a role-bearing answer.
        (True, True, 1, 1, PreviewMode.PER_SERIES, True),
        (True, False, 2, 2, PreviewMode.PER_SERIES, True),
        # Bound pairs over the cap decide without a count.
        (True, False, 3, 1, PreviewMode.ROLE_BULK, False),
        (True, False, 2, 3, PreviewMode.ROLE_BULK, True),
    ],
)
def test_preview_mode_precedence(
    search_settings,
    role_bearing: bool,
    places_defaulted: bool,
    bound_pairs: int,
    series: int,
    expected: PreviewMode,
    counted: bool,
) -> None:
    search_settings(obs_preview_max_pairs=2)
    counts: list[int] = []

    def series_count() -> int:
        counts.append(series)
        return series

    mode = preview_mode(
        role_bearing=role_bearing,
        places_defaulted=places_defaulted,
        bound_pairs=bound_pairs,
        series_count=series_count,
    )
    assert (mode, bool(counts)) == (expected, counted)


def test_too_large_names_the_count_and_the_cap(search_settings) -> None:
    search_settings(obs_preview_max_pairs=4)
    omitted = observations_too_large(9).omitted
    assert omitted is not None
    assert (omitted.reason, omitted.pairs, omitted.cap) == ("observations_preview_too_large", 9, 4)


# ---------------------------------------------------------------------------
# assemble_per_series
# ---------------------------------------------------------------------------


def test_per_series_ranks_by_latest_value_and_labels_from_one_batch() -> None:
    labels = _Labels()
    observations = assemble_per_series(
        {
            (_POP, _KEN): _read(5.0, 1.0, facets={"provenance": _PROV, "unit": "USDollar"}),
            (_POP, _GHA): _read(1.0, 9.0),
            (_POP, _NOR): _read(3.0, None),
        },
        label_source=labels,
        member_constraints={},
        target_currency_ref=None,
    )
    assert [s.entities[0].label for s in observations.series] == ["Ghana", "Kenya", "Norway"]
    kenya = observations.series[1]
    assert kenya.stat_var.label == "Population"
    assert {axis: ref.label for axis, ref in kenya.facet.items()} == {
        "provenance": "World Bank",
        "unit": "US dollar",
    }
    assert [ref.label for ref in kenya.source_refs] == ["World Bank"]
    assert (observations.total_obs, observations.truncated) == (6, False)
    assert len(labels.batches) == 1


def test_per_series_labels_a_member_by_its_constraint_values() -> None:
    observations = assemble_per_series(
        {(_POP, _KEN): _read(1.0)},
        label_source=_Labels(),
        member_constraints={_POP: {"gender": "gender-value/Female"}},
        target_currency_ref=None,
    )
    assert observations.series[0].stat_var.label == "Female"


@pytest.mark.parametrize("assemble", [assemble_per_series, assemble_role_bulk, assemble_scope_bulk])
@pytest.mark.parametrize(
    ("target", "unit"), [("CurrencyEUR", ("CurrencyEUR", "Euro")), (None, None)]
)
def test_a_converted_series_shows_the_target_unit_and_its_conversion(
    assemble, target: str | None, unit: tuple[str, str] | None
) -> None:
    observations = assemble(
        {(_POP, _KEN): _read(1.0, facets={"unit": "USDollar"}, conversion=_conversion())},
        label_source=_Labels(),
        member_constraints={},
        target_currency_ref=target,
    )
    (series,) = observations.series
    shown = series.facet.get("unit")
    assert ((shown.ref, shown.label) if shown is not None else None) == unit
    assert series.conversion is not None
    assert series.conversion.table_provenance.label == "World Bank"
    assert [(e.entity, e.fallback_entity) for e in series.conversion.fallback_examples] == [
        ("country/KEN", "Earth")
    ]


def test_per_series_over_the_inline_cap_keeps_each_series_latest_points(search_settings) -> None:
    search_settings(obs_inline_max=3, obs_preview_points=2)
    observations = assemble_per_series(
        {(_POP, _KEN): _read(1.0, 2.0, 3.0), (_POP, _GHA): _read(4.0)},
        label_source=_Labels(),
        member_constraints={},
        target_currency_ref=None,
    )
    assert [[p.value for p in s.points] for s in observations.series] == [[4.0], [2.0, 3.0]]
    assert [s.total_obs for s in observations.series] == [1, 3]
    assert (observations.total_obs, observations.truncated) == (4, True)


def test_a_row_limited_read_marks_the_preview_truncated() -> None:
    observations = assemble_per_series(
        {(_POP, _KEN): _read(1.0, total=5)},
        label_source=_Labels(),
        member_constraints={},
        target_currency_ref=None,
    )
    assert (len(observations.series[0].points), observations.total_obs) == (1, 5)
    assert observations.truncated


def test_a_label_source_error_propagates() -> None:
    class _Failing:
        def node_labels_batch(self, dcids: list[str]) -> dict[str, str]:
            raise RuntimeError("graph down")

    with pytest.raises(RuntimeError, match="graph down"):
        assemble_per_series(
            {(_POP, _KEN): _read(1.0)},
            label_source=_Failing(),
            member_constraints={},
            target_currency_ref=None,
        )


# ---------------------------------------------------------------------------
# assemble_role_bulk
# ---------------------------------------------------------------------------


def _flow(recipient: str) -> tuple[tuple[str, str], ...]:
    return (("provider", "country/USA"), ("recipient", recipient))


def test_role_bulk_shows_the_first_series_by_key_and_counts_every_row(search_settings) -> None:
    search_settings(role_bulk_series_max=2)
    labels = _Labels()
    observations = assemble_role_bulk(
        {
            (_AID, _flow("country/NOR")): _read(1.0, 2.0),
            (_AID, _flow("country/GHA")): _read(1.0),
            (_AID, _flow("country/KEN")): _read(99.0),
        },
        label_source=labels,
        member_constraints={},
        unread_obs=10,
    )
    assert [[e.label for e in s.entities] for s in observations.series] == [
        ["United States", "Ghana"],
        ["United States", "Kenya"],
    ]
    assert [e.role for e in observations.series[0].entities] == ["origin", "destination"]
    assert "country/NOR" not in labels.batches[0]
    assert (observations.total_obs, observations.truncated) == (14, False)


def test_role_bulk_over_the_inline_cap_keeps_each_series_latest_points(search_settings) -> None:
    search_settings(obs_inline_max=2, obs_preview_points=1)
    observations = assemble_role_bulk(
        {(_AID, _flow("country/KEN")): _read(1.0, 2.0, 3.0)},
        label_source=_Labels(),
        member_constraints={},
    )
    assert [p.value for p in observations.series[0].points] == [3.0]
    assert observations.truncated


# ---------------------------------------------------------------------------
# assemble_scope_bulk
# ---------------------------------------------------------------------------


def test_scope_bulk_shows_each_place_latest_point_ranked_by_value() -> None:
    observations = assemble_scope_bulk(
        {
            (_POP, _KEN): _read(50.0, 2.0, total=7),
            (_POP, _GHA): _read(1.0, 8.0, total=3),
        },
        label_source=_Labels(),
        member_constraints={},
    )
    assert [(s.entities[0].label, [p.value for p in s.points]) for s in observations.series] == [
        ("Ghana", [8.0]),
        ("Kenya", [2.0]),
    ]
    assert (observations.total_obs, observations.truncated) == (10, True)


def test_scope_bulk_is_not_truncated_when_every_place_holds_one_row() -> None:
    observations = assemble_scope_bulk(
        {(_POP, _KEN): _read(2.0), (_POP, _GHA): _read(8.0)},
        label_source=_Labels(),
        member_constraints={},
    )
    assert (observations.total_obs, observations.truncated) == (2, False)
