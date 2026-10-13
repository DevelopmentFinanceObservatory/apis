"""Tests selector validation, series grouping, minted-id lookup, and window refusals."""

from __future__ import annotations

import pytest

import assembler.series_id as series_id_module
from assembler.errors import BindingInvalid, ResourceExhausted
from assembler.frame_view import compute_structural_view
from assembler.ports import FrameRef, ResolvedFrame
from assembler.resolve import _group_by_coordinate, resolve_selectors
from assembler.series_id import mint_series_id
from assembler.testing import FIXTURE_RETRIEVED_ON, build_resolved_frame

_FRAME_REF = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)


def _row(
    *,
    stat_var="dcid:X",
    recipient="country/ETH",
    period,
    value=1.0,
    unit=None,
    method=None,
    provenance="src_a",
):
    return {
        "stat_var": stat_var,
        "recipient": recipient,
        "time_period": period,
        "value": value,
        "unit": unit,
        "measurementMethod": method,
        "provenance": provenance,
    }


def _resolved_frame(rows, *, labels=None):
    return build_resolved_frame(
        frame_ref=_FRAME_REF, role_keys=("recipient",), rows=rows, labels=labels
    )


def test_resolves_one_series_by_bound_dimensions():
    rf = _resolved_frame([_row(period="2020"), _row(period="2021")])
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2021"},
    )
    assert len(series) == 1
    assert dict(series[0].coordinates)["recipient"] == "country/ETH"
    assert len(series[0].rows) == 2


def test_unbound_entity_dimension_expands_into_multiple_series():
    rows = [
        _row(recipient="country/ETH", period="2020"),
        _row(recipient="country/GHA", period="2020"),
    ]
    rf = _resolved_frame(rows)
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "stat_var", "values": ["dcid:X"]}]}],
        window={"start": "2020", "end": "2020"},
    )
    assert len(series) == 2
    recipients = {dict(s.coordinates)["recipient"] for s in series}
    assert recipients == {"country/ETH", "country/GHA"}


_ORDER_ROWS = [
    {**_row(recipient=recipient, period="2020", method=method), "stat_var": stat_var}
    for recipient in ("country/ETH", "country/GHA", "country/KEN")
    for method in ("m1", "m2")
    for stat_var in ("dcid:X", "dcid:Y")
]


@pytest.mark.parametrize(
    ("bindings", "expected"),
    [
        pytest.param(
            [
                {"dimension": "recipient", "values": ["country/KEN", "country/ETH", "country/GHA"]},
                {"dimension": "measurementMethod", "values": ["m1"]},
            ],
            [
                ("country/KEN", "m1", "dcid:X"),
                ("country/KEN", "m1", "dcid:Y"),
                ("country/ETH", "m1", "dcid:X"),
                ("country/ETH", "m1", "dcid:Y"),
                ("country/GHA", "m1", "dcid:X"),
                ("country/GHA", "m1", "dcid:Y"),
            ],
            id="one-list-unbound-dimension-keeps-frame-order",
        ),
        pytest.param(
            [
                {"dimension": "recipient", "values": ["country/GHA", "country/ETH"]},
                {"dimension": "measurementMethod", "values": ["m2", "m1"]},
                {"dimension": "stat_var", "values": ["dcid:X"]},
            ],
            [
                ("country/GHA", "m2", "dcid:X"),
                ("country/GHA", "m1", "dcid:X"),
                ("country/ETH", "m2", "dcid:X"),
                ("country/ETH", "m1", "dcid:X"),
            ],
            id="two-lists-in-binding-order",
        ),
    ],
)
def test_bound_series_follow_the_binding_list_order(bindings, expected):
    series = resolve_selectors(
        resolved_frame=_resolved_frame(_ORDER_ROWS),
        selectors=[{"bindings": bindings}],
        window=None,
    )
    assert [
        tuple(
            dict(one.coordinates)[name] for name in ("recipient", "measurementMethod", "stat_var")
        )
        for one in series
    ] == expected


def test_unknown_dimension_refuses_with_dimension_names_as_legal_values():
    rf = _resolved_frame([_row(period="2020")])
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "bogus", "values": ["x"]}]}],
            window=None,
        )
    exc = excinfo.value
    assert exc.dimension == "bogus"
    assert "recipient" in exc.legal_values
    assert exc.pointer == "/select/bogus"


def test_unknown_value_refuses_with_the_dimensions_legal_values():
    rf = _resolved_frame([_row(recipient="country/ETH", period="2020")])
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ZZZ"]}]}],
            window=None,
        )
    exc = excinfo.value
    assert exc.dimension == "recipient"
    assert exc.legal_values == ("country/ETH",)
    assert exc.pointer == "/select/recipient"


def test_unknown_value_refusal_carries_the_frames_labels():
    rows = [
        _row(recipient="country/ETH", period="2020"),
        _row(recipient="country/GHA", period="2020"),
    ]
    rf = _resolved_frame(rows, labels={"country/ETH": "Ethiopia"})
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ZZZ"]}]}],
            window=None,
        )
    exc = excinfo.value
    assert exc.labels == {"country/ETH": "Ethiopia"}
    assert set(exc.legal_values) == {"country/ETH", "country/GHA"}


def test_unknown_dimension_pointer_escapes_pointer_metacharacters():
    rf = _resolved_frame([_row(period="2020")])
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "foo/bar~baz", "values": ["x"]}]}],
            window=None,
        )
    assert excinfo.value.pointer == "/select/foo~1bar~0baz"


def test_unknown_value_pointer_escapes_pointer_metacharacters():
    rf = build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("foo/bar~baz",),
        rows=[
            {
                "stat_var": "dcid:X",
                "foo/bar~baz": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
        ],
    )
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "foo/bar~baz", "values": ["country/ZZZ"]}]}],
            window=None,
        )
    assert excinfo.value.pointer == "/select/foo~1bar~0baz"


def test_an_unbound_facet_splitting_one_series_refuses_and_carries_structural_view():
    rows = [
        _row(period="2020", unit="unit/USD"),
        _row(period="2020", unit="unit/EUR"),
    ]
    rf = _resolved_frame(rows)
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
            window=None,
        )
    exc = excinfo.value
    assert exc.dimension == "unit"
    assert set(exc.legal_values) == {"unit/USD", "unit/EUR"}
    assert exc.structural_view is rf.structural_view


def test_binding_the_splitting_facet_avoids_the_refusal():
    rows = [
        _row(period="2020", unit="unit/USD"),
        _row(period="2020", unit="unit/EUR"),
    ]
    rf = _resolved_frame(rows)
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[
            {
                "bindings": [
                    {"dimension": "recipient", "values": ["country/ETH"]},
                    {"dimension": "unit", "values": ["unit/USD"]},
                ]
            }
        ],
        window=None,
    )
    assert len(series) == 1
    assert dict(series[0].coordinates)["unit"] == "unit/USD"


@pytest.mark.parametrize(("facet", "row_key"), [("unit", "unit"), ("measurementMethod", "method")])
def test_an_unbound_facet_set_and_null_on_one_series_refuses(facet, row_key):
    """One StatVar and place under a set facet value and under a null one is a split."""
    rows = [_row(period="2020", **{row_key: "value/A"}), _row(period="2020")]
    rf = _resolved_frame(rows)
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "stat_var", "values": ["dcid:X"]}]}],
            window=None,
        )
    exc = excinfo.value
    assert exc.pointer == f"/select/{facet}"
    assert exc.dimension == facet
    assert exc.legal_values == ("value/A",)


def test_the_split_refusal_names_the_values_of_the_ambiguous_series_only():
    rows = [
        _row(recipient="country/ETH", period="2020", unit="unit/USD"),
        _row(recipient="country/ETH", period="2020", unit="unit/EUR"),
        _row(recipient="country/GHA", period="2020", unit="unit/GBP"),
    ]
    rf = _resolved_frame(rows)
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "stat_var", "values": ["dcid:X"]}]}],
            window=None,
        )
    assert excinfo.value.legal_values == ("unit/EUR", "unit/USD")


def test_series_whose_unbound_facets_differ_resolve_without_binding_them():
    """Issue #953: GDP in USDollar beside population with a null unit is two series, one
    unit each."""
    rows = [
        _row(stat_var="dcid:GDP", period="2020", unit="USDollar"),
        _row(stat_var="dcid:Population", period="2020"),
    ]
    rf = _resolved_frame(rows)
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[
            {"bindings": [{"dimension": "stat_var", "values": ["dcid:GDP", "dcid:Population"]}]}
        ],
        window=None,
    )
    assert {(dict(s.coordinates)["stat_var"], dict(s.coordinates)["unit"]) for s in series} == {
        ("dcid:GDP", "USDollar"),
        ("dcid:Population", None),
    }


def test_a_bound_facet_split_resolves_when_another_facet_varies_with_it():
    """Binding `unit` to two values asks for both series. Provenance that differs with the unit
    distinguishes nothing further, so it is not refused."""
    rows = [
        _row(period="2020", unit="unit/USD", provenance="src_a"),
        _row(period="2020", unit="unit/EUR", provenance="src_b"),
    ]
    rf = _resolved_frame(rows)
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "unit", "values": ["unit/USD", "unit/EUR"]}]}],
        window=None,
    )
    assert {dict(s.coordinates)["provenance"] for s in series} == {"src_a", "src_b"}


def test_unbound_facets_do_not_multiply_the_series_estimate():
    """12 StatVars at 2 places, each StatVar from one of 3 sources, is 24 series."""
    rows = [
        _row(
            stat_var=f"dcid:X{index:02d}",
            recipient=recipient,
            period="2020",
            provenance=f"src_{index % 3}",
        )
        for index in range(12)
        for recipient in ("country/ETH", "country/GHA")
    ]
    rf = _resolved_frame(rows)
    stat_vars = [f"dcid:X{index:02d}" for index in range(12)]
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "stat_var", "values": stat_vars}]}],
        window=None,
    )
    assert len(series) == 24


def test_empty_cross_product_refuses():
    # Both values are individually legal (each appears somewhere in the
    # frame) but never co-occur on the same row -- a genuinely empty
    # cross-product, distinct from an unknown-value refusal.
    rows = [
        _row(recipient="country/ETH", period="2020", method="m1"),
        _row(recipient="country/GHA", period="2020", method="m2"),
    ]
    rf = _resolved_frame(rows)
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[
                {
                    "bindings": [
                        {"dimension": "recipient", "values": ["country/ETH"]},
                        {"dimension": "measurementMethod", "values": ["m2"]},
                    ]
                }
            ],
            window=None,
        )
    assert excinfo.value.pointer == "/select"


def test_series_count_estimate_refuses_before_touching_frame_rows():
    # 26 recipients exceed the 24-series limit when recipient is unbound.
    # The estimate refuses from the structural view before grouping rows.
    rows = [_row(recipient=f"country/{i:03d}", period="2020") for i in range(26)]
    rf = _resolved_frame(rows)
    selectors = [{"bindings": [{"dimension": "stat_var", "values": ["dcid:X"]}]}]
    with pytest.raises(ResourceExhausted) as excinfo:
        resolve_selectors(resolved_frame=rf, selectors=selectors, window=None)
    assert excinfo.value.limit_name == "MAX_SERIES_PER_COMPONENT"


def test_series_count_estimate_runs_before_any_row_grouping(monkeypatch):
    """The structural estimate refuses before `_group_by_coordinate` reads frame rows."""
    import assembler.resolve as resolve_module

    def _group_by_coordinate_must_not_run(**kwargs):
        raise AssertionError("_group_by_coordinate must not run once the estimate refuses")

    monkeypatch.setattr(resolve_module, "_group_by_coordinate", _group_by_coordinate_must_not_run)

    rows = [_row(recipient=f"country/{i:03d}", period="2020") for i in range(26)]
    rf = _resolved_frame(rows)
    selectors = [{"bindings": [{"dimension": "stat_var", "values": ["dcid:X"]}]}]
    with pytest.raises(ResourceExhausted) as excinfo:
        resolve_selectors(resolved_frame=rf, selectors=selectors, window=None)
    assert excinfo.value.limit_name == "MAX_SERIES_PER_COMPONENT"


def test_resolve_by_id_finds_a_batch_extended_id_for_a_collision(monkeypatch):
    """An id selector finds a collision-extended id minted for the full batch."""
    # 20 recipients fit the 24-series cap but must collide across 16 tail values.
    monkeypatch.setattr(series_id_module, "_BASE_TAIL_LENGTH", 1)
    rows = [_row(recipient=f"country/{i:03d}", period="2020") for i in range(20)]
    rf = _resolved_frame(rows)

    all_series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "stat_var", "values": ["dcid:X"]}]}],
        window=None,
    )
    base_id_length = len("s_") + 1
    extended = next(s for s in all_series if len(s.id) > base_id_length)

    resolved = resolve_selectors(resolved_frame=rf, selectors=[{"id": extended.id}], window=None)
    assert len(resolved) == 1
    assert resolved[0].coordinates == extended.coordinates


def test_too_many_selectors_refuses():
    rf = _resolved_frame([_row(period="2020")])
    selectors = [{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}] * 17
    with pytest.raises(ResourceExhausted) as excinfo:
        resolve_selectors(resolved_frame=rf, selectors=selectors, window=None)
    assert excinfo.value.limit_name == "MAX_SELECTORS"


def test_selects_series_by_minted_id():
    rf = _resolved_frame([_row(recipient="country/ETH", period="2020")])
    coordinates = tuple(
        sorted(
            (
                ("stat_var", "dcid:X"),
                ("recipient", "country/ETH"),
                ("unit", None),
                ("measurementMethod", None),
                ("provenance", "src_a"),
            )
        )
    )
    target_id = mint_series_id(coordinates=coordinates)
    series = resolve_selectors(resolved_frame=rf, selectors=[{"id": target_id}], window=None)
    assert len(series) == 1
    assert series[0].id == target_id


def test_unknown_series_id_refuses():
    rf = _resolved_frame([_row(period="2020")])
    with pytest.raises(BindingInvalid):
        resolve_selectors(
            resolved_frame=rf, selectors=[{"id": "not_a_real_id_000000"}], window=None
        )


def test_window_with_no_observation_for_any_series_refuses():
    rf = _resolved_frame([_row(recipient="country/ETH", period="2020")])
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(
            resolved_frame=rf,
            selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
            window={"start": "2030", "end": "2035"},
        )
    assert excinfo.value.pointer == "/window"


def test_window_absent_defaults_to_frame_coverage():
    rf = _resolved_frame(
        [_row(recipient="country/ETH", period="2015"), _row(recipient="country/ETH", period="2024")]
    )
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window=None,
    )
    assert len(series) == 1


def test_duplicate_resolution_across_selectors_is_deduped():
    rf = _resolved_frame([_row(recipient="country/ETH", period="2020")])
    selectors = [
        {"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]},
        {"bindings": [{"dimension": "stat_var", "values": ["dcid:X"]}]},
    ]
    series = resolve_selectors(resolved_frame=rf, selectors=selectors, window=None)
    assert len(series) == 1


# -- Converted units (`unit` cells carrying `{currency, prices, base_year}`,
# rather than `None` or a bare unit dcid string) ----------------------------
#
# Built by hand, bypassing `build_resolved_frame`: `Frame.build` stores a
# converted unit cell as a `dfo_frame_grammar.grammar.UnitRequest` object
# (already hashable), not the plain JSON dict a real frame1 document
# actually carries once `UnitRequest.as_json()` has run -- the shape
# `LiveFrameStore.load`'s `json.loads` always yields, and the shape that
# used to crash `_group_by_coordinate` with `TypeError: unhashable type:
# 'dict'` the moment a coordinate tuple carrying it was used as a dict key
# (see `tests/assembler/test_frame_view.py`'s identical note).
_UNIT_COLUMNS = ("stat_var", "recipient", "unit", "time_period", "value")


def _unit_resolved_frame(rows):
    doc = {"columns": list(_UNIT_COLUMNS), "rows": [list(row) for row in rows]}
    return ResolvedFrame(
        frame_ref=_FRAME_REF,
        frame=doc,
        structural_view=compute_structural_view(frame=doc, continuation=None),
        manifest={},
        labels={},
        retrieved_on=FIXTURE_RETRIEVED_ON,
    )


def test_group_by_coordinate_normalises_a_converted_unit_cell_for_hashing():
    """The exact shape that used to raise `TypeError: unhashable type:
    'dict'`: a converted unit's resolved unit-request object, grouped
    directly through `_group_by_coordinate` rather than the full
    `resolve_selectors` path."""
    groups = _group_by_coordinate(
        columns=["stat_var", "recipient", "unit", "time_period", "value"],
        rows=[["dcid:X", "country/FRA", {"currency": "USD", "prices": "current"}, "2020", 1.0]],
        continuation=None,
    )
    assert len(groups) == 1
    (coordinates,) = groups
    assert dict(coordinates)["unit"] == "USD/current"


def test_group_by_coordinate_mixing_converted_and_unconverted_unit_does_not_collide():
    groups = _group_by_coordinate(
        columns=["stat_var", "recipient", "unit", "time_period", "value"],
        rows=[
            ["dcid:X", "country/ETH", "USDollar", "2020", 1.0],
            ["dcid:X", "country/ETH", {"currency": "USD", "prices": "current"}, "2020", 2.0],
        ],
        continuation=None,
    )
    assert len(groups) == 2
    units = {dict(coordinates)["unit"] for coordinates in groups}
    assert units == {"USDollar", "USD/current"}


def test_group_by_coordinate_two_rows_differing_only_by_base_year_are_distinct():
    groups = _group_by_coordinate(
        columns=["stat_var", "recipient", "unit", "time_period", "value"],
        rows=[
            [
                "dcid:X",
                "country/ETH",
                {"currency": "USD", "prices": "constant", "base_year": 2010},
                "2020",
                1.0,
            ],
            [
                "dcid:X",
                "country/ETH",
                {"currency": "USD", "prices": "constant", "base_year": 2015},
                "2020",
                2.0,
            ],
        ],
        continuation=None,
    )
    assert len(groups) == 2
    units = {dict(coordinates)["unit"] for coordinates in groups}
    assert units == {"USD/constant/2010", "USD/constant/2015"}


def test_resolve_selectors_collapses_two_rows_sharing_one_converted_unit_into_one_series():
    rf = _unit_resolved_frame(
        [
            ["dcid:X", "country/ETH", {"currency": "USD", "prices": "current"}, "2020", 1.0],
            ["dcid:X", "country/ETH", {"currency": "USD", "prices": "current"}, "2021", 2.0],
        ]
    )
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[{"bindings": [{"dimension": "recipient", "values": ["country/ETH"]}]}],
        window={"start": "2020", "end": "2021"},
    )
    assert len(series) == 1
    assert dict(series[0].coordinates)["unit"] == "USD/current"
    assert len(series[0].rows) == 2


def test_resolve_selectors_binds_a_converted_unit_facet_by_its_rendered_text():
    """A `unit` facet splitting one series into a bare string and a converted
    unit must be bound to avoid the split refusal, and the selector's
    binding value is the rendered text -- the same text `DimensionView.values`
    and this resolution's own coordinate both carry -- since `select`
    bindings only ever supply strings."""
    rf = _unit_resolved_frame(
        [
            ["dcid:X", "country/ETH", "USDollar", "2020", 1.0],
            ["dcid:X", "country/ETH", {"currency": "USD", "prices": "current"}, "2020", 2.0],
        ]
    )
    series = resolve_selectors(
        resolved_frame=rf,
        selectors=[
            {
                "bindings": [
                    {"dimension": "recipient", "values": ["country/ETH"]},
                    {"dimension": "unit", "values": ["USD/current"]},
                ]
            }
        ],
        window={"start": "2020", "end": "2020"},
    )
    assert len(series) == 1
    assert series[0].rows[0]["value"] == 2.0


# The Zambia frame Datasets served for IMF WEO gross national saving: reported values for
# 2015-2024 under a null method, continued by the source's estimates for 2025-2026.
_ZAMBIA_VALUES = {
    "2015": 40.116951,
    "2016": 34.94189,
    "2017": 39.320593,
    "2018": 37.288123,
    "2019": 39.738918,
    "2020": 44.035811,
    "2021": 43.267269,
    "2022": 30.727552,
    "2023": 28.407558,
    "2024": 26.742426,
    "2025": 29.551633,
    "2026": 34.819228,
}
_ZAMBIA_ROWS = [
    {
        "stat_var": "dcid:Saving_AsAFractionOf_GDP",
        "observationAbout": "country/ZMB",
        "time_period": period,
        "value": value,
        "unit": "Percent",
        "measurementMethod": "Estimated" if period >= "2025" else None,
        "provenance": "provenance/IMFWorldEconomicOutlook",
    }
    for period, value in _ZAMBIA_VALUES.items()
]
_CONTINUED_MANIFEST = {
    "schema_version": 1,
    "trace": {
        "continuations": [
            {"axis": "measurementMethod", "method": "Estimated", "periods": ["2025", "2026"]}
        ]
    },
}


def _zambia_frame(*, continued: bool) -> ResolvedFrame:
    return build_resolved_frame(
        frame_ref=_FRAME_REF,
        role_keys=("observationAbout",),
        rows=_ZAMBIA_ROWS,
        manifest=_CONTINUED_MANIFEST if continued else None,
    )


def _method_view(rf: ResolvedFrame) -> tuple[tuple[str, ...], bool]:
    """The frame view's `measurementMethod` values and whether it marks the axis plural, the
    signal the frames door gives a caller about which dimensions split series."""
    (method,) = (one for one in rf.structural_view.dimensions if one.name == "measurementMethod")
    return method.values, method.plural


def _zambia_select(**bound: str) -> list[dict[str, object]]:
    bindings = {"observationAbout": "country/ZMB", **bound}
    return [{"bindings": [{"dimension": k, "values": [v]} for k, v in bindings.items()]}]


def test_a_series_and_the_estimates_continuing_it_resolve_as_one_series():
    """A frame continued by its source's estimates (`trace.continuations`) draws all twelve
    years as one series, each row keeping its own method cell. The frame view lists
    `Estimated` as a value a select may bind and leaves the method axis not plural, so a
    caller is not told to bind it. Issue: the Zambia chart that drew only 2025-2026."""
    rf = _zambia_frame(continued=True)
    (series,) = resolve_selectors(resolved_frame=rf, selectors=_zambia_select(), window=None)

    assert rf.structural_view.series_count == 1
    assert _method_view(rf) == (("Estimated",), False)
    assert dict(series.coordinates)["measurementMethod"] is None
    assert [row["time_period"] for row in series.rows] == list(_ZAMBIA_VALUES)
    assert [row["measurementMethod"] for row in series.rows] == [None] * 10 + ["Estimated"] * 2


def test_binding_the_continuing_method_draws_the_estimates_alone():
    rf = _zambia_frame(continued=True)
    (series,) = resolve_selectors(
        resolved_frame=rf, selectors=_zambia_select(measurementMethod="Estimated"), window=None
    )

    assert dict(series.coordinates)["measurementMethod"] == "Estimated"
    assert [row["time_period"] for row in series.rows] == ["2025", "2026"]


def test_a_frame_without_continuations_splits_on_its_methods_as_before():
    rf = _zambia_frame(continued=False)

    assert rf.structural_view.series_count == 2
    assert _method_view(rf) == (("Estimated",), True)
    with pytest.raises(BindingInvalid) as excinfo:
        resolve_selectors(resolved_frame=rf, selectors=_zambia_select(), window=None)
    assert excinfo.value.pointer == "/select/measurementMethod"
    assert excinfo.value.legal_values == ("Estimated",)


def test_a_continued_series_resolves_by_the_id_minted_for_it():
    rf = _zambia_frame(continued=True)
    (series,) = resolve_selectors(resolved_frame=rf, selectors=_zambia_select(), window=None)

    (again,) = resolve_selectors(resolved_frame=rf, selectors=[{"id": series.id}], window=None)

    assert again.rows == series.rows
