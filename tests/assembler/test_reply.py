"""Pin changed-field disclosure and reply summaries, including units, sources, and token caps."""

from __future__ import annotations

import dataclasses

import pytest

from assembler.bind import bind_all
from assembler.mint import default_window, mint_component, stamp_envelope
from assembler.reply import (
    WriteReply,
    build_changed_fields,
    build_fact_summary,
    build_reply,
    series_entry,
)
from assembler.resolve import resolve_selectors
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
)


def _selectors():
    return [
        {
            "bindings": [
                {"dimension": "stat_var", "values": ["dcid:MalariaGrantDisbursements"]},
                {"dimension": "recipient", "values": ["country/ETH"]},
            ]
        }
    ]


_COMPONENT_ID = "cmp_0123456789abcdef"


def _envelope(minted, *, release):
    """Stamp the envelope that `build_reply` reads."""
    return stamp_envelope(minted=minted, component_id=_COMPONENT_ID, revision=1, release=release)


def _select():
    return {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"}


def _series_coordinates(series):
    """Build stored `{id, coordinates}` entries without the write pipeline."""
    return tuple({"id": one.id, "coordinates": dict(one.coordinates)} for one in series)


def _resolve_and_bind(*, window=None):
    resolved_frame = malaria_ethiopia_resolved_frame()
    series = resolve_selectors(resolved_frame=resolved_frame, selectors=_selectors(), window=window)
    effective = window or default_window(resolved_frame=resolved_frame)
    bound = bind_all(resolved_frame=resolved_frame, series=series, window=effective)
    return resolved_frame, series, bound


# -- build_changed_fields -----------------------------------------------------


def test_scalar_fields_carry_prior_and_new():
    changed = build_changed_fields(
        current_intent={"title": "old", "window": {"start": "2015", "end": "2024"}},
        patch={"title": "new"},
    )
    assert changed == (type(changed[0])(field="title", prior="old", new="new"),)


def test_select_field_carries_name_only():
    """Object-valued selections disclose the field name without prior or new values."""
    changed = build_changed_fields(
        current_intent={"select": {"recipient": "country/ETH"}},
        patch={"select": {"recipient": ["country/ETH", "country/GHA"]}},
    )
    assert len(changed) == 1
    assert changed[0].field == "select"
    assert changed[0].disclosed is False
    # The serializer omits `prior` and `new` for undisclosed fields. These `None` values stay off
    # the wire.
    assert changed[0].prior is None
    assert changed[0].new is None


def test_create_carries_no_changed_fields():
    assert build_changed_fields(current_intent=None, patch={"title": "New title"}) == ()


def test_cleared_scalar_field_carries_the_genuine_prior_and_new_none():
    changed = build_changed_fields(current_intent={"title": "Old title"}, patch={"title": None})
    # A cleared scalar discloses its real `None` value.
    assert changed[0].disclosed is True
    assert changed[0].prior == "Old title"
    assert changed[0].new is None


def test_window_field_cleared_to_null_carries_name_only():
    changed = build_changed_fields(
        current_intent={"window": {"start": "2015", "end": "2024"}}, patch={"window": None}
    )
    # Object fields disclose by name, even when cleared with `null`.
    assert changed[0].disclosed is False
    assert changed[0].prior is None
    assert changed[0].new is None


# -- build_reply / fact summary -----------------------------------------------


def test_fact_summary_series_shape_and_no_cut_below_cap(release):
    resolved_frame, series, bound = _resolve_and_bind(window={"start": "2015", "end": "2024"})
    minted = mint_component(
        member="line_chart",
        merged_intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": resolved_frame.frame_ref.spec_id,
                "content_version": resolved_frame.frame_ref.content_version,
            },
            "select": _select(),
            "window": {"start": "2015", "end": "2024"},
        },
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
    )
    reply = build_reply(
        composition_id=None,
        composition_revision=None,
        changed=(),
        resolved_payload=_envelope(minted, release=release),
        series_coordinates=_series_coordinates(series),
        formatter_profile=release.formatter_profile,
    )
    summary = reply.fact_summary
    assert summary["series_cut"] == 0
    assert len(summary["series"]) == 1
    entry = summary["series"][0]
    assert entry["id"] == bound[0].id
    assert entry["label"] == "Ethiopia"
    assert entry["coordinates"] == dict(series[0].coordinates)
    assert entry["point_count"] == 9  # ten periods, one ("2018") missing
    assert entry["first_value"] == "412"
    assert entry["last_value"] == "442"
    assert summary["window"] == {"start": "2015", "end": "2024"}
    assert summary["change"] is None
    assert summary["source_refs"] is None


def _ethiopia_resolved_frame_with_unit(unit):
    rows = tuple(
        {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
            "time_period": period,
            "value": value,
            "unit": unit,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
        }
        for period, value in {"2020": 100.0, "2021": 110.0, "2022": 120.0}.items()
    )
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=rows,
        labels={"country/ETH": "Ethiopia"},
    )


def test_table_and_line_chart_fact_summary_unit_agree_for_the_same_series(release):
    """Table and line-chart summaries report the same unit for identical series (#585)."""
    resolved_frame = _ethiopia_resolved_frame_with_unit("Year")
    window = {"start": "2020", "end": "2022"}
    series = resolve_selectors(resolved_frame=resolved_frame, selectors=_selectors(), window=window)
    bound = bind_all(resolved_frame=resolved_frame, series=series, window=window)

    def _fact_summary(member):
        minted = mint_component(
            member=member,
            merged_intent={
                "type": member,
                "frame_ref": {
                    "spec_id": resolved_frame.frame_ref.spec_id,
                    "content_version": resolved_frame.frame_ref.content_version,
                },
                "select": _select(),
                "window": window,
            },
            resolved_frame=resolved_frame,
            series=series,
            bound_series=bound,
            release=release,
        )
        reply = build_reply(
            composition_id=None,
            composition_revision=None,
            changed=(),
            resolved_payload=_envelope(minted, release=release),
            series_coordinates=_series_coordinates(series),
            formatter_profile=release.formatter_profile,
        )
        return reply.fact_summary

    line_chart_unit = _fact_summary("line_chart")["unit"]
    assert line_chart_unit == "Year"
    assert _fact_summary("table")["unit"] == line_chart_unit


@pytest.mark.parametrize(
    ("labels", "expected_unit"),
    [
        pytest.param({"CurrencyUSD": "US Dollar"}, "US Dollar", id="labelled-unit-ref"),
        pytest.param({}, "CurrencyUSD", id="unlabelled-unit-ref-stays-raw"),
    ],
)
def test_stat_card_and_its_fact_summary_carry_the_unit_label(release, labels, expected_unit):
    resolved_frame = dataclasses.replace(
        _ethiopia_resolved_frame_with_unit("CurrencyUSD"), labels=labels
    )
    window = {"start": "2020", "end": "2022"}
    series = resolve_selectors(resolved_frame=resolved_frame, selectors=_selectors(), window=window)
    bound = bind_all(resolved_frame=resolved_frame, series=series, window=window)
    minted = mint_component(
        member="stat_card",
        merged_intent={
            "type": "stat_card",
            "frame_ref": {
                "spec_id": resolved_frame.frame_ref.spec_id,
                "content_version": resolved_frame.frame_ref.content_version,
            },
            "select": _select(),
            "window": window,
        },
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
    )
    reply = build_reply(
        composition_id=None,
        composition_revision=None,
        changed=(),
        resolved_payload=_envelope(minted, release=release),
        series_coordinates=_series_coordinates(series),
        formatter_profile=release.formatter_profile,
    )

    assert minted.component["unit"] == expected_unit
    assert reply.fact_summary["unit"] == expected_unit


@pytest.mark.parametrize(
    ("first_unit", "second_unit", "expected"),
    [
        pytest.param(None, None, None, id="no-unit-control"),
        pytest.param("Year", "Number", None, id="different-string-units"),
        pytest.param(
            {"currency": "USD", "magnitude": "millions"},
            {"magnitude": "millions", "currency": "USD"},
            {"currency": "USD", "magnitude": "millions"},
            id="equal-structured-units-with-different-key-order",
        ),
        pytest.param(
            {"currency": "USD", "magnitude": "millions"},
            {"currency": "EUR", "magnitude": "millions"},
            None,
            id="different-structured-units",
        ),
        pytest.param(
            {"currency": "USD", "magnitude": "millions"},
            "USD",
            None,
            id="structured-and-string-units",
        ),
        pytest.param("Year", "Year", "Year", id="equal-string-units"),
    ],
)
def test_table_fact_summary_unit_matches_row_units(release, first_unit, second_unit, expected):
    """A table reports a unit only when every row agrees (#585, #646)."""
    component = {
        "type": "table",
        "rows": [
            {
                "series_id": "s1",
                "dimensions": {},
                "period": "2020",
                "value": 1.0,
                "unit": first_unit,
            },
            {
                "series_id": "s2",
                "dimensions": {},
                "period": "2021",
                "value": 2.0,
                "unit": second_unit,
            },
        ],
    }
    summary = build_fact_summary(
        component=component,
        member="table",
        formatter_profile=release.formatter_profile,
        series_coordinates=None,
        component_id="cmp_x",
        revision=1,
        changed=(),
        catalog_version=release.catalog_version,
    )

    assert summary["unit"] == expected


def _entity_mode_resolved_frame():
    data = {
        "country/ETH": {"2020": 100.0, "2021": 110.0, "2022": 120.0},
        "country/GHA": {"2020": 200.0, "2021": 210.0, "2022": 220.0},
    }
    rows = [
        {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": recipient,
            "time_period": period,
            "value": value,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_oecd_dac",
        }
        for recipient, values in data.items()
        for period, value in values.items()
    ]
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=rows,
        labels={"country/ETH": "Ethiopia", "country/GHA": "Ghana"},
    )


def test_entity_mode_bar_chart_fact_summary_agrees_with_a_readback_of_the_minted_payload(release):
    """A create reply's per-series stats must describe only what the
    stored component carries. Entity-mode bar_chart mints one point per
    series at the shared observed period. The summary drawn from that
    minted `series[]` must equal what a later `get_component` readback
    rebuilds from the same stored payload (mirrored here via
    `series_entry`, the function both builds share)."""
    resolved_frame = _entity_mode_resolved_frame()
    selectors = [
        {
            "bindings": [
                {"dimension": "stat_var", "values": ["dcid:MalariaGrantDisbursements"]},
                {"dimension": "recipient", "values": ["country/ETH", "country/GHA"]},
            ]
        }
    ]
    window = {"start": "2020", "end": "2022"}
    series = resolve_selectors(resolved_frame=resolved_frame, selectors=selectors, window=window)
    bound = bind_all(resolved_frame=resolved_frame, series=series, window=window)
    intent = {
        "type": "bar_chart",
        "frame_ref": {
            "spec_id": resolved_frame.frame_ref.spec_id,
            "content_version": resolved_frame.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": ["country/ETH", "country/GHA"],
        },
        "window": window,
    }
    minted = mint_component(
        member="bar_chart",
        merged_intent=intent,
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
    )
    reply = build_reply(
        composition_id=None,
        composition_revision=None,
        changed=(),
        resolved_payload=_envelope(minted, release=release),
        series_coordinates=_series_coordinates(series),
        formatter_profile=release.formatter_profile,
    )

    # Draws each series entry from the stored payload's own `series[]`
    # alone -- the point_count/first_value/last_value agreement this test
    # checks, independent of `coordinates` (covered separately in
    # `tests/assembler/test_readback.py`'s write-vs-readback tests).
    readback_series = [
        series_entry(
            id=one["id"],
            label=one["label"],
            coordinates={},
            points=one["points"],
            profile=release.formatter_profile,
        )
        for one in minted.component["series"]
    ]
    write_series = reply.fact_summary["series"]
    assert [s["point_count"] for s in write_series] == [1, 1]
    assert [s["point_count"] for s in write_series] == [s["point_count"] for s in readback_series]
    assert [s["first_value"] for s in write_series] == [s["first_value"] for s in readback_series]
    assert [s["last_value"] for s in write_series] == [s["last_value"] for s in readback_series]


def test_write_reply_carries_no_affordance_field():
    """The assembler pipeline is door-agnostic. `WriteReply` names no
    door's verb names, so it carries no `affordance` field. Each door
    renders its own affordance line over the reply's `component_id` (the
    REST door's own line lives in `presentation_api.reply`)."""
    assert "affordance" not in {field.name for field in dataclasses.fields(WriteReply)}


def test_token_cap_truncates_series_before_fields(release, monkeypatch):
    import assembler.reply as reply_module

    resolved_frame, series, bound = _resolve_and_bind()
    minted = mint_component(
        member="line_chart",
        merged_intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": resolved_frame.frame_ref.spec_id,
                "content_version": resolved_frame.frame_ref.content_version,
            },
            "select": _select(),
        },
        resolved_frame=resolved_frame,
        series=series,
        bound_series=bound,
        release=release,
    )
    monkeypatch.setattr(reply_module, "FACT_SUMMARY_TOKEN_CAP", 1)
    reply = build_reply(
        composition_id=None,
        composition_revision=None,
        changed=(),
        resolved_payload=_envelope(minted, release=release),
        series_coordinates=_series_coordinates(series),
        formatter_profile=release.formatter_profile,
    )
    assert reply.fact_summary["series"] == []
    assert reply.fact_summary["series_cut"] == 1
