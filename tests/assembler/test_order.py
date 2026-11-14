"""Checks that an intent's `order` draws the series in value order, at the period it ranks
at, and that a write refuses an order it cannot honour while a render falls back."""

from __future__ import annotations

from dataclasses import replace

import pytest

from assembler.errors import BindingInvalid
from assembler.limits import PER_DIMENSION_LISTING_CAP
from assembler.pipeline import render_component, write_component
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame

_SPEC_ID = "spec_" + "b" * 32
_STAT_VAR = "dcid:MalariaGrantDisbursements"
_LABELS = {
    "country/ETH": "Ethiopia",
    "country/GHA": "Ghana",
    "country/KEN": "Kenya",
    "country/MLI": "Mali",
}
_DESCENDING = {"by": "value", "direction": "descending"}
_ASCENDING = {"by": "value", "direction": "ascending"}


def _frame(values, *, version="1", units=None):
    """One series per recipient, with `values[recipient]` mapping a period to its value."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id=_SPEC_ID, content_version="cv_" + version * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": recipient,
                "time_period": period,
                "value": value,
                "unit": (units or {}).get(recipient),
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for recipient, by_period in values.items()
            for period, value in by_period.items()
        ],
        labels=_LABELS,
    )


# Member-id order is Ethiopia, Ghana, Kenya, Mali. Values at 2024: 30, 10, 20, none; at 2023:
# 5, 50, 1, none.
_VALUES = {
    "country/ETH": {"2023": 5.0, "2024": 30.0},
    "country/GHA": {"2023": 50.0, "2024": 10.0},
    "country/KEN": {"2023": 1.0, "2024": 20.0},
    "country/MLI": {"2022": 9.0},
}


def _intent(frame, *, member, recipients, **fields):
    return {
        "type": member,
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": _STAT_VAR, "recipient": recipients},
        **fields,
    }


@pytest.fixture
def write(frame_store, revision_store, release_registry):
    def _write(frame, intent, **kwargs):
        frame_store.put(resolved_frame=frame)
        return write_component(
            intent=intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
            **kwargs,
        )

    return _write


def _labels(reply):
    return [one["label"] for one in reply.resolved_payload["component"]["series"]]


def _bar_values(reply):
    return [one["points"][0]["value"] for one in reply.resolved_payload["component"]["series"]]


@pytest.fixture
def frame():
    return _frame({k: v for k, v in _VALUES.items() if k != "country/MLI"})


_THREE = ["country/ETH", "country/GHA", "country/KEN"]


def test_a_listed_bar_chart_with_an_order_draws_in_value_order(write, frame):
    reply = write(frame, _intent(frame, member="bar_chart", recipients=_THREE, order=_DESCENDING))

    assert _bar_values(reply) == [30.0, 20.0, 10.0]


def test_an_unbound_bar_chart_with_ascending_draws_smallest_first(write, frame):
    intent = {
        "type": "bar_chart",
        "frame_ref": _intent(frame, member="bar_chart", recipients=_THREE)["frame_ref"],
        "select": {"stat_var": _STAT_VAR},
        "window": {"start": "2024", "end": "2024"},
        "order": _ASCENDING,
    }

    reply = write(frame, intent)

    assert _bar_values(reply) == [10.0, 20.0, 30.0]


def test_a_line_chart_ranks_at_the_latest_period_most_series_hold_and_ends_with_the_valueless(
    write,
):
    frame = _frame(_VALUES)
    intent = _intent(frame, member="line_chart", recipients=list(_LABELS), order=_DESCENDING)

    reply = write(frame, intent)

    # 2023 and 2024 are each held by three series, and the tie goes to 2024. Mali holds
    # neither and comes last.
    assert _labels(reply) == ["Ethiopia", "Kenya", "Ghana", "Mali"]


def test_a_stated_period_ranks_there(write):
    frame = _frame(_VALUES)
    order = {**_ASCENDING, "period": "2023"}
    intent = _intent(frame, member="table", recipients=list(_LABELS), order=order)

    reply = write(frame, intent)

    rows = reply.resolved_payload["component"]["rows"]
    firsts = list(dict.fromkeys(row["dimensions"]["recipient"] for row in rows))
    assert firsts == ["Kenya", "Ethiopia", "Ghana", "Mali"]


@pytest.mark.parametrize(
    "units",
    [
        {"country/ETH": "dcid:USDollar", "country/GHA": "dcid:Euro"},
        # Two units that share a graph name are still two units.
        {"country/ETH": "dcid:UnitA", "country/GHA": "dcid:UnitB", "country/KEN": "dcid:UnitA"},
    ],
    ids=["different-labels", "same-label"],
)
def test_a_table_of_mixed_units_refuses_at_order(write, units):
    frame = _frame(_VALUES, units=units)
    frame = replace(frame, labels={**frame.labels, "dcid:UnitA": "Dollar", "dcid:UnitB": "Dollar"})
    intent = _intent(frame, member="table", recipients=_THREE, order=_DESCENDING)

    with pytest.raises(BindingInvalid) as excinfo:
        write(frame, intent)

    assert excinfo.value.pointer == "/order"


@pytest.mark.parametrize(
    ("order", "window"),
    [
        pytest.param(
            {**_DESCENDING, "period": "2024"},
            {"start": {"anchor": "latest", "offset": -1}, "end": {"anchor": "latest", "offset": 0}},
            id="relative-window",
        ),
        pytest.param({**_DESCENDING, "period": "1999"}, None, id="period-no-series-holds"),
    ],
)
def test_an_order_period_that_cannot_stand_is_refused_at_order_period(write, frame, order, window):
    fields = {"order": order} | ({"window": window} if window else {})
    intent = _intent(frame, member="line_chart", recipients=_THREE, **fields)

    with pytest.raises(BindingInvalid) as excinfo:
        write(frame, intent)

    assert excinfo.value.pointer == "/order/period"


def test_the_held_periods_an_order_period_refusal_lists_are_capped(write):
    periods = [str(year) for year in range(1900, 1900 + PER_DIMENSION_LISTING_CAP + 10)]
    frame = _frame(
        {"country/ETH": dict.fromkeys(periods, 1.0), "country/GHA": dict.fromkeys(periods, 2.0)}
    )
    order = {**_DESCENDING, "period": "1800"}
    intent = _intent(
        frame, member="line_chart", recipients=["country/ETH", "country/GHA"], order=order
    )

    with pytest.raises(BindingInvalid) as excinfo:
        write(frame, intent)

    assert excinfo.value.pointer == "/order/period"
    assert excinfo.value.legal_values == tuple(periods[-PER_DIMENSION_LISTING_CAP:])
    assert excinfo.value.truncated is True


def test_a_revise_keeps_order_through_a_select_change_and_null_clears_it(write, frame):
    created = write(
        frame, _intent(frame, member="line_chart", recipients=_THREE, order=_DESCENDING)
    )
    assert _labels(created) == ["Ethiopia", "Kenya", "Ghana"]

    narrowed = write(
        frame,
        {"select": {"stat_var": _STAT_VAR, "recipient": ["country/GHA", "country/KEN"]}},
        component_id=created.component_id,
        write_token=created.write_token,
    )
    assert _labels(narrowed) == ["Kenya", "Ghana"]

    cleared = write(
        frame,
        {"order": None},
        component_id=created.component_id,
        write_token=created.write_token,
    )
    assert cleared.revision == 3
    assert _labels(cleared) == ["Ghana", "Kenya"]


def test_a_render_without_the_stated_period_falls_back_without_refusing(
    write, frame_store, revision_store, release_registry, frame
):
    order = {**_DESCENDING, "period": "2023"}
    created = write(frame, _intent(frame, member="line_chart", recipients=_THREE, order=order))
    assert _labels(created) == ["Ghana", "Ethiopia", "Kenya"]
    later = _frame(
        {k: {"2024": v["2024"]} for k, v in _VALUES.items() if k != "country/MLI"}, version="2"
    )
    frame_store.put(resolved_frame=later)

    rendered = render_component(
        component_id=created.component_id,
        revision=created.revision,
        target_frame=later.frame_ref,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert _labels(rendered) == ["Ethiopia", "Kenya", "Ghana"]
    assert rendered.fact_summary["order"] == {
        "direction": "descending",
        "period": "2024",
        "detail": "no drawn series holds a value at 2023, so the series are ranked at 2024",
    }


def _table_recipients(reply):
    rows = reply.resolved_payload["component"]["rows"]
    return list(dict.fromkeys(row["dimensions"]["recipient"] for row in rows))


def test_a_render_of_an_ordered_table_whose_series_now_differ_in_unit_draws_unranked(
    write, frame_store, revision_store, release_registry, frame
):
    created = write(frame, _intent(frame, member="table", recipients=_THREE, order=_DESCENDING))
    assert _table_recipients(created) == ["Ethiopia", "Kenya", "Ghana"]
    later = _frame(
        {k: v for k, v in _VALUES.items() if k != "country/MLI"},
        version="2",
        units={"country/ETH": "dcid:USDollar", "country/GHA": "dcid:Euro"},
    )
    frame_store.put(resolved_frame=later)

    rendered = render_component(
        component_id=created.component_id,
        revision=created.revision,
        target_frame=later.frame_ref,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    # Select order: the series are not ranked across units.
    assert _table_recipients(rendered) == ["Ethiopia", "Ghana", "Kenya"]
    order = rendered.fact_summary["order"]
    assert order["period"] is None
    assert order["detail"]
