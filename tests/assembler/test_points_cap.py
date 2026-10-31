"""Pin pre-bind point and row caps, plus window grain boundaries."""

from __future__ import annotations

import dataclasses

import pytest

from assembler.errors import BindingInvalid, ResourceExhausted
from assembler.pipeline import write_component
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame

from ._pipeline_support import _multi_recipient_frame, computed_record

# -- points cap ----------------------------------------------------------------


def _day_grain_frame(*, spec_digit):
    return build_resolved_frame(
        frame_ref=FrameRef(
            spec_id="spec_" + spec_digit * 32, content_version="cv_" + spec_digit * 32
        ),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020-01-01",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
        ],
    )


def test_wide_day_grain_window_refuses_resource_exhausted_without_materializing_points(
    frame_store, revision_store, release_registry, monkeypatch
):
    import assembler.pipeline.candidate as candidate_module

    def _bind_all_must_not_run(**kwargs):
        raise AssertionError("bind_all must not run once the window exceeds the points cap")

    monkeypatch.setattr(candidate_module, "bind_all", _bind_all_must_not_run)

    rf = _day_grain_frame(spec_digit="5")
    frame_store.put(resolved_frame=rf)

    with pytest.raises(ResourceExhausted) as excinfo:
        write_component(
            intent={
                "type": "line_chart",
                "frame_ref": {
                    "spec_id": rf.frame_ref.spec_id,
                    "content_version": rf.frame_ref.content_version,
                },
                "select": {"recipient": "country/ETH"},
                "window": {"start": "2020-01-01", "end": "2030-01-01"},
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/window"
    assert excinfo.value.limit_name == "MAX_POINTS_PER_SERIES"


_YEARS = range(2020, 2430)


def _wide_table_frame(*, frame_ref, years=_YEARS):
    """Five recipients, one of them in each of `years` in turn, by default 2020 to 2429: 410
    rows, the shape of a per-period top one."""
    return build_resolved_frame(
        frame_ref=frame_ref,
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": f"country/C{year % 5:02d}",
                "time_period": str(year),
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for year in years
        ],
    )


_TOP_ONE_FRAME = FrameRef(spec_id="ops_" + "6" * 32, content_version="ocv_" + "6" * 32)


def _top_one(frame):
    """`frame` with the record of a per-period top one."""
    return dataclasses.replace(
        frame,
        description_record=computed_record(
            frame_ref=_TOP_ONE_FRAME,
            steps=[
                {
                    "id": "s2",
                    "op": "rank",
                    "input": "s1",
                    "params": {"limit": 1, "order": "descending", "per_period": True},
                }
            ],
        ),
    )


@pytest.mark.parametrize(
    ("resolved_frame", "row_count"),
    [
        pytest.param(
            _wide_table_frame(
                frame_ref=FrameRef(spec_id="spec_" + "6" * 32, content_version="cv_" + "6" * 32)
            ),
            None,
            id="padded",
        ),
        pytest.param(
            _top_one(_wide_table_frame(frame_ref=_TOP_ONE_FRAME)),
            410,
            id="per-period-selection",
        ),
        pytest.param(
            _top_one(_wide_table_frame(frame_ref=_TOP_ONE_FRAME, years=range(2020, 2030))),
            None,
            id="per-period-selection-over-years-no-row-holds",
        ),
    ],
)
def test_table_row_cap_counts_the_rows_the_table_lists(
    frame_store, revision_store, release_registry, monkeypatch, resolved_frame, row_count
):
    """Five recipients across 410 years pad to 2,050 rows, above the 2,000-row cap, so the
    table refuses before `bind_all` runs. A frame keeping each period's selection lists only
    the rows it holds where some recipient holds one, 410 here, and the cap counts those. A
    year no recipient holds is a gap that keeps a missing row per recipient: 10 rows and 400
    gap years of five pad to 2,010 rows, and the cap counts those too."""
    import assembler.pipeline.candidate as candidate_module

    if row_count is None:

        def _bind_all_must_not_run(**kwargs):
            raise AssertionError("bind_all must not run once the row count exceeds MAX_TABLE_ROWS")

        monkeypatch.setattr(candidate_module, "bind_all", _bind_all_must_not_run)

    frame_store.put(resolved_frame=resolved_frame)

    def _write():
        return write_component(
            intent={
                "type": "table",
                "frame_ref": {
                    "spec_id": resolved_frame.frame_ref.spec_id,
                    "content_version": resolved_frame.frame_ref.content_version,
                },
                "select": {
                    "recipient": [f"country/C{i:02d}" for i in range(5)],
                },
                "window": {"start": "2020", "end": "2429"},
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    if row_count is None:
        with pytest.raises(ResourceExhausted) as excinfo:
            _write()
        assert excinfo.value.pointer == "/select"
        assert excinfo.value.limit_name == "MAX_TABLE_ROWS"
        return

    reply = _write()
    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    assert len(record.resolved_payload["component"]["rows"]) == row_count


def test_realistic_all_positive_entity_mode_bar_chart_succeeds(
    frame_store, revision_store, release_registry
):
    """All-positive entity-mode bars pass `bar_zero_in_domain`."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)

    reply = write_component(
        intent={
            "type": "bar_chart",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/ETH", "country/GHA", "country/KEN"],
            },
            "window": {"start": "2020", "end": "2022"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert reply.fact_summary["series"]


@pytest.mark.parametrize(
    ("periods", "window"),
    [
        pytest.param(
            ("2020", "2021"), {"start": "2020-01", "end": "2021-01"}, id="month_on_year_frame"
        ),
        pytest.param(
            ("2024-01", "2024-02"),
            {"start": "2024-01-15", "end": "2024-03-15"},
            id="day_on_month_frame",
        ),
    ],
)
def test_a_window_finer_than_the_frame_s_grain_refuses_binding_invalid(
    frame_store, revision_store, release_registry, periods, window
):
    """A finer window passes lexical observation matching but fails counting at the frame's
    grain. Return `BindingInvalid` at `/window`, not a raw `ValueError`."""
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "c" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": period,
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for period in periods
        ],
    )
    frame_store.put(resolved_frame=rf)

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent={
                "type": "line_chart",
                "frame_ref": {
                    "spec_id": rf.frame_ref.spec_id,
                    "content_version": rf.frame_ref.content_version,
                },
                "select": {"recipient": "country/ETH"},
                "window": window,
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/window"


def test_window_at_exactly_the_points_cap_passes(frame_store, revision_store, release_registry):
    rf = _day_grain_frame(spec_digit="7")
    frame_store.put(resolved_frame=rf)

    reply = write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {"recipient": "country/ETH"},
            "window": {"start": "2020-01-01", "end": "2021-05-14"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert reply.fact_summary["series"][0]["point_count"] == 1


def test_day_grain_default_window_binds_a_calendar_invalid_max_observed_period(
    frame_store, revision_store, release_registry
):
    """The default window includes observed `2024-02-30`.

    Unobserved `2024-02-29` becomes a missing point.
    """
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "9" * 32, content_version="cv_" + "9" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for period, value in [("2024-02-28", 1.0), ("2024-02-30", 2.0)]
        ],
    )
    frame_store.put(resolved_frame=rf)

    reply = write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {"recipient": "country/ETH"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    points = record.resolved_payload["component"]["series"][0]["points"]
    periods = [p["period"] for p in points]
    assert periods == ["2024-02-28", "2024-02-29", "2024-02-30"]
    points_by_period = {p["period"]: p for p in points}
    assert points_by_period["2024-02-28"] == {"period": "2024-02-28", "value": 1.0}
    assert points_by_period["2024-02-29"] == {"period": "2024-02-29", "status": "missing"}
    assert points_by_period["2024-02-30"] == {"period": "2024-02-30", "value": 2.0}


def test_day_grain_window_with_an_extra_observed_non_calendar_day_refuses_resource_exhausted(
    frame_store, revision_store, release_registry, monkeypatch
):
    """A 500-day window plus observed `2020-02-30` binds 501 points.

    The pre-bind cap must include the extra grammar-valid date.
    """
    import assembler.pipeline.candidate as candidate_module

    def _bind_all_must_not_run(**kwargs):
        raise AssertionError("bind_all must not run once the window exceeds the points cap")

    monkeypatch.setattr(candidate_module, "bind_all", _bind_all_must_not_run)

    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "e" * 32, content_version="cv_" + "e" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020-01-01",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            },
            {
                # Calendar-invalid (February 2020 has 29 days) -- lexically
                # inside the window below, but not one of its 500 calendar
                # days, so it's an extra point the union adds.
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020-02-30",
                "value": 2.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            },
        ],
    )
    frame_store.put(resolved_frame=rf)

    with pytest.raises(ResourceExhausted) as excinfo:
        write_component(
            intent={
                "type": "line_chart",
                "frame_ref": {
                    "spec_id": rf.frame_ref.spec_id,
                    "content_version": rf.frame_ref.content_version,
                },
                "select": {"recipient": "country/ETH"},
                "window": {"start": "2020-01-01", "end": "2021-05-14"},
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/window"
    assert excinfo.value.limit_name == "MAX_POINTS_PER_SERIES"


def test_default_window_with_a_calendar_invalid_earliest_observation_binds_no_earlier_point(
    frame_store, revision_store, release_registry
):
    """`default_window` preserves the calendar-invalid start `2024-02-30`.

    Date arithmetic clamps it to Feb 29, which stays outside the bound sequence.
    """
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "f" * 32, content_version="cv_" + "f" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2024-02-30",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
        ],
    )
    frame_store.put(resolved_frame=rf)

    reply = write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {"recipient": "country/ETH"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    points = record.resolved_payload["component"]["series"][0]["points"]
    periods = [p["period"] for p in points]
    assert periods == ["2024-02-30"]
