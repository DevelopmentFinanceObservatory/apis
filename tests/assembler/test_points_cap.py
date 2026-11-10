"""Pin pre-bind point and row caps, plus window grain boundaries."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import cast

import pytest

from assembler.catalogue.constraints import ConstraintFailure, ConstraintSpec
from assembler.errors import BindingInvalid, ConstraintViolation, ResourceExhausted, SchemaInvalid
from assembler.fakes import FakeReleaseRegistry
from assembler.limits import MAX_SERIES_PER_COMPONENT
from assembler.pipeline import write_component
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame

from ._pipeline_support import _multi_recipient_frame, computed_record

# -- line chart series cap ----------------------------------------------------


def _recipients_frame(*, count, stat_vars=("dcid:X",), periods=("2020", "2021")):
    """One series per variable and recipient, `count` recipients over `periods`."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "4" * 32, content_version="cv_" + "4" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var,
                "recipient": f"country/C{index:02d}",
                "time_period": period,
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for stat_var in stat_vars
            for index in range(count)
            for period in periods
        ],
    )


def _write_line_chart(*, frame, select, frame_store, revision_store, release_registry):
    frame_store.put(resolved_frame=frame)
    return write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
            "select": select,
        },
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )


@pytest.mark.parametrize(
    "count",
    [
        pytest.param(lambda cap: cap + 1, id="one-over"),
        # An estimate over MAX_SERIES_PER_COMPONENT names the line chart's cap.
        pytest.param(lambda _cap: MAX_SERIES_PER_COMPONENT + 1, id="over-the-component-cap"),
    ],
)
def test_a_line_chart_with_an_unbound_dimension_over_the_cap_refuses(
    count, frame_store, revision_store, release, release_registry
):
    with pytest.raises(ResourceExhausted) as excinfo:
        _write_line_chart(
            frame=_recipients_frame(count=count(release.line_chart_max_series)),
            select={"stat_var": "dcid:X"},
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select"
    assert excinfo.value.limit_name == "MAX_LINE_CHART_SERIES"


def _sector_frame(*, recipients, sectors, periods=("2020", "2021")):
    """One series per recipient and sector over `periods`."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "7" * 32, content_version="cv_" + "7" * 32),
        role_keys=("recipient", "sector"),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": f"country/C{index:02d}",
                "sector": f"sector/S{sector}",
                "time_period": period,
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for index in range(recipients)
            for sector in range(sectors)
            for period in periods
        ],
    )


def _unit_per_variable_frame(*, count):
    """One recipient at 2020 holding `count` variables, each in a unit of its own."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "8" * 32, content_version="cv_" + "8" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": f"dcid:V{index:02d}",
                "recipient": "country/C00",
                "time_period": "2020",
                "value": 1.0,
                "unit": f"Unit{index:02d}",
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for index in range(count)
        ],
    )


_BAR_OFFER = "as a bar_chart"
_TABLE_OFFER = "as a table"


@pytest.mark.parametrize(
    ("frame", "select", "named", "offered"),
    [
        # One recipient over the cap, over two years: a bar chart draws one period or one
        # series, so only the table is offered.
        pytest.param(
            lambda over: _recipients_frame(count=over),
            {"stat_var": "dcid:X"},
            "recipient ({over} values) is the series dimension",
            ("table",),
            id="unbound-over-two-years",
        ),
        # At one period a bar chart draws one bar per recipient.
        pytest.param(
            lambda over: _recipients_frame(count=over, periods=("2020",)),
            {"stat_var": "dcid:X"},
            "recipient ({over} values) is the series dimension",
            ("bar_chart", "table"),
            id="unbound-at-one-period",
        ),
        # Over 1700 to 2020 a table of that many recipients passes `MAX_TABLE_ROWS`.
        pytest.param(
            lambda over: _recipients_frame(count=over, periods=("1700", "2020")),
            {"stat_var": "dcid:X"},
            "recipient ({over} values) is the series dimension",
            (),
            id="unbound-over-321-years",
        ),
        # Three listed recipients and four unbound sectors: sector is the one to bind.
        pytest.param(
            lambda _over: _sector_frame(recipients=3, sectors=4),
            {"recipient": [f"country/C{index:02d}" for index in range(3)]},
            "bind sector (4 values) to one value, keeping recipient (3 values) as the series",
            ("table",),
            id="listed-and-unbound",
        ),
        # Both unbound: either may stay the series. A bar chart compares one dimension.
        pytest.param(
            lambda _over: _sector_frame(recipients=3, sectors=4),
            {"stat_var": "dcid:X"},
            "recipient (3 values) and sector (4 values) multiply",
            ("table",),
            id="two-unbound",
        ),
        # Nine variables in nine units: a bar chart's bars share a value axis, which
        # `comparable_units_required` refuses, so only the table is offered.
        pytest.param(
            lambda over: _unit_per_variable_frame(count=over),
            {"recipient": "country/C00"},
            "stat_var ({over} values) is the series dimension",
            ("table",),
            id="series-in-different-units",
        ),
    ],
)
def test_a_line_chart_refusal_names_its_dimensions_and_offers_what_draws(
    frame, select, named, offered, frame_store, revision_store, release, release_registry
):
    """The refusal names the dimensions the resolved series vary on, the listed one kept as
    the series, and offers a bar_chart or a table of the same select only where that write
    draws."""
    over = release.line_chart_max_series + 1
    frame = frame(over)
    named = named.format(over=over)
    with pytest.raises(ResourceExhausted) as excinfo:
        _write_line_chart(
            frame=frame,
            select=select,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    detail = excinfo.value.detail
    assert excinfo.value.limit_name == "MAX_LINE_CHART_SERIES"
    assert named in detail
    assert "bind recipient" not in detail
    assert tuple(member for member in ("bar_chart", "table") if f"as a {member}" in detail) == (
        offered
    )
    for member in offered:
        written = write_component(
            intent={
                "type": member,
                "frame_ref": {
                    "spec_id": frame.frame_ref.spec_id,
                    "content_version": frame.frame_ref.content_version,
                },
                "select": select,
            },
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
        component = cast("Mapping[str, object]", written.resolved_payload["component"])
        assert component["type"] == member


def _two_sources_frame(*, recipients):
    """One series per recipient from each of two sources over 2020 and 2021."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "8" * 32, content_version="cv_" + "8" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": f"country/C{index:02d}",
                "time_period": period,
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": source,
            }
            for index in range(recipients)
            for source in ("src_oecd", "src_imf")
            for period in ("2020", "2021")
        ],
    )


@pytest.mark.parametrize(
    ("frame", "select", "advice", "followed"),
    [
        # A listed facet is the series: the unbound recipient is the one to bind.
        pytest.param(
            _two_sources_frame(recipients=5),
            {"stat_var": "dcid:X", "provenance": ["src_oecd", "src_imf"]},
            "bind recipient (5 values) to one value, keeping provenance (2 values) as the series",
            {
                "stat_var": "dcid:X",
                "provenance": ["src_oecd", "src_imf"],
                "recipient": "country/C00",
            },
            id="listed-facet",
        ),
        pytest.param(
            _sector_frame(recipients=3, sectors=4),
            {"recipient": [f"country/C{index:02d}" for index in range(3)]},
            "bind sector (4 values) to one value, keeping recipient (3 values) as the series",
            {"recipient": [f"country/C{index:02d}" for index in range(3)], "sector": "sector/S0"},
            id="listed-and-unbound",
        ),
    ],
)
def test_a_line_chart_refusals_advice_draws_when_followed(
    frame, select, advice, followed, frame_store, revision_store, release_registry
):
    """The select the refusal's advice describes draws."""
    stores = {
        "frame_store": frame_store,
        "revision_store": revision_store,
        "release_registry": release_registry,
    }
    with pytest.raises(ResourceExhausted) as excinfo:
        _write_line_chart(frame=frame, select=select, **stores)
    assert advice in excinfo.value.detail

    written = _write_line_chart(frame=frame, select=followed, **stores)

    component = cast("Mapping[str, object]", written.resolved_payload["component"])
    assert component["type"] == "line_chart"


def test_a_line_chart_refusal_offers_nothing_when_the_window_refuses(
    frame_store, revision_store, release, release_registry
):
    """A window past `MAX_POINTS_PER_SERIES` draws no other type either, so none is offered."""
    frame = _recipients_frame(count=release.line_chart_max_series + 1)
    frame_store.put(resolved_frame=frame)
    with pytest.raises(ResourceExhausted) as excinfo:
        write_component(
            intent={
                "type": "line_chart",
                "frame_ref": {
                    "spec_id": frame.frame_ref.spec_id,
                    "content_version": frame.frame_ref.content_version,
                },
                "select": {"stat_var": "dcid:X"},
                "window": {"start": "1000", "end": "2021"},
            },
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.limit_name == "MAX_LINE_CHART_SERIES"
    assert "draw the same select" not in excinfo.value.detail


@pytest.mark.parametrize("blocked", ["bar_chart", "table"])
def test_a_line_chart_refusal_offers_no_type_whose_own_write_refuses(
    blocked, frame_store, revision_store, release
):
    """At one period a bar chart and a table each draw the series a line chart cannot. A
    release constraint on one of them takes that type out of the offer, and its write
    refuses, since the offer runs that type's own write checks."""

    def refuse(_context):
        return ConstraintFailure(pointer="/component", detail="blocked", fix="none")

    registry = FakeReleaseRegistry(
        release=dataclasses.replace(
            release,
            constraints=(
                *release.constraints,
                ConstraintSpec(
                    constraint_id="blocked_type",
                    encoding_classes=(),
                    component_types=(blocked,),
                    check=refuse,
                ),
            ),
        )
    )
    frame = _recipients_frame(count=release.line_chart_max_series + 1, periods=("2020",))
    stores = {
        "frame_store": frame_store,
        "revision_store": revision_store,
        "release_registry": registry,
    }
    with pytest.raises(ResourceExhausted) as excinfo:
        _write_line_chart(frame=frame, select={"stat_var": "dcid:X"}, **stores)
    offered = [m for m in ("bar_chart", "table") if f"as a {m}" in excinfo.value.detail]
    assert offered == [m for m in ("bar_chart", "table") if m != blocked]

    with pytest.raises(ConstraintViolation) as refused:
        write_component(
            intent={
                "type": blocked,
                "frame_ref": {
                    "spec_id": frame.frame_ref.spec_id,
                    "content_version": frame.frame_ref.content_version,
                },
                "select": {"stat_var": "dcid:X"},
            },
            **stores,
        )
    assert refused.value.constraint_id == "blocked_type"


@pytest.mark.parametrize(
    ("count", "refusal", "limit_name"),
    [
        # The exact count runs after the span check, so the span refusal names the bar chart.
        pytest.param(
            lambda cap: cap + 1, BindingInvalid, None, id="over-the-cap-under-the-estimate"
        ),
        # The estimate runs while the span check resolves the select, so it wins. A span
        # frame's line chart never draws, so it names the component limit of the bar chart
        # the span refusal points to.
        pytest.param(
            lambda _cap: MAX_SERIES_PER_COMPONENT + 1,
            ResourceExhausted,
            "MAX_SERIES_PER_COMPONENT",
            id="over-the-estimate",
        ),
    ],
)
def test_a_span_frame_line_chart_over_the_cap_refuses_by_which_check_runs_first(
    count, refusal, limit_name, frame_store, revision_store, release, release_registry
):
    with pytest.raises(refusal) as excinfo:
        _write_line_chart(
            frame=_recipients_frame(
                count=count(release.line_chart_max_series), periods=("2018/2022",)
            ),
            select={"stat_var": "dcid:X"},
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert getattr(excinfo.value, "limit_name", None) == limit_name


def test_a_span_frame_estimate_refusal_advises_what_the_bar_chart_draws(
    frame_store, revision_store, release_registry
):
    """A span frame's line chart never draws, so its estimate refusal names the
    `MAX_SERIES_PER_COMPONENT` series a bar chart draws, and a bar chart listing that many of
    them draws."""
    frame = _recipients_frame(count=MAX_SERIES_PER_COMPONENT + 1, periods=("2018/2022",))
    with pytest.raises(ResourceExhausted) as excinfo:
        _write_line_chart(
            frame=frame,
            select={"stat_var": "dcid:X"},
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.limit_name == "MAX_SERIES_PER_COMPONENT"
    assert f"bind it to at most {MAX_SERIES_PER_COMPONENT} values" in excinfo.value.detail
    written = write_component(
        intent={
            "type": "bar_chart",
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
            "select": {
                "recipient": [f"country/C{index:02d}" for index in range(MAX_SERIES_PER_COMPONENT)]
            },
        },
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    component = cast("Mapping[str, object]", written.resolved_payload["component"])
    assert component["type"] == "bar_chart"


def test_a_line_chart_with_an_unbound_dimension_at_the_cap_draws_every_series(
    frame_store, revision_store, release, release_registry
):
    cap = release.line_chart_max_series
    written = _write_line_chart(
        frame=_recipients_frame(count=cap),
        select={"stat_var": "dcid:X"},
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert len(written.resolved_payload["component"]["series"]) == cap


_STORED = {"stat_var": "dcid:X"}
# A select of two dimensions, which a `jsonb` column returns in its own key order.
_STORED_TWO_KEYS = {"stat_var": "dcid:X", "provenance": "src_a"}
# A list one value longer than the cap, which `_recipients_frame` holds.
_STORED_LISTED = "listed"


@pytest.mark.parametrize(
    ("stored", "patch", "refused"),
    [
        pytest.param(_STORED, {"title": "Grants by recipient"}, False, id="title-keeps-select"),
        pytest.param(
            _STORED_LISTED, {"title": "Grants by recipient"}, False, id="title-keeps-a-long-list"
        ),
        # A one-value list draws differently from a scalar (`has_arrayed_dimension`), so it
        # changes the select.
        pytest.param(
            _STORED, {"select": {"stat_var": ["dcid:X"]}}, True, id="scalar-restated-as-a-list"
        ),
        pytest.param(
            _STORED_TWO_KEYS,
            {
                "title": "Grants by recipient",
                "select": {"provenance": "src_a", "stat_var": "dcid:X"},
            },
            False,
            id="same-select-in-another-key-order",
        ),
        # The recipients `_cut` fills in, one per value the cap allows.
        pytest.param(_STORED, "cut", False, id="select-cut-to-the-cap"),
        pytest.param(
            _STORED,
            {"select": {"stat_var": "dcid:Y"}},
            True,
            id="select-changed-still-over",
        ),
    ],
)
def test_a_revision_of_a_stored_over_cap_line_chart_is_held_to_the_cap_when_it_changes_select(
    stored, patch, refused, frame_store, revision_store, release, release_registry
):
    """A chart stored over the cap takes edits that keep its select, and is refused when an
    edit changes its select to one that resolves over the cap."""
    cap = release.line_chart_max_series
    if stored == _STORED_LISTED:
        stored = {
            "stat_var": "dcid:X",
            "recipient": [f"country/C{i:02d}" for i in range(cap + 1)],
        }
    if patch == "cut":
        patch = {
            "select": {"stat_var": "dcid:X", "recipient": [f"country/C{i:02d}" for i in range(cap)]}
        }
    created = _write_line_chart(
        frame=_recipients_frame(count=cap + 1, stat_vars=("dcid:X", "dcid:Y")),
        select=stored,
        frame_store=frame_store,
        revision_store=revision_store,
        # A release whose cap the stored select fits, as an earlier release's may have.
        release_registry=FakeReleaseRegistry(
            release=dataclasses.replace(release, line_chart_max_series=cap + 1)
        ),
    )

    def revise():
        return write_component(
            intent=patch,
            component_id=created.component_id,
            write_token=created.write_token,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    if refused:
        with pytest.raises(ResourceExhausted) as excinfo:
            revise()
        assert excinfo.value.limit_name == "MAX_LINE_CHART_SERIES"
    else:
        assert revise().resolved_payload["component"]["type"] == "line_chart"


def test_a_line_chart_listing_more_values_than_the_cap_refuses_as_the_series_cap(
    frame_store, revision_store, release, release_registry
):
    """The schema's `maxItems` on the list is the line chart's series cap, so the refusal
    names it and advises what a select resolving past it is advised."""
    over = release.line_chart_max_series + 1
    stores = {
        "frame_store": frame_store,
        "revision_store": revision_store,
        "release_registry": release_registry,
    }
    frame = _recipients_frame(count=over)
    with pytest.raises(ResourceExhausted) as listed:
        _write_line_chart(
            frame=frame,
            select={
                "stat_var": "dcid:X",
                "recipient": [f"country/C{index:02d}" for index in range(over)],
            },
            **stores,
        )
    with pytest.raises(ResourceExhausted) as resolved:
        _write_line_chart(frame=frame, select={"stat_var": "dcid:X"}, **stores)

    assert (listed.value.pointer, listed.value.limit_name) == (
        "/select/recipient",
        "MAX_LINE_CHART_SERIES",
    )
    assert resolved.value.limit_name == "MAX_LINE_CHART_SERIES"
    # The resolved refusal's advice ends with the types that draw the select, which the list
    # refusal, raised before any frame is read, cannot name.
    _found, advice = listed.value.detail.split("; ", 1)
    assert resolved.value.detail.split("; ", 1)[1].startswith(advice)
    assert f"bind it to at most {release.line_chart_max_series} values" in advice


def test_a_line_chart_list_counts_its_distinct_values(
    frame_store, revision_store, release, release_registry
):
    """A list one item over the cap that repeats a value names as many series as the cap,
    so it draws them."""
    cap = release.line_chart_max_series
    recipients = [f"country/C{index:02d}" for index in range(cap)]
    written = _write_line_chart(
        frame=_recipients_frame(count=cap),
        select={"stat_var": "dcid:X", "recipient": [*recipients, recipients[0]]},
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert len(written.resolved_payload["component"]["series"]) == cap


def test_a_line_chart_list_past_the_cap_holding_a_wrong_type_is_left_to_the_schema(
    frame_store, revision_store, release, release_registry
):
    over = release.line_chart_max_series + 1
    with pytest.raises(SchemaInvalid) as excinfo:
        _write_line_chart(
            frame=_recipients_frame(count=over),
            select={
                "stat_var": "dcid:X",
                "recipient": [*(f"country/C{index:02d}" for index in range(over - 1)), 7],
            },
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select/recipient"


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
    ("resolved_frame", "row_count", "advice"),
    [
        # 2,000 // 410 periods is 4 series, and 2,000 // 5 series is 400 periods.
        pytest.param(
            _wide_table_frame(
                frame_ref=FrameRef(spec_id="spec_" + "6" * 32, content_version="cv_" + "6" * 32)
            ),
            None,
            ("5 series x 410 periods = 2050 rows", "at most 4 series", "at most 400 periods"),
            id="padded",
        ),
        pytest.param(
            _top_one(_wide_table_frame(frame_ref=_TOP_ONE_FRAME)),
            410,
            (),
            id="per-period-selection",
        ),
        pytest.param(
            _top_one(_wide_table_frame(frame_ref=_TOP_ONE_FRAME, years=range(2020, 2030))),
            None,
            ("keeps 2010 rows", "bind fewer series or narrow the window"),
            id="per-period-selection-over-years-no-row-holds",
        ),
    ],
)
def test_table_row_cap_counts_the_rows_the_table_lists(
    frame_store, revision_store, release_registry, monkeypatch, resolved_frame, row_count, advice
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
        for words in advice:
            assert words in excinfo.value.detail
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
