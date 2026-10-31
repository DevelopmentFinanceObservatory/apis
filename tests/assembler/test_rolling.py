"""Checks that relative periods in an intent resolve on the frame the component is drawn on,
that the stored intent keeps them, and that the series a rolling component leaves out are
named in its provenance."""

from __future__ import annotations

import dataclasses

import jsonschema
import pytest

from assembler.errors import BindingInvalid, ConstraintViolation
from assembler.mint.provenance import _with_left_out_warning
from assembler.mint.validate import component_registry
from assembler.pipeline import render_component, write_component
from assembler.pipeline.export_component import export_component_data
from assembler.ports import (
    FrameRef,
    FrameUnavailable,
    LeftOutSeries,
    ResolvedFrame,
    RollingAnchor,
)
from assembler.rolling import (
    ROLLING_ANCHOR_UNHELD,
    ROLLING_SPAN_FRAME,
    ROLLING_WINDOW_INVERTED,
    ROLLING_WINDOW_SIDE_OMITTED,
    RollingPeriodInvalid,
)
from assembler.testing import build_resolved_frame, malaria_ethiopia_resolved_frame

from ._pipeline_support import _malaria_intent, _stat_card_intent

_SPEC_ID = "spec_" + "a" * 32
_STAT_VAR = "dcid:MalariaGrantDisbursements"
_LABELS = {
    "country/ETH": "Ethiopia",
    "country/GHA": "Ghana",
    "country/KEN": "Kenya",
    "country/MLI": "Mali",
    "country/TCD": "Chad",
}
_LATEST = {"anchor": "latest", "offset": 0}


def _back(offset: int) -> dict[str, object]:
    return {"anchor": "latest", "offset": offset}


def _frame(periods_by_recipient, *, version: str = "1", **overrides):
    """One series per recipient, holding `periods_by_recipient[recipient]`. A value is its
    period's last two characters, plus 100 per recipient before it."""
    frame = build_resolved_frame(
        frame_ref=FrameRef(spec_id=_SPEC_ID, content_version="cv_" + version * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": recipient,
                "time_period": period,
                "value": 100.0 * index + int(period[-2:]),
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for index, (recipient, periods) in enumerate(periods_by_recipient.items())
            for period in periods
        ],
        labels=_LABELS,
    )
    return dataclasses.replace(frame, **overrides)


def _intent(frame, *, member: str, recipients, **fields):
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


def _component(reply):
    return reply.resolved_payload["component"]


def _warnings(reply):
    return reply.resolved_payload["provenance"]["warnings"]


def _relative_periods(node) -> list[object]:
    """Every `relative_period` object anywhere under `node`."""
    if isinstance(node, dict):
        found = [node] if "anchor" in node else []
        return found + [one for value in node.values() for one in _relative_periods(value)]
    if isinstance(node, list):
        return [one for value in node for one in _relative_periods(value)]
    return []


_THREE_YEARS = ("2022", "2023", "2024")
_TWO_YEARS = ("2022", "2023")


# -- the window ------------------------------------------------------------------


def test_a_relative_window_is_drawn_as_labels_and_stored_as_written(write, revision_store, release):
    frame, intent = _malaria_intent(window={"start": _back(-4), "end": _LATEST})

    reply = write(frame, intent)

    component = _component(reply)
    assert component["x_axis"]["window"] == {"start": "2020", "end": "2024"}
    assert [point["period"] for point in component["series"][0]["points"]] == [
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
    ]
    assert _relative_periods(reply.resolved_payload) == []
    jsonschema.Draft202012Validator(
        release.component_schemas["line_chart"], registry=component_registry(release)
    ).validate(component)
    stored = revision_store.read_current(component_id=reply.component_id).stored_intent
    assert stored["window"] == {"start": _back(-4), "end": _LATEST}


def test_a_window_mixes_a_stated_start_with_a_relative_end(write):
    frame, intent = _malaria_intent(window={"start": "2019", "end": _back(-1)})

    reply = write(frame, intent)

    assert _component(reply)["x_axis"]["window"] == {"start": "2019", "end": "2023"}


@pytest.mark.parametrize("window", [{"end": _LATEST}, {"start": _back(-3)}])
def test_a_relative_side_beside_an_omitted_side_is_refused_at_window(write, window):
    frame, intent = _malaria_intent(window=window)

    with pytest.raises(RollingPeriodInvalid) as excinfo:
        write(frame, intent)

    assert (excinfo.value.code, excinfo.value.pointer) == (ROLLING_WINDOW_SIDE_OMITTED, "/window")


def test_a_month_frame_moves_a_relative_period_by_months(write):
    frame = _frame({"country/ETH": ("2023-11", "2023-12", "2024-01", "2024-02")})
    intent = _intent(
        frame,
        member="line_chart",
        recipients="country/ETH",
        window={"start": _back(-2), "end": _LATEST},
    )

    reply = write(frame, intent)

    assert _component(reply)["x_axis"]["window"] == {"start": "2023-12", "end": "2024-02"}


# -- refusals --------------------------------------------------------------------

_DAYS = ("2024-03-01", "2024-03-02", "2024-03-04")
_SPAN = ("2019/2024",)


@pytest.mark.parametrize(
    ("periods_by_recipient", "fields", "overrides", "code", "pointer"),
    [
        pytest.param(
            {"country/ETH": _SPAN},
            {"window": {"start": _LATEST, "end": _LATEST}},
            {},
            ROLLING_SPAN_FRAME,
            "/window",
            id="span-frame-window",
        ),
        pytest.param(
            {"country/ETH": _SPAN},
            {"change": {"basis": [_back(-1), _LATEST]}},
            {},
            ROLLING_SPAN_FRAME,
            "/change/basis",
            id="span-frame-basis",
        ),
        pytest.param(
            {"country/ETH": _DAYS},
            {"window": {"start": _back(-1), "end": _LATEST}},
            {},
            "rolling_grain_unsupported",
            "/window",
            id="day-grain-moves",
        ),
        pytest.param(
            {"country/ETH": ("2020",), "country/GHA": ("2021",), "country/KEN": ("2022",)},
            {"window": {"start": _LATEST, "end": _LATEST}},
            {},
            "rolling_anchor_unresolved",
            "/window",
            id="no-period-held-by-half",
        ),
        pytest.param(
            {"country/ETH": _THREE_YEARS},
            {"change": {"basis": [_back(-5), _LATEST]}},
            {"anchor": RollingAnchor(period="0003", rule="half1")},
            "invalid_time_period",
            "/change/basis",
            id="moves-before-year-zero",
        ),
        pytest.param(
            {"country/ETH": _THREE_YEARS},
            {"window": {"start": "2030", "end": _LATEST}},
            {},
            ROLLING_WINDOW_INVERTED,
            "/window",
            id="start-after-end",
        ),
    ],
)
def test_an_unresolvable_relative_period_is_refused_as_binding_invalid(
    write, periods_by_recipient, fields, overrides, code, pointer
):
    frame = _frame(periods_by_recipient, **overrides)
    recipients = list(periods_by_recipient)
    intent = _intent(
        frame,
        member="stat_card" if "change" in fields else "table",
        recipients=recipients[0] if len(recipients) == 1 else recipients,
        **fields,
    )

    with pytest.raises(BindingInvalid) as excinfo:
        write(frame, intent)

    assert isinstance(excinfo.value, RollingPeriodInvalid)
    assert (excinfo.value.code, excinfo.value.pointer) == (code, pointer)


def test_a_day_frame_resolves_the_anchor_itself(write):
    frame = _frame({"country/ETH": _DAYS})
    intent = _intent(
        frame,
        member="table",
        recipients="country/ETH",
        window={"start": "2024-03-02", "end": _LATEST},
    )

    reply = write(frame, intent)

    assert [row["period"] for row in _component(reply)["rows"]] == [
        "2024-03-02",
        "2024-03-03",
        "2024-03-04",
    ]


# -- the stat card ---------------------------------------------------------------


def test_a_stat_card_with_a_change_shows_the_value_its_change_ends_at(write):
    """The malaria series runs to 2024. A basis ending at 2022 puts the headline there."""
    frame, intent = _stat_card_intent(change={"basis": ["2021", "2022"]})

    component = _component(write(frame, intent))

    assert (component["period"], component["value"]) == ("2022", 420.1)
    assert component["change"]["basis"] == ["2021", "2022"]
    assert component["change"]["value"] == pytest.approx(420.1 - 405.6)


def test_a_relative_change_basis_resolves_to_labels_the_headline_follows(
    write, revision_store, release
):
    frame, intent = _stat_card_intent(change={"basis": [_back(-2), _back(-1)]})

    reply = write(frame, intent)

    component = _component(reply)
    assert component["change"]["basis"] == ["2022", "2023"]
    assert (component["period"], component["value"]) == ("2023", 433.7)
    assert _relative_periods(reply.resolved_payload) == []
    jsonschema.Draft202012Validator(
        release.component_schemas["stat_card"], registry=component_registry(release)
    ).validate(component)
    stored = revision_store.read_current(component_id=reply.component_id).stored_intent
    assert stored["change"]["basis"] == [_back(-2), _back(-1)]


def test_a_relative_basis_period_without_a_value_fails_the_observed_basis_constraint(write):
    """The malaria series holds no 2018 row, six periods back from 2024."""
    frame, intent = _stat_card_intent(change={"basis": [_back(-6), _LATEST]})

    with pytest.raises(ConstraintViolation) as excinfo:
        write(frame, intent)

    assert excinfo.value.constraint_id == "stat_change_basis_observed"
    assert excinfo.value.pointer == "/change/basis"


# -- the bar chart ---------------------------------------------------------------

_KENYA_LAGS = {
    "country/ETH": _THREE_YEARS,
    "country/GHA": _THREE_YEARS,
    "country/KEN": _TWO_YEARS,
}


@pytest.mark.parametrize(
    ("recipients", "bars"),
    [
        pytest.param(list(_KENYA_LAGS), ["Ethiopia", "Ghana"], id="listed"),
        pytest.param(None, ["Ghana", "Ethiopia"], id="largest-first-default"),
    ],
)
def test_a_bar_chart_with_a_relative_end_draws_at_the_anchor_and_names_the_rest(
    write, recipients, bars
):
    frame = _frame(_KENYA_LAGS)
    intent = _intent(
        frame, member="bar_chart", recipients=recipients, window={"start": _LATEST, "end": _LATEST}
    )
    if recipients is None:
        del intent["select"]["recipient"]

    reply = write(frame, intent)

    component = _component(reply)
    assert component["category_axis"] == {
        "mode": "entity",
        "category_dimension": "recipient",
        "period": "2024",
    }
    assert [bar["label"] for bar in component["series"]] == bars
    assert all(bar["points"][0]["period"] == "2024" for bar in component["series"])
    assert _warnings(reply) == [
        {"kind": "coverage_gap", "detail": "1 series has no 2024 value: Kenya"}
    ]


def test_a_rolling_bar_chart_exports_the_rows_of_the_bars_it_draws(
    write, frame_store, revision_store
):
    frame = _frame(_KENYA_LAGS)
    intent = _intent(
        frame,
        member="bar_chart",
        recipients=list(_KENYA_LAGS),
        window={"start": _LATEST, "end": _LATEST},
    )
    created = write(frame, intent)

    exported = export_component_data(
        component_id=created.component_id,
        revision=created.revision,
        format="csv",
        bundle=False,
        revision_store=revision_store,
        frame_store=frame_store,
        site_url="https://example.org",
    )

    header, *rows = exported.data.decode().strip().splitlines()
    columns = header.split(",")
    cells = [dict(zip(columns, row.split(","), strict=True)) for row in rows]
    assert [(row["recipient"], row["time_period"]) for row in cells] == [
        ("country/ETH", "2024"),
        ("country/GHA", "2024"),
    ]


def test_a_bar_chart_with_a_stated_window_keeps_the_latest_shared_period(write):
    frame = _frame(_KENYA_LAGS)
    intent = _intent(
        frame,
        member="bar_chart",
        recipients=list(_KENYA_LAGS),
        window={"start": "2022", "end": "2024"},
    )

    reply = write(frame, intent)

    component = _component(reply)
    assert component["category_axis"]["period"] == "2023"
    assert [bar["label"] for bar in component["series"]] == ["Ethiopia", "Ghana", "Kenya"]
    assert _warnings(reply) == []


def test_a_bar_chart_no_series_can_draw_at_its_relative_end_is_refused(write):
    """The frame's anchor is 2024 and the listed series stop at 2023."""
    frame = _frame(
        {"country/ETH": _TWO_YEARS, "country/GHA": _TWO_YEARS},
        anchor=RollingAnchor(period="2024", rule="half1"),
    )
    intent = _intent(
        frame,
        member="bar_chart",
        recipients=["country/ETH", "country/GHA"],
        window={"start": _back(-1), "end": _LATEST},
    )

    with pytest.raises(RollingPeriodInvalid) as excinfo:
        write(frame, intent)

    assert (excinfo.value.code, excinfo.value.pointer) == (ROLLING_ANCHOR_UNHELD, "/window")


# -- left-out series -------------------------------------------------------------


def test_left_out_series_are_named_once_ahead_of_the_frame_warnings(write):
    frame = _frame(
        {
            "country/ETH": _THREE_YEARS,
            "country/GHA": _THREE_YEARS,
            "country/KEN": _TWO_YEARS,
            "country/TCD": _TWO_YEARS,
        },
        manifest={
            "schema_version": 1,
            "warnings": [{"code": "revision_unavailable", "message": "No revision endpoint."}],
        },
    )
    intent = _intent(
        frame,
        member="line_chart",
        recipients=["country/TCD", "country/ETH", "country/KEN", "country/GHA"],
        window={"start": _back(-1), "end": _LATEST},
    )

    reply = write(frame, intent)

    assert _warnings(reply) == [
        {"kind": "coverage_gap", "detail": "2 series have no 2024 value: Kenya, Chad"},
        {"kind": "provisional", "detail": "No revision endpoint."},
    ]
    assert len(_component(reply)["series"]) == 4


def test_a_rolling_component_with_every_series_at_the_anchor_carries_no_warning(write):
    frame = _frame({"country/ETH": _THREE_YEARS, "country/GHA": _THREE_YEARS})
    intent = _intent(
        frame,
        member="line_chart",
        recipients=["country/ETH", "country/GHA"],
        window={"start": _back(-1), "end": _LATEST},
    )

    assert _warnings(write(frame, intent)) == []


@pytest.mark.parametrize(
    "window",
    [
        pytest.param({"start": _back(-1), "end": _LATEST}, id="rolling"),
        pytest.param({"start": "2023", "end": "2024"}, id="fixed"),
    ],
)
def test_a_select_value_the_frame_lacks_is_named_when_rolling_and_refused_when_fixed(write, window):
    frame = _frame({"country/ETH": _THREE_YEARS, "country/GHA": _THREE_YEARS})
    intent = _intent(
        frame,
        member="line_chart",
        recipients=["country/ETH", "country/TCD", "country/GHA"],
        window=window,
    )

    if isinstance(window["end"], str):
        with pytest.raises(BindingInvalid) as excinfo:
            write(frame, intent)
        assert (excinfo.value.pointer, excinfo.value.dimension) == (
            "/select/recipient",
            "recipient",
        )
        return

    reply = write(frame, intent)

    assert [one["label"] for one in _component(reply)["series"]] == ["Ethiopia", "Ghana"]
    assert _warnings(reply) == [
        {"kind": "coverage_gap", "detail": "1 series has no 2024 value: Chad"}
    ]


def test_a_frame_anchor_sets_latest_and_names_the_series_its_recipe_left_out(write):
    """The series hold 2024, and the recipe's anchor is 2023. The select admits Chad and
    leaves Mali out, so only Chad is named."""
    frame = _frame(
        {"country/ETH": _THREE_YEARS, "country/TCD": ("2022",)},
        anchor=RollingAnchor(
            period="2023",
            rule="half1",
            left_out=(
                LeftOutSeries(root=0, coordinates={"recipient": "country/TCD"}, label="Chad"),
                LeftOutSeries(root=1, coordinates={"recipient": "country/MLI"}, label="Mali"),
            ),
        ),
    )
    intent = _intent(
        frame,
        member="line_chart",
        recipients=["country/ETH", "country/TCD"],
        window={"start": _back(-1), "end": _LATEST},
    )

    reply = write(frame, intent)

    assert _component(reply)["x_axis"]["window"] == {"start": "2022", "end": "2023"}
    assert _warnings(reply) == [
        {"kind": "coverage_gap", "detail": "1 series has no 2023 value: Chad"}
    ]


def test_the_left_out_warning_fits_the_envelope_bounds():
    names = [f"Recipient number {index:03d}" for index in range(60)]
    manifest_warnings = [{"kind": "provisional", "detail": f"w{index}"} for index in range(50)]

    warnings = _with_left_out_warning(
        provenance={"warnings": manifest_warnings}, period="2024", names=names
    )["warnings"]

    assert len(warnings) == 50
    assert warnings[1:] == manifest_warnings[:49]
    detail = warnings[0]["detail"]
    assert len(detail) <= 500
    shown = detail.removeprefix("60 series have no 2024 value: ").split(" and ")[0].split(", ")
    assert shown == names[: len(shown)]
    assert detail.endswith(f" and {60 - len(shown)} more")
    # One more name would pass the bound.
    assert len(detail) + len(", " + names[len(shown)]) > 500


# -- rendering on another frame --------------------------------------------------

_FOUR = ("country/ETH", "country/GHA", "country/KEN", "country/TCD")


def test_a_stored_relative_intent_resolves_again_on_each_frame_it_is_rendered_on(
    write, frame_store, revision_store, release_registry
):
    first = _frame(dict.fromkeys(_FOUR, _TWO_YEARS))
    intent = _intent(
        first,
        member="line_chart",
        recipients=list(_FOUR),
        window={"start": _back(-1), "end": _LATEST},
    )
    created = write(first, intent)
    assert _component(created)["x_axis"]["window"]["end"] == "2023"

    half_hold_2024 = _frame(
        {**dict.fromkeys(_FOUR[:2], _THREE_YEARS), **dict.fromkeys(_FOUR[2:], _TWO_YEARS)},
        version="2",
    )
    one_holds_2024 = _frame(
        {
            "country/ETH": _THREE_YEARS,
            "country/GHA": _TWO_YEARS,
            "country/KEN": _TWO_YEARS,
            "country/TCD": ("2022",),
        },
        version="3",
    )

    def render(frame):
        frame_store.put(resolved_frame=frame)
        return render_component(
            component_id=created.component_id,
            revision=created.revision,
            target_frame=frame.frame_ref,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    moved = render(half_hold_2024)
    assert _component(moved)["x_axis"]["window"] == {"start": "2023", "end": "2024"}
    assert _warnings(moved) == [
        {"kind": "coverage_gap", "detail": "2 series have no 2024 value: Kenya, Chad"}
    ]

    held = render(one_holds_2024)
    assert _component(held)["x_axis"]["window"] == {"start": "2022", "end": "2023"}
    assert _warnings(held) == [
        {"kind": "coverage_gap", "detail": "1 series has no 2023 value: Chad"}
    ]

    stored = revision_store.read_current(component_id=created.component_id)
    assert stored.revision == 1
    assert stored.stored_intent["window"] == {"start": _back(-1), "end": _LATEST}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            {"window": {"start": "2020", "end": "2024"}},
            {"window": {"start": _back(-4), "end": _LATEST}},
            id="window-label-to-relative",
        ),
        pytest.param(
            {"window": {"start": _back(-4), "end": _LATEST}},
            {"window": {"start": "2020", "end": "2024"}},
            id="window-relative-to-label",
        ),
        pytest.param(
            {"type": "stat_card", "change": {"basis": ["2023", "2024"]}},
            {"change": {"basis": [_back(-1), _LATEST]}},
            id="basis-label-to-relative",
        ),
        pytest.param(
            {"type": "stat_card", "change": {"basis": [_back(-1), _LATEST]}},
            {"change": {"basis": ["2023", "2024"]}},
            id="basis-relative-to-label",
        ),
    ],
)
def test_a_revise_between_a_label_and_the_relative_period_it_resolves_to_appends_a_revision(
    write, revision_store, first, second
):
    """Both forms mint the same envelope on this frame, and the stored intent must change."""
    frame, intent = _malaria_intent(**first)
    if intent["type"] == "stat_card":
        del intent["window"]
    created = write(frame, intent)

    revised = write(
        frame, second, component_id=created.component_id, write_token=created.write_token
    )
    again = write(frame, second, component_id=created.component_id, write_token=created.write_token)

    assert (revised.revision, again.revision) == (2, 2)
    assert revised.resolved_payload["component"] == created.resolved_payload["component"]
    stored = revision_store.read_current(component_id=created.component_id).stored_intent
    assert {key: stored[key] for key in second} == second


def test_a_revise_stating_the_window_the_head_leaves_to_coverage_pins_it(
    write, frame_store, revision_store, release_registry
):
    """The head omits its window, so it follows coverage. Stating that coverage mints the
    same envelope today and holds the window on a frame with one more year."""
    frame = _frame({"country/ETH": _THREE_YEARS})
    created = write(frame, _intent(frame, member="line_chart", recipients="country/ETH"))
    stated = {"window": {"start": "2022", "end": "2024"}}

    revised = write(
        frame, stated, component_id=created.component_id, write_token=created.write_token
    )
    again = write(frame, stated, component_id=created.component_id, write_token=created.write_token)

    assert (revised.revision, again.revision) == (2, 2)
    assert revised.resolved_payload["component"] == created.resolved_payload["component"]
    assert revision_store.read_current(component_id=created.component_id).stored_intent[
        "window"
    ] == {"start": "2022", "end": "2024"}

    longer = _frame({"country/ETH": (*_THREE_YEARS, "2025")}, version="2")
    frame_store.put(resolved_frame=longer)
    ends = {
        revision: render_component(
            component_id=created.component_id,
            revision=revision,
            target_frame=longer.frame_ref,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        ).resolved_payload["component"]["x_axis"]["window"]["end"]
        for revision in (1, 2)
    }
    assert ends == {1: "2025", 2: "2024"}


def test_an_intent_without_a_relative_period_keeps_its_frame_anchor_out_of_the_card(write):
    """A stated window on a frame with an anchor draws the stated periods and names no one."""
    frame = dataclasses.replace(
        malaria_ethiopia_resolved_frame(),
        anchor=RollingAnchor(
            period="2020",
            rule="half1",
            left_out=(LeftOutSeries(root=0, coordinates={}, label="Chad"),),
        ),
    )
    _, intent = _malaria_intent()

    reply = write(frame, intent)

    assert _component(reply)["x_axis"]["window"] == {"start": "2015", "end": "2024"}
    assert _warnings(reply) == []


# -- a stored select on a frame that moved ---------------------------------------


@pytest.fixture
def render(frame_store, revision_store, release_registry):
    def _render(created, frame):
        frame_store.put(resolved_frame=frame)
        return render_component(
            component_id=created.component_id,
            revision=created.revision,
            target_frame=frame.frame_ref,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    return _render


def _left_out(*recipients: str) -> tuple[LeftOutSeries, ...]:
    return tuple(
        LeftOutSeries(root=0, coordinates={"recipient": one}, label=_LABELS[one])
        for one in recipients
    )


@pytest.mark.parametrize(
    ("target_holds", "stored_refusal", "bars"),
    [
        pytest.param(_FOUR[:2], None, ["Ethiopia", "Ghana"], id="two-of-four-stay"),
        pytest.param(
            (*_FOUR[:2], "country/MLI"), None, ["Ethiopia", "Ghana", "Mali"], id="one-joins"
        ),
        pytest.param(
            (*_FOUR[:2], "country/MLI"), "io", ["Ethiopia", "Ghana"], id="stored-frame-gone"
        ),
        pytest.param(
            (*_FOUR[:2], "country/MLI"),
            "missing_rendering",
            ["Ethiopia", "Ghana"],
            id="stored-frame-malformed",
        ),
    ],
)
def test_a_select_of_every_value_follows_the_target_and_names_the_left_out_series(
    monkeypatch, write, render, frame_store, revision_store, target_holds, stored_refusal, bars
):
    """A bar chart on a growth frame, every row one span, selects all four recipients. On the
    refreshed frame the anchor moved to 2025 and two recipients hold no value there. When the
    stored frame cannot be loaded, the select keeps the values the target holds."""
    stored = _frame(dict.fromkeys(_FOUR, ("2019/2024",)))
    created = write(stored, _intent(stored, member="bar_chart", recipients=list(_FOUR)))
    target = _frame(
        dict.fromkeys(target_holds, ("2020/2025",)),
        version="2",
        anchor=RollingAnchor(period="2025", rule="half1", left_out=_left_out(*_FOUR[2:])),
    )
    if stored_refusal is not None:
        load = frame_store.load

        def refuse_stored(*, frame_ref: FrameRef) -> ResolvedFrame:
            if frame_ref == stored.frame_ref:
                raise FrameUnavailable(reason=stored_refusal, detail="exemplar")
            return load(frame_ref=frame_ref)

        monkeypatch.setattr(frame_store, "load", refuse_stored)

    rendered = render(created, target)

    assert [bar["label"] for bar in _component(rendered)["series"]] == bars
    assert _warnings(rendered) == [
        {"kind": "coverage_gap", "detail": "2 series have no 2025 value: Kenya, Chad"}
    ]
    stored_intent = revision_store.read_current(component_id=created.component_id).stored_intent
    assert stored_intent["select"]["recipient"] == list(_FOUR)
    with pytest.raises(BindingInvalid) as excinfo:
        write(
            target,
            {**stored_intent, "frame_ref": _intent(target, member="", recipients=[])["frame_ref"]},
        )
    assert excinfo.value.pointer == "/select/recipient"


def test_a_select_of_some_values_drops_the_value_the_target_lacks_and_names_it(write, render):
    stored = _frame(dict.fromkeys(_FOUR, _THREE_YEARS))
    created = write(
        stored,
        _intent(
            stored,
            member="line_chart",
            recipients=["country/ETH", "country/KEN"],
            window={"start": "2022", "end": "2024"},
        ),
    )
    target = _frame(dict.fromkeys(("country/ETH", "country/GHA"), _THREE_YEARS), version="2")

    rendered = render(created, target)

    assert [one["label"] for one in _component(rendered)["series"]] == ["Ethiopia"]
    assert _warnings(rendered) == [
        {"kind": "coverage_gap", "detail": "1 series has no 2024 value: Kenya"}
    ]


def _provenance_frame(series, *, version: str):
    """One `country/ETH` series per `(provenance, periods)` in `series`."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id=_SPEC_ID, content_version="cv_" + version * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": _STAT_VAR,
                "recipient": "country/ETH",
                "time_period": period,
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": provenance,
            }
            for provenance, periods in series.items()
            for period in periods
        ],
        labels=_LABELS,
    )


@pytest.mark.parametrize(
    ("target_holds", "drawn"),
    [
        pytest.param(("country/GHA", "country/KEN"), None, id="several-refuse"),
        pytest.param(("country/GHA",), ("2024", 24.0), id="one-follows"),
    ],
)
def test_a_scalar_whose_value_left_follows_only_a_target_holding_one_value(
    write, render, target_holds, drawn
):
    stored = _frame({"country/ETH": _THREE_YEARS})
    created = write(stored, _intent(stored, member="stat_card", recipients="country/ETH"))
    target = _frame(dict.fromkeys(target_holds, _THREE_YEARS), version="2")

    if drawn is None:
        with pytest.raises(BindingInvalid) as excinfo:
            render(created, target)
        assert excinfo.value.pointer == "/select/recipient"
        return

    component = _component(render(created, target))
    assert (component["period"], component["value"]) == drawn


def test_a_facet_binding_of_every_value_is_never_widened(write, render):
    stored = _provenance_frame({"src_a": _THREE_YEARS, "src_b": _THREE_YEARS}, version="1")
    intent = _intent(stored, member="table", recipients="country/ETH")
    intent["select"]["provenance"] = ["src_a", "src_b"]
    created = write(stored, intent)
    target = _provenance_frame({"src_a": _THREE_YEARS, "src_c": _THREE_YEARS}, version="2")

    with pytest.raises(BindingInvalid) as excinfo:
        render(created, target)

    assert excinfo.value.pointer == "/select/provenance"


@pytest.mark.parametrize(
    ("route", "recipients", "window", "warnings"),
    [
        pytest.param(
            "render",
            "country/ETH",
            {"start": _back(-1), "end": _LATEST},
            [],
            id="render-unselected-rolling",
        ),
        pytest.param(
            "render",
            "country/ETH",
            {"start": "2022", "end": "2024"},
            [],
            id="render-unselected-fixed",
        ),
        pytest.param(
            "render",
            ["country/ETH", "country/TCD"],
            {"start": "2022", "end": "2024"},
            [{"kind": "coverage_gap", "detail": "1 series has no 2024 value: Chad"}],
            id="render-selected-fixed",
        ),
        pytest.param(
            "write",
            "country/ETH",
            {"start": _back(-1), "end": _LATEST},
            [],
            id="write-unselected-rolling",
        ),
        pytest.param(
            "write",
            ["country/ETH", "country/TCD"],
            {"start": _back(-1), "end": _LATEST},
            [{"kind": "coverage_gap", "detail": "1 series has no 2024 value: Chad"}],
            id="write-selected-rolling",
        ),
    ],
)
def test_a_card_names_the_anchor_left_out_series_its_select_admits(
    write, render, route, recipients, window, warnings
):
    stored = _frame({"country/ETH": _THREE_YEARS, "country/TCD": _THREE_YEARS})
    target = _frame(
        {"country/ETH": _THREE_YEARS, "country/TCD": _TWO_YEARS},
        version="2",
        anchor=RollingAnchor(period="2024", rule="half1", left_out=_left_out("country/TCD")),
    )
    on = stored if route == "render" else target
    drawn = write(on, _intent(on, member="line_chart", recipients=recipients, window=window))
    if route == "render":
        drawn = render(drawn, target)

    assert _warnings(drawn) == warnings


def test_a_widened_bar_chart_draws_at_the_period_every_bar_shares(write, render):
    """Kenya joins in Ghana's place and stops at 2023, so Ethiopia's bar is drawn at 2023
    too. Ghana, which the stored frame held, is named at the window's end."""
    stored = _frame({"country/ETH": _THREE_YEARS, "country/GHA": _THREE_YEARS})
    created = write(
        stored, _intent(stored, member="bar_chart", recipients=["country/ETH", "country/GHA"])
    )
    target = _frame({"country/ETH": _THREE_YEARS, "country/KEN": _TWO_YEARS}, version="2")

    rendered = render(created, target)

    component = _component(rendered)
    assert component["category_axis"]["period"] == "2023"
    assert [bar["label"] for bar in component["series"]] == ["Ethiopia", "Kenya"]
    assert _warnings(rendered) == [
        {"kind": "coverage_gap", "detail": "1 series has no 2024 value: Ghana"}
    ]
