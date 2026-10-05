"""Clearing an authored field restores its default.

Scalar `title` and `scale` values are disclosed in `changed`; object fields `range` and `change`
are withheld. The scale case uses `line_chart`, since `bar_chart` rejects `log` under
`bar_zero_in_domain`.
"""

from __future__ import annotations

import pytest

from assembler.pipeline import get_component, write_component

from ._pipeline_support import (
    _bar_chart_intent,
    _malaria_intent,
    _stat_card_intent,
)


def _write(
    intent, *, frame_store, revision_store, release_registry, component_id=None, write_token=None
):
    return write_component(
        intent=intent,
        component_id=component_id,
        composition_id=None,
        write_token=write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )


def _changed_field(reply, name):
    return next(field for field in reply.changed if field.field == name)


def test_revise_clears_authored_title_reverts_to_the_generated_default(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent(title="Authored Title")
    frame_store.put(resolved_frame=frame)
    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revised = _write(
        {"title": None},
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        component_id=created.component_id,
        write_token=created.write_token,
    )

    changed_title = _changed_field(revised, "title")
    assert changed_title.prior == "Authored Title"
    assert changed_title.new is None

    record = revision_store.read_revision(
        component_id=created.component_id, revision=revised.revision
    )
    title_text = record.resolved_payload["component"]["title"]["text"]
    assert title_text == "dcid:MalariaGrantDisbursements — Ethiopia"
    assert title_text != "Authored Title"

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert "title" not in got["stored_intent"]


def test_revise_clears_authored_scale_reverts_to_linear(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent(scale="log")
    frame_store.put(resolved_frame=frame)
    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revised = _write(
        {"scale": None},
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        component_id=created.component_id,
        write_token=created.write_token,
    )

    changed_scale = _changed_field(revised, "scale")
    assert changed_scale.prior == "log"
    assert changed_scale.new is None

    record = revision_store.read_revision(
        component_id=created.component_id, revision=revised.revision
    )
    assert record.resolved_payload["component"]["y_axis"]["scale"] == "linear"

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert "scale" not in got["stored_intent"]


def _range_case_line_chart():
    return _malaria_intent(scale="linear", range={"min": 390, "max": 460}), "y_axis"


def _range_case_bar_chart():
    return _bar_chart_intent(scale="linear", range={"min": -10, "max": 460}), "value_axis"


@pytest.mark.parametrize("build_case", [_range_case_line_chart, _range_case_bar_chart])
def test_revise_clears_authored_range_reverts_to_unbounded(
    build_case, frame_store, revision_store, release_registry
):
    (frame, intent), axis_key = build_case()
    frame_store.put(resolved_frame=frame)
    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revised = _write(
        {"range": None},
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        component_id=created.component_id,
        write_token=created.write_token,
    )

    changed_range = _changed_field(revised, "range")
    assert changed_range.disclosed is False

    record = revision_store.read_revision(
        component_id=created.component_id, revision=revised.revision
    )
    assert record.resolved_payload["component"][axis_key]["range"] == {"min": None, "max": None}

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert "range" not in got["stored_intent"]


def test_revise_clears_authored_change_removes_it_from_the_stat_card(
    frame_store, revision_store, release_registry
):
    frame, intent = _stat_card_intent(change={"basis": ["2023", "2024"]})
    frame_store.put(resolved_frame=frame)
    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revised = _write(
        {"change": None},
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
        component_id=created.component_id,
        write_token=created.write_token,
    )

    changed_change = _changed_field(revised, "change")
    assert changed_change.disclosed is False
    assert revised.fact_summary["change"] is None

    record = revision_store.read_revision(
        component_id=created.component_id, revision=revised.revision
    )
    assert "change" not in record.resolved_payload["component"]

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert "change" not in got["stored_intent"]
    assert got["fact_summary"]["change"] is None
