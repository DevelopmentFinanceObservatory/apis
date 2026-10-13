"""Tests authored scales, ranges, constraints, and defaults on line and bar charts."""

from __future__ import annotations

import jsonschema
import pytest

from assembler.errors import ConstraintViolation
from assembler.mint.validate import component_registry
from assembler.pipeline import write_component
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame

from ._pipeline_support import _bar_chart_intent, _malaria_intent


def _non_positive_frame():
    """A negative observation for the log-scale constraint."""
    return build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "4" * 32, content_version="cv_" + "5" * 32),
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
            for period, value in {"2020": 10.0, "2021": -5.0, "2022": 20.0}.items()
        ],
    )


def _write(intent, *, frame_store, revision_store, release_registry):
    return write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )


@pytest.mark.parametrize("member_type", ["line_chart", "bar_chart"])
def test_authored_log_scale_over_a_non_positive_point_refuses_at_scale(
    member_type, frame_store, revision_store, release_registry
):
    frame = _non_positive_frame()
    frame_store.put(resolved_frame=frame)
    intent = {
        "type": member_type,
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:X", "recipient": "country/ETH"},
        "scale": "log",
    }

    with pytest.raises(ConstraintViolation) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "log_scale_requires_positive_values"
    assert excinfo.value.pointer == "/scale"


def test_authored_line_chart_range_that_crops_data_refuses_under_range(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent(scale="linear", range={"min": 400, "max": 460})
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ConstraintViolation) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "range_contains_visible_data"
    assert excinfo.value.pointer.startswith("/range")


def test_authored_bar_chart_range_that_crops_data_refuses_under_range(
    frame_store, revision_store, release_registry
):
    frame, intent = _bar_chart_intent(scale="linear", range={"min": None, "max": 450})
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ConstraintViolation) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "range_contains_visible_data"
    assert excinfo.value.pointer.startswith("/range")


def test_authored_bar_chart_range_excluding_zero_refuses_bar_zero_in_domain(
    frame_store, revision_store, release_registry
):
    frame, intent = _bar_chart_intent(scale="linear", range={"min": 390, "max": 460})
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ConstraintViolation) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "bar_zero_in_domain"
    assert excinfo.value.pointer == "/range"


def test_authored_bar_chart_log_scale_refuses_bar_zero_in_domain(
    frame_store, revision_store, release_registry
):
    frame, intent = _bar_chart_intent(scale="log")
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ConstraintViolation) as excinfo:
        _write(
            intent,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "bar_zero_in_domain"
    assert excinfo.value.pointer == "/scale"


def test_authored_bar_chart_range_including_zero_resolves_and_validates(
    frame_store, revision_store, release_registry
):
    frame, intent = _bar_chart_intent(scale="linear", range={"min": -10, "max": 460})
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    value_axis = record.resolved_payload["component"]["value_axis"]
    assert value_axis["scale"] == "linear"
    assert value_axis["range"] == {"min": -10, "max": 460}

    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )
    jsonschema.Draft202012Validator(
        release.component_schemas["bar_chart"], registry=registry
    ).validate(record.resolved_payload["component"])


def test_authored_line_chart_scale_and_range_containing_data_resolve_and_validate(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent(scale="linear", range={"min": 390, "max": 460})
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    y_axis = record.resolved_payload["component"]["y_axis"]
    assert y_axis["scale"] == "linear"
    assert y_axis["range"] == {"min": 390, "max": 460}

    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )
    jsonschema.Draft202012Validator(
        release.component_schemas["line_chart"], registry=registry
    ).validate(record.resolved_payload["component"])


def test_authored_bar_chart_scale_and_range_containing_data_resolve_and_validate(
    frame_store, revision_store, release_registry
):
    # A half-open bound (only `max` set) contains the visible data without
    # excluding zero, so `bar_zero_in_domain` stays quiet alongside
    # `range_contains_visible_data`.
    frame, intent = _bar_chart_intent(scale="linear", range={"min": None, "max": 460})
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    value_axis = record.resolved_payload["component"]["value_axis"]
    assert value_axis["scale"] == "linear"
    assert value_axis["range"] == {"min": None, "max": 460}

    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )
    jsonschema.Draft202012Validator(
        release.component_schemas["bar_chart"], registry=registry
    ).validate(record.resolved_payload["component"])


@pytest.mark.parametrize(
    "build_intent,axis_key",
    [(_malaria_intent, "y_axis"), (_bar_chart_intent, "value_axis")],
)
def test_intent_without_scale_or_range_keeps_the_defaults(
    build_intent, axis_key, frame_store, revision_store, release_registry
):
    frame, intent = build_intent()
    frame_store.put(resolved_frame=frame)

    created = _write(
        intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    axis = record.resolved_payload["component"][axis_key]
    assert axis["scale"] == "linear"
    assert axis["range"] == {"min": None, "max": None}
