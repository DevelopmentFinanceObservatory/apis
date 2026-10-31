"""Checks that `render_component` draws a stored revision's intent on another frame under
the stored revision's number, pins nothing, and leaves the revision store as it was."""

from __future__ import annotations

import dataclasses

import pytest

from assembler.errors import BindingInvalid, ReleaseMismatch, RevisionNotFound
from assembler.fakes import FakeReleaseRegistry
from assembler.pipeline import (
    ComponentHoldsNoFrame,
    get_component,
    render_component,
    write_component,
)
from assembler.ports import FrameRef, FrameUnavailable
from assembler.reply import format_value
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_rows,
)

from ._pipeline_support import _malaria_intent, _multi_recipient_frame

_TARGET = FrameRef(
    spec_id=malaria_ethiopia_frame_ref().spec_id,
    content_version="cv_" + "b" * 32,
)
_TARGET_BODY = {"spec_id": _TARGET.spec_id, "content_version": _TARGET.content_version}
_MOVED_BY = 1000.0


def _moved_frame(*, recipients: tuple[str, ...] = ("country/ETH",)):
    """The malaria frame under `_TARGET` for each of `recipients`, every value `_MOVED_BY`
    higher."""
    return build_resolved_frame(
        frame_ref=_TARGET,
        role_keys=("recipient",),
        rows=[
            {**row, "recipient": recipient, "value": row["value"] + _MOVED_BY}
            for recipient in recipients
            for row in malaria_ethiopia_rows()
        ],
        labels={"country/ETH": "Ethiopia"},
    )


class _AppendRefusingStore:
    """Reads through to `inner` and fails the test on any step of the append path."""

    def __init__(self, *, inner):
        self._inner = inner

    def begin_write(self, **kwargs):
        raise AssertionError("a render opened a write")

    def allocate_component_id(self):
        raise AssertionError("a render allocated a component id")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _create(*, frame_store, revision_store, release_registry, **overrides):
    frame, intent = _malaria_intent(**overrides)
    frame_store.put(resolved_frame=frame)
    created = write_component(
        intent=intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    return frame, created


def _values(payload) -> list[float]:
    (series,) = payload["component"]["series"]
    return [point["value"] for point in series["points"] if "value" in point]


def test_render_draws_the_stored_intent_on_the_target_frame_and_appends_nothing(
    frame_store, revision_store, release_registry
):
    frame, created = _create(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    frame_store.put(resolved_frame=_moved_frame())
    head_before = revision_store.read_current(component_id=created.component_id)

    rendered = render_component(
        component_id=created.component_id,
        revision=created.revision,
        target_frame=_TARGET,
        frame_store=frame_store,
        revision_store=_AppendRefusingStore(inner=revision_store),
        release_registry=release_registry,
    )

    payload = rendered.resolved_payload
    assert (rendered.component_id, rendered.revision) == (created.component_id, 1)
    assert (payload["component_id"], payload["revision"]) == (created.component_id, 1)
    assert payload["provenance"]["inputs"] == [_TARGET_BODY]
    assert payload["provenance"]["refreshed_from"] == 1
    assert _values(payload) == [value + _MOVED_BY for value in _values(created.resolved_payload)]
    assert rendered.fact_summary["series"][0]["first_value"] == format_value(
        value=412.3 + _MOVED_BY, profile=release_registry.pinned().formatter_profile
    )
    assert rendered.changed == ()
    assert (rendered.write_token, rendered.composition_write_token) == (None, None)

    assert len(revision_store._revisions[created.component_id]) == 1
    assert revision_store.read_current(component_id=created.component_id) == head_before
    assert head_before.resolved_payload["provenance"]["inputs"] == [
        {"spec_id": frame.frame_ref.spec_id, "content_version": frame.frame_ref.content_version}
    ]


def test_render_draws_the_revision_asked_for_when_a_later_one_exists(
    frame_store, revision_store, release_registry
):
    _, created = _create(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    revised = write_component(
        intent={"title": "A later title"},
        component_id=created.component_id,
        write_token=created.write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    frame_store.put(resolved_frame=_moved_frame())

    rendered = render_component(
        component_id=created.component_id,
        revision=1,
        target_frame=_TARGET,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert revised.revision == 2
    assert rendered.revision == 1
    assert rendered.resolved_payload["provenance"]["refreshed_from"] == 1
    assert (
        rendered.resolved_payload["component"]["title"]
        == created.resolved_payload["component"]["title"]
    )
    assert revision_store.read_current(component_id=created.component_id).revision == 2
    # A create and a revise are not renderings of another revision.
    assert created.resolved_payload["provenance"]["refreshed_from"] is None
    assert revised.resolved_payload["provenance"]["refreshed_from"] is None


def test_repeated_renders_pin_nothing(frame_store, revision_store, release_registry):
    _, created = _create(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    frame_store.put(resolved_frame=_moved_frame())

    for _ in range(2):
        render_component(
            component_id=created.component_id,
            revision=1,
            target_frame=_TARGET,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    assert frame_store.pins == ()


def test_a_target_frame_lacking_the_selected_series_refuses_as_a_write_does_and_pins_nothing(
    frame_store, revision_store, release_registry
):
    _, created = _create(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    # A target holding one recipient would draw it in Ethiopia's place.
    frame_store.put(resolved_frame=_moved_frame(recipients=("country/KEN", "country/GHA")))
    stored_intent = revision_store.read_current(component_id=created.component_id).stored_intent

    with pytest.raises(BindingInvalid) as from_write:
        write_component(
            intent={**stored_intent, "frame_ref": _TARGET_BODY},
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    with pytest.raises(BindingInvalid) as from_render:
        render_component(
            component_id=created.component_id,
            revision=1,
            target_frame=_TARGET,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    assert from_render.value.pointer == from_write.value.pointer == "/select/recipient"
    assert from_render.value.legal_values == from_write.value.legal_values
    assert (_TARGET, created.component_id) not in frame_store.pins


class _CompositionFailingStore:
    """Reads through to `inner` except `composition_of`, which fails as an unreachable store
    would."""

    def __init__(self, *, inner):
        self._inner = inner

    def composition_of(self, *, component_id):
        raise ConnectionError("revision store unreachable")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_render_whose_composition_lookup_fails_pins_nothing(
    frame_store, revision_store, release_registry
):
    _, created = _create(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )
    frame_store.put(resolved_frame=_moved_frame())

    with pytest.raises(ConnectionError):
        render_component(
            component_id=created.component_id,
            revision=1,
            target_frame=_TARGET,
            frame_store=frame_store,
            revision_store=_CompositionFailingStore(inner=revision_store),
            release_registry=release_registry,
        )

    assert (_TARGET, created.component_id) not in frame_store.pins


def test_an_intent_stored_under_the_first_release_renders_under_the_current_one(
    frame_store, revision_store, release
):
    """The stored revision carries an older release's label and formatter profile, an intent
    of only the fields the first intent grammar had, and a bar chart payload with the axis
    names that release wrote, which a read under the current release refuses."""
    older = dataclasses.replace(
        release,
        catalog_version="2026.7.0",
        formatter_profile={**release.formatter_profile, "significant_digits": 6},
    )
    frame = _multi_recipient_frame()
    frame_store.put(resolved_frame=frame)
    created = write_component(
        intent={
            "type": "bar_chart",
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/ETH", "country/GHA", "country/KEN"],
            },
            "window": {"start": "2020", "end": "2022"},
            "title": "Malaria grants",
        },
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=FakeReleaseRegistry(release=older),
    )
    revisions = revision_store._revisions[created.component_id]
    component = dict(revisions[0].resolved_payload["component"])
    component["x_axis"] = component.pop("category_axis")
    component["y_axis"] = component.pop("value_axis")
    revisions[0] = dataclasses.replace(
        revisions[0], resolved_payload={**revisions[0].resolved_payload, "component": component}
    )
    with pytest.raises(ReleaseMismatch):
        get_component(component_id=created.component_id, revision_store=revision_store)
    frame_store.put(resolved_frame=dataclasses.replace(frame, frame_ref=_TARGET))

    rendered = render_component(
        component_id=created.component_id,
        revision=1,
        target_frame=_TARGET,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=FakeReleaseRegistry(release=release),
    )

    assert rendered.catalog_version == release.catalog_version != "2026.7.0"
    assert rendered.resolved_payload["catalog_version"] == release.catalog_version
    assert {"category_axis", "value_axis"} <= set(rendered.resolved_payload["component"])
    assert rendered.resolved_payload["provenance"]["inputs"] == [_TARGET_BODY]
    assert revision_store.read_current(component_id=created.component_id).catalog_version == (
        "2026.7.0"
    )


def test_a_revision_minted_under_a_newer_release_refuses(frame_store, revision_store, release):
    _, created = _create(
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=FakeReleaseRegistry(
            release=dataclasses.replace(release, catalog_version="2099.1.0")
        ),
    )
    frame_store.put(resolved_frame=_moved_frame())

    with pytest.raises(ReleaseMismatch) as excinfo:
        render_component(
            component_id=created.component_id,
            revision=1,
            target_frame=_TARGET,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=FakeReleaseRegistry(release=release),
        )

    assert excinfo.value.catalog_version == "2099.1.0"
    assert frame_store.pins == ()


def test_a_note_has_no_rendering_on_a_frame(frame_store, revision_store, release_registry):
    created = write_component(
        intent={"type": "note_narration", "text": "A note."},
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    frame_store.put(resolved_frame=_moved_frame())

    with pytest.raises(ComponentHoldsNoFrame):
        render_component(
            component_id=created.component_id,
            revision=1,
            target_frame=_TARGET,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    assert frame_store.pins == ()


def test_a_missing_revision_and_an_unavailable_target_frame_refuse(
    frame_store, revision_store, release_registry
):
    _, created = _create(
        frame_store=frame_store, revision_store=revision_store, release_registry=release_registry
    )

    def render(revision: int):
        return render_component(
            component_id=created.component_id,
            revision=revision,
            target_frame=_TARGET,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    with pytest.raises(FrameUnavailable):
        render(1)
    frame_store.put(resolved_frame=_moved_frame())
    with pytest.raises(RevisionNotFound):
        render(2)
