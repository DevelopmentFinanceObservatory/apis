"""Pin member-specific write, frame, and composition behaviour."""

from __future__ import annotations

import jsonschema
import pytest

from assembler.errors import BindingInvalid, FrameUnavailable
from assembler.mint.validate import component_registry
from assembler.pipeline import get_component, write_component
from assembler.testing import malaria_ethiopia_resolved_frame

from ._pipeline_support import _malaria_intent, _stat_card_intent

# -- composition ---------------------------------------------------------------


def test_revise_carrying_composition_id_is_binding_invalid(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent={"title": "New title"},
            component_id=created.component_id,
            write_token=created.write_token,
            composition_id="cps_x",
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/composition_id"


def test_create_in_new_composition_mints_composition_and_reply_carries_both_identities(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)

    reply = write_component(
        intent=intent,
        component_id=None,
        composition_id="new",
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert reply.component_id.startswith("cmp_")
    assert reply.composition_id is not None
    assert reply.composition_id.startswith("cps_")
    assert reply.composition_revision == 1


def test_create_outside_any_composition_reports_null_composition_fields(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)

    reply = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert reply.composition_id is None
    assert reply.composition_revision is None

    got = get_component(
        component_id=reply.component_id,
        revision_store=revision_store,
    )
    assert got["composition_id"] is None
    assert got["composition_revision"] is None


def test_no_op_patch_and_get_on_a_composition_member_report_the_real_id_and_revision(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)

    created = write_component(
        intent=intent,
        component_id=None,
        composition_id="new",
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    no_op = write_component(
        intent={"title": "Malaria grant disbursements to Ethiopia"},  # already the current value
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert no_op.changed == ()
    assert no_op.composition_id == created.composition_id
    assert no_op.composition_revision == created.composition_revision

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["composition_id"] == created.composition_id
    assert got["composition_revision"] == created.composition_revision


def test_substantive_revise_of_a_composition_member_reports_the_advanced_composition_revision(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)

    created = write_component(
        intent=intent,
        component_id=None,
        composition_id="new",
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revised = write_component(
        intent={"title": "New title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert revised.changed != ()
    assert revised.composition_id == created.composition_id
    assert revised.composition_revision == created.composition_revision + 1


def test_substantive_revise_of_a_component_outside_any_composition_reports_null_composition_fields(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)

    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revised = write_component(
        intent={"title": "New title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert revised.changed != ()
    assert revised.composition_id is None
    assert revised.composition_revision is None


# -- stat_card -----------------------------------------------------------------


def test_stat_card_create_headline_is_the_latest_observed_point(
    frame_store, revision_store, release_registry
):
    frame, intent = _stat_card_intent()
    frame_store.put(resolved_frame=frame)

    reply = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert reply.fact_summary["series"][0]["last_value"] == "442"
    assert reply.fact_summary["change"] is None


# -- note_narration --------------------------------------------------------


def test_note_narration_create_pins_no_frame(frame_store, revision_store, release_registry):
    """`note_narration` has no `frame_ref`, so create resolves and pins no frame."""
    reply = write_component(
        intent={"type": "note_narration", "text": "Disbursements rose each year after 2020."},
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert reply.component_id.startswith("cmp_")
    assert frame_store.pins == ()

    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    assert record.resolved_payload["component"] == {
        "type": "note_narration",
        "text": "Disbursements rose each year after 2020.",
        "authorship": "model",
    }
    assert record.resolved_payload["provenance"]["inputs"] == []
    assert record.resolved_payload["provenance"].get("sources", []) == []

    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )


def test_note_narration_revise_updates_text(frame_store, revision_store, release_registry):
    created = write_component(
        intent={"type": "note_narration", "text": "Original text."},
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    revised = write_component(
        intent={"text": "Revised text."},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert revised.revision == 2
    record = revision_store.read_revision(component_id=created.component_id, revision=2)
    assert record.resolved_payload["component"]["text"] == "Revised text."


# -- source_card -------------------------------------------------------------


def test_source_card_create_pins_the_frame_and_names_its_sources(
    frame_store, revision_store, release_registry
):
    """`source_card` resolves and pins its frame but binds no series."""
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)

    reply = write_component(
        intent={
            "type": "source_card",
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert len(frame_store.pins) == 1
    assert frame_store.pins[0][0] == frame.frame_ref

    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    assert record.resolved_payload["component"] == {
        "type": "source_card",
        "source_refs": ["src_oecd_dac"],
    }
    assert record.resolved_payload["provenance"]["sources"][0]["id"] == "src_oecd_dac"

    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )


def test_source_card_naming_an_unknown_frame_is_frame_unavailable(
    frame_store, revision_store, release_registry
):
    """An unknown source-card frame raises `FrameUnavailable`."""
    with pytest.raises(FrameUnavailable):
        write_component(
            intent={
                "type": "source_card",
                "frame_ref": {
                    "spec_id": "spec_" + "9" * 32,
                    "content_version": "cv_" + "9" * 32,
                },
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
