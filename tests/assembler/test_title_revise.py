"""A title-only PATCH mints a new revision and reports `title` in
`changed` for every member whose grammar carries a `title`.
"""

from __future__ import annotations

import jsonschema
import pytest

from assembler.mint.validate import component_registry
from assembler.pipeline import write_component

from ._pipeline_support import (
    _bar_chart_intent,
    _malaria_intent,
    _stat_card_intent,
    _table_intent,
)

_TITLED_MEMBER_INTENTS = {
    "line_chart": _malaria_intent,
    "bar_chart": _bar_chart_intent,
    "table": _table_intent,
    "stat_card": _stat_card_intent,
}


@pytest.mark.parametrize("member", sorted(_TITLED_MEMBER_INTENTS))
def test_title_only_revise_mints_a_new_revision_and_reports_title_changed(
    member, frame_store, revision_store, release_registry
):
    build_intent = _TITLED_MEMBER_INTENTS[member]
    frame, intent = build_intent(title="Title A")
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
        intent={"title": "Title B"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert revised.revision == 2
    assert "title" in {field.field for field in revised.changed}

    record = revision_store.read_revision(component_id=created.component_id, revision=2)
    assert record.resolved_payload["component"]["title"]["text"] == "Title B"


def test_stat_card_without_a_title_carries_the_default_title_distinct_from_label(
    frame_store, revision_store, release_registry
):
    frame, intent = _stat_card_intent()
    frame_store.put(resolved_frame=frame)

    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    component = record.resolved_payload["component"]
    assert component["title"]["text"]
    assert component["title"]["text"] != component["label"]


def test_minted_stat_card_validates_against_the_component_schema(
    frame_store, revision_store, release_registry
):
    frame, intent = _stat_card_intent(title="Malaria grants to Ethiopia")
    frame_store.put(resolved_frame=frame)

    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=created.component_id, revision=1)
    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )
    jsonschema.Draft202012Validator(release.component_schemas["stat_card"]).validate(
        record.resolved_payload["component"]
    )
