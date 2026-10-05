"""Pin frame and component readbacks, including write-reply summaries and series coordinates."""

from __future__ import annotations

import dataclasses

import pytest

from assembler.errors import DraftNotFound, EditConflict, ReleaseMismatch, RevisionNotFound
from assembler.frame_view import compute_structural_view
from assembler.pipeline import get_component, get_frame_view, write_component
from assembler.ports import FrameRef, ResolvedFrame
from assembler.reply import (
    SERIES_COORDINATES_INCONSISTENT_MARKER,
    TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER,
    WriteReply,
)
from assembler.testing import (
    FIXTURE_RETRIEVED_ON,
    build_resolved_frame,
    malaria_ethiopia_resolved_frame,
)

from ._pipeline_support import (
    _bar_chart_intent,
    _donor_and_recipient_frame,
    _malaria_intent,
    _multi_recipient_frame,
    _stat_card_intent,
)

# -- get_frame_view ------------------------------------------------------------


def test_get_frame_view_structural_mode(frame_store):
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)
    view = get_frame_view(frame_ref=frame.frame_ref, frame_store=frame_store)
    names = {dimension["name"] for dimension in view["dimensions"]}
    assert "recipient" in names
    assert view["period_grain"] == "year"


def test_get_frame_view_dimension_mode_returns_full_values(frame_store):
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)
    view = get_frame_view(frame_ref=frame.frame_ref, dimension="recipient", frame_store=frame_store)
    assert view["values"] == ["country/ETH"]


def test_get_frame_view_query_mode_filters_listings(frame_store):
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)
    view = get_frame_view(frame_ref=frame.frame_ref, query="ETH", frame_store=frame_store)
    recipient = next(d for d in view["dimensions"] if d["name"] == "recipient")
    assert recipient["values"] == ["country/ETH"]


def _converted_unit_frame_ref():
    return FrameRef(spec_id="spec_" + "d" * 32, content_version="cv_" + "d" * 32)


def _converted_unit_resolved_frame():
    """Build a frame1 JSON shape with null, bare, and structured unit values.

    Real frame documents carry structured units as dictionaries, so construct the document directly.
    """
    doc = {
        "columns": ["stat_var", "recipient", "unit", "time_period", "value"],
        "rows": [
            ["dcid:X", "country/ETH", None, "2020", 1.0],
            ["dcid:X", "country/ETH", "USDollar", "2021", 2.0],
            [
                "dcid:X",
                "country/ETH",
                {"currency": "USD", "prices": "current"},
                "2022",
                3.0,
            ],
        ],
    }
    return ResolvedFrame(
        frame_ref=_converted_unit_frame_ref(),
        frame=doc,
        structural_view=compute_structural_view(frame=doc),
        manifest={},
        labels={},
        retrieved_on=FIXTURE_RETRIEVED_ON,
    )


def test_get_frame_view_dimension_mode_renders_a_converted_unit_as_text(frame_store):
    frame = _converted_unit_resolved_frame()
    frame_store.put(resolved_frame=frame)
    view = get_frame_view(frame_ref=frame.frame_ref, dimension="unit", frame_store=frame_store)
    assert view["values"] == ["USD/current", "USDollar"]


def test_get_frame_view_query_mode_matches_a_converted_units_rendered_text(frame_store):
    frame = _converted_unit_resolved_frame()
    frame_store.put(resolved_frame=frame)
    view = get_frame_view(frame_ref=frame.frame_ref, query="current", frame_store=frame_store)
    unit = next(d for d in view["dimensions"] if d["name"] == "unit")
    assert unit["values"] == ["USD/current"]


# -- get_component ---------------------------------------------------------------


def test_get_component_returns_stored_intent_and_fact_summary(
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

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["revision"] == 1
    assert got["stored_intent"]["type"] == "line_chart"
    assert got["fact_summary"]["series"][0]["point_count"] == 9
    # a readback must serve the same coordinates the create reply
    # carried, not `{}` -- line_chart mints a real series id, so this is
    # the id-keyed match.
    write_entry = created.fact_summary["series"][0]
    read_entry = got["fact_summary"]["series"][0]
    assert write_entry["coordinates"]
    assert read_entry["coordinates"] == write_entry["coordinates"]


def test_stat_card_create_reply_agrees_with_a_get_component_readback(
    frame_store, revision_store, release_registry
):
    """Create and readback summaries describe the stat card's single headline point."""
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
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    write_entry = created.fact_summary["series"][0]
    read_entry = got["fact_summary"]["series"][0]
    assert write_entry["point_count"] == 1
    assert read_entry["point_count"] == 1
    assert write_entry["first_value"] == write_entry["last_value"] == "442"
    assert read_entry["first_value"] == read_entry["last_value"] == "442"
    # Both summaries read the stat-card id from `series_coordinates`.
    assert write_entry["id"] is not None
    assert read_entry["id"] == write_entry["id"]
    assert write_entry["coordinates"]
    assert read_entry["coordinates"] == write_entry["coordinates"]


def test_stat_card_summary_id_is_none_when_stored_entry_has_no_id(
    frame_store, revision_store, release_registry
):
    """A missing stored `id` yields a `None` summary id."""
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
    stored = revision_store._revisions[created.component_id][-1].series_coordinates
    _replace_current_series_coordinates(
        revision_store=revision_store,
        component_id=created.component_id,
        series_coordinates=({"coordinates": dict(stored[0]["coordinates"])},),
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["fact_summary"]["series"][0]["id"] is None


def test_stat_card_summary_id_is_none_when_stored_entry_carries_an_empty_id(
    frame_store, revision_store, release_registry
):
    """An empty stored `id` yields summary id `None` and empty coordinates."""
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
    stored = revision_store._revisions[created.component_id][-1].series_coordinates
    _replace_current_series_coordinates(
        revision_store=revision_store,
        component_id=created.component_id,
        series_coordinates=({"id": "", "coordinates": dict(stored[0]["coordinates"])},),
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["fact_summary"]["series"][0]["id"] is None
    assert got["fact_summary"]["series"][0]["coordinates"] == {}


def test_table_create_reply_agrees_with_a_get_component_readback(
    frame_store, revision_store, release_registry
):
    """Compare table summaries from the create reply and stored rows."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)

    created = write_component(
        intent={
            "type": "table",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/ETH", "country/GHA", "country/KEN"],
            },
            # 2023 is missing for every recipient. Both summaries exclude this status-only row
            # from `point_count`.
            "window": {"start": "2020", "end": "2023"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    write_series = created.fact_summary["series"]
    read_series = got["fact_summary"]["series"]
    assert len(write_series) == 3
    assert len(read_series) == 3
    write_by_count = sorted(entry["point_count"] for entry in write_series)
    read_by_count = sorted(entry["point_count"] for entry in read_series)
    assert write_by_count == read_by_count == [3, 3, 3]
    write_values = sorted(
        (entry["point_count"], entry["first_value"], entry["last_value"]) for entry in write_series
    )
    read_values = sorted(
        (entry["point_count"], entry["first_value"], entry["last_value"]) for entry in read_series
    )
    assert write_values == read_values

    # `recipient` is the entity-role label; `stat_var` is a table column. With no labels map, raw
    # recipient refs become labels.
    write_labels = sorted(entry["label"] for entry in write_series)
    read_labels = sorted(entry["label"] for entry in read_series)
    assert write_labels == read_labels == ["country/ETH", "country/GHA", "country/KEN"]

    # Every stored row carries the `series_id` of the series it belongs to.
    # Readback groups `rows[]` by that field and keys each group to it.
    write_coordinates_by_id = {entry["id"]: entry["coordinates"] for entry in write_series}
    read_coordinates_by_id = {entry["id"]: entry["coordinates"] for entry in read_series}
    assert all(entry["id"] is not None for entry in write_series)
    assert all(entry["id"] is not None for entry in read_series)
    assert all(coordinates for coordinates in write_coordinates_by_id.values())
    assert read_coordinates_by_id == write_coordinates_by_id

    stored_rows = revision_store._revisions[created.component_id][-1].resolved_payload["component"][
        "rows"
    ]
    assert {row["series_id"] for row in stored_rows} == set(read_coordinates_by_id)


def test_table_row_label_includes_a_constant_unselected_role_dimension(
    frame_store, revision_store, release_registry
):
    """A table row's label joins every entity-role dimension's value the
    same way a line_chart series' label does (`bind._label`), including a
    role dimension `select` neither names nor sees vary. Over a frame with
    a constant `donor` role and a varying `recipient` role, the table's
    Ethiopia row and a line_chart selecting that same recipient both label
    "Global Fund / Ethiopia", on create and on GET."""
    rf = _donor_and_recipient_frame()
    frame_store.put(resolved_frame=rf)

    table_created = write_component(
        intent={
            "type": "table",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {"stat_var": "dcid:MalariaGrantDisbursements"},
            "window": {"start": "2020", "end": "2021"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    table_got = get_component(
        component_id=table_created.component_id,
        revision_store=revision_store,
    )

    line_chart_created = write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/ETH",
            },
            "window": {"start": "2020", "end": "2021"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    line_chart_got = get_component(
        component_id=line_chart_created.component_id,
        revision_store=revision_store,
    )

    expected = "Global Fund / Ethiopia"
    assert line_chart_created.fact_summary["series"][0]["label"] == expected
    assert line_chart_got["fact_summary"]["series"][0]["label"] == expected

    expected_labels = {"Global Fund / Ethiopia", "Global Fund / Ghana"}
    table_created_labels = {entry["label"] for entry in table_created.fact_summary["series"]}
    table_got_labels = {entry["label"] for entry in table_got["fact_summary"]["series"]}
    assert table_created_labels == expected_labels
    assert table_got_labels == expected_labels


def test_table_readback_keeps_series_distinct_when_two_entities_share_a_label(
    frame_store, revision_store, release_registry
):
    """Two entities share a display label, so readback groups their table rows by `series_id`."""
    rf = _dual_entity_stat_card_frame()
    frame_store.put(resolved_frame=rf)

    created = write_component(
        intent={
            "type": "table",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:X",
                "recipient": ["country/ETH", "country/GHA"],
            },
            "window": {"start": "2020", "end": "2020"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    write_series = created.fact_summary["series"]
    read_series = got["fact_summary"]["series"]
    assert len(write_series) == 2
    assert len(read_series) == 2
    write_coordinates_by_id = {entry["id"]: entry["coordinates"] for entry in write_series}
    read_coordinates_by_id = {entry["id"]: entry["coordinates"] for entry in read_series}
    assert all(entry["id"] is not None for entry in write_series)
    assert all(entry["id"] is not None for entry in read_series)
    assert read_coordinates_by_id == write_coordinates_by_id

    stored_rows = revision_store._revisions[created.component_id][-1].resolved_payload["component"][
        "rows"
    ]
    assert {row["series_id"] for row in stored_rows} == set(read_coordinates_by_id)


def test_table_readback_with_duplicate_stored_ids_withholds_coordinates_but_keeps_row_identity(
    frame_store, revision_store, release_registry, caplog
):
    """Duplicate cached ids withhold coordinates, while row `series_id` values preserve grouping."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    intent = {
        "type": "table",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": ["country/ETH", "country/GHA", "country/KEN"],
        },
        "window": {"start": "2020", "end": "2020"},
    }
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    stored = revision_store._revisions[created.component_id][-1].series_coordinates
    assert stored[0]["coordinates"] != stored[1]["coordinates"]
    duplicated = (
        dict(stored[0]),
        {**dict(stored[1]), "id": stored[0]["id"]},
        dict(stored[2]),
    )
    _replace_current_series_coordinates(
        revision_store=revision_store,
        component_id=created.component_id,
        series_coordinates=duplicated,
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    read_series = got["fact_summary"]["series"]
    assert all(entry["id"] is not None for entry in read_series)
    assert all(entry["coordinates"] == {} for entry in read_series)
    assert TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER not in caplog.text
    assert caplog.text.count(SERIES_COORDINATES_INCONSISTENT_MARKER) == 1
    assert "reason=duplicate_non_null_id" in caplog.text


def test_current_table_candidate_noops_against_historical_row_count_metadata(
    frame_store, revision_store, release_registry
):
    """The comparator accepts stored `row_count` metadata without changing it."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    select = {
        "stat_var": "dcid:MalariaGrantDisbursements",
        "recipient": ["country/ETH", "country/GHA", "country/KEN"],
    }
    created = write_component(
        intent={
            "type": "table",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": select,
            "window": {"start": "2020", "end": "2020"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    stored = revision_store._revisions[created.component_id][-1].series_coordinates
    with_row_count = tuple({**dict(entry), "row_count": 1} for entry in stored)
    _replace_current_series_coordinates(
        revision_store=revision_store,
        component_id=created.component_id,
        series_coordinates=with_row_count,
    )

    retried = write_component(
        intent={"select": select},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert retried.revision == 1
    assert len(revision_store._revisions[created.component_id]) == 1
    current = revision_store._revisions[created.component_id][-1]
    assert [entry["row_count"] for entry in current.series_coordinates] == [1, 1, 1]


@pytest.mark.parametrize(
    "series_id_shape",
    [
        pytest.param("malformed", id="all-malformed-series-id"),
        pytest.param("mixed", id="mixed-series-id"),
    ],
)
def test_table_readback_falls_back_when_rows_carry_a_malformed_or_mixed_series_id(
    frame_store, revision_store, release_registry, caplog, series_id_shape
):
    """Untrusted stored `series_id` values use dimensions-keyed grouping."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    created = write_component(
        intent={
            "type": "table",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/ETH", "country/GHA", "country/KEN"],
            },
            "window": {"start": "2020", "end": "2020"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    stored_rows = revision_store._revisions[created.component_id][-1].resolved_payload["component"][
        "rows"
    ]
    if series_id_shape == "malformed":
        mutated_rows = tuple({**dict(row), "series_id": ""} for row in stored_rows)
    else:
        mutated_rows = tuple(
            {key: value for key, value in row.items() if key != "series_id"}
            if index == 0
            else dict(row)
            for index, row in enumerate(stored_rows)
        )
    _replace_current_rows(
        revision_store=revision_store, component_id=created.component_id, rows=mutated_rows
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    read_series = got["fact_summary"]["series"]
    assert len(read_series) == 3
    assert all(entry["id"] is None for entry in read_series)
    read_labels = sorted(entry["label"] for entry in read_series)
    assert read_labels == ["country/ETH", "country/GHA", "country/KEN"]
    assert caplog.text.count(TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER) == 1
    assert "reason=mixed_or_malformed_series_id" in caplog.text


def test_table_readback_falls_back_when_a_series_id_collides_across_series(
    frame_store, revision_store, release_registry, caplog
):
    """A row's `series_id` corrupted to collide with a different series'
    id, while keeping its own distinct `dimensions`, must not be silently
    merged into that other series' chunk. Readback falls back to
    `dimensions`-keyed grouping and logs the inconsistency, the same as
    every other corruption class this function guards against."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    created = write_component(
        intent={
            "type": "table",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/ETH", "country/GHA", "country/KEN"],
            },
            "window": {"start": "2020", "end": "2020"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    stored_rows = revision_store._revisions[created.component_id][-1].resolved_payload["component"][
        "rows"
    ]
    assert len(stored_rows) == 3
    colliding_rows = tuple(
        {**dict(row), "series_id": stored_rows[0]["series_id"]} if index == 2 else dict(row)
        for index, row in enumerate(stored_rows)
    )
    _replace_current_rows(
        revision_store=revision_store, component_id=created.component_id, rows=colliding_rows
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    read_series = got["fact_summary"]["series"]
    assert len(read_series) == 3
    assert all(entry["id"] is None for entry in read_series)
    read_labels = sorted(entry["label"] for entry in read_series)
    assert read_labels == ["country/ETH", "country/GHA", "country/KEN"]
    assert caplog.text.count(TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER) == 1
    assert "reason=heterogeneous_series_id_chunk" in caplog.text


def test_table_readback_with_no_series_id_on_any_row_is_a_pre_fix_payload(
    frame_store, revision_store, release_registry, caplog
):
    """Legacy table rows without `series_id` group by dimensions without an inconsistency log."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    created = write_component(
        intent={
            "type": "table",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": ["country/ETH", "country/GHA", "country/KEN"],
            },
            "window": {"start": "2020", "end": "2020"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    stored_rows = revision_store._revisions[created.component_id][-1].resolved_payload["component"][
        "rows"
    ]
    pre_fix_rows = tuple(
        {key: value for key, value in row.items() if key != "series_id"} for row in stored_rows
    )
    _replace_current_rows(
        revision_store=revision_store, component_id=created.component_id, rows=pre_fix_rows
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    read_series = got["fact_summary"]["series"]
    assert len(read_series) == 3
    assert all(entry["id"] is None for entry in read_series)
    read_labels = sorted(entry["label"] for entry in read_series)
    assert read_labels == ["country/ETH", "country/GHA", "country/KEN"]
    assert TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER not in caplog.text


def test_entity_mode_bar_chart_create_reply_agrees_with_a_get_component_readback_on_coordinates(
    frame_store, revision_store, release_registry
):
    """Entity-mode bar-chart series carry ids, so readback matches coordinates by id."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)

    created = write_component(
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
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )

    write_series = created.fact_summary["series"]
    read_series = got["fact_summary"]["series"]
    assert len(write_series) == 3
    write_coordinates_by_id = {entry["id"]: entry["coordinates"] for entry in write_series}
    read_coordinates_by_id = {entry["id"]: entry["coordinates"] for entry in read_series}
    assert all(entry["id"] is not None for entry in write_series)
    assert all(coordinates for coordinates in write_coordinates_by_id.values())
    assert read_coordinates_by_id == write_coordinates_by_id


def test_readback_of_a_bar_chart_payload_missing_the_renamed_axis_keys_raises_release_mismatch(
    frame_store, revision_store, release_registry
):
    """A bar chart using `x_axis` and `y_axis` fails the release schema with `ReleaseMismatch`."""
    intent = _bar_chart_case(frame_store)
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    stored_component = revision_store._revisions[created.component_id][-1].resolved_payload[
        "component"
    ]
    unmigrated = dict(stored_component)
    unmigrated["x_axis"] = unmigrated.pop("category_axis")
    unmigrated["y_axis"] = unmigrated.pop("value_axis")
    _replace_current_component(
        revision_store=revision_store, component_id=created.component_id, component=unmigrated
    )

    with pytest.raises(ReleaseMismatch) as excinfo:
        get_component(
            component_id=created.component_id,
            revision_store=revision_store,
        )
    assert excinfo.value.catalog_version == created.catalog_version
    assert "bar_chart" in excinfo.value.detail
    assert "category_axis" in excinfo.value.detail


def test_period_mode_bar_chart_create_reply_agrees_with_readback_on_window(
    frame_store, revision_store, release_registry
):
    """Period-mode `category_axis.window` supplies the summary window on create and readback."""
    frame, intent = _bar_chart_intent()
    frame_store.put(resolved_frame=frame)
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert created.fact_summary["window"] == intent["window"]
    assert got["fact_summary"]["window"] == intent["window"]


@pytest.mark.parametrize(
    ("recipient", "defaults"),
    [(None, ["/select/recipient"]), (["country/ETH", "country/KEN"], [])],
    ids=["recipient_unbound", "recipient_listed"],
)
def test_bar_chart_states_its_default_on_create_and_readback(
    frame_store, revision_store, release_registry, recipient, defaults
):
    """A bar chart whose select lists no values, over a window of one period, compares every
    recipient and says so at `/select/recipient`. A listed select states no default."""
    frame = _multi_recipient_frame()
    frame_store.put(resolved_frame=frame)
    select: dict[str, object] = {"stat_var": "dcid:MalariaGrantDisbursements"}
    if recipient is not None:
        select["recipient"] = recipient
    created = write_component(
        intent={
            "type": "bar_chart",
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
            "select": select,
            "window": {"start": "2022", "end": "2022"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(component_id=created.component_id, revision_store=revision_store)

    assert [one["pointer"] for one in created.fact_summary["defaults"]] == defaults
    assert got["fact_summary"]["defaults"] == created.fact_summary["defaults"]


@pytest.mark.parametrize(
    ("created_recipient", "revised_recipient", "defaults"),
    [
        (None, ["country/ETH", "country/GHA", "country/KEN"], []),
        (["country/ETH", "country/GHA", "country/KEN"], None, ["/select/recipient"]),
    ],
    ids=["list_added", "list_removed"],
)
def test_bar_chart_revise_that_toggles_the_default_states_the_revised_defaults(
    frame_store, revision_store, release_registry, created_recipient, revised_recipient, defaults
):
    """The recipients fall in value from first to last in frame order, so listing them in
    that order draws and resolves them exactly as the default does. The revise still stores
    the new intent, so the reply and readback state the defaults of the select just sent."""
    values = {"country/ETH": 320.0, "country/GHA": 220.0, "country/KEN": 120.0}
    frame = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "1" * 32, content_version="cv_" + "1" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": recipient,
                "time_period": "2022",
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for recipient, value in values.items()
        ],
    )
    frame_store.put(resolved_frame=frame)

    def select(recipient):
        chosen: dict[str, object] = {"stat_var": "dcid:MalariaGrantDisbursements"}
        if recipient is not None:
            chosen["recipient"] = recipient
        return chosen

    created = write_component(
        intent={
            "type": "bar_chart",
            "frame_ref": {
                "spec_id": frame.frame_ref.spec_id,
                "content_version": frame.frame_ref.content_version,
            },
            "select": select(created_recipient),
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    revised = write_component(
        intent={"select": select(revised_recipient)},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(component_id=created.component_id, revision_store=revision_store)

    assert revised.revision == 2
    assert [one["pointer"] for one in revised.fact_summary["defaults"]] == defaults
    assert got["fact_summary"]["defaults"] == revised.fact_summary["defaults"]


def _replace_current_series_coordinates(*, revision_store, component_id, series_coordinates):
    """Replace stored series coordinates to test readback of absent or corrupted metadata."""
    revisions = revision_store._revisions[component_id]
    revisions[-1] = dataclasses.replace(revisions[-1], series_coordinates=series_coordinates)


def _replace_current_component(*, revision_store, component_id, component):
    """Replace the stored component to test readback of a modified payload."""
    revisions = revision_store._revisions[component_id]
    current = revisions[-1]
    payload = dict(current.resolved_payload)
    payload["component"] = component
    revisions[-1] = dataclasses.replace(current, resolved_payload=payload)


def _replace_current_rows(*, revision_store, component_id, rows):
    """Replace stored table rows to test missing, malformed, or inconsistent `series_id` values."""
    revisions = revision_store._revisions[component_id]
    current = revisions[-1]
    component = dict(current.resolved_payload["component"])
    component["rows"] = rows
    payload = dict(current.resolved_payload)
    payload["component"] = component
    revisions[-1] = dataclasses.replace(current, resolved_payload=payload)


def test_pre_fix_revision_with_no_series_coordinates_reads_back_empty_coordinates(
    frame_store, revision_store, release_registry
):
    """Missing stored series coordinates yield `{}` for each readback series."""
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
    _replace_current_series_coordinates(
        revision_store=revision_store, component_id=created.component_id, series_coordinates=None
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["fact_summary"]["series"][0]["coordinates"] == {}


def test_table_readback_tolerates_a_stored_coordinates_list_shorter_than_the_series(
    frame_store, revision_store, release_registry, caplog
):
    """A missing coordinate entry leaves row ids intact and returns `{}` only for that lookup."""
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    created = write_component(
        intent={
            "type": "table",
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
    assert len(created.fact_summary["series"]) == 3
    stored = revision_store._revisions[created.component_id][-1].series_coordinates
    _replace_current_series_coordinates(
        revision_store=revision_store,
        component_id=created.component_id,
        series_coordinates=tuple(dict(entry) for entry in stored[:2]),
    )

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    read_series = got["fact_summary"]["series"]
    assert len(read_series) == 3
    assert all(entry["id"] is not None for entry in read_series)
    with_coordinates = [entry for entry in read_series if entry["coordinates"]]
    without_coordinates = [entry for entry in read_series if not entry["coordinates"]]
    assert len(with_coordinates) == 2
    assert len(without_coordinates) == 1
    assert TABLE_ROW_SERIES_ID_INCONSISTENT_MARKER not in caplog.text
    assert SERIES_COORDINATES_INCONSISTENT_MARKER not in caplog.text


def _dual_entity_stat_card_frame():
    """Give two entities identical values, period, unit, and label; only their coordinates differ."""
    frame_ref = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "c" * 32)
    rows = [
        {
            "stat_var": "dcid:X",
            "recipient": recipient,
            "time_period": "2020",
            "value": 442.0,
            "unit": None,
            "measurementMethod": None,
            "provenance": "src_a",
        }
        for recipient in ("country/ETH", "country/GHA")
    ]
    return build_resolved_frame(
        frame_ref=frame_ref,
        role_keys=("recipient",),
        rows=rows,
        labels={"country/ETH": "Same Label", "country/GHA": "Same Label"},
    )


def test_revise_to_a_structurally_identical_member_with_different_coordinates_mints_and_serves_the_new_ones(
    frame_store, revision_store, release_registry
):
    """Changing coordinates mints a revision even when the stat-card payload is unchanged.

    Readback serves the selected entity's coordinates.
    """
    rf = _dual_entity_stat_card_frame()
    frame_store.put(resolved_frame=rf)
    created = write_component(
        intent={
            "type": "stat_card",
            "frame_ref": {
                "spec_id": rf.frame_ref.spec_id,
                "content_version": rf.frame_ref.content_version,
            },
            "select": {"stat_var": "dcid:X", "recipient": "country/ETH"},
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert created.fact_summary["series"][0]["coordinates"]["recipient"] == "country/ETH"

    revised = write_component(
        intent={"select": {"stat_var": "dcid:X", "recipient": "country/GHA"}},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    # A no-op reply always carries `changed=()`; a non-empty tuple here
    # proves this took the substantive path, not the no-op one.
    assert revised.changed != ()
    assert revised.revision == 2
    assert revised.fact_summary["series"][0]["coordinates"]["recipient"] == "country/GHA"
    assert revision_store.read_current(component_id=created.component_id).revision == 2

    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["revision"] == 2
    assert got["fact_summary"]["series"][0]["coordinates"]["recipient"] == "country/GHA"


def test_revise_with_no_stored_coordinates_backfills_then_no_ops_on_repeat(
    frame_store, revision_store, release_registry
):
    """An identical revise backfills missing coordinates, then the next identical revise no-ops."""
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
    _replace_current_series_coordinates(
        revision_store=revision_store, component_id=created.component_id, series_coordinates=None
    )

    backfilled = write_component(
        intent={"title": "Malaria grant disbursements to Ethiopia"},  # already the current value
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert backfilled.revision == 2
    assert backfilled.changed != ()
    current = revision_store.read_current(component_id=created.component_id)
    assert current.revision == 2
    assert current.series_coordinates is not None

    no_op = write_component(
        intent={"title": "Malaria grant disbursements to Ethiopia"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert no_op.revision == 2
    assert no_op.changed == ()
    assert revision_store.read_current(component_id=created.component_id).revision == 2


def test_get_component_by_revision_after_a_revise(frame_store, revision_store, release_registry):
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
    write_component(
        intent={"title": "Revised title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    rev1 = get_component(
        component_id=created.component_id,
        revision=1,
        revision_store=revision_store,
    )
    assert rev1["stored_intent"]["title"] == "Malaria grant disbursements to Ethiopia"


def _two_revision_component(frame_store, revision_store, release_registry) -> WriteReply:
    """Create and revise a component so revision 2 is its head. Returns the create reply."""
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
    write_component(
        intent={"title": "Revised title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    return created


@pytest.mark.parametrize("revision", [3, 999])
def test_get_component_by_a_revision_number_past_the_head_raises_revision_not_found(
    revision, frame_store, revision_store, release_registry
):
    """An unminted revision above the head raises `RevisionNotFound` with both numbers."""
    created = _two_revision_component(frame_store, revision_store, release_registry)
    component_id = created.component_id

    with pytest.raises(RevisionNotFound) as excinfo:
        get_component(
            component_id=component_id,
            revision=revision,
            revision_store=revision_store,
        )
    assert excinfo.value.component_id == component_id
    assert excinfo.value.revision == revision
    assert str(revision) in excinfo.value.detail
    assert "2" in excinfo.value.detail


def test_get_component_at_the_head_revision_number_succeeds(
    frame_store, revision_store, release_registry
):
    created = _two_revision_component(frame_store, revision_store, release_registry)
    component_id = created.component_id

    got = get_component(
        component_id=component_id,
        revision=2,
        revision_store=revision_store,
    )
    assert got["revision"] == 2


def test_a_revise_of_a_component_that_vanished_mid_flight_gets_edit_conflict(
    frame_store, revision_store, release_registry
):
    """A missing current revision raises `EditConflict` during revise.

    A concurrent revision advance instead rebases against fresh state.
    """
    created = _two_revision_component(frame_store, revision_store, release_registry)
    component_id = created.component_id
    revision_store._revisions[component_id] = []

    with pytest.raises(EditConflict):
        write_component(
            intent={"title": "one more revise"},
            component_id=component_id,
            write_token=created.write_token,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )


def test_get_component_scoped_read_raises_edit_conflict_when_the_component_vanished(
    frame_store, revision_store, release_registry
):
    """If revision and head reads both return empty, `get_component` raises `EditConflict`."""
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
    revision_store._revisions[created.component_id] = []

    with pytest.raises(EditConflict):
        get_component(
            component_id=created.component_id,
            revision=1,
            revision_store=revision_store,
        )

    with pytest.raises(EditConflict):
        get_component(
            component_id=created.component_id,
            revision_store=revision_store,
        )


@pytest.mark.parametrize("revision", [None, 1])
def test_get_component_of_an_unknown_id_raises_draft_not_found(
    revision, revision_store, release_registry
):
    with pytest.raises(DraftNotFound):
        get_component(
            component_id="cmp_never_created",
            revision=revision,
            revision_store=revision_store,
        )


# -- one builder for both paths ----------------------------------------------


def _line_chart_case(frame_store):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    return intent


def _bar_chart_case(frame_store):
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    return {
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
    }


def _table_case(frame_store):
    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    return {
        "type": "table",
        "frame_ref": {
            "spec_id": rf.frame_ref.spec_id,
            "content_version": rf.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": ["country/ETH", "country/GHA", "country/KEN"],
        },
        "window": {"start": "2020", "end": "2022"},
    }


def _stat_card_case(frame_store):
    frame, intent = _stat_card_intent()
    frame_store.put(resolved_frame=frame)
    return intent


def _note_narration_case(frame_store):
    return {"type": "note_narration", "text": "Disbursements rose each year after 2020."}


def _source_card_case(frame_store):
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)
    return {
        "type": "source_card",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
    }


_MEMBER_CASES = {
    "line_chart": _line_chart_case,
    "bar_chart": _bar_chart_case,
    "table": _table_case,
    "stat_card": _stat_card_case,
    "note_narration": _note_narration_case,
    "source_card": _source_card_case,
}


@pytest.mark.parametrize("member", sorted(_MEMBER_CASES))
def test_create_reply_fact_summary_equals_get_component_readback(
    member, frame_store, revision_store, release_registry
):
    """Create and readback use the same fact-summary builder for every member."""
    intent = _MEMBER_CASES[member](frame_store)
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["fact_summary"] == created.fact_summary


@pytest.mark.parametrize("member", sorted(_MEMBER_CASES))
def test_create_reply_resolved_payload_equals_get_component_and_the_stored_row(
    member, frame_store, revision_store, release_registry
):
    """Create, readback, and storage carry the same stamped envelope."""
    intent = _MEMBER_CASES[member](frame_store)
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    stored = revision_store.read_revision(component_id=created.component_id, revision=1)
    assert created.resolved_payload == stored.resolved_payload
    assert got["resolved_payload"] == stored.resolved_payload
    assert stored.resolved_payload["component_id"] == created.component_id
    assert stored.resolved_payload["revision"] == 1
    assert stored.resolved_payload["component"]["type"] == member


def test_a_revision_scoped_get_returns_that_revisions_own_envelope(
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
        intent={"title": "Revised title"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    def _read(revision):
        return get_component(
            component_id=created.component_id,
            revision=revision,
            revision_store=revision_store,
        )["resolved_payload"]

    assert _read(1) == created.resolved_payload
    assert _read(1)["revision"] == 1
    assert _read(1)["component"]["title"]["text"] == "Malaria grant disbursements to Ethiopia"
    assert _read(2) == revised.resolved_payload
    assert _read(2)["revision"] == 2
    assert _read(2)["component"]["title"]["text"] == "Revised title"
    assert _read(None) == revised.resolved_payload


def test_a_no_op_revise_reply_carries_the_heads_stored_envelope(
    frame_store, revision_store, release_registry
):
    """A no-op reply returns the head envelope with its original `provenance.generated_at`."""
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
    no_op = write_component(
        intent={"title": "Malaria grant disbursements to Ethiopia"},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert no_op.revision == 1
    assert no_op.changed == ()
    stored = revision_store.read_revision(component_id=created.component_id, revision=1)
    assert no_op.resolved_payload == stored.resolved_payload
    assert no_op.resolved_payload["revision"] == 1


def test_source_card_create_and_readback_both_carry_source_refs(
    frame_store, revision_store, release_registry
):
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)
    created = write_component(
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
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert created.fact_summary["source_refs"] == ["src_oecd_dac"]
    assert got["fact_summary"]["source_refs"] == ["src_oecd_dac"]
    expected_keys = {
        "changed",
        "series",
        "series_cut",
        "window",
        "unit",
        "change",
        "source_refs",
        "defaults",
    }
    assert set(created.fact_summary) == expected_keys
    assert set(got["fact_summary"]) == expected_keys


def test_note_narration_summary_source_refs_is_null_on_both_paths(
    frame_store, revision_store, release_registry
):
    created = write_component(
        intent={"type": "note_narration", "text": "Disbursements rose each year after 2020."},
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert created.fact_summary["source_refs"] is None
    assert got["fact_summary"]["source_refs"] is None


def test_token_cap_truncates_series_on_get_the_same_as_on_create(
    frame_store, revision_store, release_registry, monkeypatch
):
    """GET summaries apply the same token cap as create replies."""
    import assembler.reply as reply_module

    rf = _multi_recipient_frame()
    frame_store.put(resolved_frame=rf)
    created = write_component(
        intent={
            "type": "table",
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
    assert len(created.fact_summary["series"]) == 3

    monkeypatch.setattr(reply_module, "FACT_SUMMARY_TOKEN_CAP", 1)
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["fact_summary"]["series"] == []
    assert got["fact_summary"]["series_cut"] == 3
