"""Create and envelope validity, then revise, no-op retry, and idempotent
retry after a simulated transport failure -- exercised through
`write_component` alone over the in-memory fakes, the way a door
eventually will call it, rather than any of its private stages directly.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses

import jsonschema
import pytest

from assembler.errors import BindingInvalid, ReleaseMismatch, SchemaInvalid
from assembler.fakes import FakeReleaseRegistry
from assembler.mint import MintedComponent, stamp_envelope
from assembler.mint.validate import component_registry
from assembler.pipeline import get_component, write_component
from assembler.pipeline.write import (
    _series_coordinates_equal,
    _series_coordinates_for_storage,
)
from assembler.reply import format_value
from assembler.resolve import ResolvedSeries
from assembler.testing import (
    build_resolved_frame,
    malaria_ethiopia_frame_ref,
    malaria_ethiopia_resolved_frame,
    malaria_ethiopia_rows,
)
from presentation_api.refusals import binding_invalid_spec

from ._pipeline_support import _malaria_intent

# -- create / envelope validity -----------------------------------------------


def _current_coordinate_candidate() -> tuple[dict[str, object], ...]:
    return _series_coordinates_for_storage(
        series=(
            ResolvedSeries(
                id="series_a",
                coordinates=(("recipient", "country/ETH"),),
                rows=(),
            ),
        )
    )


def test_current_coordinate_candidate_accepts_historical_stored_metadata() -> None:
    """The comparator accepts stored `row_count` metadata without changing it."""
    current_candidate = _current_coordinate_candidate()
    metadata_bearing_stored = (
        {
            "id": "series_a",
            "coordinates": {"recipient": "country/ETH"},
            "row_count": 3,
        },
    )

    assert _series_coordinates_equal(candidate=current_candidate, stored=metadata_bearing_stored)


def test_current_coordinate_shape_is_exactly_compatible_with_pre_row_count_workers() -> None:
    """The two-key wire shape gives every supported writer an exact match."""
    current_candidate = _current_coordinate_candidate()
    pre_row_count_stored = [{"id": "series_a", "coordinates": {"recipient": "country/ETH"}}]

    assert [dict(one) for one in current_candidate] == [dict(one) for one in pre_row_count_stored]
    assert _series_coordinates_equal(candidate=current_candidate, stored=pre_row_count_stored)


def test_create_line_chart_from_the_worked_intent_validates_against_schemas(
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

    assert reply.component_id.startswith("cmp_")
    assert reply.revision == 1
    assert reply.changed == ()

    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )


@pytest.mark.parametrize(("period", "refused"), [("2016", False), ("2018/2022", True)])
def test_the_line_chart_schema_refuses_a_span_period(
    frame_store, revision_store, release_registry, period, refused
):
    """The catalogue schema holds a line chart's periods to points, so a span that reaches a
    line chart past its builder fails `stamp_envelope`. The point row is the control, and the
    two rows differ only in that one period."""
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
    payload = revision_store.read_revision(
        component_id=reply.component_id, revision=1
    ).resolved_payload
    component = copy.deepcopy(payload["component"])
    component["series"][0]["points"][0]["period"] = period
    minted = MintedComponent(
        component=component, provenance=payload["provenance"], observed_periods=frozenset()
    )

    with pytest.raises(SchemaInvalid) if refused else contextlib.nullcontext():
        stamp_envelope(
            minted=minted,
            component_id=reply.component_id,
            revision=2,
            release=release_registry.pinned(),
        )


def _span_frame(*, period="2018/2022"):
    """Three recipients whose rows share one span, as a verb's collapsed output."""
    return build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": recipient,
                "time_period": period,
                "value": value,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_oecd_dac",
            }
            for recipient, value in (
                ("country/ETH", 500.0),
                ("country/GHA", 400.0),
                ("country/KEN", 300.0),
            )
        ],
    )


_SPAN_RECIPIENTS = ["country/ETH", "country/GHA", "country/KEN"]


def _span_intent(frame, *, member="bar_chart", recipient=_SPAN_RECIPIENTS, window=None):
    intent = {
        "type": member,
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": recipient},
    }
    if window is not None:
        intent["window"] = window
    return intent


def _shown_periods(component):
    """Every period the component states, in payload order."""
    if component["type"] == "table":
        return [row["period"] for row in component["rows"]]
    if component["type"] == "stat_card":
        return [component["period"]]
    points = [point["period"] for one in component["series"] for point in one["points"]]
    return [component["category_axis"]["period"], *points]


@pytest.mark.parametrize("window", [None, {"start": "2018", "end": "2022"}])
@pytest.mark.parametrize(
    ("member", "recipient", "stated"),
    [
        ("bar_chart", _SPAN_RECIPIENTS, 4),
        ("table", _SPAN_RECIPIENTS, 3),
        ("stat_card", "country/ETH", 1),
    ],
)
def test_a_span_frame_binds_the_span_as_its_one_period(
    frame_store, revision_store, release_registry, member, recipient, stated, window
):
    """An entity-mode bar chart, a table, and a stat card state the span wherever they
    state a period: the category axis and each bar, each row, and the card."""
    frame = _span_frame()
    frame_store.put(resolved_frame=frame)

    reply = write_component(
        intent=_span_intent(frame, member=member, recipient=recipient, window=window),
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    component = revision_store.read_revision(
        component_id=reply.component_id, revision=1
    ).resolved_payload["component"]
    assert _shown_periods(component) == ["2018/2022"] * stated


@pytest.mark.parametrize("window_from_coverage", [False, True], ids=["no_window", "coverage"])
def test_a_day_span_table_binds_with_its_coverage_as_the_window(
    frame_store, revision_store, release_registry, window_from_coverage
):
    """A day-grain span is 21 characters, the longest period the frame grammar parses. A table
    over one binds with no window, and again with the frame's coverage written back as its
    window, which the intent grammar's window bounds must admit."""
    span = "2024-01-01/2024-01-31"
    frame = _span_frame(period=span)
    frame_store.put(resolved_frame=frame)
    assert frame.structural_view.coverage == (span, span)
    window = None
    if window_from_coverage:
        start, end = frame.structural_view.coverage
        window = {"start": start, "end": end}

    reply = write_component(
        intent=_span_intent(frame, member="table", window=window),
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    component = revision_store.read_revision(
        component_id=reply.component_id, revision=1
    ).resolved_payload["component"]
    assert _shown_periods(component) == [span] * 3


@pytest.mark.parametrize(
    ("intent_fields", "pointer", "alternatives"),
    [
        ({"window": {"start": "2018", "end": "2021"}}, "/window", []),
        ({"member": "line_chart"}, "/type", ["bar_chart", "table", "stat_card"]),
        ({"recipient": "country/ETH"}, "/type", ["bar_chart", "table", "stat_card"]),
        (
            {"member": "line_chart", "window": {"start": "2019", "end": "2022"}},
            "/type",
            ["bar_chart", "table", "stat_card"],
        ),
    ],
    ids=[
        "window_leaves_out_part_of_the_span",
        "line_chart",
        "period_mode_bar_chart",
        "line_chart_refuses_its_type_before_its_window",
    ],
)
def test_a_span_frame_refuses_binding_invalid_where_it_cannot_bind(
    frame_store, revision_store, release_registry, intent_fields, pointer, alternatives
):
    """A window that leaves out part of the span, a line chart, and a period-mode bar chart
    refuse. The two drawn kinds offer the component types that show a span, and refuse at
    `/type` even when their window would also refuse."""
    frame = _span_frame()
    frame_store.put(resolved_frame=frame)

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=_span_intent(frame, **intent_fields),
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    problem = binding_invalid_spec(excinfo.value)
    assert (problem.code, problem.pointer) == ("binding_invalid", pointer)
    assert [one.ref for one in problem.alternatives or ()] == alternatives


def test_create_with_manifest_warnings_translates_into_envelope_provenance_warnings(
    frame_store, revision_store, release_registry
):
    """A frame manifest carrying warnings (`{code, severity, message}`, the
    datasets materialiser's own shape) must not break `write_component` --
    the pipeline's mint boundary translates them into the envelope's closed
    `{kind, detail}` warning vocabulary before the write is validated and
    persisted.
    """
    frame = build_resolved_frame(
        frame_ref=malaria_ethiopia_frame_ref(),
        role_keys=("recipient",),
        rows=malaria_ethiopia_rows(),
        labels={"country/ETH": "Ethiopia"},
        manifest={
            "schema_version": 1,
            "warnings": [
                {
                    "code": "revision_unavailable",
                    "severity": "info",
                    "message": "no live revision endpoint is available",
                },
                {
                    "code": "processing_level_undeclared",
                    "severity": "info",
                    "message": "a source StatVar lacks a processing-level declaration",
                },
            ],
        },
    )
    intent = {
        "type": "line_chart",
        "frame_ref": {
            "spec_id": frame.frame_ref.spec_id,
            "content_version": frame.frame_ref.content_version,
        },
        "select": {
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": "country/ETH",
        },
        "window": {"start": "2015", "end": "2024"},
        "title": "Malaria grant disbursements to Ethiopia",
    }
    frame_store.put(resolved_frame=frame)

    reply = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    assert record.resolved_payload["provenance"]["warnings"] == [
        {"kind": "provisional", "detail": "no live revision endpoint is available"},
        {"kind": "provisional", "detail": "a source StatVar lacks a processing-level declaration"},
    ]
    release = release_registry.pinned()
    registry = component_registry(release)
    jsonschema.Draft202012Validator(release.envelope_schema, registry=registry).validate(
        record.resolved_payload
    )


def test_line_chart_resolves_linear_scale_and_unbounded_range_when_the_intent_omits_them(
    frame_store, revision_store, release_registry
):
    """A resolved line_chart carries `scale: "linear"` and an unbounded
    `range` when the intent carries neither field itself."""
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
    record = revision_store.read_revision(component_id=reply.component_id, revision=1)
    y_axis = record.resolved_payload["component"]["y_axis"]
    assert y_axis["scale"] == "linear"
    assert y_axis["range"] == {"min": None, "max": None}


# -- revise / no-op / retry ---------------------------------------------------


def test_revise_title_mints_revision_2_byte_identical_outside_the_write_set(
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
    assert revised.revision == 2
    assert revised.changed == (
        revised.changed[0].__class__(
            field="title", prior="Malaria grant disbursements to Ethiopia", new="Revised title"
        ),
    )

    rev1 = revision_store.read_revision(component_id=created.component_id, revision=1)
    rev2 = revision_store.read_revision(component_id=created.component_id, revision=2)
    component_1 = dict(rev1.resolved_payload["component"])
    component_2 = dict(rev2.resolved_payload["component"])
    component_1["title"] = dict(component_1["title"])
    component_2["title"] = dict(component_2["title"])
    component_1["title"].pop("text")
    component_2["title"].pop("text")
    assert component_1 == component_2


def test_no_op_patch_returns_same_revision_and_nothing_minted(
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
    assert revision_store.read_current(component_id=created.component_id).revision == 1
    # A title-only no-op preserves revision 1 coordinates for later readback.
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert (
        got["fact_summary"]["series"][0]["coordinates"]
        == created.fact_summary["series"][0]["coordinates"]
    )


def test_revise_under_a_new_catalogue_release_mints_a_new_revision_under_it(
    frame_store, revision_store, release_registry
):
    """A changed catalog or schema version mints a revision even when intent is unchanged.

    The reply uses the new formatter profile, and readback matches its summary apart from `changed`.
    """
    release_x = release_registry.pinned()
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

    release_y = dataclasses.replace(
        release_x,
        catalog_version="2026.9.0",
        schema_version=release_x.schema_version + 1,
        formatter_profile={**release_x.formatter_profile, "significant_digits": 6},
    )
    registry_y = FakeReleaseRegistry(release=release_y)

    revised = write_component(
        intent={"title": intent["title"]},  # already the stored value
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=registry_y,
    )

    assert revised.revision == created.revision + 1
    assert revised.catalog_version == release_y.catalog_version
    assert revised.resolved_payload["catalog_version"] == release_y.catalog_version
    assert revised.resolved_payload["schema_version"] == release_y.schema_version

    stored = revision_store.read_revision(
        component_id=created.component_id, revision=revised.revision
    )
    assert stored.catalog_version == release_y.catalog_version
    assert stored.schema_version == release_y.schema_version

    # 412.3 formats as 412 at 3 significant digits and 412.3 at 6.
    expected_first_value = format_value(value=412.3, profile=release_y.formatter_profile)
    assert expected_first_value == "412.3"
    assert revised.fact_summary["series"][0]["first_value"] == expected_first_value

    # A plain readback under the same registry must agree with the PATCH
    # reply on every field `build_fact_summary` derives from the stored
    # payload. `changed` is excluded: a read always reports it empty,
    # while this write's reply names the patched field, `title`, even
    # though the value it carries is unchanged.
    got = get_component(
        component_id=created.component_id,
        revision_store=revision_store,
    )
    assert got["fact_summary"]["series"][0]["first_value"] == expected_first_value
    assert {k: v for k, v in got["fact_summary"].items() if k != "changed"} == {
        k: v for k, v in revised.fact_summary.items() if k != "changed"
    }

    # Revision 1 keeps release X. A read under release Y formats it with X's stored profile.
    previous = get_component(
        component_id=created.component_id,
        revision=created.revision,
        revision_store=revision_store,
    )
    assert previous["catalog_version"] == release_x.catalog_version
    assert previous["fact_summary"]["series"][0]["first_value"] == "412"


def test_an_idempotent_create_replay_formats_with_revision_1s_stored_profile(
    frame_store, revision_store, release_registry
):
    """A replay on an instance pinning another release reads revision 1's stored profile."""
    release_x = release_registry.pinned()
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    create_write_token = "cmw_" + "a" * 22
    created = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        create_write_token=create_write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    release_y = dataclasses.replace(
        release_x,
        catalog_version="2026.9.0",
        schema_version=release_x.schema_version + 1,
        formatter_profile={**release_x.formatter_profile, "significant_digits": 6},
    )
    replayed = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        create_write_token=create_write_token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=FakeReleaseRegistry(release=release_y),
    )

    assert replayed.component_id == created.component_id
    assert replayed.catalog_version == release_x.catalog_version
    # 412.3 formats as 412 at X's 3 significant digits and 412.3 at Y's 6.
    assert replayed.fact_summary["series"][0]["first_value"] == "412"


def _release_at(release, catalog_version: str):
    return dataclasses.replace(release, catalog_version=catalog_version)


def test_a_revise_on_an_older_instance_refuses_and_the_head_keeps_its_release(
    frame_store, revision_store, release_registry
):
    """CalVer compares as integers, so a 2026.10.0 head is newer than a 2026.9.0 instance."""
    release = release_registry.pinned()
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    created = write_component(
        intent=intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=FakeReleaseRegistry(release=_release_at(release, "2026.10.0")),
    )

    with pytest.raises(ReleaseMismatch) as excinfo:
        write_component(
            intent={"title": "Older instance"},
            component_id=created.component_id,
            write_token=created.write_token,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=FakeReleaseRegistry(release=_release_at(release, "2026.9.0")),
        )

    assert excinfo.value.catalog_version == "2026.10.0"
    head = revision_store.read_current(component_id=created.component_id)
    assert (head.revision, head.catalog_version) == (1, "2026.10.0")


class _NewerWriteOnLoadFrameStore:
    """Runs `newer_write` once, on the first `load`, so the head moves to a newer release
    after the writer under test has read it and before its write transaction opens."""

    def __init__(self, *, inner, newer_write):
        self._inner = inner
        self._newer_write = newer_write

    def load(self, *, frame_ref):
        if self._newer_write is not None:
            newer_write, self._newer_write = self._newer_write, None
            newer_write()
        return self._inner.load(frame_ref=frame_ref)

    def pin(self, *, frame_ref, component_id):
        return self._inner.pin(frame_ref=frame_ref, component_id=component_id)


def test_a_revise_refuses_when_a_newer_release_moves_the_head_during_the_write(
    frame_store, revision_store, release_registry
):
    release = release_registry.pinned()
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    created = write_component(
        intent=intent,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    def newer_write() -> None:
        write_component(
            intent={"title": "Newer instance"},
            component_id=created.component_id,
            write_token=created.write_token,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=FakeReleaseRegistry(release=_release_at(release, "2099.1.0")),
        )

    with pytest.raises(ReleaseMismatch):
        write_component(
            intent={"title": "Older instance"},
            component_id=created.component_id,
            write_token=created.write_token,
            frame_store=_NewerWriteOnLoadFrameStore(inner=frame_store, newer_write=newer_write),
            revision_store=revision_store,
            release_registry=release_registry,
        )

    head = revision_store.read_current(component_id=created.component_id)
    assert (head.revision, head.catalog_version) == (2, "2099.1.0")
    assert head.stored_intent["title"] == "Newer instance"


def test_identical_retry_after_simulated_transport_failure_is_idempotent(
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

    patch = {"title": "Retried title"}
    first = write_component(
        intent=patch,
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert first.revision == 2

    # Simulate a lost response followed by the same patch, without an idempotency key.
    retry = write_component(
        intent=patch,
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert retry.revision == 2
    assert retry.changed == ()


def test_source_card_revise_with_empty_patch_is_a_no_op_and_still_resolves_the_frame(
    frame_store, revision_store, release_registry, monkeypatch
):
    """An empty patch keeps the revision but still resolves the source_card frame."""
    frame = malaria_ethiopia_resolved_frame()
    frame_store.put(resolved_frame=frame)

    load_calls: list[object] = []
    original_load = frame_store.load

    def _counting_load(*, frame_ref):
        load_calls.append(frame_ref)
        return original_load(frame_ref=frame_ref)

    monkeypatch.setattr(frame_store, "load", _counting_load)

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
    assert len(load_calls) == 1

    revised = write_component(
        intent={},
        component_id=created.component_id,
        write_token=created.write_token,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )
    assert len(load_calls) == 2  # the frame is re-resolved, even though nothing changed
    assert revised.revision == created.revision
    assert revised.changed == ()


def test_source_card_revise_naming_frame_ref_refuses(frame_store, revision_store, release_registry):
    """A revise containing `frame_ref` refuses at `/frame_ref` before merging."""
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

    with pytest.raises(SchemaInvalid) as excinfo:
        write_component(
            intent={
                "frame_ref": {
                    "spec_id": "spec_" + "9" * 32,
                    "content_version": "cv_" + "9" * 32,
                }
            },
            component_id=created.component_id,
            write_token=created.write_token,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/frame_ref"
