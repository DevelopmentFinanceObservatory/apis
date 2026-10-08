"""Pin `write_component` schema, binding, unknown-id, frame, and resource refusals."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

import assembler.pipeline.candidate as candidate_module
from assembler.errors import (
    BindingInvalid,
    ConstraintViolation,
    DraftNotFound,
    FrameUnavailable,
    ReleaseMismatch,
    ResourceExhausted,
    SchemaInvalid,
)
from assembler.fakes import FakeReleaseRegistry, FakeRevisionStore
from assembler.pipeline import ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER, write_component
from assembler.ports import FrameRef
from assembler.testing import build_resolved_frame, malaria_ethiopia_rows

from ._pipeline_support import _malaria_intent, _stat_card_intent, _stored_resolved_frame


def _release_excluding(release, member):
    """Remove one served member while preserving the grammar's six-member union.

    This creates a valid release mismatch case for a member the packaged release serves.
    """
    return dataclasses.replace(
        release,
        component_schemas={k: v for k, v in release.component_schemas.items() if k != member},
        intent_schemas={k: v for k, v in release.intent_schemas.items() if k != member},
        intent_schemas_tolerant={
            k: v for k, v in release.intent_schemas_tolerant.items() if k != member
        },
        minimal_intents={k: v for k, v in release.minimal_intents.items() if k != member},
        encoding_classes={k: v for k, v in release.encoding_classes.items() if k != member},
    )


# -- schema / merge refusals --------------------------------------------------


def test_patch_with_one_illegal_field_among_three_applies_nothing(
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

    with pytest.raises(SchemaInvalid) as excinfo:
        write_component(
            intent={
                "title": "New title",
                "window": {"start": "2016", "end": "2024"},
                "not_a_real_field": 1,
            },
            component_id=created.component_id,
            write_token=created.write_token,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/not_a_real_field"
    assert excinfo.value.legal_fields is not None

    current = revision_store.read_current(component_id=created.component_id)
    assert current.revision == 1
    assert current.stored_intent["title"] == "Malaria grant disbursements to Ethiopia"


def test_create_missing_select_carries_minimal_intent(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    del intent["select"]

    with pytest.raises(SchemaInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select"
    assert excinfo.value.minimal_intent == release_registry.pinned().minimal_intents["line_chart"]
    assert excinfo.value.legal_fields is None


def test_create_missing_type_is_correct_and_retry_naming_the_legal_types(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    del intent["type"]

    with pytest.raises(SchemaInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/type"
    # With no member selected, disclosure lists every legal type instead of one minimal intent.
    assert excinfo.value.legal_fields is None
    assert excinfo.value.minimal_intent == {
        "type": ("bar_chart", "line_chart", "note_narration", "source_card", "stat_card", "table")
    }


def test_create_with_a_garbage_type_is_schema_invalid_not_release_mismatch(
    frame_store, revision_store, release_registry
):
    """Unknown types raise `SchemaInvalid`. Unserved known types raise `ReleaseMismatch`."""
    frame, intent = _malaria_intent(type="pie_chart")
    frame_store.put(resolved_frame=frame)

    with pytest.raises(SchemaInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/type"
    assert excinfo.value.minimal_intent == {
        "type": ("bar_chart", "line_chart", "note_narration", "source_card", "stat_card", "table")
    }


def test_create_naming_a_grammar_valid_unshipped_type_is_release_mismatch(
    frame_store, revision_store, release
):
    """The grammar includes `source_card`, and the packaged release serves it.

    A synthetic release removes its per-member entry but keeps grammar membership, so the refusal
    names deployment support rather than malformed input.
    """
    reduced = _release_excluding(release, "source_card")
    release_registry = FakeReleaseRegistry(release=reduced)
    frame, intent = _malaria_intent(type="source_card")
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ReleaseMismatch) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.catalog_version == reduced.catalog_version


def test_stat_card_change_over_an_unobserved_basis_period_refuses_at_change_basis(
    frame_store, revision_store, release_registry
):
    """An unobserved 2018 basis period raises `stat_change_basis_observed` at `/change/basis`."""
    frame, intent = _stat_card_intent(
        change={"derivation": "percent_change", "basis": ["2018", "2024"]}
    )
    frame_store.put(resolved_frame=frame)

    with pytest.raises(ConstraintViolation) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "stat_change_basis_observed"
    assert excinfo.value.pointer == "/change/basis"


# -- select shape refusals -----------------------------------------------------


def test_empty_select_refuses_binding_invalid(frame_store, revision_store, release_registry):
    frame, intent = _malaria_intent(select={})
    frame_store.put(resolved_frame=frame)

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select"


def test_two_arrayed_dimensions_in_line_chart_select_refuses_binding_invalid(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent(
        select={
            "stat_var": "dcid:MalariaGrantDisbursements",
            "recipient": ["country/ETH"],
            "prices": ["current", "constant"],
        }
    )
    frame_store.put(resolved_frame=frame)

    with pytest.raises(BindingInvalid) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/select/recipient"


# -- binding / constraint refusals --------------------------------------------


def test_unbound_facet_splitting_one_series_carries_structural_view(
    frame_store, revision_store, release_registry
):
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "1" * 32, content_version="cv_" + "2" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_a",
            },
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2021",
                "value": 2.0,
                "unit": None,
                "measurementMethod": None,
                "provenance": "src_b",
            },
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
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.dimension == "provenance"
    assert excinfo.value.structural_view is not None


def test_comparable_units_required_is_still_reachable_from_the_write_path(
    frame_store, revision_store, release_registry
):
    """`comparable_units_required` uses frame units, so it fires at write time. The change-basis
    constraint cannot, because the builder refuses first. Bind `unit`, which splits the one
    series, to reach it without an earlier facet refusal.
    """
    rf = build_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "8" * 32, content_version="cv_" + "9" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 1.0,
                "unit": "unit/USD",
                "measurementMethod": None,
                "provenance": "src_a",
            },
            {
                "stat_var": "dcid:X",
                "recipient": "country/ETH",
                "time_period": "2020",
                "value": 2.0,
                "unit": "unit/EUR",
                "measurementMethod": None,
                "provenance": "src_a",
            },
        ],
    )
    frame_store.put(resolved_frame=rf)

    with pytest.raises(ConstraintViolation) as excinfo:
        write_component(
            intent={
                "type": "line_chart",
                "frame_ref": {
                    "spec_id": rf.frame_ref.spec_id,
                    "content_version": rf.frame_ref.content_version,
                },
                "select": {"stat_var": "dcid:X", "unit": ["unit/USD", "unit/EUR"]},
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "comparable_units_required"


_PER_PERSON = {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}}
_PER_WOMAN = {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person_Female"}}


@pytest.mark.parametrize(
    ("member", "units"),
    [
        pytest.param("line_chart", ("USDollar", None), id="line_chart-set-and-null"),
        pytest.param("bar_chart", ("USDollar", None), id="bar_chart-set-and-null"),
        pytest.param("line_chart", (_PER_PERSON, _PER_WOMAN), id="line_chart-composed"),
    ],
)
def test_a_chart_over_series_with_different_units_refuses_comparable_units(
    member, units, frame_store, revision_store, release_registry
):
    """Issue #953: GDP in USDollar beside population with a null unit resolves as two series.
    A chart over them refuses on its units, after resolution lets them through. Two composed
    units that differ in one side refuse the same way."""
    rf = _stored_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "6" * 32, content_version="cv_" + "7" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var,
                "recipient": "country/GTM",
                "time_period": "2020",
                "value": 1.0,
                "unit": unit,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for stat_var, unit in zip(("dcid:GDP", "dcid:Population"), units, strict=True)
        ],
    )
    frame_store.put(resolved_frame=rf)

    with pytest.raises(ConstraintViolation) as excinfo:
        write_component(
            intent={
                "type": member,
                "frame_ref": {
                    "spec_id": rf.frame_ref.spec_id,
                    "content_version": rf.frame_ref.content_version,
                },
                "select": {"stat_var": ["dcid:GDP", "dcid:Population"]},
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.constraint_id == "comparable_units_required"


@pytest.mark.parametrize(
    ("units", "labels", "table_unit"),
    [
        pytest.param(
            ("U" * 200 + "a" * 100, "U" * 200 + "b" * 100),
            {},
            None,
            id="labels-cut-to-the-same-200-characters",
        ),
        pytest.param(
            ("USDollar", "CurrencyUSD"),
            {"USDollar": "US Dollar", "CurrencyUSD": "US Dollar"},
            None,
            id="two-refs-the-graph-names-alike",
        ),
        pytest.param(
            ("USDollar", "USDollar"), {"USDollar": "US Dollar"}, "US Dollar", id="one-unit-control"
        ),
    ],
)
def test_series_units_compare_by_their_frame_cells_not_their_labels(
    units, labels, table_unit, frame_store, revision_store, release_registry
):
    """Two units that differ in the frame differ on a card, whatever their labels read: a
    line_chart over them refuses and a table reports no shared unit."""
    rf = _stored_resolved_frame(
        frame_ref=FrameRef(spec_id="spec_" + "4" * 32, content_version="cv_" + "5" * 32),
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var,
                "recipient": "country/GTM",
                "time_period": "2020",
                "value": 1.0,
                "unit": unit,
                "measurementMethod": None,
                "provenance": "src_a",
            }
            for stat_var, unit in zip(("dcid:A", "dcid:B"), units, strict=True)
        ],
        labels=labels,
    )
    frame_store.put(resolved_frame=rf)

    def write(member):
        return write_component(
            intent={
                "type": member,
                "frame_ref": {
                    "spec_id": rf.frame_ref.spec_id,
                    "content_version": rf.frame_ref.content_version,
                },
                "select": {"stat_var": ["dcid:A", "dcid:B"]},
            },
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )

    assert write("table").fact_summary["unit"] == table_unit
    if table_unit is None:
        with pytest.raises(ConstraintViolation) as excinfo:
            write("line_chart")
        assert excinfo.value.constraint_id == "comparable_units_required"
    else:
        assert write("line_chart").fact_summary["unit"] == table_unit


# -- unknown id / frame refusals ------------------------------------------------


def test_revising_an_unknown_component_id_raises_draft_not_found(
    frame_store, revision_store, release_registry
):
    with pytest.raises(DraftNotFound) as excinfo:
        write_component(
            intent={"title": "New title"},
            component_id="cmp_never_created",
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.subject_kind == "component"


def test_creating_into_an_unknown_composition_id_raises_draft_not_found(
    frame_store, revision_store, release_registry
):
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)
    with pytest.raises(DraftNotFound) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id="cps_never_minted",
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.subject_kind == "composition"


def test_ops_ref_resolves_once_the_frame_store_holds_it(
    frame_store, revision_store, release_registry
):
    """An `ops_` spec id resolves through the frame store after registration.

    The `cv_` content version shows that the pipeline gate uses the spec-id prefix.
    """
    frame_ref = FrameRef(spec_id="ops_" + "a" * 32, content_version="cv_" + "b" * 32)
    frame_store.put(
        resolved_frame=build_resolved_frame(
            frame_ref=frame_ref, role_keys=("recipient",), rows=malaria_ethiopia_rows()
        )
    )

    created = write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": frame_ref.spec_id,
                "content_version": frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/ETH",
            },
            "window": {"start": "2015", "end": "2024"},
            "title": "Derived frame",
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert created.component_id


def test_ops_ref_with_an_ocv_content_version_resolves_once_the_frame_store_holds_it(
    frame_store, revision_store, release_registry
):
    """Derived frame versions use `ocv_`.

    The envelope accepts `cv_` or `ocv_` followed by 6 to 64 hex digits.
    """
    frame_ref = FrameRef(spec_id="ops_" + "a" * 32, content_version="ocv_" + "b" * 32)
    frame_store.put(
        resolved_frame=build_resolved_frame(
            frame_ref=frame_ref, role_keys=("recipient",), rows=malaria_ethiopia_rows()
        )
    )

    created = write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": frame_ref.spec_id,
                "content_version": frame_ref.content_version,
            },
            "select": {
                "stat_var": "dcid:MalariaGrantDisbursements",
                "recipient": "country/ETH",
            },
            "window": {"start": "2015", "end": "2024"},
            "title": "Derived frame",
        },
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=release_registry,
    )

    assert created.component_id


def test_unpublished_ops_ref_still_refuses_with_ops_unsupported(revision_store, release_registry):
    """An unregistered `ops_` reference propagates `ops_unsupported` from `FrameStore.load`.

    The stub isolates that reason from `FakeFrameStore`'s generic `io` refusal.
    """

    class _UnpublishedOpsFrameStore:
        def load(self, *, frame_ref):
            raise FrameUnavailable(
                reason="ops_unsupported",
                detail=f"{frame_ref.spec_id!r} was minted but never published to the platform",
            )

        def pin(self, *, frame_ref, component_id):
            raise AssertionError("pin should not run after an ops_unsupported refusal")

    with pytest.raises(FrameUnavailable) as excinfo:
        write_component(
            intent={
                "type": "line_chart",
                "frame_ref": {"spec_id": "ops_" + "a" * 32, "content_version": "ocv_" + "b" * 32},
                "select": {"recipient": "country/ETH"},
            },
            component_id=None,
            composition_id=None,
            frame_store=_UnpublishedOpsFrameStore(),
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.reason == "ops_unsupported"


# -- envelope size refusals -----------------------------------------------------


def _minted_bytes(payload):
    return len(
        json.dumps(
            {"component": payload["component"], "provenance": payload["provenance"]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def test_an_oversized_create_refuses_before_pinning_or_storing(
    frame_store, revision_store, release_registry, monkeypatch, caplog
):
    """The cap runs in `_finish_candidate`, ahead of the frame pin and the
    write transaction, so an oversized create leaves no pin, no revision,
    and no reconciliation marker behind."""
    monkeypatch.setattr(candidate_module, "MAX_ENVELOPE_BYTES", 100)
    frame, intent = _malaria_intent()
    frame_store.put(resolved_frame=frame)

    with caplog.at_level(logging.ERROR), pytest.raises(ResourceExhausted) as excinfo:
        write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.pointer == "/"
    assert excinfo.value.limit_name == "MAX_ENVELOPE_BYTES"
    assert frame_store.pins == ()
    assert revision_store._revisions == {}
    assert ORPHANED_PIN_RECONCILIATION_REQUIRED_MARKER not in caplog.text


def test_an_oversized_revise_refuses_and_leaves_the_head_revision(
    frame_store, revision_store, release_registry, monkeypatch
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
    monkeypatch.setattr(candidate_module, "MAX_ENVELOPE_BYTES", 100)

    with pytest.raises(ResourceExhausted) as excinfo:
        write_component(
            intent={"title": "New title"},
            component_id=created.component_id,
            write_token=created.write_token,
            composition_id=None,
            frame_store=frame_store,
            revision_store=revision_store,
            release_registry=release_registry,
        )
    assert excinfo.value.limit_name == "MAX_ENVELOPE_BYTES"
    assert len(revision_store._revisions[created.component_id]) == 1


@pytest.mark.parametrize(("slack", "admitted"), [(0, True), (-1, False)])
def test_the_envelope_cap_measures_compact_utf8_json(
    frame_store, release_registry, monkeypatch, slack, admitted
):
    """The accented title distinguishes UTF-8 bytes from `ensure_ascii` output.

    Compact separators set the measured size. The exact size admits; one byte less refuses.
    """
    frame, intent = _malaria_intent(title="Paludisme en Éthiopie : décaissements")
    frame_store.put(resolved_frame=frame)
    measuring_store = FakeRevisionStore()
    measured = write_component(
        intent=intent,
        component_id=None,
        composition_id=None,
        frame_store=frame_store,
        revision_store=measuring_store,
        release_registry=release_registry,
    )
    size = _minted_bytes(measuring_store._revisions[measured.component_id][-1].resolved_payload)
    monkeypatch.setattr(candidate_module, "MAX_ENVELOPE_BYTES", size + slack)

    def write():
        return write_component(
            intent=intent,
            component_id=None,
            composition_id=None,
            frame_store=frame_store,
            revision_store=FakeRevisionStore(),
            release_registry=release_registry,
        )

    if admitted:
        assert write().revision == 1
    else:
        with pytest.raises(ResourceExhausted):
            write()
