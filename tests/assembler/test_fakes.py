"""Pin frame-store, revision-store, and release-registry fake behaviour."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from assembler.fakes import FakeFrameStore, FakeReleaseRegistry, FakeRevisionStore
from assembler.ports import (
    CompositionAppend,
    DraftNotFound,
    EditConflict,
    FrameRef,
    FrameUnavailable,
    PriorCreate,
    RevisionRecord,
    WriteTokenInUse,
)
from assembler.testing import build_resolved_frame

_FRAME_REF = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)


_FIRST_COMMIT = datetime(2026, 10, 4, 9, 0, tzinfo=UTC)


def _record(*, component_id="cmp_x", revision=1, created_at=None):
    return RevisionRecord(
        component_id=component_id,
        revision=revision,
        stored_intent={"type": "line_chart"},
        resolved_payload={"schema_version": 1},
        catalog_version="2026.8.0",
        schema_version=1,
        formatter_profile={"significant_digits": 3},
        created_at=created_at,
    )


def _ticking_store() -> FakeRevisionStore:
    """A store whose clock reads one second later at each commit."""
    ticks = itertools.count()
    return FakeRevisionStore(clock=lambda: _FIRST_COMMIT + timedelta(seconds=next(ticks)))


def _stored(*, revision: int) -> RevisionRecord:
    """Revision `revision` as `_ticking_store` holds it when it was the store's nth commit."""
    return _record(revision=revision, created_at=_FIRST_COMMIT + timedelta(seconds=revision - 1))


def _resolved_frame():
    return build_resolved_frame(
        frame_ref=_FRAME_REF,
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
            }
        ],
    )


# -- FakeFrameStore ----------------------------------------------------------


def test_frame_store_loads_a_registered_frame():
    store = FakeFrameStore()
    resolved = _resolved_frame()
    store.put(resolved_frame=resolved)
    assert store.load(frame_ref=_FRAME_REF) is resolved


def test_frame_store_refuses_unknown_frame():
    store = FakeFrameStore()
    with pytest.raises(FrameUnavailable):
        store.load(frame_ref=_FRAME_REF)


def test_frame_store_pin_is_idempotent_and_recorded():
    store = FakeFrameStore()
    store.put(resolved_frame=_resolved_frame())
    store.pin(frame_ref=_FRAME_REF, component_id="cmp_x")
    store.pin(frame_ref=_FRAME_REF, component_id="cmp_x")
    assert store.pins == ((_FRAME_REF, "cmp_x"),)


def test_frame_store_refuses_pinning_unknown_frame():
    store = FakeFrameStore()
    with pytest.raises(FrameUnavailable):
        store.pin(frame_ref=_FRAME_REF, component_id="cmp_x")


def test_frame_store_release_pin_removes_a_recorded_pin():
    store = FakeFrameStore()
    store.put(resolved_frame=_resolved_frame())
    store.pin(frame_ref=_FRAME_REF, component_id="cmp_x")
    store.release_pin(frame_ref=_FRAME_REF, component_id="cmp_x")
    assert store.pins == ()


def test_frame_store_release_pin_is_a_no_op_for_an_unrecorded_pin():
    store = FakeFrameStore()
    store.release_pin(frame_ref=_FRAME_REF, component_id="cmp_x")  # does not raise
    assert store.pins == ()


def test_frame_store_release_pin_raises_an_injected_error():
    store = FakeFrameStore(release_pin_error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        store.release_pin(frame_ref=_FRAME_REF, component_id="cmp_x")


# -- FakeRevisionStore: no caller check, unknown ids raise DraftNotFound ------


def test_unknown_component_raises_draft_not_found():
    store = FakeRevisionStore()
    with pytest.raises(DraftNotFound) as excinfo:
        store.read_current(component_id="cmp_x")
    assert excinfo.value.subject_kind == "component"
    assert excinfo.value.subject_id == "cmp_x"
    with pytest.raises(DraftNotFound):
        store.read_revision(component_id="cmp_x", revision=1)


def test_begin_write_revise_of_an_unknown_component_raises_draft_not_found():
    store = FakeRevisionStore()
    with (
        pytest.raises(DraftNotFound),
        store.begin_write(component_id="cmp_x", is_create=False, composition_id=None),
    ):
        pass


def test_any_holder_of_the_id_reads_a_created_component():
    store = _ticking_store()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)
    assert store.read_current(component_id="cmp_x") == _stored(revision=1)


# -- FakeRevisionStore: begin_write / WriteContext ----------------------------


def test_begin_write_create_mints_component_id_and_current_is_none():
    store = FakeRevisionStore()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx:
        assert ctx.current is None
        assert ctx.component_id == "cmp_x"


def test_begin_write_new_composition_mints_composition_id():
    store = FakeRevisionStore()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id="new") as ctx:
        assert ctx.composition_id.startswith("cps_")


def test_append_commits_only_on_clean_exit():
    store = _ticking_store()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)
        # Not yet committed inside the context.
        with pytest.raises(DraftNotFound):
            store.read_current(component_id="cmp_x")
    assert store.read_current(component_id="cmp_x") == _stored(revision=1)


def test_append_does_not_commit_on_exception():
    store = FakeRevisionStore()
    with (
        pytest.raises(ValueError),
        store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx,
    ):
        ctx.append(record=_record(revision=1), composition_entry=None)
        raise ValueError("caller-side failure mid-transaction")
    with pytest.raises(DraftNotFound):
        store.read_current(component_id="cmp_x")


def test_stale_append_raises_edit_conflict():
    store = FakeRevisionStore()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)
    with store.begin_write(component_id="cmp_x", is_create=False, composition_id=None) as racer:
        with store.begin_write(
            component_id="cmp_x", is_create=False, composition_id=None
        ) as winner:
            winner.append(record=_record(revision=2), composition_entry=None)
        # `racer` entered before `winner` committed revision 2; its own
        # captured `current` (revision 1) is now stale.
        with pytest.raises(EditConflict):
            racer.append(record=_record(revision=2), composition_entry=None)


def test_second_revision_after_first_commits_succeeds():
    store = _ticking_store()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)
    with store.begin_write(component_id="cmp_x", is_create=False, composition_id=None) as ctx:
        assert ctx.current == _stored(revision=1)
        ctx.append(record=_record(revision=2), composition_entry=None)
    assert store.read_current(component_id="cmp_x") == _stored(revision=2)


def test_read_revision_returns_a_specific_past_revision():
    store = _ticking_store()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)
    with store.begin_write(component_id="cmp_x", is_create=False, composition_id=None) as ctx:
        ctx.append(record=_record(revision=2), composition_entry=None)
    assert store.read_revision(component_id="cmp_x", revision=1) == _stored(revision=1)
    assert store.read_revision(component_id="cmp_x", revision=99) is None


def test_composition_revision_advances_on_every_write_even_when_the_component_repeats():
    """Every committed write advances the composition revision, even for a repeated component.

    The block list can remain shorter because repeated component entries update in place.
    """
    store = FakeRevisionStore()

    with store.begin_write(component_id="cmp_x", is_create=True, composition_id="new") as ctx:
        composition_id = ctx.composition_id
        entry = CompositionAppend(composition_id=composition_id, actor="model")
        first_revision = ctx.append(
            record=_record(component_id="cmp_x", revision=1), composition_entry=entry
        )
    assert first_revision == 1

    with store.begin_write(
        component_id="cmp_x", is_create=False, composition_id=composition_id
    ) as ctx:
        second_revision = ctx.append(
            record=_record(component_id="cmp_x", revision=2), composition_entry=entry
        )
    assert second_revision == 2


def test_begin_write_refuses_a_non_new_composition_id_with_no_prior_mint():
    store = FakeRevisionStore()
    with (
        pytest.raises(DraftNotFound),
        store.begin_write(component_id="cmp_x", is_create=True, composition_id="cps_never_minted"),
    ):
        pass


def test_composition_of_reports_the_composition_id_and_head_revision_after_create():
    store = FakeRevisionStore()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id="new") as ctx:
        composition_id = ctx.composition_id
        ctx.append(
            record=_record(revision=1),
            composition_entry=CompositionAppend(composition_id=composition_id, actor="model"),
        )

    membership = store.composition_of(component_id="cmp_x")
    assert membership is not None
    assert membership.composition_id == composition_id
    assert membership.revision == 1


def test_composition_of_returns_none_for_a_component_outside_any_composition():
    store = FakeRevisionStore()
    with store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)

    assert store.composition_of(component_id="cmp_x") is None


def test_composition_of_an_unknown_component_raises_draft_not_found():
    store = FakeRevisionStore()
    with pytest.raises(DraftNotFound):
        store.composition_of(component_id="cmp_never_created")


def test_composition_of_inside_an_open_write_raises():
    """`composition_of` rejects same-thread calls inside an open write.

    The guard prevents reads of uncommitted state.
    """
    store = FakeRevisionStore()
    with (
        pytest.raises(RuntimeError, match="composition_of"),
        store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx,
    ):
        store.composition_of(component_id=ctx.component_id)


def test_a_create_stores_the_write_token_digests_it_was_given():
    store = FakeRevisionStore()
    with store.begin_write(
        component_id="cmp_x",
        is_create=True,
        composition_id="new",
        write_token_sha256=b"c" * 32,
        composition_write_token_sha256=b"p" * 32,
    ) as ctx:
        composition_id = ctx.composition_id
        ctx.append(
            record=_record(revision=1),
            composition_entry=CompositionAppend(composition_id=composition_id, actor="model"),
        )
    with store.begin_write(
        component_id="cmp_x", is_create=False, composition_id=None, write_token_sha256=b"z" * 32
    ) as ctx:
        ctx.append(
            record=_record(revision=2),
            composition_entry=CompositionAppend(composition_id=composition_id, actor="model"),
        )

    assert store.write_token_digest(subject_kind="component", subject_id="cmp_x") == b"c" * 32
    assert (
        store.write_token_digest(subject_kind="composition", subject_id=composition_id) == b"p" * 32
    )


def test_a_forgotten_write_token_digest_reads_back_none():
    store = FakeRevisionStore()
    with store.begin_write(
        component_id="cmp_x", is_create=True, composition_id=None, write_token_sha256=b"c" * 32
    ) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)

    store.forget_write_token(subject_kind="component", subject_id="cmp_x")

    assert store.write_token_digest(subject_kind="component", subject_id="cmp_x") is None


@pytest.mark.parametrize("subject_kind", ["component", "composition"])
def test_write_token_digest_of_an_unknown_id_raises_draft_not_found(subject_kind):
    with pytest.raises(DraftNotFound):
        FakeRevisionStore().write_token_digest(subject_kind=subject_kind, subject_id="never")


def test_prior_create_names_the_component_holding_a_write_token_digest():
    store = FakeRevisionStore()
    with store.begin_write(
        component_id="cmp_x",
        is_create=True,
        composition_id="new",
        write_token_sha256=b"c" * 32,
        create_request_sha256=b"r" * 32,
    ) as ctx:
        composition_id = ctx.composition_id
        ctx.append(
            record=_record(revision=1),
            composition_entry=CompositionAppend(composition_id=composition_id, actor="model"),
        )

    assert store.prior_create(write_token_sha256=b"c" * 32) == PriorCreate(
        component_id="cmp_x", composition_id=composition_id, create_request_sha256=b"r" * 32
    )
    assert store.prior_create(write_token_sha256=b"d" * 32) is None


def test_a_create_with_a_committed_write_token_digest_raises_on_entry():
    store = FakeRevisionStore()
    with store.begin_write(
        component_id="cmp_x", is_create=True, composition_id=None, write_token_sha256=b"c" * 32
    ) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)

    with (
        pytest.raises(WriteTokenInUse),
        store.begin_write(
            component_id="cmp_y",
            is_create=True,
            composition_id=None,
            write_token_sha256=b"c" * 32,
        ),
    ):
        pass

    with pytest.raises(DraftNotFound):
        store.read_current(component_id="cmp_y")


def test_the_second_of_two_open_creates_with_one_digest_raises_on_commit():
    store = FakeRevisionStore()
    first = store.begin_write(
        component_id="cmp_x", is_create=True, composition_id=None, write_token_sha256=b"c" * 32
    )
    second = store.begin_write(
        component_id="cmp_y", is_create=True, composition_id=None, write_token_sha256=b"c" * 32
    )
    first_ctx = first.__enter__()
    second_ctx = second.__enter__()
    first_ctx.append(record=_record(component_id="cmp_x", revision=1), composition_entry=None)
    second_ctx.append(record=_record(component_id="cmp_y", revision=1), composition_entry=None)
    first.__exit__(None, None, None)

    with pytest.raises(WriteTokenInUse):
        second.__exit__(None, None, None)

    prior = store.prior_create(write_token_sha256=b"c" * 32)
    assert prior is not None
    assert prior.component_id == "cmp_x"
    with pytest.raises(DraftNotFound):
        store.read_current(component_id="cmp_y")


def test_a_revise_stores_no_create_request_hash():
    store = FakeRevisionStore()
    with store.begin_write(
        component_id="cmp_x", is_create=True, composition_id=None, write_token_sha256=b"c" * 32
    ) as ctx:
        ctx.append(record=_record(revision=1), composition_entry=None)
    with store.begin_write(
        component_id="cmp_x",
        is_create=False,
        composition_id=None,
        create_request_sha256=b"r" * 32,
    ) as ctx:
        ctx.append(record=_record(revision=2), composition_entry=None)

    prior = store.prior_create(write_token_sha256=b"c" * 32)
    assert prior is not None
    assert prior.create_request_sha256 is None


def test_prior_create_inside_an_open_write_raises():
    store = FakeRevisionStore()
    with (
        pytest.raises(RuntimeError, match="prior_create"),
        store.begin_write(component_id="cmp_x", is_create=True, composition_id=None),
    ):
        store.prior_create(write_token_sha256=b"c" * 32)


# -- FakeReleaseRegistry -------------------------------------------------------


def test_fake_release_registry_defaults_to_the_release_it_was_given(release):
    registry = FakeReleaseRegistry(release=release)
    assert registry.pinned().catalog_version == release.catalog_version
