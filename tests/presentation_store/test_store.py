"""Checks `LivePresentationStore` id-grant, transaction, revision, and reclaim behaviour."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import presentation_store.store as store_module
from assembler.ports import (
    CompositionAppend,
    DraftNotFound,
    EditConflict,
    PriorCreate,
    RevisionRecord,
    WriteTokenInUse,
)
from presentation_store.store import LivePresentationStore
from presentation_store.store_errors import GeneratedIdCollision, PresentationStoreUnavailable

from .conftest import (
    CommitExplodingFakePresentationStore,
    DriverExplodingFakePresentationStore,
    FakePresentationStore,
    SqlstateExplodingFakePresentationStore,
    TransactionEntryExplodingFakePresentationStore,
)


def _make_store(
    *, fake: FakePresentationStore | None = None
) -> tuple[LivePresentationStore, FakePresentationStore]:
    fake = fake or FakePresentationStore()
    return LivePresentationStore(store=fake), fake


def _minimal_record(*, component_id: str, revision: int) -> RevisionRecord:
    return RevisionRecord(
        component_id=component_id,
        revision=revision,
        stored_intent={"type": "stat_card"},
        resolved_payload={"type": "stat_card", "revision": revision},
        catalog_version="2026.8.0",
        schema_version=1,
        formatter_profile={"significant_digits": 3},
    )


def _create(store: LivePresentationStore, *, composition_id: str | None = None):
    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=composition_id,
    ) as ctx:
        record = _minimal_record(component_id=ctx.component_id, revision=1)
        composition_entry = None
        if ctx.composition_id is not None:
            from assembler.ports import CompositionAppend

            composition_entry = CompositionAppend(composition_id=ctx.composition_id, actor="model")
        ctx.composition_revision = ctx.append(record=record, composition_entry=composition_entry)
    return ctx


# -- no caller check: unknown ids raise DraftNotFound


def test_read_current_raises_draft_not_found_for_an_unknown_component():
    store, _fake = _make_store()
    with pytest.raises(DraftNotFound) as excinfo:
        store.read_current(component_id="cmp_never_created")
    assert excinfo.value.subject_kind == "component"
    assert excinfo.value.subject_id == "cmp_never_created"


def test_read_revision_raises_draft_not_found_for_an_unknown_component():
    store, _fake = _make_store()
    with pytest.raises(DraftNotFound):
        store.read_revision(component_id="cmp_never_created", revision=1)


def test_read_revision_of_a_missing_revision_on_a_known_component_is_none():
    store, _fake = _make_store()
    ctx = _create(store)
    assert store.read_revision(component_id=ctx.component_id, revision=99) is None


def test_a_second_store_over_the_same_database_reads_and_revises_by_id():
    """A store holding no state of its own reads and revises a draft another store created."""
    creator, fake = _make_store()
    ctx = _create(creator)
    other = LivePresentationStore(store=FakePresentationStore(fake.state))

    assert other.read_current(component_id=ctx.component_id) is not None
    with other.begin_write(
        component_id=ctx.component_id, is_create=False, composition_id=None
    ) as write:
        write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=2),
            composition_entry=None,
        )
    current = creator.read_current(component_id=ctx.component_id)
    assert current is not None
    assert current.revision == 2


def test_begin_write_revise_raises_draft_not_found_for_an_unknown_component():
    store, _fake = _make_store()
    with (
        pytest.raises(DraftNotFound),
        store.begin_write(component_id="cmp_never_created", is_create=False, composition_id=None),
    ):
        pass


def test_the_store_issues_no_draft_lease_statement():
    """Reads, creates and revises leave `presentation.draft_lease` untouched."""
    store, fake = _make_store()
    ctx = _create(store, composition_id="new")
    with store.begin_write(
        component_id=ctx.component_id, is_create=False, composition_id=None
    ) as write:
        write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=2),
            composition_entry=None,
        )
    store.read_current(component_id=ctx.component_id)
    store.read_revision(component_id=ctx.component_id, revision=1)
    store.composition_of(component_id=ctx.component_id)

    assert fake.statement_log
    assert not [sql for sql in fake.statement_log if "draft_lease" in sql]


# -- write-token digests: stored once at create, returned by write_token_digest


def test_a_create_stores_the_component_and_new_composition_write_token_digests():
    store, _fake = _make_store()
    component_digest = b"\x01" * 32
    composition_digest = b"\x02" * 32
    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id="new",
        write_token_sha256=component_digest,
        composition_write_token_sha256=composition_digest,
    ) as ctx:
        from assembler.ports import CompositionAppend

        ctx.append(
            record=_minimal_record(component_id=ctx.component_id, revision=1),
            composition_entry=CompositionAppend(composition_id=ctx.composition_id, actor="model"),
        )

    assert (
        store.write_token_digest(subject_kind="component", subject_id=ctx.component_id)
        == component_digest
    )
    assert (
        store.write_token_digest(subject_kind="composition", subject_id=ctx.composition_id)
        == composition_digest
    )


def test_a_revise_leaves_the_stored_write_token_digest_unchanged():
    store, _fake = _make_store()
    digest = b"\x03" * 32
    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=None,
        write_token_sha256=digest,
    ) as ctx:
        ctx.append(
            record=_minimal_record(component_id=ctx.component_id, revision=1),
            composition_entry=None,
        )
    with store.begin_write(
        component_id=ctx.component_id,
        is_create=False,
        composition_id=None,
        write_token_sha256=b"\x04" * 32,
    ) as write:
        write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=2),
            composition_entry=None,
        )

    assert store.write_token_digest(subject_kind="component", subject_id=ctx.component_id) == digest


def test_a_draft_created_without_a_digest_reads_back_none():
    store, _fake = _make_store()
    ctx = _create(store, composition_id="new")

    assert store.write_token_digest(subject_kind="component", subject_id=ctx.component_id) is None
    assert (
        store.write_token_digest(subject_kind="composition", subject_id=ctx.composition_id) is None
    )


@pytest.mark.parametrize("subject_kind", ["component", "composition"])
def test_write_token_digest_raises_draft_not_found_for_an_unknown_id(subject_kind):
    store, _fake = _make_store()
    with pytest.raises(DraftNotFound) as excinfo:
        store.write_token_digest(subject_kind=subject_kind, subject_id="never_created")
    assert excinfo.value.subject_kind == subject_kind


# -- reads: current vs a specific revision


# -- idempotent create: the write-token digest is unique and names its component


def _create_with_token(
    store: LivePresentationStore,
    *,
    write_token_sha256: bytes,
    create_request_sha256: bytes | None = None,
    composition_id: str | None = None,
):
    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=composition_id,
        write_token_sha256=write_token_sha256,
        create_request_sha256=create_request_sha256,
    ) as ctx:
        composition_entry = (
            CompositionAppend(composition_id=ctx.composition_id, actor="model")
            if ctx.composition_id is not None
            else None
        )
        ctx.append(
            record=_minimal_record(component_id=ctx.component_id, revision=1),
            composition_entry=composition_entry,
        )
    return ctx


def test_prior_create_names_the_component_its_request_hash_and_composition():
    store, _fake = _make_store()
    ctx = _create_with_token(
        store,
        write_token_sha256=b"\x05" * 32,
        create_request_sha256=b"\x06" * 32,
        composition_id="new",
    )

    assert store.prior_create(write_token_sha256=b"\x05" * 32) == PriorCreate(
        component_id=ctx.component_id,
        composition_id=ctx.composition_id,
        create_request_sha256=b"\x06" * 32,
    )


def test_prior_create_reads_none_for_a_digest_no_component_stores():
    store, _fake = _make_store()
    _create_with_token(store, write_token_sha256=b"\x05" * 32)

    assert store.prior_create(write_token_sha256=b"\x07" * 32) is None


def test_prior_create_of_a_create_without_a_request_hash_reads_it_back_none():
    store, _fake = _make_store()
    ctx = _create_with_token(store, write_token_sha256=b"\x05" * 32)

    assert store.prior_create(write_token_sha256=b"\x05" * 32) == PriorCreate(
        component_id=ctx.component_id, composition_id=None, create_request_sha256=None
    )


def test_a_revise_leaves_the_stored_create_request_hash_unchanged():
    store, _fake = _make_store()
    ctx = _create_with_token(
        store, write_token_sha256=b"\x05" * 32, create_request_sha256=b"\x06" * 32
    )
    with store.begin_write(
        component_id=ctx.component_id,
        is_create=False,
        composition_id=None,
        create_request_sha256=b"\x08" * 32,
    ) as write:
        write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=2),
            composition_entry=None,
        )

    prior = store.prior_create(write_token_sha256=b"\x05" * 32)
    assert prior is not None
    assert prior.create_request_sha256 == b"\x06" * 32


def test_a_second_create_with_a_stored_write_token_digest_raises_write_token_in_use():
    store, fake = _make_store()
    first = _create_with_token(store, write_token_sha256=b"\x05" * 32, composition_id="new")
    component_id = store.allocate_component_id()

    with (
        pytest.raises(WriteTokenInUse) as excinfo,
        store.begin_write(
            component_id=component_id,
            is_create=True,
            composition_id="new",
            write_token_sha256=b"\x05" * 32,
        ),
    ):
        pass

    assert excinfo.value.__context__ is None
    assert component_id not in fake.state.components
    assert fake.state.compositions == {first.composition_id}


def test_creates_with_distinct_write_token_digests_both_land():
    store, fake = _make_store()
    _create_with_token(store, write_token_sha256=b"\x05" * 32)
    _create_with_token(store, write_token_sha256=b"\x07" * 32)

    assert len(fake.state.components) == 2


def test_read_current_returns_the_latest_revision():
    store, _fake = _make_store()
    ctx = _create(store)

    current = store.read_current(component_id=ctx.component_id)

    assert current is not None
    assert current.revision == 1
    assert current.resolved_payload == {"type": "stat_card", "revision": 1}


def test_series_coordinates_and_formatter_profile_round_trip_through_create_and_read():
    store, _fake = _make_store()
    coordinates = ({"id": "srs_abc", "coordinates": {"stat_var": "dcid:X"}},)
    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=None,
    ) as ctx:
        record = RevisionRecord(
            component_id=ctx.component_id,
            revision=1,
            stored_intent={"type": "stat_card"},
            resolved_payload={"type": "stat_card", "revision": 1},
            catalog_version="2026.8.0",
            schema_version=1,
            formatter_profile={"significant_digits": 6},
            series_coordinates=coordinates,
        )
        component_id = ctx.component_id
        ctx.append(record=record, composition_entry=None)

    current = store.read_current(component_id=component_id)
    assert current is not None
    assert current.series_coordinates == [{"id": "srs_abc", "coordinates": {"stat_var": "dcid:X"}}]
    assert current.formatter_profile == {"significant_digits": 6}


def test_series_coordinates_is_none_when_the_record_never_set_it():
    """`_minimal_record` builds a `RevisionRecord` naming no
    `series_coordinates` -- the same shape a revision written before this
    field existed carries -- and the store round-trips that absence as
    `None`, not an empty list."""
    store, _fake = _make_store()
    ctx = _create(store)

    current = store.read_current(component_id=ctx.component_id)

    assert current is not None
    assert current.series_coordinates is None


def test_every_read_returns_the_instant_the_insert_stamped():
    store, _fake = _make_store()
    before = datetime.now(UTC)
    ctx = _create(store)
    after = datetime.now(UTC)

    current = store.read_current(component_id=ctx.component_id)
    revision = store.read_revision(component_id=ctx.component_id, revision=1)

    assert current is not None and revision is not None
    assert current.created_at is not None
    assert before <= current.created_at <= after
    assert revision.created_at == current.created_at


# -- composition_of: membership plus the composition's current head revision


def test_composition_of_returns_the_composition_id_and_head_revision():
    store, _fake = _make_store()
    ctx = _create(store, composition_id="new")

    membership = store.composition_of(component_id=ctx.component_id)

    assert membership is not None
    assert membership.composition_id == ctx.composition_id
    assert membership.revision == 1


def test_composition_of_returns_none_for_a_component_outside_any_composition():
    store, _fake = _make_store()
    ctx = _create(store)

    assert store.composition_of(component_id=ctx.component_id) is None


def test_composition_of_reports_the_composition_s_latest_revision_after_a_second_write():
    from assembler.ports import CompositionAppend

    store, _fake = _make_store()
    first = _create(store, composition_id="new")

    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=first.composition_id,
    ) as second:
        second.append(
            record=_minimal_record(component_id=second.component_id, revision=1),
            composition_entry=CompositionAppend(composition_id=first.composition_id, actor="model"),
        )

    membership = store.composition_of(component_id=first.component_id)
    assert membership is not None
    assert membership.revision == 2


def test_composition_of_an_unknown_component_raises_draft_not_found():
    store, _fake = _make_store()
    with pytest.raises(DraftNotFound):
        store.composition_of(component_id="cmp_never_created")


def test_composition_of_while_a_begin_write_context_is_open_is_unavailable():
    # `begin_write` and `composition_of` both open `self._store.transaction()`
    # on the same underlying store. A second, nested open on the same
    # thread is exactly what the fake's own same-thread guard (mirroring
    # `LiveDatasetStore.transaction`) refuses.
    store, _fake = _make_store()
    ctx = _create(store)

    with (
        store.begin_write(component_id=ctx.component_id, is_create=False, composition_id=None),
        pytest.raises(PresentationStoreUnavailable),
    ):
        store.composition_of(component_id=ctx.component_id)


# -- lock order and locking


def test_create_in_composition_locks_composition_before_component():
    store, fake = _make_store()

    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id="new",
    ):
        pass

    statements = fake.statement_log
    composition_index = next(
        i
        for i, sql in enumerate(statements)
        if sql.startswith("INSERT INTO presentation.composition (")
    )
    component_index = next(
        i
        for i, sql in enumerate(statements)
        if sql.startswith("INSERT INTO presentation.component (")
    )
    assert composition_index < component_index


def test_revise_locks_the_latest_revision_row_for_update():
    store, fake = _make_store()
    ctx = _create(store)
    before = len(fake.statement_log)

    with store.begin_write(component_id=ctx.component_id, is_create=False, composition_id=None):
        pass

    issued = fake.statement_log[before:]
    assert any("FOR UPDATE" in sql and "component_revision" in sql for sql in issued), issued


def test_touching_an_existing_composition_locks_it_for_update():
    store, fake = _make_store()
    ctx = _create(store, composition_id="new")
    composition_id = ctx.composition_id
    before = len(fake.statement_log)

    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=composition_id,
    ):
        pass

    issued = fake.statement_log[before:]
    assert any(
        "FOR UPDATE" in sql and "FROM presentation.composition WHERE" in sql for sql in issued
    ), issued


# -- compositions: a create names an existing `cps_` id to append to it


def test_create_into_an_existing_composition_by_id_appends_to_it():
    store, _fake = _make_store()
    first = _create(store, composition_id="new")

    second = _create(store, composition_id=first.composition_id)

    assert second.composition_id == first.composition_id
    assert second.composition_revision == 2


def test_create_into_an_unknown_composition_id_raises_draft_not_found():
    store, _fake = _make_store()

    with (
        pytest.raises(DraftNotFound) as excinfo,
        store.begin_write(
            component_id=store.allocate_component_id(),
            is_create=True,
            composition_id="cps_never_minted",
        ),
    ):
        pass
    assert excinfo.value.subject_kind == "composition"


# -- revision insert, N+1, and the PK-backstop conflict conversion


def test_append_inserts_revision_two_on_a_revise():
    store, _fake = _make_store()
    ctx = _create(store)

    with store.begin_write(
        component_id=ctx.component_id, is_create=False, composition_id=None
    ) as write:
        assert write.current is not None
        assert write.current.revision == 1
        write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=2),
            composition_entry=None,
        )

    current = store.read_current(component_id=ctx.component_id)
    assert current is not None
    assert current.revision == 2


def test_append_raises_edit_conflict_when_the_revision_already_exists():
    store, _fake = _make_store()
    ctx = _create(store)

    with (
        store.begin_write(
            component_id=ctx.component_id, is_create=False, composition_id=None
        ) as write,
        pytest.raises(EditConflict),
    ):
        # Revision 1 already exists -- simulates the PK backstop firing
        # for a caller (or a would-be race) that computed a stale target.
        write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=1),
            composition_entry=None,
        )


# -- composition block synthesis


def test_composition_touch_creates_a_first_revision_with_one_block():
    store, fake = _make_store()
    ctx = _create(store, composition_id="new")

    with fake.state.lock:
        revisions = fake.state.composition_revisions[ctx.composition_id]
    assert len(revisions) == 1
    assert revisions[0]["revision"] == 1
    assert revisions[0]["blocks"] == [{"component_id": ctx.component_id, "revision": 1}]
    assert revisions[0]["actor"] == "model"


def test_a_second_component_in_the_same_composition_appends_a_second_block():
    from assembler.ports import CompositionAppend

    store, fake = _make_store()
    first = _create(store, composition_id="new")

    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=first.composition_id,
    ) as write:
        write.append(
            record=_minimal_record(component_id=write.component_id, revision=1),
            composition_entry=CompositionAppend(composition_id=first.composition_id, actor="model"),
        )
        second_component_id = write.component_id

    with fake.state.lock:
        revisions = fake.state.composition_revisions[first.composition_id]
    assert len(revisions) == 2
    assert revisions[-1]["blocks"] == [
        {"component_id": first.component_id, "revision": 1},
        {"component_id": second_component_id, "revision": 1},
    ]


def test_revising_a_component_already_in_a_composition_updates_its_block_in_place():
    from assembler.ports import CompositionAppend

    store, fake = _make_store()
    ctx = _create(store, composition_id="new")

    with store.begin_write(
        component_id=ctx.component_id, is_create=False, composition_id=None
    ) as write:
        write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=2),
            composition_entry=CompositionAppend(composition_id=ctx.composition_id, actor="model"),
        )

    with fake.state.lock:
        revisions = fake.state.composition_revisions[ctx.composition_id]
    assert len(revisions) == 2
    # Position preserved -- one block, its revision advanced, not a second
    # entry for the same component.
    assert revisions[-1]["blocks"] == [{"component_id": ctx.component_id, "revision": 2}]


def test_revise_recovers_the_persisted_composition_id_without_the_caller_supplying_one():
    """A revise reaches `begin_write` with no `composition_id` of its own
    -- the pipeline raises before `begin_write` is ever called if a revise
    supplies one. Recovering the component's own persisted composition
    membership lets it advance the composition's own blocks anyway.
    """
    store, _fake = _make_store()
    created = _create(store, composition_id="new")

    with store.begin_write(
        component_id=created.component_id, is_create=False, composition_id=None
    ) as write:
        assert write.composition_id == created.composition_id


def test_revise_of_a_standalone_component_leaves_composition_id_none():
    store, _fake = _make_store()
    ctx = _create(store)

    with store.begin_write(
        component_id=ctx.component_id, is_create=False, composition_id=None
    ) as write:
        assert write.composition_id is None


def test_append_returns_the_minted_composition_revision_number():
    from assembler.ports import CompositionAppend

    store, _fake = _make_store()
    first = _create(store, composition_id="new")
    assert first.composition_revision == 1

    with store.begin_write(
        component_id=first.component_id, is_create=False, composition_id=None
    ) as write:
        second_revision = write.append(
            record=_minimal_record(component_id=first.component_id, revision=2),
            composition_entry=CompositionAppend(composition_id=first.composition_id, actor="model"),
        )
    assert second_revision == 2


def test_append_returns_none_when_no_composition_is_touched():
    store, _fake = _make_store()
    ctx = _create(store)

    with store.begin_write(
        component_id=ctx.component_id, is_create=False, composition_id=None
    ) as write:
        result = write.append(
            record=_minimal_record(component_id=ctx.component_id, revision=2),
            composition_entry=None,
        )
    assert result is None


# -- create-in-composition atomicity


def test_fresh_component_id_collision_is_distinguished_from_other_integrity_failures():
    store, _fake = _make_store()
    existing = _create(store)

    with (
        pytest.raises(GeneratedIdCollision) as excinfo,
        store.begin_write(
            component_id=existing.component_id,
            is_create=True,
            composition_id=None,
        ),
    ):
        pass

    assert excinfo.value.subject_kind == "component"
    assert excinfo.value.__context__ is None


def test_fresh_composition_id_collision_is_distinguished_and_rolls_back_component(
    monkeypatch: pytest.MonkeyPatch,
):
    store, fake = _make_store()
    existing = _create(store, composition_id="new")
    monkeypatch.setattr(store_module, "mint_composition_id", lambda: existing.composition_id)
    component_id = store.allocate_component_id()

    with (
        pytest.raises(GeneratedIdCollision) as excinfo,
        store.begin_write(
            component_id=component_id,
            is_create=True,
            composition_id="new",
        ),
    ):
        pass

    assert excinfo.value.subject_kind == "composition"
    assert excinfo.value.__context__ is None
    assert component_id not in fake.state.components


def test_create_in_composition_lands_neither_row_when_the_component_insert_fails():
    fake = FakePresentationStore(
        fail_before=lambda sql, params: sql.startswith("INSERT INTO presentation.component ("),
    )
    store = LivePresentationStore(store=fake)

    with (
        pytest.raises(PresentationStoreUnavailable),
        store.begin_write(
            component_id=store.allocate_component_id(),
            is_create=True,
            composition_id="new",
        ),
    ):
        pass

    assert fake.state.compositions == set()
    assert fake.state.components == {}


def test_list_orphaned_presentation_references_surfaces_a_manufactured_orphan():
    store, _fake = _make_store()
    ctx = _create(store)

    orphaned = store.list_orphaned_presentation_references(
        referenced_component_ids=[ctx.component_id, "cmp_never_created"]
    )

    assert orphaned == ("cmp_never_created",)


def test_list_orphaned_presentation_references_stays_empty_when_every_referenced_id_exists():
    store, _fake = _make_store()
    ctx = _create(store)

    orphaned = store.list_orphaned_presentation_references(
        referenced_component_ids=[ctx.component_id]
    )

    assert orphaned == ()


def test_list_orphaned_presentation_references_skips_the_query_for_an_empty_input():
    store, fake = _make_store()

    orphaned = store.list_orphaned_presentation_references(referenced_component_ids=[])

    assert orphaned == ()
    assert fake.statement_log == []


# -- driver failure conversion


def test_read_current_wraps_a_driver_failure_with_no_leaked_context():
    distinctive = "distinctive_driver_boom_presentation_read_f7a1"
    store = LivePresentationStore(
        store=DriverExplodingFakePresentationStore(message=distinctive),
    )

    with pytest.raises(PresentationStoreUnavailable) as excinfo:
        store.read_current(component_id="cmp_x")

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert distinctive not in str(exc)


def test_a_driver_failure_carrying_a_sqlstate_attaches_it_to_the_raised_exception():
    """A pg8000-shaped driver exception's own SQLSTATE (`args[0]["C"]`)
    reaches `PresentationStoreUnavailable.sqlstate`, extracted via
    `dfo_platform.store_errors.extract_sqlstate` -- not lost the way a bare
    `RuntimeError(message)` (`DriverExplodingFakePresentationStore`'s own
    shape) would leave it.
    """
    store = LivePresentationStore(
        store=SqlstateExplodingFakePresentationStore(sqlstate="42601"),
    )

    with pytest.raises(PresentationStoreUnavailable) as excinfo:
        store.read_current(component_id="cmp_x")

    assert excinfo.value.sqlstate == "42601"


def test_a_bare_driver_failure_with_no_sqlstate_leaves_sqlstate_none():
    store = LivePresentationStore(
        store=DriverExplodingFakePresentationStore(message="distinctive_driver_boom_no_sqlstate"),
    )

    with pytest.raises(PresentationStoreUnavailable) as excinfo:
        store.read_current(component_id="cmp_x")

    assert excinfo.value.sqlstate is None


def test_begin_write_wraps_a_driver_failure_with_no_leaked_context():
    distinctive = "distinctive_driver_boom_presentation_write_a291"
    store = LivePresentationStore(
        store=DriverExplodingFakePresentationStore(message=distinctive),
    )

    with (
        pytest.raises(PresentationStoreUnavailable) as excinfo,
        store.begin_write(component_id="cmp_x", is_create=True, composition_id=None),
    ):
        pass

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert distinctive not in str(exc)


def test_begin_write_wraps_a_failure_borrowing_the_transactions_own_connection():
    """Distinct from `test_begin_write_wraps_a_driver_failure_...` above:
    that one fails once the block issues its first statement; this fails
    opening the transaction itself, before `_prepare_write` ever runs.
    """
    distinctive = "distinctive_driver_boom_presentation_borrow_c481"
    store = LivePresentationStore(
        store=TransactionEntryExplodingFakePresentationStore(message=distinctive),
    )

    with (
        pytest.raises(PresentationStoreUnavailable) as excinfo,
        store.begin_write(component_id="cmp_x", is_create=True, composition_id=None),
    ):
        pass

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert distinctive not in str(exc)


def test_begin_write_wraps_a_failure_committing_after_a_clean_exit():
    """The block's own body succeeds -- every statement it issues lands --
    but the transaction's own commit-on-clean-exit itself fails.
    """
    distinctive = "distinctive_driver_boom_presentation_commit_e930"
    store = LivePresentationStore(
        store=CommitExplodingFakePresentationStore(message=distinctive),
    )

    with (
        pytest.raises(PresentationStoreUnavailable) as excinfo,
        store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx,
    ):
        ctx.append(record=_minimal_record(component_id="cmp_x", revision=1), composition_entry=None)

    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert distinctive not in str(exc)


def test_begin_write_attaches_the_sqlstate_of_a_failure_committing_after_a_clean_exit():
    """A pg8000-shaped commit-on-clean-exit failure reaches
    `PresentationStoreUnavailable.sqlstate` the same way a read-path
    failure does (`test_a_driver_failure_carrying_a_sqlstate_attaches_it_
    to_the_raised_exception`), with no leaked `__context__`.
    """
    store = LivePresentationStore(
        store=CommitExplodingFakePresentationStore(message="commit boom", sqlstate="42601"),
    )

    with (
        pytest.raises(PresentationStoreUnavailable) as excinfo,
        store.begin_write(component_id="cmp_x", is_create=True, composition_id=None) as ctx,
    ):
        ctx.append(record=_minimal_record(component_id="cmp_x", revision=1), composition_entry=None)

    exc = excinfo.value
    assert exc.sqlstate == "42601"
    assert exc.__context__ is None


def test_begin_write_lets_the_callers_own_exception_propagate_through_rollback_unwrapped():
    """The caller's own business logic, raised from inside the
    `with begin_write(...)` block, must reach the caller unchanged: the
    same rollback path that wraps a genuine commit failure leaves it
    alone rather than reclassifying it as `PresentationStoreUnavailable`.
    """
    store, _fake = _make_store()

    class _CallerOwnError(Exception):
        pass

    with (
        pytest.raises(_CallerOwnError),
        store.begin_write(
            component_id=store.allocate_component_id(),
            is_create=True,
            composition_id=None,
        ),
    ):
        raise _CallerOwnError("business logic failed, not a store failure")
