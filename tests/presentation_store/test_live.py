"""Live migration and store tests; set `DFO_PRESENTATION_STORE_LIVE_DSN` to run them."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import pytest

from assembler.catalogue import BakedReleaseRegistry
from assembler.fakes import FakeFrameStore
from assembler.pipeline import write_component
from assembler.ports import (
    DraftNotFound,
    EditConflict,
    FrameRef,
    PriorCreate,
    RevisionRecord,
    WriteTokenInUse,
)
from assembler.testing import malaria_ethiopia_resolved_frame
from assembler.write_token import write_token_digest
from dataset_store import ConnectionPool, LiveDatasetStore, connect
from dataset_store.migrate import apply_pending
from presentation_store.migrate import MIGRATIONS_DIR
from presentation_store.shares import (
    ImportedShare,
    LiveShareStore,
    ShareCard,
    ShareIdTaken,
    ShareNotFound,
    SlotCheck,
)
from presentation_store.store import LivePresentationStore
from presentation_store.store_errors import GeneratedIdCollision

_DSN = os.environ.get("DFO_PRESENTATION_STORE_LIVE_DSN")
_BOOKKEEPING_SCHEMA = "presentation"
_BOOKKEEPING_TABLE = "presentation.presentation_store_schema_migrations"

# The bootstrap migration grants this role, which must exist before the migration runs.
_RUNTIME_USER_PLACEHOLDER = "presentation-runtime-placeholder@ci.dev"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _DSN, reason="set DFO_PRESENTATION_STORE_LIVE_DSN to run live presentation_store tests"
    ),
]


def _connect():
    parts = urlsplit(_DSN)
    return connect(
        connector=None,
        instance_connection_name=None,
        host=parts.hostname,
        port=parts.port or 5432,
        database=parts.path.lstrip("/"),
        user=parts.username,
        password=parts.password,
    )


def _pool(*, size: int = 2) -> ConnectionPool:
    return ConnectionPool(connect=_connect, size=size)


@pytest.fixture
def live_store():
    pool = _pool()
    try:
        yield LiveDatasetStore(connections=pool)
    finally:
        pool.close_all()


def _apply(migrations_dir=MIGRATIONS_DIR) -> list[str]:
    return apply_pending(
        connect=_connect,
        migrations_dir=migrations_dir,
        substitutions={"RUNTIME_DB_USER": _RUNTIME_USER_PLACEHOLDER},
        bookkeeping_schema=_BOOKKEEPING_SCHEMA,
        bookkeeping_table=_BOOKKEEPING_TABLE,
    )


def test_apply_pending_applies_every_migration_and_a_second_run_is_a_no_op(live_store):
    # Migrations may already be applied; the second call must always be a no-op.
    _apply()
    second = _apply()
    assert second == []

    on_disk = {path.name for path in MIGRATIONS_DIR.glob("*.sql")}
    with live_store.transaction() as tx:
        rows = tx.fetch_all(f"SELECT version FROM {_BOOKKEEPING_TABLE}")
    assert {row["version"] for row in rows} == on_disk

    with live_store.transaction() as tx:
        reg = tx.fetch_one("SELECT to_regclass('presentation.component_revision') AS reg")
    assert reg == {"reg": "presentation.component_revision"}


def _seed_component(tx, component_id: str) -> None:
    tx.execute(
        "INSERT INTO presentation.component (component_id) VALUES (%s) "
        "ON CONFLICT (component_id) DO NOTHING",
        (component_id,),
    )


def _seed_revision(tx, *, component_id: str, resolved_payload: dict) -> None:
    tx.execute(
        "INSERT INTO presentation.component_revision "
        "(component_id, revision, stored_intent, resolved_payload, catalog_version, "
        "schema_version) VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (component_id, revision) DO UPDATE SET "
        "resolved_payload = EXCLUDED.resolved_payload",
        (component_id, 1, {}, resolved_payload, "test", 1),
    )


def test_bar_chart_axes_migration_rewrites_only_legacy_bar_chart_rows(live_store):
    _apply()

    legacy_bar_chart_id = "comp:test-live-bar-chart-legacy"
    line_chart_id = "comp:test-live-line-chart"
    migrated_bar_chart_id = "comp:test-live-bar-chart-migrated"

    legacy_payload = {
        "component_id": legacy_bar_chart_id,
        "revision": 1,
        "provenance": {"source": "test"},
        "component": {
            "type": "bar_chart",
            "title": "Legacy bar chart",
            "x_axis": {"field": "year"},
            "y_axis": {"field": "value"},
        },
    }
    line_chart_payload = {
        "component_id": line_chart_id,
        "revision": 1,
        "provenance": {"source": "test"},
        "component": {
            "type": "line_chart",
            "title": "Line chart",
            "x_axis": {"field": "year"},
            "y_axis": {"field": "value"},
        },
    }
    migrated_payload = {
        "component_id": migrated_bar_chart_id,
        "revision": 1,
        "provenance": {"source": "test"},
        "component": {
            "type": "bar_chart",
            "title": "Already migrated bar chart",
            "category_axis": {"field": "year"},
            "value_axis": {"field": "value"},
        },
    }

    # Seed rows before reapplying the 0006 data rewrite.
    with live_store.transaction() as tx:
        for component_id in (legacy_bar_chart_id, line_chart_id, migrated_bar_chart_id):
            _seed_component(tx, component_id)
        _seed_revision(tx, component_id=legacy_bar_chart_id, resolved_payload=legacy_payload)
        _seed_revision(tx, component_id=line_chart_id, resolved_payload=line_chart_payload)
        _seed_revision(tx, component_id=migrated_bar_chart_id, resolved_payload=migrated_payload)

    # Remove the bookkeeping row so `apply_pending` reruns migration 0006.
    with live_store.transaction() as tx:
        tx.execute(
            f"DELETE FROM {_BOOKKEEPING_TABLE} WHERE version = %s", ("0006_bar_chart_axes.sql",)
        )
    applied = _apply()
    assert applied == ["0006_bar_chart_axes.sql"]

    with live_store.transaction() as tx:
        legacy_row = tx.fetch_one(
            "SELECT resolved_payload FROM presentation.component_revision "
            "WHERE component_id = %s AND revision = %s",
            (legacy_bar_chart_id, 1),
        )
        line_chart_row = tx.fetch_one(
            "SELECT resolved_payload FROM presentation.component_revision "
            "WHERE component_id = %s AND revision = %s",
            (line_chart_id, 1),
        )
        migrated_row = tx.fetch_one(
            "SELECT resolved_payload FROM presentation.component_revision "
            "WHERE component_id = %s AND revision = %s",
            (migrated_bar_chart_id, 1),
        )

    legacy_component = legacy_row["resolved_payload"]["component"]
    assert legacy_component["category_axis"] == {"field": "year"}
    assert legacy_component["value_axis"] == {"field": "value"}
    assert "x_axis" not in legacy_component
    assert "y_axis" not in legacy_component
    assert legacy_component["type"] == "bar_chart"
    assert legacy_component["title"] == "Legacy bar chart"
    assert legacy_row["resolved_payload"]["provenance"] == {"source": "test"}

    assert line_chart_row["resolved_payload"] == line_chart_payload
    assert migrated_row["resolved_payload"] == migrated_payload

    # The second run leaves legacy bar-chart rows unchanged; line-chart axes remain untouched.
    second = _apply()
    assert second == []
    with live_store.transaction() as tx:
        remaining = tx.fetch_all(
            "SELECT component_id FROM presentation.component_revision "
            "WHERE resolved_payload -> 'component' ->> 'type' = 'bar_chart' "
            "AND (resolved_payload -> 'component' ? 'x_axis' "
            "OR resolved_payload -> 'component' ? 'y_axis')"
        )
    assert remaining == []


class _RollBack(Exception):
    """Raised to roll back a transaction that replays a migration."""


@pytest.mark.parametrize(("catalog_version", "refuses"), [("2026.8.0", False), ("2026.7.0", True)])
def test_formatter_profile_migration_fills_2026_8_0_rows_and_refuses_any_other(
    live_store, catalog_version, refuses
):
    """0010 replays inside one transaction on a table returned to its 0009 shape, then rolls
    back. A 2026.8.0 row takes the 2026.8.0 profile, as does an insert that omits the column;
    a row stamped with any other release fails the migration."""
    _apply()
    component_id = f"comp:test-live-formatter-profile-{catalog_version}"
    sql_text = (MIGRATIONS_DIR / "0010_revision_formatter_profile.sql").read_text()
    profiles: list[object] = []

    with pytest.raises(_RollBack), live_store.transaction() as tx:
        tx.execute(
            "DELETE FROM presentation.component_revision WHERE catalog_version <> '2026.8.0'"
        )
        tx.execute("ALTER TABLE presentation.component_revision DROP COLUMN formatter_profile")
        _seed_component(tx, component_id)
        tx.execute(
            "INSERT INTO presentation.component_revision (component_id, revision, "
            "stored_intent, resolved_payload, catalog_version, schema_version) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (component_id, 1, {}, {}, catalog_version, 1),
        )
        try:
            tx.execute(sql_text)
        except Exception as exc:
            assert refuses, exc
            assert "contains null values" in str(exc)
            raise _RollBack from exc
        tx.execute(
            "INSERT INTO presentation.component_revision (component_id, revision, "
            "stored_intent, resolved_payload, catalog_version, schema_version) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (component_id, 2, {}, {}, catalog_version, 1),
        )
        rows = tx.fetch_all(
            "SELECT formatter_profile FROM presentation.component_revision "
            "WHERE component_id = %s ORDER BY revision",
            (component_id,),
        )
        profiles = [row["formatter_profile"] for row in rows]
        raise _RollBack

    if not refuses:
        assert len(profiles) == 2
        assert profiles[0] == profiles[1]
        assert profiles[0]["significant_digits"] == 3


# -- idempotent create against Postgres: the write-token index and the race it settles


@pytest.fixture
def presentation_store():
    _apply()
    pool = _pool(size=4)
    try:
        yield LivePresentationStore(store=LiveDatasetStore(connections=pool))
    finally:
        pool.close_all()


def _record(component_id: str) -> RevisionRecord:
    return RevisionRecord(
        component_id=component_id,
        revision=1,
        stored_intent={"type": "stat_card"},
        resolved_payload={"type": "stat_card"},
        catalog_version="test",
        schema_version=1,
        formatter_profile={"significant_digits": 3},
    )


def _create(
    store: LivePresentationStore, *, write_token_sha256: bytes, create_request_sha256: bytes
) -> str:
    with store.begin_write(
        component_id=store.allocate_component_id(),
        is_create=True,
        composition_id=None,
        write_token_sha256=write_token_sha256,
        create_request_sha256=create_request_sha256,
    ) as ctx:
        ctx.append(record=_record(ctx.component_id), composition_entry=None)
    return ctx.component_id


def test_prior_create_reads_back_the_component_its_digest_names(presentation_store):
    digest = secrets.token_bytes(32)
    request_digest = secrets.token_bytes(32)
    component_id = _create(
        presentation_store, write_token_sha256=digest, create_request_sha256=request_digest
    )

    assert presentation_store.prior_create(write_token_sha256=digest) == PriorCreate(
        component_id=component_id, composition_id=None, create_request_sha256=request_digest
    )
    assert presentation_store.prior_create(write_token_sha256=secrets.token_bytes(32)) is None


def test_a_read_revision_carries_the_column_default_s_creation_instant(presentation_store):
    component_id = _create(
        presentation_store,
        write_token_sha256=secrets.token_bytes(32),
        create_request_sha256=secrets.token_bytes(32),
    )

    current = presentation_store.read_current(component_id=component_id)
    revision = presentation_store.read_revision(component_id=component_id, revision=1)

    assert current is not None and revision is not None
    assert current.created_at is not None and current.created_at.utcoffset() is not None
    assert revision.created_at == current.created_at


def test_a_committed_digest_refuses_a_second_create_as_write_token_in_use(presentation_store):
    digest = secrets.token_bytes(32)
    _create(presentation_store, write_token_sha256=digest, create_request_sha256=b"\x01" * 32)
    loser_id = presentation_store.allocate_component_id()

    with (
        pytest.raises(WriteTokenInUse),
        presentation_store.begin_write(
            component_id=loser_id,
            is_create=True,
            composition_id=None,
            write_token_sha256=digest,
        ),
    ):
        pass

    with pytest.raises(DraftNotFound):
        presentation_store.read_current(component_id=loser_id)


def test_a_component_id_collision_stays_a_generated_id_collision(presentation_store):
    component_id = _create(
        presentation_store,
        write_token_sha256=secrets.token_bytes(32),
        create_request_sha256=b"\x01" * 32,
    )

    with (
        pytest.raises(GeneratedIdCollision),
        presentation_store.begin_write(
            component_id=component_id,
            is_create=True,
            composition_id=None,
            write_token_sha256=secrets.token_bytes(32),
        ),
    ):
        pass


def _race(presentation_store, *, digest: bytes, winner_commits: bool):
    """Hold the winner's create open on one connection while the loser inserts the same digest
    on another, check the loser is still waiting, then let the winner commit or roll back.
    Return the winner's component id, the loser's exception, and the loser's component id."""
    winner_inserted = threading.Event()
    release_winner = threading.Event()

    class _Abort(Exception):
        pass

    def winner() -> str:
        with (
            contextlib.suppress(_Abort),
            presentation_store.begin_write(
                component_id=presentation_store.allocate_component_id(),
                is_create=True,
                composition_id=None,
                write_token_sha256=digest,
                create_request_sha256=b"\x01" * 32,
            ) as ctx,
        ):
            ctx.append(record=_record(ctx.component_id), composition_entry=None)
            winner_inserted.set()
            assert release_winner.wait(timeout=30)
            if not winner_commits:
                raise _Abort
        return ctx.component_id

    def loser() -> str:
        with presentation_store.begin_write(
            component_id=presentation_store.allocate_component_id(),
            is_create=True,
            composition_id=None,
            write_token_sha256=digest,
            create_request_sha256=b"\x02" * 32,
        ) as ctx:
            ctx.append(record=_record(ctx.component_id), composition_entry=None)
        return ctx.component_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner_future = pool.submit(winner)
        assert winner_inserted.wait(timeout=30)
        loser_future = pool.submit(loser)
        try:
            # The loser's insert waits on the winner's uncommitted index entry.
            with pytest.raises(FuturesTimeout):
                loser_future.result(timeout=1.0)
        finally:
            release_winner.set()
        winner_id = winner_future.result(timeout=30)
        loser_error = loser_future.exception(timeout=30)
    loser_id = loser_future.result() if loser_error is None else None
    return winner_id, loser_error, loser_id


def test_the_loser_of_a_create_race_raises_write_token_in_use_after_the_winner_commits(
    presentation_store,
):
    digest = secrets.token_bytes(32)

    winner_id, loser_error, _loser_id = _race(
        presentation_store, digest=digest, winner_commits=True
    )

    assert isinstance(loser_error, WriteTokenInUse)
    prior = presentation_store.prior_create(write_token_sha256=digest)
    assert prior is not None
    assert prior.component_id == winner_id
    assert prior.create_request_sha256 == b"\x01" * 32


def test_the_loser_of_a_create_race_proceeds_when_the_winner_rolls_back(presentation_store):
    digest = secrets.token_bytes(32)

    _winner_id, loser_error, loser_id = _race(
        presentation_store, digest=digest, winner_commits=False
    )

    assert loser_error is None
    prior = presentation_store.prior_create(write_token_sha256=digest)
    assert prior is not None
    assert prior.component_id == loser_id
    assert prior.create_request_sha256 == b"\x02" * 32


# -- idempotent create through `write_component`: a retried create keyed by a client token


def _create_with_client_token(revision_store, *, token: str, frame_store: FakeFrameStore):
    frame_ref = malaria_ethiopia_resolved_frame().frame_ref
    return write_component(
        intent={
            "type": "line_chart",
            "frame_ref": {
                "spec_id": frame_ref.spec_id,
                "content_version": frame_ref.content_version,
            },
            "select": {"stat_var": "dcid:MalariaGrantDisbursements", "recipient": "country/ETH"},
            "window": {"start": "2015", "end": "2024"},
            "title": "Malaria grant disbursements to Ethiopia",
        },
        create_write_token=token,
        frame_store=frame_store,
        revision_store=revision_store,
        release_registry=BakedReleaseRegistry(),
    )


def _components_holding(live_store, *, token: str) -> list[str]:
    with live_store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT component_id FROM presentation.component WHERE write_token_sha256 = %s",
            (write_token_digest(token),),
        )
    return [row["component_id"] for row in rows]


@pytest.fixture
def malaria_frame_store() -> FakeFrameStore:
    frame_store = FakeFrameStore()
    frame_store.put(resolved_frame=malaria_ethiopia_resolved_frame())
    return frame_store


class _HeldOpenRevisionStore:
    """Wraps a `LivePresentationStore`. `begin_write` signals `inserted` once the caller's body
    has run and holds the transaction open until `release` is set."""

    def __init__(self, *, inner, inserted: threading.Event, release: threading.Event):
        self._inner = inner
        self._inserted = inserted
        self._release = release

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @contextlib.contextmanager
    def begin_write(self, **kwargs):
        with self._inner.begin_write(**kwargs) as ctx:
            yield ctx
            self._inserted.set()
            assert self._release.wait(timeout=30)


def test_a_retried_create_with_one_client_token_returns_the_first_component(
    presentation_store, live_store, malaria_frame_store
):
    token = f"cmw_{secrets.token_urlsafe(16)}"

    first = _create_with_client_token(
        presentation_store, token=token, frame_store=malaria_frame_store
    )
    again = _create_with_client_token(
        presentation_store, token=token, frame_store=malaria_frame_store
    )

    assert again.component_id == first.component_id
    assert again.write_token == token
    assert _components_holding(live_store, token=token) == [first.component_id]


def test_two_racing_creates_with_one_client_token_leave_one_component(
    presentation_store, live_store, malaria_frame_store
):
    token = f"cmw_{secrets.token_urlsafe(16)}"
    inserted = threading.Event()
    release = threading.Event()
    held_open = _HeldOpenRevisionStore(inner=presentation_store, inserted=inserted, release=release)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            _create_with_client_token, held_open, token=token, frame_store=malaria_frame_store
        )
        assert inserted.wait(timeout=30)
        # The second create passes the digest lookup, since the first has not committed, and
        # its insert waits on the first's uncommitted index entry.
        second = pool.submit(
            _create_with_client_token,
            presentation_store,
            token=token,
            frame_store=malaria_frame_store,
        )
        try:
            with pytest.raises(FuturesTimeout):
                second.result(timeout=1.0)
        finally:
            release.set()
        first_reply = first.result(timeout=30)
        second_reply = second.result(timeout=30)

    assert second_reply.component_id == first_reply.component_id
    assert (second_reply.revision, second_reply.write_token) == (1, token)
    assert _components_holding(live_store, token=token) == [first_reply.component_id]


def _share_card(component_id: str) -> ShareCard:
    return ShareCard(
        component_id=component_id,
        revision=1,
        frame=FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32),
    )


def test_a_share_keeps_its_slots_across_replacements_and_reads_absent_once_deleted(live_store):
    _apply()
    shares = LiveShareStore(store=live_store)
    digest = secrets.token_bytes(32)

    created = shares.create_share(
        kind="card",
        mode="live",
        title="Live share",
        write_token_sha256=digest,
        cards=[_share_card("cmp_a"), ShareCard(component_id="cmp_n", revision=2, frame=None)],
    )
    replaced = shares.replace_share(share_id=created.id, revision=1, cards=[_share_card("cmp_c")])

    assert shares.read_share(share_id=created.id) == replaced
    assert replaced.revision == 2
    assert replaced.write_token_sha256 == digest
    assert [(slot.slot, slot.card) for slot in replaced.slots] == [
        (1, _share_card("cmp_c")),
        (2, None),
    ]
    assert created.slots[1].card == ShareCard(component_id="cmp_n", revision=2, frame=None)

    shares.delete_share(share_id=created.id)
    with pytest.raises(ShareNotFound):
        shares.read_share(share_id=created.id)


def test_a_share_mint_moves_past_an_id_already_stored(live_store):
    _apply()
    taken = LiveShareStore(store=live_store).create_share(
        kind="card", mode="snapshot", title=None, write_token_sha256=None, cards=[]
    )
    minted = iter([taken.id, "Zq" + secrets.token_hex(3)])
    shares = LiveShareStore(store=live_store, mint_id=lambda: next(minted))

    created = shares.create_share(
        kind="card", mode="snapshot", title=None, write_token_sha256=None, cards=[]
    )

    assert created.id != taken.id
    assert shares.read_share(share_id=taken.id).write_token_sha256 is None


def test_a_live_share_converts_to_a_snapshot_keeping_the_frames_its_slots_serve(live_store):
    _apply()
    shares = LiveShareStore(store=live_store)
    created = shares.create_share(
        kind="canvas",
        mode="live",
        title=None,
        write_token_sha256=None,
        cards=[_share_card("cmp_a"), _share_card("cmp_b")],
        payload=[{"slot": 0}],
    )
    newer = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "c" * 32)

    snapshot = shares.snapshot_share(share_id=created.id, revision=1, frames={1: newer})

    assert (snapshot.mode, snapshot.revision, snapshot.payload) == ("snapshot", 2, [{"slot": 0}])
    assert [slot.card.frame for slot in snapshot.slots if slot.card] == [
        newer,
        _share_card("cmp_b").frame,
    ]
    assert shares.snapshot_share(share_id=created.id, revision=2, frames={}) == snapshot
    with pytest.raises(EditConflict):
        other = shares.create_share(
            kind="card", mode="live", title=None, write_token_sha256=None, cards=[]
        )
        shares.replace_share(share_id=other.id, revision=1, cards=[_share_card("cmp_c")])
        shares.snapshot_share(share_id=other.id, revision=1, frames={})


def test_an_import_keeps_its_id_and_time_once_and_refuses_an_id_holding_another_share(
    live_store,
):
    _apply()
    shares = LiveShareStore(store=live_store)
    at = datetime(2026, 9, 1, 10, 0, 0, 1, tzinfo=UTC)
    imported = ImportedShare(
        id="Im" + secrets.token_hex(3),
        kind="search",
        mode="live",
        title="Aid",
        owner="user_1",
        payload={"query": "aid", "parts": []},
        created_at=at,
        write_token_sha256=None,
        cards=(),
    )

    assert shares.import_share(imported) is True
    assert shares.import_share(dataclasses.replace(imported, title="Other")) is False
    with pytest.raises(ShareIdTaken):
        shares.import_share(dataclasses.replace(imported, created_at=at + timedelta(seconds=1)))
    stored = shares.read_share(share_id=imported.id)
    assert (stored.title, stored.owner, stored.created_at, stored.revised_at) == (
        "Aid",
        "user_1",
        at,
        at,
    )


def test_a_slot_check_claim_and_served_frames_round_trip(live_store):
    _apply()
    shares = LiveShareStore(store=live_store)
    card = _share_card("cmp_a")
    share = shares.create_share(
        kind="card", mode="live", title=None, write_token_sha256=None, cards=[card]
    )
    at = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
    newer = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "c" * 32)
    stored = card.frame
    assert stored is not None

    shares.record_slot_check(
        share_id=share.id,
        slot=1,
        card=card,
        check=SlotCheck(frame=stored, freshness="pending", checked_at=at, next_check_at=at),
    )
    shares.record_slot_check(
        share_id=share.id,
        slot=1,
        card=card,
        check=SlotCheck(
            frame=newer,
            freshness="current",
            checked_at=at,
            next_check_at=at + timedelta(seconds=30),
        ),
    )
    later = at + timedelta(seconds=30)
    until = later + timedelta(minutes=15)
    claims = [
        shares.claim_slot_check(share_id=share.id, slot=1, card=card, now=now, claim_until=until)
        for now in (at + timedelta(seconds=29), later, later)
    ]
    for frame in (newer, newer, stored):
        shares.record_served(
            share_id=share.id, slot=1, card=card, frame=frame, rendering={"columns": ["a"]}
        )

    first_claims = [
        shares.claim_slot_check(share_id=share.id, slot=2, card=card, now=at, claim_until=until)
        for _ in range(2)
    ]

    assert claims == [False, True, False]
    assert first_claims == [True, False]
    assert shares.slot_check(share_id=share.id, slot=1, card=card) == SlotCheck(
        frame=newer, freshness="current", checked_at=at, next_check_at=until
    )
    assert shares.slot_check(share_id=share.id, slot=2, card=card) == SlotCheck(
        frame=stored, freshness="pending", checked_at=None, next_check_at=until
    )
    served = shares.served_frames(share_id=share.id)
    assert [entry.frame for entry in served] == [stored, newer]
    assert served[0].rendering == {"columns": ["a"]}
