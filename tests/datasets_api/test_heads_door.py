"""`GET /api/datasets/heads` answers, for each spec its query names, the version a
`latest: "stored"` read of it would serve from its stored head now, or null, read-only and
cacheable for 300 s.

The registry rows, heads, version records and revocations sit in one SQLite store
(`FakeVersionRecordDatasetStore`) that the real `LiveHeadStore` reads, with the operation
store over it too, so a head check or any other write would show in its change count.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

import datasets_api.doors.heads as heads_door
from code_index.composites import COMPOSITES
from code_index.shared import get_index
from dataset_store.snapshot import StoredVersion, VersionState
from datasets_api.config import DatasetsBuildSettings
from datasets_api.current import stored_head_servable, stored_head_version
from datasets_api.doors.definition import definition_deps
from datasets_api.doors.heads import HEADS_MAX, head_content_version
from datasets_api.kinds.spec_id import served_spec_id
from datasets_api.operation_store import LiveOperationStore
from datasets_api.operations.submit import OperationSubmitted
from datasets_api.operations.task_client import FakeTaskClient
from datasets_api.operations.task_create_pool import InlineTaskCreates
from datasets_api.version_store import LiveHeadStore
from datasets_api.versions import StoredHeadRead
from dfo_platform.timings import count
from graph_release.context import current_release
from materialise.trace import UNKNOWN_FAMILY
from spec_registry import LiveSpecRegistry, recompute_identity
from spec_registry.reader import LiveSpecReader, StoredDefinition
from tests.code_index.aid_identity import CRS_PROVENANCE
from tests.spec_registry.conftest import shape_for

from ._app_factories import make_client
from ._deps_doubles import fake_deps
from ._version_record_store import (
    TEST_BASIS,
    FakeVersionRecordDatasetStore,
    materialisation_record,
)
from .conftest import minimal_identity
from .test_mint_canonical_identity import _identity as _aid_identity

_PATH = "/api/datasets/heads"
_NOW = datetime(2026, 6, 1, tzinfo=UTC)
# Inside the 900 s recheck window.
_FRESH = timedelta(minutes=1)
# Past the recheck window and the 24 h head max age, inside the 14-day serve-stale ceiling.
_PAST_WINDOW = timedelta(days=3)
# Past the 14-day serve-stale ceiling.
_PAST_CEILING = timedelta(days=15)


def _cv(n: int) -> str:
    return f"cv_{n:032x}"


def _unregistered(n: int) -> str:
    return f"spec_{n:032x}"


class _Rig:
    """One SQLite store holds the registry, heads, versions and operations, and backs the
    app's registry, spec reader, head store and operation store, so any write a read made
    would move its change count."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        head_writer: LiveHeadStore | None = None,
        read_max_concurrency: int | None = None,
    ) -> None:
        self.store = FakeVersionRecordDatasetStore()
        self.transactions = 0
        transaction = self.store.transaction

        @contextmanager
        def counted() -> Iterator[Any]:
            # As `LiveDatasetStore.transaction` counts on the call's timings.
            count("db_transactions")
            self.transactions += 1
            with transaction() as tx:
                yield tx

        monkeypatch.setattr(self.store, "transaction", counted)
        settings = DatasetsBuildSettings.from_env()
        self.task_client = FakeTaskClient()
        self.client = make_client(
            registry=LiveSpecRegistry(store=self.store),
            spec_reader=LiveSpecReader(store=self.store),
            head_writer=head_writer or LiveHeadStore(store=self.store, now=lambda: _NOW),
            read_max_concurrency=read_max_concurrency,
            operation_store=LiveOperationStore(store=self.store, settings=settings),
            task_client=self.task_client,
            task_creates=InlineTaskCreates(),
            now=lambda: _NOW,
        )
        self._shape = 0

    def register(self, identity: dict | None = None) -> str:
        """Register a definition of `identity`, a fresh one by default, and return its id."""
        if identity is None:
            self._shape += 1
            identity = minimal_identity(shape=shape_for(f"heads_{self._shape}"))
        recomputed = recompute_identity(identity)
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.spec_registry "
                "(spec_id, identity_version, identity_bytes, identity_digest) "
                "VALUES (%s, %s, %s, %s)",
                (
                    recomputed.spec_id,
                    "spec1",
                    recomputed.identity_bytes,
                    recomputed.identity_digest,
                ),
            )
            tx.execute(
                "INSERT INTO datasets.spec_definition (spec_id, schema_version, definition) "
                "VALUES (%s, %s, %s)",
                (recomputed.spec_id, 1, json.dumps({"identity": recomputed.identity})),
            )
        return recomputed.spec_id

    def head(
        self,
        subject_id: str,
        content_version: str,
        *,
        age: timedelta = _FRESH,
        imported_data: str | None = None,
        recorded: bool = True,
    ) -> None:
        """Write `subject_id`'s head at `content_version`, materialised and confirmed `age`
        before `_NOW`, keyed on `imported_data` when given, with its version record unless
        `recorded` is false."""
        record = materialisation_record(subject_id=subject_id, content_version=content_version)
        if imported_data is not None:
            record = replace(
                record, dependencies={**record.dependencies, "imported_data": imported_data}
            )
        written = LiveHeadStore(store=self.store, now=lambda: _NOW - age).advance_head(
            materialisation=record,
            family=UNKNOWN_FAMILY,
            basis=TEST_BASIS,
            expected_materialisation_id=None,
        )
        assert written is not None
        if recorded:
            with self.store.transaction() as tx:
                tx.execute(
                    "INSERT INTO datasets.version_record (subject_id, subject_kind, "
                    "content_version, schema_version, immutable_body, recorded_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (subject_id, "spec", content_version, 1, {}, _NOW - age),
                )

    def revoke(self, subject_id: str, content_version: str) -> None:
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.version_revocation "
                "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, %s)",
                (subject_id, content_version, "test", _NOW),
            )

    def quarantine(self, spec_id: str) -> None:
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.identity_quarantine (id_kind, subject_id) VALUES (%s, %s)",
                ("spec", spec_id),
            )

    def get(self, spec_ids: list[str], *, extra: tuple[tuple[str, str], ...] = ()):
        self.transactions = 0
        return self.client.get(
            _PATH, params=[*(("spec_id", spec_id) for spec_id in spec_ids), *extra]
        )


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> _Rig:
    return _Rig(monkeypatch)


def test_each_spec_answers_the_version_a_stored_read_serves_or_null(rig: _Rig) -> None:
    fresh = rig.register()
    rig.head(fresh, _cv(1))
    past_window = rig.register()
    rig.head(past_window, _cv(2), age=_PAST_WINDOW)
    keyed_on_bound = rig.register()
    rig.head(keyed_on_bound, _cv(3), imported_data=current_release().imported_data)
    no_head = rig.register()
    replaced = rig.register()
    rig.head(replaced, _cv(4), imported_data="sha256:replaced-import")
    past_ceiling = rig.register()
    rig.head(past_ceiling, _cv(5), age=_PAST_CEILING)
    revoked = rig.register()
    rig.head(revoked, _cv(6))
    rig.revoke(revoked, _cv(6))
    unrecorded = rig.register()
    rig.head(unrecorded, _cv(7), recorded=False)
    quarantined = rig.register()
    rig.head(quarantined, _cv(8))
    rig.quarantine(quarantined)
    # A non-canonical posted id reads the head its canonical served id holds, as a stored
    # read resolves it (`resolve_served_definition`).
    posted_identity = _aid_identity(pins_bound=False)
    posted = rig.register(posted_identity)
    served = rig.register(_aid_identity(pins_bound=True))
    assert posted != served
    rig.head(served, _cv(9))
    unregistered = _unregistered(1)
    expected = {
        fresh: _cv(1),
        past_window: _cv(2),
        keyed_on_bound: _cv(3),
        no_head: None,
        replaced: None,
        past_ceiling: None,
        revoked: None,
        unrecorded: None,
        quarantined: None,
        posted: _cv(9),
        unregistered: None,
    }

    resp = rig.get(list(expected))

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "heads": [
            {"spec_id": spec_id, "content_version": version}
            for spec_id, version in expected.items()
        ]
    }


def test_a_version_revoked_under_the_asked_id_or_as_a_base_answers_null(rig: _Rig) -> None:
    under_posted = rig.register(_aid_identity(pins_bound=False))
    served = rig.register(_aid_identity(pins_bound=True))
    rig.head(served, _cv(1))
    rig.revoke(under_posted, _cv(1))
    with_base = rig.register()
    rig.head(with_base, _cv(2))
    with rig.store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET base_content_version = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (_cv(3), with_base, _cv(2)),
        )
    rig.revoke(with_base, _cv(3))

    resp = rig.get([under_posted, served, with_base])

    assert [head["content_version"] for head in resp.json()["heads"]] == [None, _cv(1), None]


def test_the_answer_keeps_query_order(rig: _Rig) -> None:
    ids = [rig.register() for _ in range(3)]
    for n, spec_id in enumerate(ids):
        rig.head(spec_id, _cv(n))
    order = [ids[2], ids[0], ids[1]]

    resp = rig.get(order)

    assert [head["spec_id"] for head in resp.json()["heads"]] == order
    assert [head["content_version"] for head in resp.json()["heads"]] == [_cv(2), _cv(0), _cv(1)]


def test_a_batch_reads_in_one_transaction_and_writes_nothing(rig: _Rig) -> None:
    ids = [rig.register() for _ in range(5)]
    rig.head(ids[0], _cv(0))
    # Past the recheck window, where a stored read submits the head's check.
    rig.head(ids[1], _cv(1), age=_PAST_WINDOW)
    # Past the ceiling and with no head, where a stored read queues a rebuild.
    rig.head(ids[2], _cv(2), age=_PAST_CEILING)
    # A non-canonical posted id, which a stored read would resolve by registering its served
    # definition and binding it (`resolve_served_definition`).
    posted = rig.register(_aid_identity(pins_bound=False))
    canonical = recompute_identity(_aid_identity(pins_bound=True)).spec_id
    rig.head(canonical, _cv(5))
    ids.append(posted)
    changes = rig.store.total_changes

    resp = rig.get(ids)

    assert resp.status_code == 200, resp.text
    assert rig.transactions == 1
    assert "db_transactions;desc=1" in resp.headers["server-timing"]
    assert resp.json()["heads"][-1] == {"spec_id": posted, "content_version": _cv(5)}
    assert rig.store.total_changes == changes
    assert rig.task_client.enqueued == []


def test_the_answer_is_cacheable_for_five_minutes(rig: _Rig) -> None:
    resp = rig.get([rig.register()])

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "public, max-age=300"


@pytest.mark.parametrize(
    "query",
    [
        pytest.param([], id="none"),
        pytest.param([_unregistered(n) for n in range(HEADS_MAX + 1)], id="past-the-cap"),
        pytest.param([_unregistered(1), _unregistered(1)], id="repeated"),
        pytest.param(["spec_ABC"], id="malformed"),
        pytest.param([f"spec_{'A' * 32}"], id="uppercase"),
        pytest.param([f"{_unregistered(1)}\n"], id="trailing-newline"),
    ],
)
def test_a_query_outside_the_grammar_refuses_request_invalid_with_no_store(
    rig: _Rig, query: list[str]
) -> None:
    resp = rig.get(query)

    assert resp.status_code == 400
    assert resp.json()["code"] == "request_invalid"
    assert resp.headers["cache-control"] == "no-store"
    assert rig.transactions == 0


def test_the_cap_itself_is_answered(rig: _Rig) -> None:
    resp = rig.get([_unregistered(n) for n in range(HEADS_MAX)])

    assert resp.status_code == 200
    assert len(resp.json()["heads"]) == HEADS_MAX


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param((("cb", "123"),), id="cache-buster"),
        pytest.param((("spec_ids", _unregistered(2)),), id="near-miss-name"),
    ],
)
def test_a_query_naming_another_parameter_refuses_request_invalid_with_no_store(
    rig: _Rig, extra: tuple[tuple[str, str], ...]
) -> None:
    resp = rig.get([_unregistered(1)], extra=extra)

    assert resp.status_code == 400
    assert resp.json()["code"] == "request_invalid"
    assert resp.headers["cache-control"] == "no-store"
    assert rig.transactions == 0


def _unresolvable_identity() -> dict:
    """An identity whose provenance set names a composite, which canonicalisation refuses
    (`code_index.composites.CompositeNotServable`), as a stored read of it does."""
    return minimal_identity(
        shape=shape_for("heads_unresolvable"),
        facets={"provenance": {"state": "set", "refs": [COMPOSITES[0].provenance, CRS_PROVENANCE]}},
    )


def test_a_spec_with_no_served_id_answers_null_beside_the_others(rig: _Rig) -> None:
    first, last = rig.register(), rig.register()
    rig.head(first, _cv(1))
    rig.head(last, _cv(2))
    unresolvable = rig.register(_unresolvable_identity())

    resp = rig.get([first, unresolvable, last])

    assert resp.status_code == 200, resp.text
    assert [head["content_version"] for head in resp.json()["heads"]] == [_cv(1), None, _cv(2)]


def test_any_other_failure_resolving_a_served_id_is_not_a_store_failure(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(stored: StoredDefinition, *, code_index: object) -> str:
        raise RuntimeError("canonicalisation bug")

    monkeypatch.setattr(heads_door, "served_spec_id", broken)

    resp = rig.get([rig.register()])

    assert resp.status_code == 500
    assert resp.json()["code"] == "internal_error"


def test_a_quarantined_served_id_answers_null_for_a_clean_asked_id(rig: _Rig) -> None:
    posted = rig.register(_aid_identity(pins_bound=False))
    canonical = recompute_identity(_aid_identity(pins_bound=True)).spec_id
    rig.head(canonical, _cv(1))
    rig.quarantine(canonical)

    resp = rig.get([posted])

    assert resp.json()["heads"] == [{"spec_id": posted, "content_version": None}]


def test_a_revocation_under_the_served_id_answers_null_for_the_asked_id(rig: _Rig) -> None:
    posted = rig.register(_aid_identity(pins_bound=False))
    canonical = recompute_identity(_aid_identity(pins_bound=True)).spec_id
    rig.head(canonical, _cv(1))
    rig.revoke(canonical, _cv(1))

    resp = rig.get([posted])

    assert resp.json()["heads"] == [{"spec_id": posted, "content_version": None}]


class _GatedHeadStore(LiveHeadStore):
    """Holds every stored-head read until `gate` is set; `entered` is set when one waits."""

    def __init__(self, *, store: FakeVersionRecordDatasetStore) -> None:
        super().__init__(store=store, now=lambda: _NOW)
        self.gate = threading.Event()
        self.entered = threading.Event()

    def read_stored_heads(self, spec_ids, *, served_id):
        self.entered.set()
        assert self.gate.wait(5)
        return super().read_stored_heads(spec_ids, served_id=served_id)


def test_a_full_read_executor_refuses_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head_store = _GatedHeadStore(store=FakeVersionRecordDatasetStore())
    rig = _Rig(monkeypatch, head_writer=head_store, read_max_concurrency=1)
    held: list[Any] = []
    first = threading.Thread(target=lambda: held.append(rig.get([_unregistered(1)])))
    first.start()
    try:
        assert head_store.entered.wait(5)

        resp = rig.get([_unregistered(2)])
    finally:
        head_store.gate.set()
        first.join(5)

    assert resp.status_code == 503
    assert resp.json()["code"] == "service_unavailable"
    assert resp.headers["cache-control"] == "no-store"
    assert held[0].status_code == 200


def test_heads_is_never_read_as_a_dataset_id(rig: _Rig) -> None:
    """The literal reaches the heads door, and the id-taking routes never see it: a POST
    answers 405 from the heads route, where the malformed-id fallback would answer 404."""
    assert rig.get([_unregistered(1)]).status_code == 200
    assert rig.client.post(_PATH).status_code == 405


class _RecordedVersions:
    """Every version is recorded and none is revoked, as `read_version_state` reads them."""

    def read_version_state(
        self, subject_id: str, content_version: str, *, revoked_under: tuple[str, ...] = ()
    ) -> VersionState:
        return VersionState(version=cast(StoredVersion, None), revocation=None, locations=[])


@pytest.mark.parametrize(
    ("age", "imported_data", "head_exists"),
    [
        pytest.param(_FRESH, None, True, id="inside-the-window"),
        pytest.param(_PAST_WINDOW, None, True, id="past-the-window-inside-the-ceiling"),
        pytest.param(_PAST_CEILING, None, True, id="past-the-ceiling"),
        pytest.param(_FRESH, "sha256:replaced-import", True, id="replaced-import"),
        pytest.param(_FRESH, "bound", True, id="keyed-on-the-bound-import"),
        pytest.param(_FRESH, None, False, id="no-head"),
    ],
)
def test_the_check_and_stored_head_version_judge_a_head_alike(
    age: timedelta, imported_data: str | None, head_exists: bool
) -> None:
    store = FakeVersionRecordDatasetStore()
    head_store = LiveHeadStore(store=store, now=lambda: _NOW - age)
    spec_id = _unregistered(7)
    if head_exists:
        record = materialisation_record(subject_id=spec_id, content_version=_cv(1))
        if imported_data is not None:
            keyed = current_release().imported_data if imported_data == "bound" else imported_data
            record = replace(record, dependencies={**record.dependencies, "imported_data": keyed})
        head_store.advance_head(
            materialisation=record,
            family=UNKNOWN_FAMILY,
            basis=TEST_BASIS,
            expected_materialisation_id=None,
        )
    settings = DatasetsBuildSettings.from_env()
    deps = definition_deps(
        fake_deps(
            settings=settings,
            head_store=head_store,
            snapshot_reader=_RecordedVersions(),
            operation_store=LiveOperationStore(store=store, settings=settings),
            now=lambda: _NOW,
        ),
        deadline=None,
    )
    stored = _definition_under(spec_id)
    try:
        served: str | None = stored_head_version(stored, deps=deps).content_version
    except OperationSubmitted:
        served = None

    head = head_store.read_head(spec_id)
    checked = head_content_version(
        StoredHeadRead(head=head),
        now=_NOW,
        recheck_interval=deps.head_recheck_interval,
        serve_stale_max_age=timedelta(seconds=settings.head_serve_stale_max_age_seconds),
    )

    assert checked == served
    assert stored_head_servable(
        head,
        now=_NOW,
        recheck_interval=deps.head_recheck_interval,
        serve_stale_max_age=timedelta(seconds=settings.head_serve_stale_max_age_seconds),
    ) == (served is not None)


def _definition_under(spec_id: str) -> StoredDefinition:
    identity = minimal_identity()
    recomputed = recompute_identity(identity)
    return StoredDefinition(
        spec_id=spec_id,
        schema_version=1,
        identity=identity,
        identity_bytes=recomputed.identity_bytes,
        identity_digest=recomputed.identity_digest,
    )


def test_served_spec_id_is_the_canonical_id_a_stored_read_keys_its_head_on() -> None:
    posted = recompute_identity(_aid_identity(pins_bound=False))
    canonical = recompute_identity(_aid_identity(pins_bound=True))
    definition = StoredDefinition(
        spec_id=posted.spec_id,
        schema_version=1,
        identity=_aid_identity(pins_bound=False),
        identity_bytes=posted.identity_bytes,
        identity_digest=posted.identity_digest,
    )

    assert served_spec_id(definition, code_index=get_index()) == canonical.spec_id
