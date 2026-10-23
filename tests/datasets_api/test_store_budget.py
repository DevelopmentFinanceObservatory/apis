"""I14: the store budget of a request, counted on fake stores.

A head-hit GET under a canonical id makes at most four transactions: the definition read, the
head read, the version read and the retention read (`datasets_api.retention`), which every
Manifest response makes once. A GET under a non-canonical posted id, which resolves to its
canonical served id, makes one more, the transaction that reads the served definition and
binds it to the posted id (`kinds.spec_id.resolve_served_definition`).

Every store the surface reads and writes sits behind one `_CountingStore` view sharing a
`_Budget`, as production shares one pool. The artifact store is the real `LiveArtifactStore`
over a fake GCS client and the real `LiveArtifactInventory`, so its own inventory reads count.
A transaction counts as a borrow from the moment it is asked for, before the fake's lock, so
two transactions one request wants at once count as two borrows even though the fakes
serialise them.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import LiveSnapshotReader
from dataset_store.store import DatasetStore, Transaction
from datasets_api.version_store import LiveHeadStore, LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from dfo_platform.timings import count
from spec_registry import LiveSpecRegistry, recompute_identity
from spec_registry.reader import LiveSpecReader
from tests.artifact_store.conftest import FakeGCSClient
from tests.code_index.aid_identity import IMPUTED_PROVENANCE

from . import test_mint_canonical_identity as canonical_fixture
from ._app_factories import make_client
from ._frame_doubles import _BASE_URL, _body, _coverage_service, _graph, _sdmx_transport
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door import _coverage_service as _definition_coverage_service
from .test_pin_door import FakeArtifactReferenceWriter

_MINT_TRANSACTIONS = 7
_HEAD_HIT_TRANSACTIONS = 4
# A first mint under a non-canonical posted id adds the served definition's registration and
# binding.
_ALIAS_MINT_TRANSACTIONS = 8
# A head hit under a non-canonical posted id adds the served definition's read and binding.
_ALIAS_HEAD_HIT_TRANSACTIONS = 5
# Past the recheck window a head hit also confirms the head: the definition read, the head read,
# the confirmation, the version read and the retention read.
_CONFIRMED_HEAD_HIT_TRANSACTIONS = 5
# The served ids the posted id of the version-addressed read is bound to.
_BOUND_SERVED_IDS = 2
# Past the 15-minute recheck window and inside the 24-hour maximum age.
_PAST_RECHECK = timedelta(minutes=20)


@dataclass
class _Budget:
    """Transactions, concurrent borrows and signing calls since the last `reset`."""

    transactions: list[str] = field(default_factory=list)
    in_flight: int = 0
    max_in_flight: int = 0
    signings: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        with self._lock:
            self.transactions.clear()
            self.max_in_flight = 0
            self.signings = 0

    @contextmanager
    def borrow(self) -> Iterator[None]:
        caller = _caller()
        with self._lock:
            self.transactions.append(caller)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            yield
        finally:
            with self._lock:
                self.in_flight -= 1


def _caller() -> str:
    """The innermost `src` frame outside the store modules, naming which read or write a
    transaction served when an assertion fails."""
    frames = [
        frame
        for frame in traceback.extract_stack()
        if "/src/" in frame.filename and "/dataset_store/store.py" not in frame.filename
    ]
    return f"{frames[-1].name} ({frames[-2].name})" if len(frames) > 1 else "?"


class _CountingStore:
    """`DatasetStore` view of `inner` that counts each transaction against `budget`, and on the
    call's `db_transactions` count as `LiveDatasetStore` does."""

    def __init__(self, inner: DatasetStore, *, budget: _Budget) -> None:
        self._inner = inner
        self._budget = budget

    @contextmanager
    def transaction(self) -> Generator[Transaction, None, None]:
        count("db_transactions")
        with self._budget.borrow(), self._inner.transaction() as tx:
            yield tx


@dataclass
class _Surface:
    client: object
    budget: _Budget
    registry: LiveSpecRegistry
    # The surface's clock reads `start + offset[0]`.
    offset: list[timedelta]


@contextmanager
def _built_surface(
    monkeypatch: pytest.MonkeyPatch, *, graph, sdmx_transport, coverage
) -> Iterator[_Surface]:
    budget = _Budget()

    def counted_sign(**kwargs: object) -> str:
        budget.signings += 1
        return f"https://storage.googleapis.com/signed/{kwargs['object_name']}"

    monkeypatch.setattr("artifact_store.store.sign_get_url", counted_sign)
    start = datetime.now(UTC)
    offset = [timedelta(0)]
    version_store = FakeVersionRecordDatasetStore()
    store = _CountingStore(version_store, budget=budget)
    spec_store = _CountingStore(FakeDatasetStore(), budget=budget)
    artifact_store = LiveArtifactStore(
        buckets={Tier.working: "budget-working", Tier.pinned: "budget-pinned"},
        inventory=LiveArtifactInventory(store=store),
        client=FakeGCSClient(),
    )
    registry = LiveSpecRegistry(store=spec_store)
    client = make_client(
        registry=registry,
        spec_reader=LiveSpecReader(store=spec_store),
        graph=graph,
        sdmx=SdmxClient(base_url=_BASE_URL, transport=sdmx_transport),
        artifact_store=artifact_store,
        version_writer=LiveVersionRecordWriter(store=store),
        head_writer=LiveHeadStore(store=store),
        snapshot_reader=LiveSnapshotReader(store=store),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage,
        now=lambda: start + offset[0],
    )
    try:
        yield _Surface(client=client, budget=budget, registry=registry, offset=offset)
    finally:
        client.__exit__(None, None, None)
        version_store.close()


@pytest.fixture
def surface(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Surface]:
    with _built_surface(
        monkeypatch, graph=_graph(), sdmx_transport=_sdmx_transport(), coverage=_coverage_service()
    ) as built:
        yield built


def _counted(budget: _Budget) -> str:
    return "\n".join(budget.transactions)


def test_a_first_mint_makes_at_most_seven_transactions_one_borrow_at_a_time(surface):
    budget = surface.budget
    budget.reset()

    resp = surface.client.post("/api/datasets", json=_body(format="csv", bundle=False))

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"
    assert len(budget.transactions) <= _MINT_TRANSACTIONS, _counted(budget)
    assert f"db_transactions;desc={len(budget.transactions)}" in resp.headers["server-timing"]
    assert budget.max_in_flight == 1, _counted(budget)
    assert budget.signings == 0


def test_a_first_mint_of_a_spec_id_with_its_definition_makes_at_most_seven_transactions(surface):
    """The mint a Search answer hands over: an unregistered `spec_id` with its definition
    sidecar, whose read, registration and read-back precede the serve."""
    budget = surface.budget
    registered = surface.client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert registered.status_code == 200, registered.text
    definition = dict(registered.json()["definition"])
    identity = {**definition["identity"], "requested_window": {"kind": "range", "start_year": 2019}}
    recomputed = recompute_identity(identity)
    sidecar = {"schema_version": 1, "spec_id": recomputed.spec_id, "identity": recomputed.identity}
    budget.reset()

    resp = surface.client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {"kind": "spec_id", "spec_id": recomputed.spec_id, "definition": sidecar},
            "options": {"format": "csv", "bundle": False},
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"
    assert len(budget.transactions) <= _MINT_TRANSACTIONS, _counted(budget)
    assert budget.max_in_flight == 1, _counted(budget)
    assert budget.signings == 0


def test_a_head_hit_get_makes_at_most_four_transactions_and_signs_nothing(surface):
    budget = surface.budget
    minted = surface.client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    budget.reset()

    resp = surface.client.get(f"/api/datasets/{spec_id}?format=csv&bundle=false")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert resp.json()["artifacts"]
    assert len(budget.transactions) <= _HEAD_HIT_TRANSACTIONS, _counted(budget)
    assert budget.max_in_flight == 1, _counted(budget)
    assert budget.signings == 0


def test_a_head_hit_past_the_recheck_window_adds_only_its_confirmation(surface):
    budget = surface.budget
    minted = surface.client.post("/api/datasets", json=_body(format="csv", bundle=False))
    assert minted.status_code == 200, minted.text
    spec_id = minted.json()["definition"]["spec_id"]
    surface.offset[0] = _PAST_RECHECK
    budget.reset()

    resp = surface.client.get(f"/api/datasets/{spec_id}?format=csv&bundle=false")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert len(budget.transactions) <= _CONFIRMED_HEAD_HIT_TRANSACTIONS, _counted(budget)
    assert any("confirm_head" in caller for caller in budget.transactions), _counted(budget)
    assert budget.max_in_flight == 1, _counted(budget)
    assert budget.signings == 0


@contextmanager
def _alias_surface(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_Surface, dict]]:
    """A surface serving the imputed aid table, and a mint body whose posted identity leaves
    the table's invariants unbound, so its canonical served id differs from the posted id."""
    populated = canonical_fixture._populated()
    with _built_surface(
        monkeypatch,
        graph=FakeMintGraphReader(
            labels={canonical_fixture._PROVIDER: "France", canonical_fixture._RECIPIENT: "Kenya"}
        ),
        sdmx_transport=canonical_fixture._sdmx_transport(populated),
        coverage=_definition_coverage_service(
            stat_vars=populated,
            donor_role="financingProvider",
            donor=canonical_fixture._PROVIDER,
            recipient=canonical_fixture._RECIPIENT,
            recipient_role="financingRecipient",
            provenance=IMPUTED_PROVENANCE,
            years=canonical_fixture._YEARS,
        ),
    ) as built:
        posted = canonical_fixture._input(canonical_fixture._identity(pins_bound=False))
        yield built, {"schema_version": 1, "input": posted, "options": {"bundle": False}}


def test_a_first_mint_under_a_non_canonical_posted_id_makes_at_most_eight_transactions(
    monkeypatch,
):
    """The seven of a first mint, and the transaction that registers the served definition and
    binds it to the posted id."""
    with _alias_surface(monkeypatch) as (built, mint):
        built.budget.reset()

        resp = built.client.post("/api/datasets", json=mint)

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "materialise"
    assert resp.json()["definition"]["spec_id"] != mint["input"]["spec_id"]
    assert len(built.budget.transactions) <= _ALIAS_MINT_TRANSACTIONS, _counted(built.budget)
    assert built.budget.max_in_flight == 1, _counted(built.budget)
    assert built.budget.signings == 0


def test_a_head_hit_get_under_a_non_canonical_posted_id_makes_at_most_five_transactions(
    monkeypatch,
):
    """The GET a client makes with the id it posted after a mint served the canonical id: the
    posted definition's read, the served definition's read and binding, the head read, the
    version read and the retention read."""
    with _alias_surface(monkeypatch) as (built, mint):
        minted = built.client.post("/api/datasets", json=mint)
        assert minted.status_code == 200, minted.text
        assert minted.json()["definition"]["spec_id"] != mint["input"]["spec_id"]
        built.budget.reset()

        resp = built.client.get(f"/api/datasets/{mint['input']['spec_id']}?bundle=false")

    assert resp.status_code == 200, resp.text
    assert resp.json()["trace"]["served_from"] == "head"
    assert len(built.budget.transactions) <= _ALIAS_HEAD_HIT_TRANSACTIONS, _counted(built.budget)
    assert built.budget.max_in_flight == 1, _counted(built.budget)
    assert built.budget.signings == 0


def test_a_snapshot_under_a_posted_id_makes_one_version_read_per_bound_served_id(monkeypatch):
    """The posted id is bound to two served ids and the version is recorded under the older
    binding, so the read makes the bindings' transaction, a version read under the newer
    served id that finds nothing, the one that finds the version, and the retention read."""
    with _alias_surface(monkeypatch) as (built, mint):
        minted = built.client.post("/api/datasets", json=mint)
        assert minted.status_code == 200, minted.text
        posted_id, content_version = mint["input"]["spec_id"], minted.json()["content_version"]
        newer = recompute_identity(canonical_fixture._identity(pins_bound=False, named=False))
        built.registry.register_served_definition(
            served=newer, schema_version=1, posted_spec_id=posted_id
        )
        built.budget.reset()

        resp = built.client.get(f"/api/datasets/{posted_id}/{content_version}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["content_version"] == content_version
    version_reads = [c for c in built.budget.transactions if "read_servable_version" in c]
    assert len(version_reads) == _BOUND_SERVED_IDS, _counted(built.budget)
    assert len(built.budget.transactions) == 1 + _BOUND_SERVED_IDS + 1, _counted(built.budget)
    assert built.budget.max_in_flight == 1, _counted(built.budget)
