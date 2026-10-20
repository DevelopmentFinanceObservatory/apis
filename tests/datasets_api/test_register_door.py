"""Register-door tests cover the composed route, identity checks, registry writes, and
injected store failures.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from datasets_api import door_executor
from datasets_api.config import DatasetsBuildSettings
from datasets_api.doors import register as register_door
from datasets_api.doors.register import RegistryBackpressure, build_register_door
from datasets_api.models import RegisterRequest
from datasets_api.operation_store import NullOperationStore
from spec_registry import LiveSpecRegistry
from spec_registry.identity import _MAX_IDENTITY_DOCUMENT_BYTES, recompute_identity
from spec_registry.reader import LiveSpecReader
from spec_registry.registry import IDENTITY_COLLISION_QUARANTINED_MARKER

from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps_cell
from ._executor_doubles import BrokenSubmitExecutor, NeverStartingExecutor
from ._identity_builders import described_identity, pinned_identity, register_body
from ._snapshot_doubles import FakeSnapshotReader
from ._store_fault_doubles import (
    BlockingFakeDatasetStore,
    DriverExplodingFakeDatasetStore,
    PoolTimeoutFakeDatasetStore,
)
from .conftest import FakeDatasetStore, identity_of_encoded_size, wire_definition_for
from .test_pin_door import FakeArtifactReferenceWriter

# The composed lifespan reads these settings in the test that enters the client.
_SETTINGS_ENV = {
    "DATASETS_GRAPH_BASE_URL": "https://graph.example.org",
    "DATASETS_SDMX_BASE_URL": "https://sdmx.example.org",
    "DATASETS_ARTIFACT_BUCKET_WORKING": "working-bucket",
    "DATASETS_ARTIFACT_BUCKET_PINNED": "pinned-bucket",
}


def test_register_definition_is_contract_shaped() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    identity = pinned_identity(stat_var="s1")
    spec_id = recompute_identity(identity).spec_id

    resp = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
    }


def test_register_refuses_an_identity_that_carries_entity_groups() -> None:
    """A named group is request context beside an identity. The door refuses it inside the
    identity as an unknown member and stores nothing, and registers the same identity
    without it."""
    store = FakeDatasetStore()
    client = make_client(registry=LiveSpecRegistry(store=store))
    identity = pinned_identity(entities={"origin": ["country/EST", "country/LTU", "country/LVA"]})
    grouped = {**identity, "entity_groups": {"origin": {"label": "Baltic states"}}}
    spec_id = recompute_identity(identity).spec_id

    refused = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=grouped)),
    )
    assert refused.status_code == 400, refused.text
    assert refused.json()["code"] == "request_invalid"
    assert store.state.spec_registry == {}

    accepted = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )

    assert accepted.status_code == 200, accepted.text
    assert store.state.spec_definition[(spec_id, 1)]["definition"]["identity"] == identity


def test_register_definition_idempotent_repeat() -> None:
    store = FakeDatasetStore()
    client = make_client(registry=LiveSpecRegistry(store=store))
    identity = pinned_identity(stat_var="s2")
    spec_id = recompute_identity(identity).spec_id
    body = register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity))

    first = client.post("/api/datasets/register", json=body)
    second = client.post("/api/datasets/register", json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert len(store.state.spec_registry) == 1


def test_spec_id_mismatch_refuses_422() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    identity = pinned_identity(stat_var="mismatch")

    resp = client.post(
        "/api/datasets/register",
        json=register_body(
            definition=wire_definition_for(
                spec_id="spec_deliberately_wrong0000000000", identity=identity
            )
        ),
    )

    assert resp.status_code == 422
    assert resp.json()["code"] == "spec_id_mismatch"


def test_identity_collision_refuses_500_and_quarantines(caplog) -> None:
    store = FakeDatasetStore()
    identity = pinned_identity(stat_var="collide")
    spec_id = recompute_identity(identity).spec_id
    store.state.spec_registry[spec_id] = {
        "identity_version": "spec1",
        "identity_bytes": b"different bytes entirely",
        "identity_digest": "0" * 64,
        "trace_id": None,
    }
    client = make_client(registry=LiveSpecRegistry(store=store))

    with caplog.at_level(logging.ERROR):
        resp = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )

    assert resp.status_code == 500
    assert resp.json()["code"] == "identity_collision"
    assert IDENTITY_COLLISION_QUARANTINED_MARKER in caplog.text
    assert len(store.state.identity_quarantine) == 1


@pytest.mark.parametrize(
    "identity",
    [
        # unsupported_identity_version fails dfo_spec_kit's own
        # schema validation before any spec_id_mismatch check runs.
        {**pinned_identity(), "identity_version": "spec2"},
        described_identity(entities={"creditor": ["country/CHN"]}, open_roles=["origin"]),
    ],
    ids=["unsupported-version", "role-bound-and-open-on-one-axis"],
)
def test_malformed_identity_document_refuses_request_invalid(identity: dict) -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.post(
        "/api/datasets/register",
        json=register_body(
            definition={
                "schema_version": 1,
                "spec_id": "spec_whatever00000000000000000000",
                "identity": identity,
            }
        ),
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "request_invalid"


def test_driver_exception_never_leaks_through_the_composed_platform_app(
    caplog: pytest.LogCaptureFixture,
) -> None:
    distinctive_driver_text = "distinctive_driver_boom_f30c9b"
    store = DriverExplodingFakeDatasetStore(message=distinctive_driver_text)
    client = make_client(registry=LiveSpecRegistry(store=store))
    identity = pinned_identity(stat_var="explode")
    spec_id = recompute_identity(identity).spec_id

    with caplog.at_level(logging.ERROR):
        resp = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )

    assert resp.status_code == 503
    assert resp.json()["code"] == "service_unavailable"
    assert resp.json()["retryable"] is True
    assert distinctive_driver_text not in resp.text
    assert distinctive_driver_text not in caplog.text


def test_identity_document_over_the_byte_cap_refuses_request_invalid() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    over_cap_identity = identity_of_encoded_size(encoded_bytes=_MAX_IDENTITY_DOCUMENT_BYTES + 1)

    resp = client.post(
        "/api/datasets/register",
        json=register_body(
            definition=wire_definition_for(
                spec_id="spec_whatever00000000000000000000", identity=over_cap_identity
            )
        ),
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "request_invalid"


@pytest.mark.parametrize("with_definition", [True, False], ids=["beside-definition", "alone"])
def test_a_request_carrying_members_refuses_request_invalid(with_definition: bool) -> None:
    """The register door takes one `definition`; a `members` list refuses at body validation."""
    store = FakeDatasetStore()
    client = make_client(registry=LiveSpecRegistry(store=store))
    identity = pinned_identity(stat_var="both")
    spec_id = recompute_identity(identity).spec_id
    definition = wire_definition_for(spec_id=spec_id, identity=identity)
    body: dict[str, object] = {"schema_version": 1, "members": [definition]}
    if with_definition:
        body["definition"] = definition

    resp = client.post("/api/datasets/register", json=body)

    assert resp.status_code == 400
    assert resp.json()["code"] == "request_invalid"
    assert spec_id not in store.state.spec_registry


def test_request_invalid_when_no_definition_is_supplied() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))

    resp = client.post("/api/datasets/register", json={"schema_version": 1})

    assert resp.status_code == 400
    assert resp.json()["code"] == "request_invalid"


def test_unsupported_schema_version_wrong_value() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    identity = pinned_identity(stat_var="version")
    spec_id = recompute_identity(identity).spec_id

    resp = client.post(
        "/api/datasets/register",
        json={
            "schema_version": 2,
            "definition": wire_definition_for(spec_id=spec_id, identity=identity),
        },
    )

    assert resp.status_code == 422
    assert resp.json()["code"] == "unsupported_schema_version"


def test_unsupported_schema_version_missing_field() -> None:
    client = make_client(registry=LiveSpecRegistry(store=FakeDatasetStore()))
    identity = pinned_identity(stat_var="version_missing")
    spec_id = recompute_identity(identity).spec_id

    resp = client.post(
        "/api/datasets/register",
        json={"definition": wire_definition_for(spec_id=spec_id, identity=identity)},
    )

    assert resp.status_code == 422
    assert resp.json()["code"] == "unsupported_schema_version"


def test_identity_quarantined_refuses_500_not_retryable() -> None:
    store = FakeDatasetStore()
    identity = pinned_identity(stat_var="quarantined")
    spec_id = recompute_identity(identity).spec_id
    store.state.spec_registry[spec_id] = {
        "identity_version": "spec1",
        "identity_bytes": recompute_identity(identity).identity_bytes,
        "identity_digest": recompute_identity(identity).identity_digest,
        "trace_id": None,
    }
    store.state.identity_quarantine.append(
        {
            "id_kind": "spec",
            "subject_id": spec_id,
            "held_bytes": recompute_identity(identity).identity_bytes,
            "incoming_bytes": b"some other colliding bytes",
            "trace_id": "trace-quarantine",
        }
    )
    client = make_client(registry=LiveSpecRegistry(store=store))

    resp = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "identity_quarantined"
    assert body["retryable"] is False


def test_registry_pool_timeout_refuses_503_retryable() -> None:
    client = make_client(registry=LiveSpecRegistry(store=PoolTimeoutFakeDatasetStore()))
    identity = pinned_identity(stat_var="pool_timeout")
    spec_id = recompute_identity(identity).spec_id

    resp = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )

    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True


def test_backpressure_refuses_503_retryable_when_executor_is_already_at_capacity(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var, value in _SETTINGS_ENV.items():
        monkeypatch.setenv(var, value)
    store = BlockingFakeDatasetStore()
    # The concurrent requests need a shared client loop. Injected dependencies keep lifespan
    # startup offline; `_SETTINGS_ENV` supplies its required settings.
    client = make_client(
        registry=LiveSpecRegistry(store=store),
        read_max_concurrency=1,
        spec_reader=LiveSpecReader(store=store),
        artifact_store=FakeArtifactStore(),
        snapshot_reader=FakeSnapshotReader(),
        reference_writer=FakeArtifactReferenceWriter(),
        operation_store=NullOperationStore(),
    )
    identity_a = pinned_identity(stat_var="backpressure_a")
    spec_id_a = recompute_identity(identity_a).spec_id
    identity_b = pinned_identity(stat_var="backpressure_b")
    spec_id_b = recompute_identity(identity_b).spec_id

    first_response: list = []

    first_thread = threading.Thread(
        target=lambda: first_response.append(
            client.post(
                "/api/datasets/register",
                json=register_body(
                    definition=wire_definition_for(spec_id=spec_id_a, identity=identity_a)
                ),
            )
        )
    )
    first_thread.start()
    # The first request holds the sole executor slot and backpressure permit.
    assert store.entered.wait(timeout=5.0), "first request never reached the blocking store"

    with caplog.at_level(logging.WARNING):
        resp = client.post(
            "/api/datasets/register",
            json=register_body(
                definition=wire_definition_for(spec_id=spec_id_b, identity=identity_b)
            ),
        )

    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert body["retryable"] is True
    assert "write capacity gate refused admission" in caplog.text

    store.release.set()
    first_thread.join(timeout=5.0)
    assert first_response and first_response[0].status_code == 200


def test_registry_work_runs_on_the_surfaces_read_executor() -> None:
    recorded_thread_names: list[str] = []

    class _ThreadNameRecordingFakeDatasetStore(FakeDatasetStore):
        def transaction(self):
            recorded_thread_names.append(threading.current_thread().name)
            return super().transaction()

    client = make_client(registry=LiveSpecRegistry(store=_ThreadNameRecordingFakeDatasetStore()))
    identity = pinned_identity(stat_var="thread_name")
    spec_id = recompute_identity(identity).spec_id

    resp = client.post(
        "/api/datasets/register",
        json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
    )

    assert resp.status_code == 200
    assert recorded_thread_names
    assert all(name.startswith("datasets-read") for name in recorded_thread_names)
    assert not any(name.startswith("dfo-io") for name in recorded_thread_names)


def _request_for(*, spec_id: str, identity: dict) -> RegisterRequest:
    return RegisterRequest.model_validate(
        register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity))
    )


def _only_endpoint(door):
    """Returns the endpoint so tests can cancel tasks and exercise shutdown directly."""
    [route] = door.router.routes
    return route.endpoint


def test_cancellation_while_running_keeps_the_permit_held() -> None:
    """Cancelling the await keeps the permit held until the blocking store call finishes."""
    store = BlockingFakeDatasetStore()
    door = build_register_door(
        deps=fake_deps_cell(
            settings=DatasetsBuildSettings.from_env().model_copy(
                update={"read_max_concurrency": 1}
            ),
            registry=LiveSpecRegistry(store=store),
        )
    )
    endpoint = _only_endpoint(door)
    identity = pinned_identity(stat_var="cancel_running")
    body = _request_for(spec_id=recompute_identity(identity).spec_id, identity=identity)

    async def scenario() -> None:
        task = asyncio.ensure_future(endpoint(body))
        await asyncio.get_running_loop().run_in_executor(None, store.entered.wait, 5.0)
        assert store.entered.is_set(), "the register call never reached the blocking store"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The worker holds its permit while blocked, so a second call refuses admission.
        with pytest.raises(RegistryBackpressure):
            await endpoint(body)

        store.release.set()

        # Poll for release because the worker's wakeup time is uncontrolled.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while loop.time() < deadline:
            try:
                resp = await endpoint(body)
            except RegistryBackpressure:
                await asyncio.sleep(0.01)
                continue
            assert resp.status_code == 200
            return
        pytest.fail("permit was never released after the blocked call finished")

    asyncio.run(scenario())


def test_shutdown_cancels_a_queued_call_and_frees_its_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown cancels a queued task before execution and frees its permit immediately.

    `NeverStartingExecutor` reproduces this state deterministically.
    """
    monkeypatch.setattr(door_executor, "ContextCarryingExecutor", NeverStartingExecutor)
    store = FakeDatasetStore()
    cell = fake_deps_cell(
        settings=DatasetsBuildSettings.from_env().model_copy(update={"read_max_concurrency": 1}),
        registry=LiveSpecRegistry(store=store),
    )
    endpoint = _only_endpoint(build_register_door(deps=cell))
    identity = pinned_identity(stat_var="queued_cancel")
    body = _request_for(spec_id=recompute_identity(identity).spec_id, identity=identity)

    async def scenario() -> None:
        task = asyncio.ensure_future(endpoint(body))
        await asyncio.sleep(0)  # let the task run up to its own suspension point

        # The queued call holds the permit, so a second call refuses admission.
        with pytest.raises(RegistryBackpressure):
            await endpoint(body)

        cell.get().read_executor.executor.shutdown(wait=False, cancel_futures=True)

        with pytest.raises(asyncio.CancelledError):
            await task

        # Shutdown cancels the future and its callback releases the permit.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(endpoint(body), timeout=0.2)

    asyncio.run(scenario())


def test_broken_thread_pool_surfaces_as_internal_error_not_backpressure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken pool returns a non-retryable 500. `BrokenThreadPool` subclasses `RuntimeError`,
    so it needs separate classification from post-shutdown errors.
    """
    monkeypatch.setattr(door_executor, "ContextCarryingExecutor", BrokenSubmitExecutor)
    client = make_client(
        registry=LiveSpecRegistry(store=FakeDatasetStore()), read_max_concurrency=1
    )
    identity = pinned_identity(stat_var="broken_pool")
    spec_id = recompute_identity(identity).spec_id

    with caplog.at_level(logging.ERROR):
        resp = client.post(
            "/api/datasets/register",
            json=register_body(definition=wire_definition_for(spec_id=spec_id, identity=identity)),
        )

    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["code"] == "internal_error"
    assert body["retryable"] is False
    assert "permanently broken" in caplog.text


def test_post_shutdown_request_refuses_backpressure_not_500() -> None:
    """A request after the read executor shuts down maps its `RuntimeError` to retryable 503."""
    store = FakeDatasetStore()
    cell = fake_deps_cell(
        settings=DatasetsBuildSettings.from_env().model_copy(update={"read_max_concurrency": 1}),
        registry=LiveSpecRegistry(store=store),
    )
    endpoint = _only_endpoint(build_register_door(deps=cell))
    identity = pinned_identity(stat_var="post_shutdown")
    body = _request_for(spec_id=recompute_identity(identity).spec_id, identity=identity)

    async def scenario() -> None:
        cell.get().read_executor.executor.shutdown(wait=False, cancel_futures=True)

        with pytest.raises(RegistryBackpressure):
            await endpoint(body)

    asyncio.run(scenario())

    problem = register_door.classify_register_exception(RegistryBackpressure())
    assert problem is not None
    assert problem.status == 503
    assert problem.code == "service_unavailable"
    assert problem.retryable is True
