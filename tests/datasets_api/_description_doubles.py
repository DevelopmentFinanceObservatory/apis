"""Shared harness for the description record, publish steps, description route and backfill tests.

One `FakeVersionRecordDatasetStore` backs the version writer, the snapshot reader, the artifact
inventory and the description store, so a record written in the version-row transaction is the
record the route and the backfill read. `datasets_client` composes the whole Datasets surface
over that store for one of two mint fixtures: `definition`, one variable at one recipient
(`test_definition_door`), or `family`, two members on the summable `testAxis` that a groupby
collapses (`test_serve_groupby_collapse`). Either client also publishes `ops_` frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from artifact_store import ArtifactStore, LiveArtifactInventory
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from origin_catalogue.service import CatalogueService
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader

from . import test_definition_door as definition_fixture
from . import test_serve_groupby_collapse as family_fixture
from ._app_factories import RunningClient, make_client
from ._frame_doubles import link_frame1_only
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door_currency import _FakeHeadStore, _RenderableArtifactStore
from .test_pin_door import FakeArtifactReferenceWriter

if TYPE_CHECKING:
    from description.facts import DescriptionRecord

PRINCIPAL = "publisher-principal"
AUTH_HEADERS = {"x-dfo-principal": PRINCIPAL}

FAMILY_MEMBERS = (family_fixture._MEMBER_A, family_fixture._MEMBER_B)
FAMILY_AXIS = "testAxis"
GROUPBY_OPTIONS = {"bundle": False, "groupby": {"by": ["recipient"], "op": "sum"}}
GROUPBY_QUERY = "bundle=false&groupby.by=recipient&groupby.op=sum"

# Two retrieved versions and the frame computed from them, in the forms the step schema accepts.
NUMERATOR = ("spec_" + "a1" * 16, "cv_" + "b2" * 16)
DENOMINATOR = ("spec_" + "c3" * 16, "cv_" + "d4" * 16)
OPS_ID = "ops_" + "e5" * 16
OCV = "ocv_" + "f6" * 16


def read_record(
    store: FakeVersionRecordDatasetStore, *, subject_id: str, content_version: str
) -> DescriptionRecord | None:
    return LiveDescriptionStore(store=store).read_description(
        id=subject_id, content_version=content_version
    )


def forget_record(
    store: FakeVersionRecordDatasetStore, *, subject_id: str, content_version: str
) -> None:
    """Delete a version's record, leaving the version as one made before records existed."""
    with store.transaction() as tx:
        tx.execute(
            "DELETE FROM datasets.description_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )


def store_as_frame1_era(
    store: FakeVersionRecordDatasetStore,
    artifact_store: ArtifactStore,
    *,
    subject_id: str,
    content_version: str,
) -> None:
    """Leave the version as one recorded before frame hashes and canonical frames: no
    `frame_sha256` on its row, and its frame stored as a partial frame1 document
    (`link_frame1_only`), whose digest is not the frame's hash."""
    link_frame1_only(
        store, artifact_store, subject_id=subject_id, content_version=content_version, partial=True
    )
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET frame_sha256 = NULL "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )


def revoke(store: FakeVersionRecordDatasetStore, *, subject_id: str, content_version: str) -> None:
    with store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, trace_id, revoked_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (subject_id, content_version, "test", None, datetime.now(UTC)),
        )


def version_rows(store: FakeVersionRecordDatasetStore, subject_id: str) -> set[str]:
    with store.transaction() as tx:
        rows = tx.fetch_all(
            "SELECT content_version FROM datasets.version_record WHERE subject_id = %s",
            (subject_id,),
        )
    return {str(row["content_version"]) for row in rows}


def datasets_client(
    store: FakeVersionRecordDatasetStore,
    *,
    fixture: str = "definition",
    with_description_store: bool = False,
    artifact_store: _RenderableArtifactStore | None = None,
    head_store: _FakeHeadStore | None = None,
    graph_reader: FakeMintGraphReader | None = None,
    catalogue: CatalogueService | None = None,
) -> RunningClient:
    """The Datasets surface over `store`, minting `fixture`'s variables and publishing frames.

    `with_description_store` passes a description store over `store` to `make_client`, which
    the description route reads through. `head_store` lets a test age the subject head the
    client advances. `graph_reader` replaces the fixture's graph, and `catalogue` the
    process-wide catalogue service each request binds."""
    graph, sdmx_transport, coverage, code_index = {
        "definition": (
            definition_fixture._graph,
            definition_fixture._sdmx_transport,
            definition_fixture._coverage_service,
            lambda: None,
        ),
        "family": (
            family_fixture._graph,
            family_fixture._sdmx_transport,
            family_fixture._coverage_service,
            family_fixture._code_index,
        ),
    }[fixture]
    registry_store = FakeDatasetStore()
    kwargs: dict[str, Any] = {}
    if with_description_store:
        kwargs["description_store"] = LiveDescriptionStore(store=store)
    index = code_index()
    if index is not None:
        kwargs["code_index"] = index
    return make_client(
        registry=LiveSpecRegistry(store=registry_store),
        spec_reader=LiveSpecReader(store=registry_store),
        snapshot_reader=LiveSnapshotReader(store=store),
        graph=graph_reader if graph_reader is not None else graph(),
        sdmx=SdmxClient(base_url=definition_fixture._BASE_URL, transport=sdmx_transport()),
        artifact_store=artifact_store
        if artifact_store is not None
        else _RenderableArtifactStore(inventory=LiveArtifactInventory(store=store)),
        version_writer=LiveVersionRecordWriter(store=store),
        head_writer=head_store if head_store is not None else _FakeHeadStore(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=coverage(),
        publish_allowed_principals=frozenset({PRINCIPAL}),
        catalogue=catalogue,
        **kwargs,
    )


def definition_mint(client: httpx.Client | RunningClient) -> httpx.Response:
    """Mint the `definition` fixture's one variable at its one recipient."""
    return client.post(
        "/api/datasets",
        json={
            "schema_version": 1,
            "input": {
                "kind": "statvar_set",
                "statvars": [definition_fixture._STAT_VAR_REF],
                "scope": definition_fixture._SCOPE,
            },
            "options": {"bundle": False},
        },
    )


def mint_definition(client: httpx.Client | RunningClient) -> dict[str, Any]:
    """`definition_mint`'s manifest, after checking it succeeded."""
    response = definition_mint(client)
    assert response.status_code == 200, response.text
    return response.json()


def mint_family(
    client: httpx.Client | RunningClient, *, options: Mapping[str, object]
) -> dict[str, Any]:
    """Mint the `family` fixture's two members under `options`."""
    spec_id, sidecar = family_fixture._spec_id_and_sidecar()
    return family_fixture._mint(client, spec_id=spec_id, sidecar=sidecar, options=dict(options))


def quotient_steps(*, kept_entities: int = 38, numerator_label: str = "Gross ODA disbursements"):
    """The design's quotient chain over `NUMERATOR` and `DENOMINATOR`, in canonical form."""
    return [
        {
            "id": "s1",
            "op": "retrieve",
            "input": {
                "id": NUMERATOR[0],
                "content_version": NUMERATOR[1],
                "label": numerator_label,
            },
            "options": {},
        },
        {
            "id": "s2",
            "op": "retrieve",
            "input": {
                "id": DENOMINATOR[0],
                "content_version": DENOMINATOR[1],
                "label": "Gross national income",
            },
            "options": {},
        },
        {
            "id": "s3",
            "op": "quotient",
            "inputs": {"denominator": "s2", "numerator": "s1"},
            "policy": "quotient_of_intersect_periods",
            "params": {"numerator_by": ["recipient"]},
            "kept": {"entities": kept_entities, "entities_with_fewer_periods": 6, "periods": 15},
            "left_out": [
                {"reason": "unmatched_key", "count": 9, "of": "entities"},
                {"reason": "period_unmatched", "count": 5, "of": "entities"},
                {"reason": "zero_denominator", "count": 2, "of": "entities"},
            ],
            "unit": {"numerator": {"ref": "USDollar"}, "denominator": {"ref": "USDollar"}},
        },
    ]


def publish_body(*, steps: list[dict[str, object]] | None = None, value: float = 1.0) -> dict:
    """A publish of the frame computed from `NUMERATOR` and `DENOMINATOR`; `steps` is left out
    when `None`."""
    body: dict[str, object] = {
        "schema_version": 1,
        "spec_id": OPS_ID,
        "content_version": OCV,
        "role_keys": ["recipient"],
        "rows": [
            {
                "stat_var": "dcid:Exemplar",
                "time_period": "2020",
                "value": value,
                "recipient": "country/KEN",
            }
        ],
        "input_content_versions": [NUMERATOR[1], DENOMINATOR[1]],
    }
    if steps is not None:
        body["steps"] = steps
    return body


def publish(client: httpx.Client | RunningClient, body: Mapping[str, object]) -> httpx.Response:
    return client.post("/api/datasets/frames/publish", json=dict(body), headers=AUTH_HEADERS)


def served_steps(record: DescriptionRecord) -> list[dict[str, object]] | None:
    """A record's step records as JSON, or `None` while it holds none."""
    if record.steps is None:
        return None
    return [step.model_dump(mode="json") for step in record.steps]
