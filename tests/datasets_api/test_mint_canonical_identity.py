"""The spec door mints a definition in canonical form, whether the posted identity binds the
invariants of the dataset serving its family or leaves them unbound.

The imputed multilateral table declares `crsTypeOfFlow` and `dac2aAidType` as invariants.
A definition posted with both bound and one posted with both unbound mint and
report one `spec_id`, the canonical identity's (`code_index.members.canonical_identity`),
which binds both, when the definition names the table. One that names no table keeps the bindings it was
posted with.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from artifact_store import LiveArtifactInventory
from code_index.shared import get_index
from dataset_store.snapshot import VERSION_FACT_COLUMNS
from datasets_api.adapter import resolve_definition_members
from datasets_api.config import DatasetsBuildSettings
from datasets_api.version_store import LiveHeadStore
from spec_registry import LiveSpecRegistry, recompute_identity
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    AID_TYPE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_PROVENANCE,
    imputed_education_aid_identity,
)

from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door import _client, _coverage_service
from .test_snapshot_lifecycle import _InventoryRecordingArtifactStore

_PROVIDER = "country/FRA"
_RECIPIENT = "country/KEN"
_YEARS = (2020, 2021)
_LIVE = {"x-dfo-principal": "live-refresh-principal"}
_PINS = ("crsTypeOfFlow", AID_TYPE)

_CSV_HEADER = (
    "STRUCTURE,STRUCTURE_ID,ACTION,variableMeasured,financingProvider,financingRecipient,unit,"
    "measurementMethod,observationPeriod,provenance,TIME_PERIOD,OBS_VALUE,scalingFactor,"
    "facetId\r\n"
)


def _identity(*, pins_bound: bool, named: bool = True) -> dict:
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE if named else None)
    if not pins_bound:
        for slot in identity["slots"]:
            if slot["property"] in _PINS:
                slot["binding"] = {"state": "unbound"}
    return identity


def _populated() -> tuple[str, ...]:
    graph = FakeMintGraphReader()
    resolved = resolve_definition_members(
        _identity(pins_bound=True),
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=get_index(),
        bounds=mint_bounds(),
    )
    assert len(resolved.shapes) == IMPUTED_MEMBER_COUNT
    return tuple(shape.stat_var.removeprefix("dcid:") for shape in resolved.shapes[:2])


def _sdmx_transport(populated: tuple[str, ...]) -> httpx.MockTransport:
    def _respond(request: httpx.Request) -> httpx.Response:
        named = str(request.url.params.get("c[variableMeasured]", "")).split(",")
        body = _CSV_HEADER + "".join(
            f"dataflow,DC:DF_OBS(1.0.0),I,{dcid},{_PROVIDER},{_RECIPIENT},USD,GrantEquivalent,"
            f"NotApplicable,{IMPUTED_PROVENANCE},{year},{100 + position},,facet1\r\n"
            for position, dcid in enumerate(populated)
            if dcid in named
            for year in _YEARS
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "application/vnd.sdmx.data+csv;version=2.0.0"},
        )

    return httpx.MockTransport(_respond)


def _input(identity: dict) -> dict:
    spec_id = recompute_identity(identity).spec_id
    return {
        "kind": "spec_id",
        "spec_id": spec_id,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
    }


@pytest.mark.parametrize("named", [True, False], ids=["named", "unnamed"])
@pytest.mark.parametrize("pins_bound", [True, False], ids=["pins-bound", "pins-unbound"])
def test_a_definition_mints_and_reports_its_canonical_spec_id(pins_bound: bool, named: bool):
    canonical = _identity(pins_bound=pins_bound or named, named=named)
    canonical_spec_id = recompute_identity(canonical).spec_id
    posted = _input(_identity(pins_bound=pins_bound, named=named))
    assert (posted["spec_id"] == canonical_spec_id) is (pins_bound or not named)
    populated = _populated()
    store = FakeDatasetStore()
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        store=store,
        sdmx_transport=_sdmx_transport(populated),
        coverage=_coverage_service(
            stat_vars=populated,
            donor_role="financingProvider",
            donor=_PROVIDER,
            recipient=_RECIPIENT,
            recipient_role="financingRecipient",
            provenance=IMPUTED_PROVENANCE,
            years=_YEARS,
        ),
    )

    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": posted, "options": {"bundle": False}}
    )

    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    assert manifest["definition"]["spec_id"] == canonical_spec_id
    assert manifest["definition"]["identity"] == recompute_identity(canonical).identity
    assert manifest["coverage"]["rows"] == len(populated) * len(_YEARS)
    assert canonical_spec_id in store.state.spec_registry


class _CountingSdmxTransport(httpx.MockTransport):
    """`_sdmx_transport`'s responses, counting every request."""

    def __init__(self, populated: tuple[str, ...]) -> None:
        self.requests = 0
        respond = _sdmx_transport(populated).handler

        def _counted(request: httpx.Request) -> httpx.Response:
            self.requests += 1
            return respond(request)

        super().__init__(_counted)


def _head_client(
    populated: tuple[str, ...],
    transport: httpx.MockTransport,
    version_store: FakeVersionRecordDatasetStore | None = None,
    store: FakeDatasetStore | None = None,
):
    """`_client` with a head store and a version store, so a mint writes a head a later call
    can serve. The refresh route admits `_LIVE`."""
    version_store = version_store if version_store is not None else FakeVersionRecordDatasetStore()
    return _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        store=store if store is not None else FakeDatasetStore(),
        sdmx_transport=transport,
        version_store=version_store,
        head_writer=LiveHeadStore(store=version_store),
        coverage=_coverage_service(
            stat_vars=populated,
            donor_role="financingProvider",
            donor=_PROVIDER,
            recipient=_RECIPIENT,
            recipient_role="financingRecipient",
            provenance=IMPUTED_PROVENANCE,
            years=_YEARS,
        ),
        settings=DatasetsBuildSettings(live_refresh_principal=_LIVE["x-dfo-principal"]),
    )


def test_a_get_of_the_posted_id_serves_the_head_its_canonical_mint_wrote():
    """The posted id of a non-canonical definition resolves to the served id, so a GET under it
    serves the head the mint wrote, with no SDMX request and the mint's content version."""
    posted = _input(_identity(pins_bound=False))
    populated = _populated()
    transport = _CountingSdmxTransport(populated)
    client = _head_client(populated, transport)
    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": posted, "options": {"bundle": False}}
    )
    assert minted.status_code == 200, minted.text
    assert minted.json()["definition"]["spec_id"] != posted["spec_id"]
    transport.requests = 0

    got = client.get(f"/api/datasets/{posted['spec_id']}?bundle=false")

    assert got.status_code == 200, got.text
    assert got.json()["trace"]["served_from"] == "head"
    assert transport.requests == 0
    assert got.json()["content_version"] == minted.json()["content_version"]
    assert got.json()["definition"]["spec_id"] == minted.json()["definition"]["spec_id"]


def test_a_repeat_mint_of_a_current_head_serves_it_with_no_sdmx_request():
    posted = _input(_identity(pins_bound=False))
    populated = _populated()
    transport = _CountingSdmxTransport(populated)
    client = _head_client(populated, transport)
    body = {"schema_version": 1, "input": posted, "options": {"bundle": False}}
    minted = client.post("/api/datasets", json=body)
    assert minted.status_code == 200, minted.text
    transport.requests = 0

    reminted = client.post("/api/datasets", json=body)

    assert reminted.status_code == 200, reminted.text
    assert transport.requests == 0
    assert reminted.json()["trace"]["served_from"] == "head"
    assert reminted.json()["content_version"] == minted.json()["content_version"]
    assert [a["artifact_digest"] for a in reminted.json()["artifacts"]] == [
        a["artifact_digest"] for a in minted.json()["artifacts"]
    ]


def test_a_repeat_mint_over_the_sync_budget_serves_a_current_head():
    """A mint reads the head before admission, so a current head answers a repeat mint whose
    estimate is over the sync budget with the stored manifest, submitting no operation."""
    posted = _input(_identity(pins_bound=False))
    populated = _populated()
    transport = _CountingSdmxTransport(populated)
    store = FakeDatasetStore()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    body = {"schema_version": 1, "input": posted, "options": {"bundle": False}}

    def client(settings: DatasetsBuildSettings):
        return _client(
            graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
            store=store,
            artifact_store=artifact_store,
            sdmx_transport=transport,
            version_store=version_store,
            head_writer=LiveHeadStore(store=version_store),
            coverage=_coverage_service(
                stat_vars=populated,
                donor_role="financingProvider",
                donor=_PROVIDER,
                recipient=_RECIPIENT,
                recipient_role="financingRecipient",
                provenance=IMPUTED_PROVENANCE,
                years=_YEARS,
            ),
            settings=settings,
        )

    minted = client(DatasetsBuildSettings()).post("/api/datasets", json=body)
    assert minted.status_code == 200, minted.text
    transport.requests = 0

    reminted = client(DatasetsBuildSettings(sync_budget_rows=1, sync_budget_requests=1)).post(
        "/api/datasets", json=body
    )

    assert reminted.status_code == 200, reminted.text
    assert reminted.json()["trace"]["served_from"] == "head"
    assert reminted.json()["content_version"] == minted.json()["content_version"]
    assert transport.requests == 0


def test_a_frame_recorded_under_the_posted_id_keeps_resolving():
    """A version recorded under a non-canonical posted id, as retrieval keyed it before every
    door served the canonical id, stays readable by snapshot and refresh. Refresh reads it
    under the posted id and answers `moved` to the served dataset's current frame."""
    posted = _input(_identity(pins_bound=False))
    populated = _populated()
    version_store = FakeVersionRecordDatasetStore()
    client = _head_client(populated, _CountingSdmxTransport(populated), version_store)
    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": posted, "options": {"bundle": False}}
    )
    assert minted.status_code == 200, minted.text
    served_id = minted.json()["definition"]["spec_id"]
    content_version = minted.json()["content_version"]
    old_version = "cv_recorded_under_the_posted_id"
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_record (subject_id, subject_kind, content_version, "
            "schema_version, immutable_body, trace_id, recorded_at, base_content_version, "
            f"{', '.join(VERSION_FACT_COLUMNS)}) "
            "SELECT %s, subject_kind, %s, schema_version, immutable_body, trace_id, "
            f"recorded_at, base_content_version, {', '.join(VERSION_FACT_COLUMNS)} "
            "FROM datasets.version_record "
            "WHERE subject_id = %s AND content_version = %s",
            (posted["spec_id"], old_version, served_id, content_version),
        )
        tx.execute(
            "INSERT INTO datasets.version_artifact (subject_id, content_version, artifact_digest, "
            "rendering_key, rendering, media_type) SELECT %s, %s, artifact_digest, rendering_key, "
            "rendering, media_type FROM datasets.version_artifact "
            "WHERE subject_id = %s AND content_version = %s",
            (posted["spec_id"], old_version, served_id, content_version),
        )

    snapshot = client.get(f"/api/datasets/{posted['spec_id']}/{old_version}")
    refreshed = client.post(
        "/api/datasets/frames/refresh",
        json={"spec_id": posted["spec_id"], "content_version": old_version},
        headers=_LIVE,
    )

    assert snapshot.status_code == 200, snapshot.text
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json() == {
        "status": "moved",
        "frame": {"spec_id": served_id, "content_version": content_version},
    }


def test_a_revocation_under_the_posted_id_stops_its_head_hit_and_the_links_it_handed_out():
    """A request naming the posted id, a mint by `spec_id` among them, honours revocations under
    the posted and the served id, and its links name the posted id, so a takedown written
    against the posted id refuses the posted id's head hit, its snapshot and every link a
    posted-id response handed out, the mint's included. A request naming the served id honours
    revocations under the served id alone."""
    posted = _input(_identity(pins_bound=False))
    populated = _populated()
    version_store = FakeVersionRecordDatasetStore()
    artifact_store = _InventoryRecordingArtifactStore(
        inventory=LiveArtifactInventory(store=version_store)
    )
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        store=FakeDatasetStore(),
        artifact_store=artifact_store,
        sdmx_transport=_CountingSdmxTransport(populated),
        version_store=version_store,
        head_writer=LiveHeadStore(store=version_store),
        coverage=_coverage_service(
            stat_vars=populated,
            donor_role="financingProvider",
            donor=_PROVIDER,
            recipient=_RECIPIENT,
            recipient_role="financingRecipient",
            provenance=IMPUTED_PROVENANCE,
            years=_YEARS,
        ),
    )
    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": posted, "options": {"bundle": False}}
    )
    assert minted.status_code == 200, minted.text
    served_id, content_version = (
        minted.json()["definition"]["spec_id"],
        minted.json()["content_version"],
    )
    posted_id = posted["spec_id"]
    got = client.get(f"/api/datasets/{posted_id}?bundle=false")
    snapshot = client.get(f"/api/datasets/{posted_id}/{content_version}")
    assert (got.status_code, snapshot.status_code) == (200, 200), got.text
    links = [
        a["url"] for body in (minted.json(), got.json(), snapshot.json()) for a in body["artifacts"]
    ]
    assert links
    assert all(link.startswith(f"/api/datasets/{posted_id}/") for link in links)
    with version_store.transaction() as tx:
        tx.execute(
            "INSERT INTO datasets.version_revocation "
            "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, %s)",
            (posted_id, content_version, "takedown", datetime.now(UTC)),
        )
    signings = len(artifact_store.signings)

    head_hit = client.get(f"/api/datasets/{posted_id}?bundle=false")
    remint = client.post(
        "/api/datasets", json={"schema_version": 1, "input": posted, "options": {"bundle": False}}
    )
    refused = [client.get(link, follow_redirects=False) for link in links]
    refused.append(client.get(f"/api/datasets/{posted_id}/{content_version}"))

    assert [(r.status_code, r.json()["code"]) for r in (head_hit, remint)] == [
        (410, "snapshot_revoked")
    ] * 2
    assert [(r.status_code, r.json()["code"]) for r in refused] == [
        (410, "snapshot_revoked")
    ] * len(refused)
    assert len(artifact_store.signings) == signings
    assert client.get(f"/api/datasets/{served_id}/{content_version}").status_code == 200


@pytest.mark.parametrize("pins_bound", [False, True], ids=["non-canonical", "own-canonical-form"])
def test_a_posted_id_with_an_earlier_served_id_quarantined_refuses_on_every_door(pins_bound: bool):
    """A posted id is bound to a served id an earlier code index resolved, and that served id
    is quarantined. Every door that reads the definition under the posted id refuses, as the
    links a mint would hand out do, whether the current index resolves the posted id to
    another served id or to itself. A current served id other than the posted one still
    answers."""
    posted = _input(_identity(pins_bound=pins_bound))
    populated = _populated()
    store = FakeDatasetStore()
    client = _head_client(populated, _CountingSdmxTransport(populated), store=store)
    mint = {"schema_version": 1, "input": posted, "options": {"bundle": False}}
    minted = client.post("/api/datasets", json=mint)
    assert minted.status_code == 200, minted.text
    served_id, content_version = (
        minted.json()["definition"]["spec_id"],
        minted.json()["content_version"],
    )
    assert (served_id == posted["spec_id"]) is pins_bound
    link = minted.json()["artifacts"][0]["url"]
    earlier = recompute_identity(_identity(pins_bound=False, named=False))
    LiveSpecRegistry(store=store).register_served_definition(
        served=earlier, schema_version=1, posted_spec_id=posted["spec_id"]
    )
    store.state.identity_quarantine.append(
        {
            "id_kind": "spec",
            "subject_id": earlier.spec_id,
            "held_bytes": b"a",
            "incoming_bytes": b"b",
        }
    )

    refused = [
        client.get(f"/api/datasets/{posted['spec_id']}?bundle=false"),
        client.get(f"/api/datasets/{posted['spec_id']}/options"),
        client.post(f"/api/datasets/{posted['spec_id']}/members", json={}),
        client.post(
            "/api/datasets/frames/refresh",
            json={"spec_id": posted["spec_id"], "content_version": content_version},
            headers=_LIVE,
        ),
        client.post("/api/datasets", json=mint),
        client.get(f"/api/datasets/{posted['spec_id']}/{content_version}"),
        client.get(link, follow_redirects=False),
    ]

    assert [(r.status_code, r.json()["code"]) for r in refused] == [
        (500, "identity_quarantined")
    ] * len(refused)
    assert all(earlier.spec_id in r.text for r in refused)
    if not pins_bound:
        assert client.get(f"/api/datasets/{served_id}/{content_version}").status_code == 200


def test_a_scoped_mint_by_spec_id_names_its_links_under_the_served_id():
    """Scope surgery gives a served id no click on the posted id resolves to, so a scoped mint
    names its links under the served id, which resolve."""
    posted = {
        **_input(_identity(pins_bound=False)),
        "scope": {"entities": {"financingRecipient": [_RECIPIENT]}},
    }
    populated = _populated()
    client = _head_client(populated, _CountingSdmxTransport(populated))

    minted = client.post(
        "/api/datasets", json={"schema_version": 1, "input": posted, "options": {"bundle": False}}
    )

    assert minted.status_code == 200, minted.text
    served_id = minted.json()["definition"]["spec_id"]
    assert served_id != posted["spec_id"]
    links = [a["url"] for a in minted.json()["artifacts"]]
    assert links
    assert all(link.startswith(f"/api/datasets/{served_id}/") for link in links)
    assert client.get(links[0], follow_redirects=False).status_code == 302
