"""`GET /api/datasets/{id}/versions` is open to every caller and lists the versions the snapshot
door serves under the path id: under every served id bound to it and the posted id itself,
each once, newest first, with revoked versions left out, one page at a time.

One `FakeVersionRecordDatasetStore` backs the version writer and the version lister, so the
listing SQL runs against the rows the writer stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from dataset_store.snapshot import FrameLineage, LiveSnapshotReader
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.doors.versions import VERSIONS_PAGE_MAX
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister, LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from spec_registry import IdentityQuarantined, LiveSpecRegistry
from tests.artifact_store.conftest import FakeGCSClient

from ._app_factories import make_client
from ._snapshot_doubles import FakeSpecReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore

_FRAME1_RENDERING = {"format": "frame1"}
_STATISTICS_RENDERING = {"statistics": "series"}
_PLAIN_RENDERING = {"format": "parquet", "bundle": False, "layout": "frame"}
_GROUPBY = {"by": ["recipient"], "op": "sum"}
_LEVELLED_GROUPBY = {"by": ["recipient", "sector"], "op": "sum", "levels": {"sector": 0}}

_VERSIONS_PATH = "/api/datasets/spec_a/versions"


def _cv(n: int) -> str:
    return f"cv_{n:032d}"


class _ServingSpecReader(FakeSpecReader):
    """Binds each id in `served` to its served ids, and quarantines the ids in
    `quarantined`, as the registry's `read_served_spec_ids` does."""

    def __init__(
        self,
        *,
        spec_ids: Sequence[str],
        served: Mapping[str, tuple[str, ...]] | None = None,
        quarantined: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(spec_ids=spec_ids)
        self._served = dict(served or {})
        self._quarantined = quarantined

    def read_served_spec_ids(self, spec_id: str) -> tuple[str, ...] | None:
        if spec_id in self._quarantined:
            raise IdentityQuarantined(subject_id=spec_id, id_kind="spec")
        known = super().read_served_spec_ids(spec_id)
        return None if known is None else self._served.get(spec_id, known)


class _Rig:
    def __init__(self) -> None:
        self.store = FakeVersionRecordDatasetStore()
        self.artifact_store = LiveArtifactStore(
            buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
            inventory=LiveArtifactInventory(store=self.store),
            client=FakeGCSClient(),
        )
        self.now = datetime(2026, 6, 1, tzinfo=UTC)
        self.version_writer = LiveVersionRecordWriter(store=self.store, now=lambda: self.now)

    def client(self, spec_reader: FakeSpecReader | None = None) -> TestClient:
        return make_client(
            registry=LiveSpecRegistry(store=FakeDatasetStore()),
            spec_reader=spec_reader or FakeSpecReader(spec_ids=["spec_a", "spec_b"]),
            snapshot_reader=LiveSnapshotReader(store=self.store),
            artifact_store=self.artifact_store,
            version_writer=self.version_writer,
            version_lister=LiveVersionLister(store=self.store),
            recipe_store=LiveRecipeStore(store=self.store),
            description_store=LiveDescriptionStore(store=self.store),
        )

    def record_version(
        self,
        *,
        content_version: str,
        subject_id: str = "spec_a",
        base_content_version: str | None = None,
        renderings: tuple[dict[str, object], ...] = (),
    ) -> None:
        """Record a version with one stored file linked under it per entry of `renderings`,
        an hour after the version recorded before it."""
        self.now += timedelta(hours=1)
        artifacts = tuple(
            ArtifactRecord(
                artifact_digest=self.artifact_store.put(
                    f"{subject_id}:{content_version}:{index}".encode(), tier=Tier.working
                ).digest,
                rendering=rendering,
                media_type="application/vnd.apache.parquet",
            )
            for index, rendering in enumerate(renderings)
        )
        self.version_writer.record_version(
            VersionRecord(
                subject_id=subject_id,
                subject_kind="spec",
                content_version=content_version,
                schema_version=1,
                facts=FrameLineage(input_content_versions=[]),
                artifacts=artifacts,
                base_content_version=base_content_version,
            )
        )

    def revoke(self, *, content_version: str, subject_id: str = "spec_a") -> None:
        with self.store.transaction() as tx:
            tx.execute(
                "INSERT INTO datasets.version_revocation "
                "(subject_id, content_version, reason, revoked_at) VALUES (%s, %s, %s, %s)",
                (subject_id, content_version, "takedown", self.now),
            )


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


def _listed(response) -> list[str]:
    assert response.status_code == 200, response.text
    return [version["content_version"] for version in response.json()["versions"]]


def test_versions_list_is_newest_first_with_each_version_s_rendering_for_any_caller(
    rig: _Rig,
) -> None:
    rig.record_version(
        content_version=_cv(1),
        base_content_version=_cv(0),
        renderings=(_FRAME1_RENDERING, {**_PLAIN_RENDERING, "groupby": _LEVELLED_GROUPBY}),
    )
    # A raw version lists as plain whatever is linked under it.
    rig.record_version(
        content_version=_cv(3),
        renderings=(_FRAME1_RENDERING, _PLAIN_RENDERING, {**_PLAIN_RENDERING, "groupby": _GROUPBY}),
    )
    rig.record_version(
        content_version=_cv(2),
        base_content_version=_cv(3),
        renderings=(
            _STATISTICS_RENDERING,
            _FRAME1_RENDERING,
            {**_PLAIN_RENDERING, "groupby": _GROUPBY},
        ),
    )
    # A collapse holding only its frame1 and statistics files records no groupby.
    rig.record_version(
        content_version=_cv(4),
        base_content_version=_cv(3),
        renderings=(_FRAME1_RENDERING, _STATISTICS_RENDERING),
    )
    # A party-totals cut records the option on its file, alone or beside the groupby.
    rig.record_version(
        content_version=_cv(5),
        base_content_version=_cv(3),
        renderings=(_FRAME1_RENDERING, {**_PLAIN_RENDERING, "without_party_totals": True}),
    )
    rig.record_version(
        content_version=_cv(6),
        base_content_version=_cv(3),
        renderings=(
            _FRAME1_RENDERING,
            {**_PLAIN_RENDERING, "groupby": _GROUPBY, "without_party_totals": True},
        ),
    )
    rig.record_version(
        subject_id="spec_b",
        content_version=_cv(9),
        base_content_version=_cv(8),
        renderings=({**_PLAIN_RENDERING, "groupby": {"by": ["donor"], "op": "sum"}},),
    )

    # The request sends no principal header, since the route is open.
    response = rig.client().get(_VERSIONS_PATH)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "schema_version": 1,
        "spec_id": "spec_a",
        "versions": [
            {
                "content_version": _cv(6),
                "base_content_version": _cv(3),
                "rendering": "groupby",
                "groupby": _GROUPBY,
                "without_party_totals": True,
            },
            {
                "content_version": _cv(5),
                "base_content_version": _cv(3),
                "rendering": "without_party_totals",
                "groupby": None,
                "without_party_totals": True,
            },
            {
                "content_version": _cv(4),
                "base_content_version": _cv(3),
                "rendering": "not_recorded",
                "groupby": None,
            },
            {
                "content_version": _cv(2),
                "base_content_version": _cv(3),
                "rendering": "groupby",
                "groupby": _GROUPBY,
            },
            {
                "content_version": _cv(3),
                "base_content_version": None,
                "rendering": "plain",
                "groupby": None,
            },
            {
                "content_version": _cv(1),
                "base_content_version": _cv(0),
                "rendering": "groupby",
                "groupby": _LEVELLED_GROUPBY,
            },
        ],
        "next_cursor": None,
        "complete": True,
    }


def test_versions_list_is_empty_for_a_registered_dataset_with_no_stored_version(
    rig: _Rig,
) -> None:
    response = rig.client().get(_VERSIONS_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["versions"] == []
    assert response.json()["complete"] is True


@pytest.mark.parametrize(
    ("spec_reader", "status", "code"),
    [
        pytest.param(FakeSpecReader(), 404, "spec_not_found", id="unregistered"),
        pytest.param(
            _ServingSpecReader(spec_ids=["spec_a"], quarantined=frozenset({"spec_a"})),
            500,
            "identity_quarantined",
            id="quarantined",
        ),
    ],
)
def test_versions_list_refuses_an_id_the_snapshot_door_refuses(
    rig: _Rig, spec_reader: FakeSpecReader, status: int, code: str
) -> None:
    rig.record_version(content_version=_cv(1))

    response = rig.client(spec_reader).get(_VERSIONS_PATH)

    assert response.status_code == status, response.text
    assert response.json()["code"] == code


def test_versions_list_refuses_an_id_outside_the_dataset_grammar(rig: _Rig) -> None:
    response = rig.client().get("/api/datasets/ops_abc123/versions")

    assert response.status_code == 404, response.text
    assert response.json()["code"] == "not_found"


@pytest.mark.parametrize(
    ("revoked", "subject_id"),
    [
        pytest.param(_cv(2), "spec_a", id="the-version"),
        pytest.param(_cv(1), "spec_a", id="its-base-version"),
        pytest.param(_cv(2), "spec_served", id="under-a-served-id"),
    ],
)
def test_versions_list_leaves_out_a_revoked_version(
    rig: _Rig, revoked: str, subject_id: str
) -> None:
    """A version revoked, or whose base version is revoked, under the path id or a served id
    bound to it lists nowhere, as the snapshot door refuses it."""
    rig.record_version(content_version=_cv(1))
    rig.record_version(content_version=_cv(2), base_content_version=_cv(1))
    rig.record_version(content_version=_cv(3))
    rig.revoke(content_version=revoked, subject_id=subject_id)
    reader = _ServingSpecReader(spec_ids=["spec_a"], served={"spec_a": ("spec_served", "spec_a")})

    listed = _listed(rig.client(reader).get(_VERSIONS_PATH))

    assert _cv(2) not in listed
    assert _cv(3) in listed
    assert (_cv(1) in listed) is (revoked != _cv(1))


def test_versions_list_reads_under_the_served_ids_bound_to_a_posted_id(rig: _Rig) -> None:
    """A posted id lists the versions recorded under its served id and under itself, each
    once, newest first. The served id lists only its own."""
    rig.record_version(subject_id="spec_a", content_version=_cv(1))
    rig.record_version(subject_id="spec_served", content_version=_cv(2))
    rig.record_version(subject_id="spec_a", content_version=_cv(3))
    rig.record_version(subject_id="spec_served", content_version=_cv(3))
    reader = _ServingSpecReader(
        spec_ids=["spec_a", "spec_served"], served={"spec_a": ("spec_served", "spec_a")}
    )
    client = rig.client(reader)

    assert _listed(client.get(_VERSIONS_PATH)) == [_cv(3), _cv(2), _cv(1)]
    assert _listed(client.get("/api/datasets/spec_served/versions")) == [_cv(3), _cv(2)]


def test_versions_list_pages_with_a_cursor_until_complete(rig: _Rig) -> None:
    for n in range(1, 6):
        rig.record_version(content_version=_cv(n))
    client = rig.client()

    pages = []
    cursor = None
    while True:
        params = {"limit": 2} | ({"cursor": cursor} if cursor is not None else {})
        response = client.get(_VERSIONS_PATH, params=params)
        assert response.status_code == 200, response.text
        page = response.json()
        pages.append(([v["content_version"] for v in page["versions"]], page["complete"]))
        cursor = page["next_cursor"]
        assert (cursor is None) is page["complete"]
        if cursor is None:
            break

    assert pages == [
        ([_cv(5), _cv(4)], False),
        ([_cv(3), _cv(2)], False),
        ([_cv(1)], True),
    ]


def test_a_page_that_ends_the_list_exactly_is_complete(rig: _Rig) -> None:
    rig.record_version(content_version=_cv(1))
    rig.record_version(content_version=_cv(2))

    page = rig.client().get(_VERSIONS_PATH, params={"limit": 2}).json()

    assert page["complete"] is True
    assert page["next_cursor"] is None


@pytest.mark.parametrize(
    ("params", "pointer"),
    [
        pytest.param({"limit": 0}, "/limit", id="limit-zero"),
        pytest.param({"limit": VERSIONS_PAGE_MAX + 1}, "/limit", id="limit-over-max"),
        pytest.param({"cursor": "not base64 json"}, "/cursor", id="cursor-garbled"),
        pytest.param({"cursor": "other-dataset"}, "/cursor", id="cursor-of-another-id"),
    ],
)
def test_versions_list_refuses_a_bad_page_query(
    rig: _Rig, params: dict[str, object], pointer: str
) -> None:
    for n in range(1, 4):
        rig.record_version(content_version=_cv(n))
        rig.record_version(subject_id="spec_b", content_version=_cv(n))
    client = rig.client()
    if params.get("cursor") == "other-dataset":
        other = client.get("/api/datasets/spec_b/versions", params={"limit": 1}).json()
        params = {"cursor": other["next_cursor"]}

    response = client.get(_VERSIONS_PATH, params=params)

    assert response.status_code == 400, response.text
    assert response.json()["code"] == "request_invalid"
    assert response.json()["pointer"] == pointer
