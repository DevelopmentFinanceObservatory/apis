"""`GET /api/datasets/{id}/versions` is open to every caller and lists the versions the snapshot
door serves under the path id: under every served id bound to it and the posted id itself,
each once, newest first, with revoked versions left out, one page at a time.

One `FakeVersionRecordDatasetStore` backs the version writer and the version lister, so the
listing SQL runs against the rows the writer stored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from artifact_store import LiveArtifactInventory, LiveArtifactStore, Tier
from code_index.shared import get_index
from dataset_store.snapshot import FrameLineage, LiveSnapshotReader, ManifestFacts, VersionFacts
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.doors.versions import VERSIONS_PAGE_MAX
from datasets_api.recipe_store import LiveRecipeStore
from datasets_api.version_store import LiveVersionLister, LiveVersionRecordWriter
from datasets_api.versions import NO_PARTY_ROLE, ArtifactRecord, VersionRecord, VersionsPageKey
from spec_registry import IdentityQuarantined, LiveSpecRegistry
from tests.artifact_store.conftest import FakeGCSClient

from ._app_factories import make_client
from ._snapshot_doubles import FakeSpecReader
from ._version_facts import facts_from_body, golden_cases
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
        facts: VersionFacts | None = None,
        enrichment_digest: str | None = None,
    ) -> None:
        """Record a version with one stored file linked under it per entry of `renderings`,
        each salted with `enrichment_digest`, an hour after the version recorded before it.
        `facts` default to a published frame's."""
        self.now += timedelta(hours=1)
        artifacts = tuple(
            ArtifactRecord(
                artifact_digest=self.artifact_store.put(
                    f"{subject_id}:{content_version}:{index}".encode(), tier=Tier.working
                ).digest,
                rendering=rendering,
                media_type="application/vnd.apache.parquet",
                enrichment_digest=enrichment_digest,
            )
            for index, rendering in enumerate(renderings)
        )
        self.version_writer.record_version(
            VersionRecord(
                subject_id=subject_id,
                subject_kind="spec",
                content_version=content_version,
                schema_version=1,
                facts=facts if facts is not None else FrameLineage(input_content_versions=[]),
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
    # No version here stored an output profile or a `NO_PARTY_ROLE` link, so each lists the
    # digest.
    digest = get_index().enrichment_digest
    assert response.json() == {
        "schema_version": 1,
        "spec_id": "spec_a",
        "enrichment_stamp": digest,
        "versions": [
            {
                "content_version": _cv(6),
                "base_content_version": _cv(3),
                "rendering": "groupby",
                "groupby": _GROUPBY,
                "without_party_totals": True,
                "enrichment_stamp": digest,
            },
            {
                "content_version": _cv(5),
                "base_content_version": _cv(3),
                "rendering": "without_party_totals",
                "groupby": None,
                "without_party_totals": True,
                "enrichment_stamp": digest,
            },
            {
                "content_version": _cv(4),
                "base_content_version": _cv(3),
                "rendering": "not_recorded",
                "groupby": None,
                "enrichment_stamp": digest,
            },
            {
                "content_version": _cv(2),
                "base_content_version": _cv(3),
                "rendering": "groupby",
                "groupby": _GROUPBY,
                "enrichment_stamp": digest,
            },
            {
                "content_version": _cv(3),
                "base_content_version": None,
                "rendering": "plain",
                "groupby": None,
                "enrichment_stamp": digest,
            },
            {
                "content_version": _cv(1),
                "base_content_version": _cv(0),
                "rendering": "groupby",
                "groupby": _LEVELLED_GROUPBY,
                "enrichment_stamp": digest,
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
    assert response.json()["enrichment_stamp"] == get_index().enrichment_digest


_LONG_RENDERING = {"format": "parquet", "bundle": False}


def _dataset_facts(*roles: str) -> VersionFacts:
    """A dataset version's facts whose output profile lists `roles` as its role columns."""
    facts = facts_from_body(golden_cases()["definition"]["body"])
    assert isinstance(facts, ManifestFacts)
    profile = {
        "schema_version": 1,
        "rows": 1,
        "columns": [
            {"name": role, "kind": "role", "droppable": False, "distinct": 1} for role in roles
        ],
        "pivots": [],
        "enrichments": [],
    }
    return replace(facts, output_profile=profile)


@pytest.mark.parametrize(
    "spec_reader",
    [
        pytest.param(None, id="one-subject"),
        pytest.param(
            _ServingSpecReader(spec_ids=["spec_a"], served={"spec_a": ("spec_served", "spec_a")}),
            id="served-ids",
        ),
    ],
)
def test_each_listed_version_states_the_stamp_a_kind_file_of_it_carries_now(
    rig: _Rig, spec_reader: FakeSpecReader | None
) -> None:
    """A version whose stored output profile holds no party role, or whose kind file is linked
    under `NO_PARTY_ROLE`, lists no stamp; every other version lists the serving digest. The
    keyset query of one subject and the grouped query of several read the same facts."""
    digest = get_index().enrichment_digest
    rig.record_version(content_version=_cv(1), facts=_dataset_facts("observationAbout"))
    rig.record_version(content_version=_cv(2), facts=_dataset_facts("donor", "recipient"))
    # A version with no output profile is read from the salt of its kind files' links.
    rig.record_version(
        content_version=_cv(3), renderings=(_LONG_RENDERING,), enrichment_digest=NO_PARTY_ROLE
    )
    rig.record_version(content_version=_cv(4), renderings=(_LONG_RENDERING,))

    response = rig.client(spec_reader).get(_VERSIONS_PATH)

    assert response.status_code == 200, response.text
    stamps = {
        version["content_version"]: version.get("enrichment_stamp")
        for version in response.json()["versions"]
    }
    assert stamps == {_cv(4): digest, _cv(3): None, _cv(2): digest, _cv(1): None}
    assert response.json()["enrichment_stamp"] == digest


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
    if subject_id != "spec_a":
        # A revocation names a version recorded under its subject.
        rig.record_version(content_version=revoked, subject_id=subject_id)
    rig.revoke(content_version=revoked, subject_id=subject_id)
    reader = _ServingSpecReader(spec_ids=["spec_a"], served={"spec_a": ("spec_served", "spec_a")})

    listed = _listed(rig.client(reader).get(_VERSIONS_PATH))

    assert _cv(2) not in listed
    assert _cv(3) in listed
    assert (_cv(1) in listed) is (revoked != _cv(1))


_SERVED_FIRST = _ServingSpecReader(
    spec_ids=["spec_a", "spec_served"], served={"spec_a": ("spec_served", "spec_a")}
)


@pytest.mark.parametrize(
    ("bases", "revoked", "served"),
    [
        pytest.param({"spec_a": _cv(1)}, None, True, id="first-record-missing"),
        pytest.param({"spec_a": _cv(1)}, _cv(1), False, id="only-record-s-base-revoked"),
        pytest.param(
            {"spec_served": _cv(1), "spec_a": _cv(8)},
            _cv(1),
            False,
            id="first-record-s-base-revoked",
        ),
        pytest.param(
            {"spec_served": _cv(1), "spec_a": _cv(8)},
            _cv(8),
            True,
            id="later-record-s-base-revoked",
        ),
        pytest.param(
            {"spec_served": _cv(1), "spec_a": _cv(8)},
            _cv(2),
            False,
            id="version-revoked-under-the-posted-id",
        ),
    ],
)
def test_versions_list_holds_a_version_exactly_when_the_snapshot_door_serves_it(
    rig: _Rig, bases: Mapping[str, str], revoked: str | None, served: bool
) -> None:
    """`_cv(2)` is recorded as a dataset version under each subject of `bases`, on the base
    version it names, which is recorded under the posted id, and `revoked`, when set, is
    revoked under the posted id. The snapshot door reads the first subject in served order
    that records the version and checks revocation against that record, and the history
    lists the version exactly when that read serves it, as `served` states."""
    for base in sorted(set(bases.values())):
        rig.record_version(content_version=base)
    for subject_id in ("spec_served", "spec_a"):
        if subject_id in bases:
            rig.record_version(
                subject_id=subject_id,
                content_version=_cv(2),
                base_content_version=bases[subject_id],
                renderings=(_FRAME1_RENDERING,),
                facts=_dataset_facts("observationAbout"),
            )
    if revoked is not None:
        rig.revoke(content_version=revoked)
    client = rig.client(_SERVED_FIRST)

    snapshot = client.get(f"/api/datasets/spec_a/{_cv(2)}")

    assert snapshot.status_code == (200 if served else 410), snapshot.text
    assert (_cv(2) in _listed(client.get(_VERSIONS_PATH))) is served


def test_a_version_recorded_under_several_ids_lists_the_chosen_record_s_lineage_and_files(
    rig: _Rig,
) -> None:
    """A version recorded under a served id and the posted id lists the base version and the
    files of the record the snapshot door reads, the served id's, and none of the other's."""
    rig.record_version(
        subject_id="spec_a",
        content_version=_cv(2),
        base_content_version=_cv(9),
        renderings=({**_PLAIN_RENDERING, "groupby": _GROUPBY},),
        facts=_dataset_facts("observationAbout"),
    )
    rig.record_version(
        subject_id="spec_served",
        content_version=_cv(2),
        base_content_version=_cv(1),
        renderings=(_FRAME1_RENDERING,),
    )

    response = rig.client(_SERVED_FIRST).get(_VERSIONS_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["versions"] == [
        {
            "content_version": _cv(2),
            "base_content_version": _cv(1),
            "rendering": "not_recorded",
            "groupby": None,
            "enrichment_stamp": get_index().enrichment_digest,
        }
    ]


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


def test_one_subject_pages_the_same_through_the_keyset_and_the_grouped_query(
    rig: _Rig,
) -> None:
    """One subject reads its page with a keyset query and several read it grouped by
    content version. An extra id with no rows sends the same rows through the grouped query,
    and both list the same pages: newest first, `content_version` ordering a tie in
    `recorded_at`, revoked versions and versions on a revoked base left out."""
    for n in range(1, 7):
        if n == 4:
            rig.now -= timedelta(hours=1)  # records cv 4 at cv 3's instant
        rig.record_version(content_version=_cv(n), base_content_version=_cv(2) if n == 6 else None)
    rig.revoke(content_version=_cv(2))
    lister = LiveVersionLister(store=rig.store)

    def pages(subject_ids: list[str]) -> list[list[str]]:
        listed: list[list[str]] = []
        after = None
        while page := lister.list_servable_versions(subject_ids, after=after, limit=2):
            listed.append([version.content_version for version in page])
            after = VersionsPageKey(
                recorded_at=page[-1].recorded_at, content_version=page[-1].content_version
            )
        return listed

    expected = [[_cv(5), _cv(4)], [_cv(3), _cv(1)]]
    assert pages(["spec_a"]) == expected
    assert pages(["spec_a", "spec_unbound"]) == expected


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
