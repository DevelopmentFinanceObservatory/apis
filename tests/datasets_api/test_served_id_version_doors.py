"""The version-addressed doors (snapshot, files, pin, observations) read a version under the
served ids the registry binds to the path id, and under the posted id itself for a version
recorded under it (`doors.snapshot.read_under_served_id`). A version recorded under both ids
and revoked under either refuses `snapshot_revoked` on every door.

The posted identity leaves the imputed multilateral table's invariants unbound, so its
canonical form, the served identity, binds them and carries another `spec_id`
(`code_index.members.canonical_identity`). `kinds.spec_id.resolve_served_definition` binds the
served id to the posted one, and the doors read the stored bindings, so a code index export
that changes the canonical form leaves every issued address resolving."""

from __future__ import annotations

import functools
from dataclasses import replace

import pytest
from httpx import Response

from artifact_store import Tier
from code_index import CodeIndex
from code_index.shared import get_index
from dataset_store.snapshot import ArtifactNotFound, Revocation, SnapshotRevoked
from datasets_api.codes_snapshot import ObservationsUnavailable
from datasets_api.config import DatasetsBuildSettings
from datasets_api.deps import DepsCell
from datasets_api.doors.files import build_files_door
from datasets_api.doors.observations import build_observations_door, read_snapshot_observations
from datasets_api.doors.pin import build_pin_door, pin_artifact
from datasets_api.doors.refresh import build_refresh_door
from datasets_api.doors.snapshot import build_snapshot_door
from datasets_api.kinds.spec_id import resolve_served_definition
from dfo_platform.problem import Refusal
from spec_registry import (
    IdentityQuarantined,
    LiveSpecReader,
    LiveSpecRegistry,
    SpecReader,
    recompute_identity,
)
from tests.code_index.aid_identity import (
    AID_TYPE,
    IMPUTED_PROVENANCE,
    imputed_education_aid_identity,
)

from ._app_factories import offline_allowances
from ._artifact_doubles import FakeArtifactStore
from ._deps_doubles import fake_deps, fake_deps_cell
from ._snapshot_doubles import (
    _NOW,
    FakeSnapshotReader,
    _store_and_locate,
    _version,
    make_client_for_door,
)
from .conftest import FakeDatasetStore
from .test_pin_door import FakeArtifactReferenceWriter


def _posted_identity() -> dict:
    identity = imputed_education_aid_identity(source=IMPUTED_PROVENANCE)
    for slot in identity["slots"]:
        if slot["property"] in ("crsTypeOfFlow", AID_TYPE):
            slot["binding"] = {"state": "unbound"}
    return identity


_POSTED = recompute_identity(_posted_identity())


@functools.cache
def _prior_index() -> CodeIndex:
    """The fixture index as an export before any dataset declared the aid-type invariant, so
    the posted identity's canonical form under it leaves that slot unbound."""
    index = get_index()
    return CodeIndex(
        stamp=index.stamp,
        declarations=index.declarations.values(),
        codes=index.codes,
        statvars=index.statvars,
        datasets=[
            replace(
                dataset,
                invariants={
                    prop: code for prop, code in dataset.invariants.items() if prop != AID_TYPE
                },
            )
            for dataset in index.datasets.values()
        ],
        party_types=index.party_types,
        parties=index.parties.values(),
        listed_statvars=index.listed_statvars,
    )


def _registered(*indexes: CodeIndex) -> tuple[FakeDatasetStore, list[str]]:
    """A registry store holding the posted definition, resolved to its served definition under
    each of `indexes` in turn, and the served id each resolution bound."""
    store = FakeDatasetStore()
    registry = LiveSpecRegistry(store=store)
    registry.register_definition(
        identity_definition=_POSTED.identity, schema_version=1, supplied_spec_id=_POSTED.spec_id
    )
    stored = LiveSpecReader(store=store).read_definition(_POSTED.spec_id)
    assert stored is not None
    served = [
        resolve_served_definition(stored, code_index=index, registry=registry).spec_id
        for index in indexes
    ]
    return store, served


@functools.cache
def _bound() -> tuple[SpecReader, str]:
    """A reader of a registry binding the posted id to its served id under the fixture
    index, and that served id."""
    store, [served] = _registered(get_index())
    assert served != _POSTED.spec_id
    return LiveSpecReader(store=store), served


def _spec_reader() -> SpecReader:
    return _bound()[0]


def _served_spec_id() -> str:
    return _bound()[1]


def _recorded(case: str, store: FakeArtifactStore, *, tier: Tier = Tier.working, base=None):
    """A reader holding version cv1, with one artifact, as `case` names: recorded under the
    served or the posted id, or under both and revoked under one (`both-revoked-<id>`).
    Returns the reader, the subject id the doors answer under, and the artifact's location."""
    served, posted = _served_spec_id(), _POSTED.spec_id
    location = _store_and_locate(store, b"csv-bytes", tier=tier)
    recorded = {"served": (served,), "posted": (posted,)}.get(case, (served, posted))
    reader = FakeSnapshotReader(
        versions={
            (subject_id, "cv1"): replace(_version(base_content_version=base), subject_id=subject_id)
            for subject_id in recorded
        },
        locations={(subject_id, "cv1"): [location] for subject_id in recorded},
    )
    if case.startswith("both-revoked-"):
        revoked = served if case == "both-revoked-served" else posted
        reader.revocations[(revoked, "cv1")] = Revocation(reason="takedown", revoked_at=_NOW)
    return reader, recorded[0], location


_CASES = pytest.mark.parametrize(
    "case", ["served", "posted", "both-revoked-served", "both-revoked-posted"]
)


def _revoked(case: str) -> bool:
    return case.startswith("both-revoked-")


@_CASES
def test_the_snapshot_door_answers_under_the_posted_id(case: str):
    store = FakeArtifactStore()
    reader, _subject_id, location = _recorded(case, store)
    door = build_snapshot_door(
        deps=fake_deps_cell(spec_reader=_spec_reader(), snapshot_reader=reader, now=lambda: _NOW)
    )

    resp = make_client_for_door(door).get(f"/api/datasets/{_POSTED.spec_id}/cv1")

    if _revoked(case):
        assert resp.status_code == 410, resp.text
        assert resp.json()["code"] == "snapshot_revoked"
        return
    assert resp.status_code == 200, resp.text
    [artifact] = resp.json()["artifacts"]
    # The link names the id the request named, so a click honours the same revocations.
    assert artifact["url"] == (
        f"/api/datasets/{_POSTED.spec_id}/cv1/files/{location.artifact_digest}"
    )


@_CASES
def test_the_files_door_resolves_a_link_under_the_posted_id(case: str):
    store = FakeArtifactStore()
    reader, _subject_id, location = _recorded(case, store)
    door = build_files_door(
        deps=fake_deps_cell(
            spec_reader=_spec_reader(),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )

    resp = make_client_for_door(door).get(
        f"/api/datasets/{_POSTED.spec_id}/cv1/files/{location.artifact_digest}",
        follow_redirects=False,
    )

    if _revoked(case):
        assert resp.status_code == 410, resp.text
        assert resp.json()["code"] == "snapshot_revoked"
        assert store.signings == []
        return
    assert resp.status_code == 302, resp.text
    assert resp.headers["location"] == f"fake://artifact/working/{location.artifact_digest}"


def _pin(reader: FakeSnapshotReader, store, location, references) -> None:
    pin_artifact(
        id=_POSTED.spec_id,
        content_version="cv1",
        artifact_digest=location.artifact_digest,
        principal="principal",
        spec_reader=_spec_reader(),
        snapshot_reader=reader,
        artifact_store=store,
        reference_writer=references,
        allowances=offline_allowances(),
        now=lambda: _NOW,
    )


@_CASES
def test_a_pin_under_the_posted_id_references_the_id_the_version_is_recorded_under(case: str):
    store = FakeArtifactStore()
    reader, subject_id, location = _recorded(case, store, tier=Tier.pinned)
    references = FakeArtifactReferenceWriter()

    if _revoked(case):
        with pytest.raises(SnapshotRevoked):
            _pin(reader, store, location, references)
        assert references.records == {}
        return
    _pin(reader, store, location, references)
    assert [key[2] for key in references.records] == [subject_id]


def test_a_pin_rechecks_a_revocation_under_the_posted_id_at_the_write():
    """A revocation under the posted id that lands after the read refuses at the reference
    write of a version read under the served id."""
    store = FakeArtifactStore()
    reader, _subject_id, location = _recorded("served", store, tier=Tier.pinned)
    references = FakeArtifactReferenceWriter(revoked={(_POSTED.spec_id, "cv1"): "takedown"})

    with pytest.raises(SnapshotRevoked):
        _pin(reader, store, location, references)
    assert references.records == {}


@_CASES
def test_observations_under_the_posted_id_read_the_recorded_version(case: str):
    """A groupby collapse refuses `observations_unavailable` only once its version is read, so
    the refusal shows the read found it."""
    reader, _subject_id, _location = _recorded(case, FakeArtifactStore(), base="cv0")
    deps = fake_deps(spec_reader=_spec_reader(), snapshot_reader=reader, now=lambda: _NOW)

    expected = SnapshotRevoked if _revoked(case) else ObservationsUnavailable
    with pytest.raises(expected):
        read_snapshot_observations(id=_POSTED.spec_id, content_version="cv1", deps=deps)


def _recorded_with_two_digests(store: FakeArtifactStore, *, tier: Tier = Tier.working):
    """A reader holding cv1 under both ids, each record linking its own digest, as renderings
    that embed the subject id differ. Returns the reader and the posted record's location."""
    served, posted = _served_spec_id(), _POSTED.spec_id
    locations = {
        subject_id: _store_and_locate(store, f"bundle-of-{subject_id}".encode(), tier=tier)
        for subject_id in (served, posted)
    }
    reader = FakeSnapshotReader(
        versions={
            (subject_id, "cv1"): replace(_version(), subject_id=subject_id)
            for subject_id in locations
        },
        locations={(subject_id, "cv1"): [location] for subject_id, location in locations.items()},
    )
    return reader, locations[posted]


def test_a_link_to_a_digest_only_the_posted_record_links_resolves():
    """The served id's record of the version does not link the digest, so the read moves on
    to the posted id's record, which does."""
    store = FakeArtifactStore()
    reader, location = _recorded_with_two_digests(store)
    door = build_files_door(
        deps=fake_deps_cell(
            spec_reader=_spec_reader(),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )

    resp = make_client_for_door(door).get(
        f"/api/datasets/{_POSTED.spec_id}/cv1/files/{location.artifact_digest}",
        follow_redirects=False,
    )

    assert resp.status_code == 302, resp.text
    assert resp.headers["location"] == f"fake://artifact/working/{location.artifact_digest}"


def test_a_pin_of_a_digest_only_the_posted_record_links_references_the_posted_id():
    store = FakeArtifactStore()
    reader, location = _recorded_with_two_digests(store, tier=Tier.pinned)
    references = FakeArtifactReferenceWriter()

    _pin(reader, store, location, references)

    assert [key[2] for key in references.records] == [_POSTED.spec_id]


@pytest.mark.parametrize(
    ("inventoried", "status", "code"),
    [(True, 410, "artifact_expired"), (False, 404, "artifact_not_found")],
    ids=["replaced", "unknown"],
)
def test_a_posted_id_link_to_a_digest_the_served_record_does_not_link(
    inventoried: bool, status: int, code: str
):
    """The version is recorded under the served id alone. A digest it does not link refuses
    as the digest it is, replaced bytes `artifact_expired` and an unknown digest
    `artifact_not_found`, never `snapshot_not_found` from the posted id's missing record."""
    store = FakeArtifactStore()
    reader, _subject_id, _location = _recorded("served", store)
    digest = "sha256:not-linked"
    if inventoried:
        reader.inventoried.add(digest)
    door = build_files_door(
        deps=fake_deps_cell(
            spec_reader=_spec_reader(),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )

    resp = make_client_for_door(door).get(
        f"/api/datasets/{_POSTED.spec_id}/cv1/files/{digest}", follow_redirects=False
    )

    assert (resp.status_code, resp.json()["code"]) == (status, code), resp.text
    assert store.signings == []


def test_a_pin_under_the_posted_id_of_a_digest_the_served_record_does_not_link():
    store = FakeArtifactStore()
    reader, _subject_id, location = _recorded("served", store, tier=Tier.pinned)
    references = FakeArtifactReferenceWriter()

    with pytest.raises(ArtifactNotFound):
        _pin(reader, store, replace(location, artifact_digest="sha256:not-linked"), references)
    assert references.records == {}


@pytest.mark.parametrize("own_under", ["served", "posted"])
def test_a_version_revocation_wins_over_its_base_revocation_under_another_id(own_under: str):
    """A collapsed version revoked under one id while its base version is revoked under the
    other refuses with the version's own revocation, on the files and the pin door alike."""
    store = FakeArtifactStore()
    reader, _subject_id, location = _recorded(
        "both-revoked-served", store, tier=Tier.pinned, base="cv0"
    )
    served, posted = _served_spec_id(), _POSTED.spec_id
    own, base = (served, posted) if own_under == "served" else (posted, served)
    reader.revocations = {
        (own, "cv1"): Revocation(reason="own takedown", revoked_at=_NOW),
        (base, "cv0"): Revocation(reason="base takedown", revoked_at=_NOW),
    }
    door = build_files_door(
        deps=fake_deps_cell(
            spec_reader=_spec_reader(),
            snapshot_reader=reader,
            artifact_store=store,
            now=lambda: _NOW,
        )
    )

    resp = make_client_for_door(door).get(
        f"/api/datasets/{posted}/cv1/files/{location.artifact_digest}", follow_redirects=False
    )
    with pytest.raises(SnapshotRevoked) as pinned:
        _pin(reader, store, location, FakeArtifactReferenceWriter())

    assert resp.status_code == 410, resp.text
    assert resp.json()["detail"].endswith("own takedown")
    assert pinned.value.reason == "own takedown"


_DOORS = pytest.mark.parametrize("door", ["snapshot", "files", "pin", "observations"])


def _outcome(door: str, *, spec_store: FakeDatasetStore, recorded_under: str) -> tuple[str, list]:
    """What `door` answers for cv1 addressed under the posted id, over the registry in
    `spec_store` and a version recorded under `recorded_under` alone: `found`, or the refusal's
    code. Also returns the artifact store's signings."""
    store = FakeArtifactStore()
    location = _store_and_locate(
        store, b"csv-bytes", tier=Tier.pinned if door == "pin" else Tier.working
    )
    base = "cv0" if door == "observations" else None
    reader = FakeSnapshotReader(
        versions={
            (recorded_under, "cv1"): replace(
                _version(base_content_version=base), subject_id=recorded_under
            )
        },
        locations={(recorded_under, "cv1"): [location]},
    )
    spec_reader = LiveSpecReader(store=spec_store)
    deps = {"spec_reader": spec_reader, "snapshot_reader": reader, "now": lambda: _NOW}
    path = f"/api/datasets/{_POSTED.spec_id}/cv1"
    if door == "snapshot":
        resp = make_client_for_door(build_snapshot_door(deps=fake_deps_cell(**deps))).get(path)
        return ("found" if resp.status_code == 200 else resp.json()["code"]), store.signings
    if door == "files":
        files = build_files_door(deps=fake_deps_cell(artifact_store=store, **deps))
        resp = make_client_for_door(files).get(
            f"{path}/files/{location.artifact_digest}", follow_redirects=False
        )
        return ("found" if resp.status_code == 302 else resp.json()["code"]), store.signings
    try:
        if door == "pin":
            pin_artifact(
                id=_POSTED.spec_id,
                content_version="cv1",
                artifact_digest=location.artifact_digest,
                principal="principal",
                spec_reader=spec_reader,
                snapshot_reader=reader,
                artifact_store=store,
                reference_writer=FakeArtifactReferenceWriter(),
                allowances=offline_allowances(),
                now=lambda: _NOW,
            )
        else:
            read_snapshot_observations(
                id=_POSTED.spec_id, content_version="cv1", deps=fake_deps(**deps)
            )
    except ObservationsUnavailable:
        # A groupby collapse refuses only once its version is read.
        return "found", store.signings
    except Refusal as exc:
        return exc.code, store.signings
    return "found", store.signings


@_DOORS
def test_an_address_under_the_posted_id_outlives_a_code_index_export(door: str):
    """An index export that adds an invariant moves the posted id's served id. A version
    recorded under the earlier served id still answers under the posted id, which is bound to
    both."""
    spec_store, [earlier, current] = _registered(_prior_index(), get_index())
    assert len({_POSTED.spec_id, earlier, current}) == 3

    outcome, _signings = _outcome(door, spec_store=spec_store, recorded_under=earlier)

    assert outcome == "found"


@_DOORS
def test_a_posted_id_reads_no_version_under_a_served_id_it_is_not_bound_to(door: str):
    """The posted id is registered and never resolved, so no address under it was issued for
    a version recorded under its canonical form."""
    spec_store, [] = _registered()

    outcome, signings = _outcome(door, spec_store=spec_store, recorded_under=_served_spec_id())

    assert outcome == "snapshot_not_found"
    assert signings == []


@_DOORS
def test_a_posted_id_bound_to_a_quarantined_served_id_refuses(door: str):
    """The posted id is live and its served id quarantined, so the version recorded under the
    served id refuses under the posted id as it does under the served one, and signs nothing."""
    spec_store, [served] = _registered(get_index())
    spec_store.state.identity_quarantine.append(
        {"id_kind": "spec", "subject_id": served, "held_bytes": b"a", "incoming_bytes": b"b"}
    )

    outcome, signings = _outcome(door, spec_store=spec_store, recorded_under=served)

    assert outcome == "identity_quarantined"
    assert signings == []


def test_resolving_a_served_definition_binds_it_to_the_posted_id_once():
    spec_store, [first, second] = _registered(get_index(), get_index())

    assert first == second != _POSTED.spec_id
    assert spec_store.state.spec_alias == [(_POSTED.spec_id, first)]


class _QuarantinedSpecReader:
    """Raises `IdentityQuarantined` for every id, as the live reader does for a quarantined one."""

    def read_definition(self, spec_id: str, *, schema_version: int | None = None):
        raise IdentityQuarantined(subject_id=spec_id, id_kind="spec")

    def read_served_spec_ids(self, spec_id: str):
        raise IdentityQuarantined(subject_id=spec_id, id_kind="spec")


_PRINCIPAL = "quarantine-principal"
_AS_PRINCIPAL = {"x-dfo-principal": _PRINCIPAL}
_PIN_BODY = {"artifact_digest": "sha256:d"}
_REFRESH_BODY = {"spec_id": "spec_a", "content_version": "cv_1"}


def _quarantined_responses(cell: DepsCell) -> dict[str, Response]:
    """Each version-addressed route's answer to a request over `cell`."""
    settings = DatasetsBuildSettings.from_env()
    pin = build_pin_door(deps=cell, settings=settings, allowed_principals=frozenset({_PRINCIPAL}))
    refresh = build_refresh_door(deps=cell, settings=settings, principal=_PRINCIPAL)
    snapshot = make_client_for_door(build_snapshot_door(deps=cell))
    files = make_client_for_door(build_files_door(deps=cell))
    observations = make_client_for_door(build_observations_door(deps=cell))
    return {
        "snapshot": snapshot.get("/api/datasets/spec_a/cv1"),
        "files": files.get("/api/datasets/spec_a/cv1/files/sha256:d"),
        "observations": observations.get("/api/datasets/spec_a/cv1/observations"),
        "pin": make_client_for_door(pin).post(
            "/api/datasets/spec_a/cv1/pin", json=_PIN_BODY, headers=_AS_PRINCIPAL
        ),
        "refresh": make_client_for_door(refresh).post(
            "/api/datasets/frames/refresh", json=_REFRESH_BODY, headers=_AS_PRINCIPAL
        ),
    }


@pytest.mark.parametrize("route", ["snapshot", "files", "observations", "pin", "refresh"])
def test_a_version_route_refuses_a_quarantined_id_as_identity_quarantined(route: str):
    """Every version-addressed route reads the path id through the spec reader, which raises
    `IdentityQuarantined`; the code registry maps it 500 on each of these routes."""
    cell = fake_deps_cell(
        spec_reader=_QuarantinedSpecReader(), snapshot_reader=FakeSnapshotReader()
    )

    resp = _quarantined_responses(cell)[route]

    assert resp.status_code == 500, resp.text
    assert resp.json()["code"] == "identity_quarantined"
