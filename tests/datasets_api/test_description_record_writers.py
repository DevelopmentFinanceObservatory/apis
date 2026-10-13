"""Every call that writes a version row writes the version's description record in the same
transaction.

The four writers are a `spec_` mint, a collapse at mint of a multi-contributor version, a
collapsed head served from the subject head (`manifest.record_collapsed_head_version`), and an
`ops_` publish. Each record names its basis. A collapsed version's record derives from its raw
version's record, plus one step that names what was summed. A computed version's inputs are the
roots its `retrieve` steps name, and without steps each input's `id` is `null`. A failed record
insert is a store failure: it rolls the version row back and answers 503 `service_unavailable`.
A defect while building the record is not: the version is stored and served without a record,
the defect logged once at error level, for the backfill to describe once it is fixed.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager

import httpx
import pytest

from datasets_api import version_descriptions
from datasets_api.description_record import publish_record
from datasets_api.description_store import DESCRIPTION_RECORD_INVALID_MARKER
from datasets_api.version_descriptions import MintDescriber, PublishDescriber
from datasets_api.versions import DESCRIPTION_BUILD_FAILED_MARKER
from dc_graph import GraphCapExceededError, GraphDeadlineExceededError
from description.facts import SCHEMA_VERSION, ComposedVariable
from materialise import EnrichmentLookup

from ._description_doubles import (
    DENOMINATOR,
    FAMILY_AXIS,
    FAMILY_MEMBERS,
    GROUPBY_OPTIONS,
    GROUPBY_QUERY,
    NUMERATOR,
    OCV,
    OPS_ID,
    datasets_client,
    definition_mint,
    forget_record,
    mint_definition,
    mint_family,
    publish,
    publish_body,
    quotient_steps,
    read_record,
    served_steps,
    version_rows,
)
from ._version_record_store import FakeVersionRecordDatasetStore, _VersionRecordFakeTransaction


def test_a_spec_mint_writes_its_record_with_basis_mint() -> None:
    store = FakeVersionRecordDatasetStore()
    manifest = mint_definition(datasets_client(store))
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]

    record = read_record(store, subject_id=spec_id, content_version=content_version)

    assert record is not None
    assert (record.id, record.content_version, record.schema_version, record.basis) == (
        spec_id,
        content_version,
        1,
        "mint",
    )
    assert record.computed is None
    assert record.variables is not None
    assert (record.variables.admitted_at_mint, record.variables.with_data) == (1, 1)
    assert record.series.series == manifest["series_count"]
    assert record.evidence.prepared.isoformat() == manifest["prepared_at"][:10]
    assert [source.ref for source in record.evidence.sources] == [
        source["ref"] for source in manifest["citation"]["sources"]
    ]
    assert record.late_fields == []


def _summed_steps(record) -> list:
    return [step for step in record.steps if step.model_dump(mode="json").get("summed")]


def _assert_derived_from_raw(collapsed, raw) -> None:
    """A collapsed version's record keeps its raw version's population, evidence and steps, and
    adds one step naming the two members it summed. The groupby by recipient sums the test
    axis away, so the collapse holds no dimension."""
    assert collapsed.basis == "mint"
    assert raw.variables is not None
    assert [dimension.property for dimension in raw.variables.dimensions or ()] == [FAMILY_AXIS]
    assert collapsed.variables == raw.variables.model_copy(update={"dimensions": []})
    assert collapsed.evidence.prepared == raw.evidence.prepared
    assert collapsed.evidence.sources == raw.evidence.sources
    assert raw.steps is not None
    assert collapsed.steps is not None
    assert collapsed.steps[: len(raw.steps)] == raw.steps
    assert len(collapsed.steps) == len(raw.steps) + 1
    (summed,) = _summed_steps(collapsed)
    assert summed.summed.count == 2
    assert sorted(ref.ref for ref in summed.summed.variables) == sorted(FAMILY_MEMBERS)
    assert _summed_steps(raw) == []


def test_a_collapse_at_mint_writes_a_record_derived_from_its_raw_versions_record() -> None:
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store, fixture="family")

    collapsed = mint_family(client, options=GROUPBY_OPTIONS)
    spec_id = collapsed["definition"]["spec_id"]
    (raw_version,) = version_rows(store, spec_id) - {collapsed["content_version"]}

    raw = read_record(store, subject_id=spec_id, content_version=raw_version)
    derived = read_record(store, subject_id=spec_id, content_version=collapsed["content_version"])

    assert raw is not None
    assert derived is not None
    _assert_derived_from_raw(derived, raw)


def test_a_collapsed_head_writes_a_record_derived_from_its_base_versions_record() -> None:
    """A plain mint primes the subject head, so the groupby read that follows collapses the
    head through `record_collapsed_head_version`, which creates the collapsed version's row."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store, fixture="family")
    raw_manifest = mint_family(client, options={"bundle": False})
    spec_id = raw_manifest["definition"]["spec_id"]

    response = client.get(f"/api/datasets/{spec_id}?{GROUPBY_QUERY}")
    assert response.status_code == 200, response.text
    served = response.json()
    assert served["trace"]["served_from"] == "head"

    raw = read_record(store, subject_id=spec_id, content_version=raw_manifest["content_version"])
    derived = read_record(store, subject_id=spec_id, content_version=served["content_version"])

    assert raw is not None
    assert derived is not None
    _assert_derived_from_raw(derived, raw)


def test_a_collapsed_head_whose_base_holds_no_record_is_logged_without_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A base the previous image minted holds no record until the backfill runs, so its
    collapsed head gets none, and a warning names both content versions."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store, fixture="family")
    raw_manifest = mint_family(client, options={"bundle": False})
    spec_id, base_version = raw_manifest["definition"]["spec_id"], raw_manifest["content_version"]
    forget_record(store, subject_id=spec_id, content_version=base_version)

    with caplog.at_level(logging.WARNING, logger=version_descriptions.__name__):
        response = client.get(f"/api/datasets/{spec_id}?{GROUPBY_QUERY}")

    assert response.status_code == 200, response.text
    content_version = response.json()["content_version"]
    assert read_record(store, subject_id=spec_id, content_version=content_version) is None
    (warning,) = [r for r in caplog.records if r.name == version_descriptions.__name__]
    assert warning.levelno == logging.WARNING
    assert f"content_version={content_version}" in warning.getMessage()
    assert f"base_content_version={base_version}" in warning.getMessage()


def test_an_ops_publish_writes_its_record_with_its_step_records_and_basis_publish() -> None:
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    steps = quotient_steps()

    response = publish(client, publish_body(steps=steps))
    assert response.status_code == 200, response.text

    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert (record.id, record.content_version, record.basis) == (OPS_ID, OCV, "publish")
    assert record.variables is None
    assert record.computed is not None
    assert sorted((ref.id, ref.content_version) for ref in record.computed.inputs) == sorted(
        [NUMERATOR, DENOMINATOR]
    )
    assert served_steps(record) == steps
    assert record.series.series == 1
    assert record.late_fields == []


def test_an_ops_publish_without_steps_records_its_inputs_without_ids() -> None:
    """With no `retrieve` step to name them, the inputs keep their content versions and an
    `id` of `null`."""
    store = FakeVersionRecordDatasetStore()

    response = publish(datasets_client(store), publish_body())
    assert response.status_code == 200, response.text

    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert record.basis == "publish"
    assert record.steps is None
    assert record.computed is not None
    assert [(ref.id, ref.content_version) for ref in record.computed.inputs] == [
        (None, NUMERATOR[1]),
        (None, DENOMINATOR[1]),
    ]
    assert (record.caveats, record.evidence.labels) == (None, None)


class _RecordInsertFailingStore(FakeVersionRecordDatasetStore):
    """Fails any statement that writes `datasets.description_record`, inside the transaction
    that writes the version row."""

    @contextmanager
    def transaction(self) -> Generator[_FailingRecordWrites, None, None]:
        with super().transaction() as tx:
            yield _FailingRecordWrites(tx)


class _FailingRecordWrites:
    def __init__(self, tx: _VersionRecordFakeTransaction) -> None:
        self._tx = tx

    @staticmethod
    def _refuse(sql: str) -> None:
        if "description_record" in sql and sql.lstrip().upper().startswith("INSERT"):
            raise RuntimeError("record insert failed")

    def execute(self, sql: str, params: Sequence[object] = ()) -> None:
        self._refuse(sql)
        self._tx.execute(sql, params)

    def fetch_one(self, sql: str, params: Sequence[object] = ()):
        self._refuse(sql)
        return self._tx.fetch_one(sql, params)

    def fetch_all(self, sql: str, params: Sequence[object] = ()):
        self._refuse(sql)
        return self._tx.fetch_all(sql, params)


def test_a_failed_record_insert_rolls_back_the_minted_version_row() -> None:
    store = _RecordInsertFailingStore()

    response = definition_mint(datasets_client(store))

    _assert_store_unavailable(response)
    with store.transaction() as tx:
        assert tx.fetch_all("SELECT subject_id FROM datasets.version_record") == []


def test_a_failed_record_insert_rolls_back_the_published_version_row() -> None:
    store = _RecordInsertFailingStore()

    response = publish(datasets_client(store), publish_body())

    _assert_store_unavailable(response)
    assert version_rows(store, OPS_ID) == set()


def _assert_store_unavailable(response: httpx.Response) -> None:
    assert response.status_code == 503, response.text
    body = response.json()
    assert (body["code"], body["retryable"]) == ("service_unavailable", True)


def _defect(*args: object, **kwargs: object) -> None:
    raise RuntimeError("a defect in the record builder")


def _mint(client) -> tuple[str, str]:
    manifest = mint_definition(client)
    return manifest["definition"]["spec_id"], manifest["content_version"]


def _publish(client) -> tuple[str, str]:
    response = publish(client, publish_body(steps=quotient_steps()))
    assert response.status_code == 200, response.text
    return OPS_ID, OCV


@pytest.mark.parametrize(
    ("broken", "write"),
    [
        pytest.param((version_descriptions, "mint_record"), _mint, id="mint-before-the-write"),
        pytest.param((MintDescriber, "describe"), _mint, id="mint-in-the-write"),
        pytest.param(
            (version_descriptions, "publish_record"), _publish, id="publish-before-the-write"
        ),
        pytest.param((PublishDescriber, "describe"), _publish, id="publish-in-the-write"),
    ],
)
def test_a_record_builder_defect_stores_and_serves_the_version_without_a_record(
    broken: tuple[object, str],
    write: Callable[[object], tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A builder that raises, before the version write or inside its transaction, leaves the
    version stored and served with no record, and logs the defect once at error level."""
    monkeypatch.setattr(*broken, _defect)
    store = FakeVersionRecordDatasetStore()

    with caplog.at_level(logging.ERROR):
        subject_id, content_version = write(datasets_client(store))

    assert version_rows(store, subject_id) == {content_version}
    assert read_record(store, subject_id=subject_id, content_version=content_version) is None
    failures = [r for r in caplog.records if DESCRIPTION_BUILD_FAILED_MARKER in r.getMessage()]
    assert len(failures) == 1
    assert failures[0].levelno == logging.ERROR
    assert f"content_version={content_version}" in failures[0].getMessage()


def _damage_record(
    store: FakeVersionRecordDatasetStore, *, subject_id: str, content_version: str
) -> None:
    """Leave the version's stored record as JSON that fails validation."""
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.description_record SET record = %s "
            "WHERE subject_id = %s AND content_version = %s",
            ('{"id": "damaged"}', subject_id, content_version),
        )


@pytest.mark.parametrize(
    ("write", "basis"),
    [pytest.param(_mint, "mint", id="re-mint"), pytest.param(_publish, "publish", id="republish")],
)
def test_a_rewrite_replaces_a_stored_record_that_no_longer_validates(
    write: Callable[[object], tuple[str, str]], basis: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A second write of a version whose stored record fails validation is served and stores
    the record it builds in the damaged one's place, logging the damaged one."""
    store = FakeVersionRecordDatasetStore()
    subject_id, content_version = write(datasets_client(store))
    written = read_record(store, subject_id=subject_id, content_version=content_version)
    _damage_record(store, subject_id=subject_id, content_version=content_version)

    # A fresh client holds no subject head, so the re-mint writes the version again.
    with caplog.at_level(logging.ERROR):
        assert write(datasets_client(store)) == (subject_id, content_version)

    repaired = read_record(store, subject_id=subject_id, content_version=content_version)
    assert written is not None
    assert repaired is not None
    assert repaired.basis == basis
    assert repaired == written
    invalid = [r for r in caplog.records if DESCRIPTION_RECORD_INVALID_MARKER in r.getMessage()]
    assert invalid
    assert all(f"content_version={content_version}" in r.getMessage() for r in invalid)


@pytest.mark.parametrize(
    "write", [pytest.param(_mint, id="re-mint"), pytest.param(_publish, id="republish")]
)
def test_a_rewrite_leaves_a_record_of_a_later_schema_as_it_stands(
    write: Callable[[object], tuple[str, str]],
) -> None:
    """A version a later image recorded, rewritten by this image during a rollout or after a
    rollback, is served and keeps the later image's record."""
    store = FakeVersionRecordDatasetStore()
    subject_id, content_version = write(datasets_client(store))
    later = {"id": subject_id, "schema_version": SCHEMA_VERSION + 1, "a_later_member": True}
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.description_record SET schema_version = %s, record = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (SCHEMA_VERSION + 1, json.dumps(later), subject_id, content_version),
        )

    # A fresh client holds no subject head, so the re-mint writes the version again.
    assert write(datasets_client(store)) == (subject_id, content_version)

    with store.transaction() as tx:
        row = tx.fetch_one(
            "SELECT schema_version, record FROM datasets.description_record "
            "WHERE subject_id = %s AND content_version = %s",
            (subject_id, content_version),
        )
    assert row is not None
    assert (row["schema_version"], row["record"]) == (SCHEMA_VERSION + 1, later)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(GraphDeadlineExceededError("deadline passed"), id="past-its-deadline"),
        pytest.param(
            GraphCapExceededError("body too large", cap_kind="bytes", cap=1, observed=2),
            id="past-its-size-cap",
        ),
    ],
)
def test_a_publish_label_read_is_bounded_and_a_failed_one_still_stores_the_record(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publish's label read runs under the description route's 10 s bound, and a read that
    fails leaves the labels unread and the record stored."""
    deadlines: list[float | None] = []

    def failing(self: EnrichmentLookup, refs: object, *, deadline: float | None = None) -> None:
        deadlines.append(deadline)
        raise error

    monkeypatch.setattr(EnrichmentLookup, "lookup", failing)
    store = FakeVersionRecordDatasetStore()
    # A sum step names a graph ref, which the record's labels are read for.
    steps = [
        *quotient_steps(),
        {
            "id": "s4",
            "op": "sum_variables",
            "input": "s3",
            "summed": {"count": 1, "variables": [{"ref": "Count_Person"}]},
        },
    ]
    started = time.monotonic()

    response = publish(datasets_client(store), publish_body(steps=steps))

    assert response.status_code == 200, response.text
    assert len(deadlines) == 1
    assert deadlines[0] is not None
    assert started < deadlines[0] <= time.monotonic() + 10
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert served_steps(record) == steps
    assert record.evidence.labels == []


def test_a_publish_builds_its_record_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The record built to read its labels before the write is the one the write stores."""
    built: list[object] = []

    def counted(*args: object, **kwargs: object) -> object:
        built.append(args)
        return publish_record(*args, **kwargs)

    monkeypatch.setattr(version_descriptions, "publish_record", counted)
    store = FakeVersionRecordDatasetStore()

    response = publish(datasets_client(store), publish_body(steps=quotient_steps()))

    assert response.status_code == 200, response.text
    assert len(built) == 1
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert served_steps(record) == quotient_steps()


def test_a_republish_describes_the_composed_variables_its_stored_lineage_holds() -> None:
    """A version whose stored lineage holds composed variables, as one published while the
    door stored them does, gets them in the record a republish writes for it, though the
    publish posts none."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store)
    assert publish(client, publish_body()).status_code == 200
    forget_record(store, subject_id=OPS_ID, content_version=OCV)
    composed = {
        "stat_var": "dcid:Exemplar",
        "relation": "quotient",
        "first": ["dcid:Amount_GDP"],
        "second": ["dcid:Count_Person"],
    }
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET composed_stat_vars = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps([composed]), OPS_ID, OCV),
        )

    response = publish(client, publish_body())

    assert response.status_code == 200, response.text
    record = read_record(store, subject_id=OPS_ID, content_version=OCV)
    assert record is not None
    assert record.computed is not None
    assert record.computed.composed == [
        ComposedVariable(
            variable="Exemplar", relation="quotient", first=["Amount_GDP"], second=["Count_Person"]
        )
    ]
