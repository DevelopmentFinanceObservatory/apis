"""`datasets-maintenance backfill-descriptions` writes a record for every version made before
records existed, from the stored version, its artifacts and today's code index.

A backfilled record names its basis `backfill` and the code index it derived dimensions
through. A dimension is partial when a variable with data has left the index, and the population
admitted at mint is kept whole. A body stored without observation counts reads `with_data` as
unknown, and one without a family member count takes its resolved variables as the population.
Labels read from today's index carry the backfill's date. An `ops_` version recovers its inputs
and composed variables, each input's `id` is `null`, and its steps stay `null`. Stored
documentation gives a source's download date and pipeline steps. A second run writes nothing.

The counts only series statistics hold come from the version's series statistics artifact while a
tier holds it, else from a body's inline statistics list, else they are `null`. Stored warning
codes become the steps and caveats `WARNING_DESTINATIONS` routes them to, and a message naming
nothing readable, or a code the trace holds no step for, is left out with a warning. An `ops_` version whose frame no tier holds
has an unknown series count. A multi-contributor collapse is derived from its base version's
record, as at mint; while its base holds no record the collapse is logged and left for a later
run. A version the backfill cannot build a record for, whatever it raises, is logged by id with
its traceback and skipped, and the run goes on; it fails only when it reads no version and the
store failed at least one. A run past its budget leaves the rest to the next run.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pg8000.exceptions
import pytest
from google.api_core.exceptions import ServiceUnavailable

from artifact_store import (
    ArtifactStore,
    LiveArtifactInventory,
    LiveArtifactStore,
    Tier,
)
from artifact_store.errors import ArtifactIntegrityError, ArtifactInventoryError
from dataset_store.errors import DatasetStoreConnectionError
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import maintenance
from datasets_api.description_record import backfill_record
from datasets_api.maintenance import run_backfill_descriptions
from datasets_api.origin import _withheld_warning, un_unavailable_warnings
from datasets_api.render import STATISTICS_MEDIA_TYPE
from datasets_api.version_store import LiveVersionRecordWriter
from datasets_api.versions import ArtifactRecord, VersionRecord
from description.facts import SCHEMA_VERSION
from materialise.formats import CANONICAL_FRAME_MEDIA_TYPE
from materialise.frame_build import build_frame
from materialise.statistics import SeriesStatisticsSink, compute_series_statistics_table
from origin_catalogue.context import UnBlockedPlaces
from tests.artifact_store.conftest import FakeGCSClient
from tests.materialise._canonical_frame import (
    CANONICAL_FRAME_RENDERING,
    canonical_frame_parquet,
    stored_canonical_frame_959f8bc,
)

from . import test_serve_groupby_collapse as family_fixture
from . import test_serve_party_totals as party_fixture
from ._artifact_doubles import FakeArtifactStore
from ._description_doubles import (
    FAMILY_AXIS,
    GROUPBY_OPTIONS,
    datasets_client,
    forget_record,
    mint_definition,
    mint_family,
    read_record,
    version_rows,
)
from ._version_facts import facts_from_body, golden_cases
from ._version_record_store import FakeVersionRecordDatasetStore

_BACKFILL_AT = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)
# The family fixture's definition, whose stored bodies these versions are.
_SPEC_ID = family_fixture._spec_id_and_sidecar()[0]
_STALE_STAMP = {
    "content_hash": "sha256:the-mints-index",
    "exported_at": "2026-09-01T00:00:00Z",
    "graph_endpoint": "https://fixture.example.org",
}
_GONE = "FamilyMemberGone"
_DOCUMENTED_AT_MINT = {
    "last_update": "2026-10-08",
    "transformation_log": [
        {"transformation": "Filter to bilateral flows", "description": "Keeps bilateral rows."}
    ],
}


def _backfill(
    store: FakeVersionRecordDatasetStore, artifact_store: ArtifactStore | None = None
) -> int:
    return run_backfill_descriptions(
        store=store,
        artifact_store=artifact_store if artifact_store is not None else FakeArtifactStore(),
        code_index=family_fixture._code_index(),
        now=lambda: _BACKFILL_AT,
    )


def _live_artifact_store(store: FakeVersionRecordDatasetStore) -> LiveArtifactStore:
    """An artifact store whose writes the version store inventories, so a linked artifact has a
    location a backfill can read."""
    return LiveArtifactStore(
        buckets={Tier.working: "working-bucket", Tier.pinned: "pinned-bucket"},
        inventory=LiveArtifactInventory(store=store),
        client=FakeGCSClient(),
    )


def _entry(stat_var: str, observations: int | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"ref": f"dcid:{stat_var}", "roles": ["recipient"]}
    if observations is not None:
        entry["observation_count"] = observations
    return entry


def _family_body(
    *,
    prepared_at: str = "2026-10-08T09:00:00Z",
    entries: list[dict[str, Any]] | None = None,
    family_member_count: int | None = 5,
    documentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A stored body of the family fixture's definition, as a mint of `prepared_at` held it."""
    body = copy.deepcopy(golden_cases()["definition"]["body"])
    _spec_id, sidecar = family_fixture._spec_id_and_sidecar()
    body["definition"] = {"identity": sidecar["identity"], "schema_version": 1, "spec_id": _SPEC_ID}
    body["prepared_at"] = prepared_at
    body["series_count"] = 2
    body["trace"]["resolved_stat_vars"] = (
        entries
        if entries is not None
        else [
            _entry(family_fixture._MEMBER_A, 3),
            _entry(family_fixture._MEMBER_B, 0),
            _entry(_GONE, 2),
        ]
    )
    body["trace"]["family_member_count"] = family_member_count
    body["trace"]["code_index_stamp"] = _STALE_STAMP
    source = {**body["citation"]["sources"][0], "documentation": documentation}
    body["citation"] = {"text": source["label"], "sources": [source]}
    return body


def _stored_without_record(
    store: FakeVersionRecordDatasetStore,
    *,
    subject_id: str,
    content_version: str,
    body: dict,
    artifacts: tuple[ArtifactRecord, ...] = (),
) -> None:
    """Record a version from `body`, which it keeps as its stored body, leaving it without a
    description record, as a version made before records existed."""
    LiveVersionRecordWriter(store=store).record_version(
        VersionRecord(
            subject_id=subject_id,
            subject_kind="spec",
            content_version=content_version,
            schema_version=1,
            facts=facts_from_body(body),
            artifacts=artifacts,
        )
    )
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.version_record SET immutable_body = %s "
            "WHERE subject_id = %s AND content_version = %s",
            (json.dumps(body), subject_id, content_version),
        )
    forget_record(store, subject_id=subject_id, content_version=content_version)


def _backfilled(
    body: dict[str, Any],
    *,
    content_version: str = "cv_" + "2b" * 16,
    statistics: bytes | None = None,
):
    """`body`'s backfilled record, with `statistics` linked as its series statistics artifact."""
    store = FakeVersionRecordDatasetStore()
    artifact_store = _live_artifact_store(store)
    artifacts = ()
    if statistics is not None:
        stored = artifact_store.put(statistics, tier=Tier.working)
        artifacts = (
            ArtifactRecord(
                artifact_digest=stored.digest,
                rendering={"statistics": "series"},
                media_type=STATISTICS_MEDIA_TYPE,
            ),
        )
    _stored_without_record(
        store,
        subject_id=_SPEC_ID,
        content_version=content_version,
        body=body,
        artifacts=artifacts,
    )
    assert _backfill(store, artifact_store) == 0
    record = read_record(store, subject_id=_SPEC_ID, content_version=content_version)
    assert record is not None
    return record


def test_a_backfilled_record_names_its_basis_and_todays_index() -> None:
    record = _backfilled(_family_body())

    assert record.basis == "backfill"
    assert record.index_stamp is not None
    assert record.index_stamp.content_hash == family_fixture._code_index().stamp.content_hash
    assert record.evidence.prepared == date(2026, 10, 8)
    assert record.series.series == 2


def test_a_dimension_is_partial_when_a_variable_with_data_left_the_index() -> None:
    """Of three resolved variables, `FamilyMemberB` holds no data and `FamilyMemberGone`, which
    holds data, is gone from today's index. The population is the family's five members."""
    record = _backfilled(_family_body())

    assert record.variables is not None
    assert (
        record.variables.admitted_at_mint,
        record.variables.admitted_from,
        record.variables.with_data,
    ) == (5, "family_member_count", 2)
    assert record.variables.dimensions is not None
    (axis,) = [dim for dim in record.variables.dimensions if dim.property == FAMILY_AXIS]
    assert axis.partial is True
    assert [(value.value, value.variables) for value in axis.values] == [("TestAxisA", 1)]


def test_a_dimension_is_whole_when_only_variables_without_data_left_the_index() -> None:
    body = _family_body(
        entries=[
            _entry(family_fixture._MEMBER_A, 3),
            _entry(family_fixture._MEMBER_B, 1),
            _entry(_GONE, 0),
        ]
    )

    record = _backfilled(body)

    assert record.variables is not None
    assert record.variables.dimensions is not None
    (axis,) = [dim for dim in record.variables.dimensions if dim.property == FAMILY_AXIS]
    assert axis.partial is False
    assert [(value.value, value.variables) for value in axis.values] == [
        ("TestAxisA", 1),
        ("TestAxisB", 1),
    ]


def test_a_body_without_observation_counts_or_a_member_count_keeps_its_population() -> None:
    body = _family_body(
        entries=[
            _entry(stat_var, None)
            for stat_var in (family_fixture._MEMBER_A, family_fixture._MEMBER_B, _GONE)
        ],
        family_member_count=None,
    )

    record = _backfilled(body)

    assert record.variables is not None
    assert (
        record.variables.admitted_at_mint,
        record.variables.admitted_from,
        record.variables.with_data,
    ) == (3, "resolved_variables", None)


def test_labels_read_from_todays_index_carry_the_backfill_date() -> None:
    record = _backfilled(_family_body())

    labels = {(label.ref, label.label): label.read for label in record.evidence.labels}
    assert labels[("TestAxisA", "Axis value A")] == _BACKFILL_AT.date()


def test_documentation_a_mint_read_gives_the_download_date_and_pipeline_steps() -> None:
    record = _backfilled(_family_body(documentation=_DOCUMENTED_AT_MINT))

    (source,) = record.evidence.sources
    assert source.downloaded == date(2026, 10, 8)
    assert source.pipeline_steps is not None
    assert [step.title for step in source.pipeline_steps] == ["Filter to bilateral flows"]


def _backfilled_ops(*, frame: bytes | None):
    """The backfilled record of the golden published body, linking `frame` as its canonical
    frame when it is not `None`."""
    published = golden_cases()["published"]["body"]
    store = FakeVersionRecordDatasetStore()
    artifact_store = _live_artifact_store(store)
    ops_id, ocv = "ops_" + "0f" * 16, "ocv_" + "1e" * 16
    artifacts = ()
    if frame is not None:
        stored = artifact_store.put(frame, tier=Tier.working)
        artifacts = (
            ArtifactRecord(
                artifact_digest=stored.digest,
                rendering=dict(CANONICAL_FRAME_RENDERING),
                media_type=CANONICAL_FRAME_MEDIA_TYPE,
            ),
        )
    _stored_without_record(
        store, subject_id=ops_id, content_version=ocv, body=published, artifacts=artifacts
    )

    assert _backfill(store, artifact_store) == 0

    record = read_record(store, subject_id=ops_id, content_version=ocv)
    assert record is not None
    return published, record


def _two_series_frame() -> bytes:
    return canonical_frame_parquet(
        (
            build_frame(
                role_keys=("recipient",),
                rows=[
                    {
                        "stat_var": "dcid:Share_A",
                        "recipient": place,
                        "time_period": "2020",
                        "value": 0.5,
                    }
                    for place in _PLACES
                ],
            ),
        )
    )


def test_an_ops_version_recovers_its_inputs_without_ids_and_its_composed_variables() -> None:
    """Its frame is not readable, so its series count is unknown and the run still succeeds."""
    published, record = _backfilled_ops(frame=None)

    assert record.basis == "backfill"
    assert record.steps is None
    assert record.computed is not None
    assert [(ref.id, ref.content_version) for ref in record.computed.inputs] == [
        (None, content_version) for content_version in published["input_content_versions"]
    ]
    assert (record.caveats, record.evidence.labels) == (None, None)
    assert record.computed.composed is not None
    assert [(composed.variable, composed.relation) for composed in record.computed.composed] == [
        ("Share_A", "share")
    ]
    assert (record.series.series, record.series.rows, record.series.periods) == (None, None, None)


@pytest.mark.parametrize(
    ("frame", "counts"),
    [
        pytest.param(_two_series_frame, (2, 2, 1, 2), id="this-release"),
        pytest.param(
            lambda: stored_canonical_frame_959f8bc()[0], (6, 12, 2, 6), id="stored-by-kit-959f8bc"
        ),
    ],
)
def test_an_ops_version_with_a_readable_frame_counts_its_series_from_the_frame(
    frame: Callable[[], bytes], counts: tuple[int, int, int, int]
) -> None:
    """A canonical frame stored under frame grammar kit 959f8bc, whose unit `request` has no
    `recipe`, is read as one this release stored."""
    _published, record = _backfilled_ops(frame=frame())

    series = record.series
    assert series.entities_by_role is not None
    assert [(role.role, role.entities) for role in series.entities_by_role] == [
        ("recipient", counts[3])
    ]
    assert (series.series, series.rows, series.periods) == counts[:3]


def test_a_second_run_writes_nothing() -> None:
    store = FakeVersionRecordDatasetStore()
    content_version = "cv_" + "3c" * 16
    _stored_without_record(
        store, subject_id=_SPEC_ID, content_version=content_version, body=_family_body()
    )
    assert _backfill(store) == 0
    first = read_record(store, subject_id=_SPEC_ID, content_version=content_version)

    assert _backfill(store) == 0

    assert read_record(store, subject_id=_SPEC_ID, content_version=content_version) == first


def test_a_record_that_fails_validation_is_backfilled_in_its_place(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stored record that no longer validates counts as none: the run replaces it, and the
    next run finds nothing to do."""
    store = FakeVersionRecordDatasetStore()
    content_version = "cv_" + "3c" * 16
    _stored_without_record(
        store, subject_id=_SPEC_ID, content_version=content_version, body=_family_body()
    )
    assert _backfill(store) == 0
    first = read_record(store, subject_id=_SPEC_ID, content_version=content_version)
    with store.transaction() as tx:
        tx.execute("UPDATE datasets.description_record SET record = %s", ('{"id": "damaged"}',))
    capsys.readouterr()

    assert _backfill(store) == 0
    assert _backfill(store) == 0

    assert read_record(store, subject_id=_SPEC_ID, content_version=content_version) == first
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        line["backfilled"] for line in lines if line["event"] == "datasets_descriptions_backfilled"
    ] == [1, 0]


def test_a_record_of_a_later_schema_is_neither_queued_nor_replaced(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A later image's record, which this image cannot validate, is no damage to repair: a
    rollback's run leaves it as the later image wrote it."""
    store = FakeVersionRecordDatasetStore()
    content_version = "cv_" + "3c" * 16
    _stored_without_record(
        store, subject_id=_SPEC_ID, content_version=content_version, body=_family_body()
    )
    assert _backfill(store) == 0
    first = read_record(store, subject_id=_SPEC_ID, content_version=content_version)
    assert first is not None
    later = {
        **first.model_dump(mode="json"),
        "schema_version": SCHEMA_VERSION + 1,
        "a_later_member": True,
    }
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.description_record SET schema_version = %s, basis = %s, record = %s",
            (SCHEMA_VERSION + 1, "mint", json.dumps(later)),
        )
    capsys.readouterr()

    assert maintenance._unrecorded_versions(store) == []
    assert _backfill(store) == 0

    with store.transaction() as tx:
        row = tx.fetch_one("SELECT schema_version, basis, record FROM datasets.description_record")
    assert row is not None
    assert (row["schema_version"], row["basis"], row["record"]) == (
        SCHEMA_VERSION + 1,
        "mint",
        later,
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        (line["backfilled"], line["failed"])
        for line in lines
        if line["event"] == "datasets_descriptions_backfilled"
    ] == [(0, 0)]


def test_a_version_holding_a_minted_record_keeps_it() -> None:
    store = FakeVersionRecordDatasetStore()
    manifest = mint_definition(datasets_client(store))
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    minted = read_record(store, subject_id=spec_id, content_version=content_version)

    assert _backfill(store) == 0

    kept = read_record(store, subject_id=spec_id, content_version=content_version)
    assert kept == minted
    assert kept is not None
    assert kept.basis == "mint"


_PLACES = ("country/KEN", "country/TGO")
_PERIODS = ("2019", "2020")
_FACET = {"measurementMethod": "Estimated", "provenance": "dc/base/Fixture", "unit": "USDollar"}


def _statistics_artifact() -> bytes:
    """The series statistics of two estimated series of `FamilyMemberA`, 2019 to 2020."""
    frame = build_frame(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": f"dcid:{family_fixture._MEMBER_A}",
                "recipient": place,
                "time_period": period,
                "value": 1.0,
                **_FACET,
            }
            for place in _PLACES
            for period in _PERIODS
        ],
    )
    sink = SeriesStatisticsSink(continuing=())
    sink.write(compute_series_statistics_table(frame))
    return sink.finish().data


# The same two series as a body stored before the series statistics artifact held them.
_INLINE_STATISTICS = [
    {
        "stat_var": f"dcid:{family_fixture._MEMBER_A}",
        "entities": [{"role": "recipient", "ref": place}],
        "facet": _FACET,
        "observations": 2,
        "first": {"value": 1.0, "period": "2019"},
        "latest": {"value": 1.0, "period": "2020"},
        "minimum": {"value": 1.0, "period": "2019"},
        "maximum": {"value": 1.0, "period": "2019"},
        "change": {"absolute": 0.0, "percent": 0.0},
    }
    for place in _PLACES
]


@pytest.mark.parametrize("held", ["artifact", "inline", "neither"])
def test_series_facts_come_from_the_series_statistics_while_they_are_held(held: str) -> None:
    body = _family_body()
    if held == "inline":
        body["statistics"] = _INLINE_STATISTICS

    record = _backfilled(body, statistics=_statistics_artifact() if held == "artifact" else None)

    series = record.series
    assert (series.series, series.rows) == (2, body["coverage"]["rows"])
    figures = (
        series.span,
        series.latest_period,
        series.series_reaching_latest,
        series.by_source_unit,
        (series.status.estimated, series.status.provisional),
    )
    if held == "neither":
        assert figures == (None, None, None, None, (None, None))
        assert record.evidence.builds.method_status is None
        return
    span, latest, reaching, by_source_unit, status = figures
    assert span is not None
    assert (span.start, span.end, latest, reaching, status) == ("2019", "2020", "2020", 2, (4, 0))
    assert by_source_unit is not None
    assert [(count.source, count.series) for count in by_source_unit] == [("dc/base/Fixture", 2)]


@pytest.mark.parametrize("held", ["artifact", "inline", "neither"])
def test_an_open_role_is_a_dimension_known_while_the_series_statistics_are_held(
    held: str,
) -> None:
    """The definition leaves the recipient open. Each of its two places is held by
    `FamilyMemberA` alone. With no statistics held, the recipient is still a dimension, its
    values unknown."""
    body = _family_body()
    body["definition"]["identity"] = {**body["definition"]["identity"], "entities": {}}
    if held == "inline":
        body["statistics"] = _INLINE_STATISTICS

    record = _backfilled(body, statistics=_statistics_artifact() if held == "artifact" else None)

    assert record.variables is not None
    assert record.variables.dimensions is not None
    (recipient,) = [dim for dim in record.variables.dimensions if dim.property == "recipient"]
    if held == "neither":
        assert recipient.values is None
        return
    assert recipient.values is not None
    assert [(value.value, value.variables) for value in recipient.values] == [
        ("country/KEN", 1),
        ("country/TGO", 1),
    ]
    assert recipient.partial is False


_NAMED_PLACES = (
    "country/KEN",
    "country/NGA",
    "country/UGA",
    "country/TZA",
    "country/RWA",
    "country/BDI",
    "country/SSD",
)


@pytest.mark.parametrize(
    ("places", "count"), [(_NAMED_PLACES[:2], None), (_NAMED_PLACES, 7)], ids=["all", "capped"]
)
def test_stored_warning_codes_become_the_steps_and_caveats_they_route_to(
    places: tuple[str, ...], count: int | None
) -> None:
    """A registry withholding and an unanswered UN read are coverage gaps naming what their
    messages list, in the one format each producer has written, a sum is a step, and a code
    routed to the manifest adds nothing. A message naming more than five places names five and
    counts the rest."""
    blocked = UnBlockedPlaces()
    blocked.add(places)
    withheld = _withheld_warning(
        f"dcid:{family_fixture._MEMBER_A}", dict.fromkeys(places, ("dc/withheld",))
    )
    body = _family_body()
    body["warnings"] = [
        withheld.model_dump(mode="json"),
        *(warning.model_dump(mode="json") for warning in un_unavailable_warnings(blocked)),
        {
            "code": "stat_vars_summed",
            "severity": "info",
            "message": "summed",
            "contributing_member_count": 3,
            "stat_var_ref": f"dcid:{_SPEC_ID}",
        },
        {"code": "coverage_below_scope", "severity": "warn", "message": "below scope"},
    ]

    record = _backfilled(body)

    named = list(places[:5])
    assert [
        (caveat.kind, caveat.series, caveat.entities, caveat.count) for caveat in record.caveats
    ] == [
        ("coverage_gap", [family_fixture._MEMBER_A], named, count),
        ("coverage_gap", [], named, count),
    ]
    assert record.steps is not None
    (step,) = record.steps
    assert step.op == "sum_variables"
    assert step.summed is not None
    assert step.summed.count == 3
    assert [ref.ref for ref in step.summed.variables] == [family_fixture._MEMBER_A, _GONE]


def test_a_withholding_and_an_unanswered_read_the_trace_records_are_stated_once_from_it() -> None:
    """A version whose trace records the places (`trace.origin_gaps`) states each gap from the
    trace, naming every place, and its stored warnings add no second caveat."""
    blocked = UnBlockedPlaces()
    blocked.add(_NAMED_PLACES)
    withheld = _withheld_warning(
        f"dcid:{family_fixture._MEMBER_A}", dict.fromkeys(_NAMED_PLACES, ("dc/withheld",))
    )
    body = _family_body()
    body["warnings"] = [
        withheld.model_dump(mode="json"),
        *(warning.model_dump(mode="json") for warning in un_unavailable_warnings(blocked)),
    ]
    body["trace"]["origin_gaps"] = [
        {
            "code": "withheld_by_registry",
            "variable": family_fixture._MEMBER_A,
            "entities": list(_NAMED_PLACES),
            "provenances": ["dc/withheld"],
        },
        {"code": "un_system_unavailable", "entities": list(_NAMED_PLACES)},
    ]

    record = _backfilled(body)

    assert [
        (caveat.kind, caveat.series, caveat.entities, caveat.count) for caveat in record.caveats
    ] == [
        ("coverage_gap", [family_fixture._MEMBER_A], list(_NAMED_PLACES), None),
        ("coverage_gap", [], list(_NAMED_PLACES), None),
    ]


def test_a_stored_message_naming_nothing_readable_leaves_its_caveat_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Old prose is no defect: the version gains its record without the caveat, the run
    succeeds, and one warning names the version and code."""
    store = FakeVersionRecordDatasetStore()
    content_version = "cv_" + "6f" * 16
    body = _family_body()
    body["warnings"] = [
        {"code": "withheld_by_registry", "severity": "warn", "message": "GrantB withheld at KEN"},
        {"code": "un_system_unavailable", "severity": "warn", "message": "UN unavailable"},
    ]
    _stored_without_record(store, subject_id=_SPEC_ID, content_version=content_version, body=body)

    assert _backfill(store) == 0

    record = read_record(store, subject_id=_SPEC_ID, content_version=content_version)
    assert record is not None
    assert record.caveats == []
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        (line["severity"], line["subject_id"], line["content_version"], line["warning_code"])
        for line in lines
        if line["event"] == "datasets_description_backfill_warning_unstated"
    ] == [
        ("WARNING", _SPEC_ID, content_version, "withheld_by_registry"),
        ("WARNING", _SPEC_ID, content_version, "un_system_unavailable"),
    ]


# A conversion whose projected rates make its values estimated, as a trace records it.
_PROJECTED_CONVERSION = {"members": [{"native_unit": "dcid:EUR"}], "projected": {"count": 3}}


@pytest.mark.parametrize(
    ("code", "conversion", "steps"),
    [
        ("facet_default_selected", None, None),
        ("facet_default_selected", _PROJECTED_CONVERSION, None),
        ("currency_unmapped_omitted", None, None),
        ("deflator_missing_omitted", None, None),
        ("deflator_projected", None, []),
    ],
    ids=["choice", "choice-beside-a-conversion", "unmapped", "deflator-missing", "projected"],
)
def test_a_stored_warning_the_trace_contradicts_is_left_out_with_a_warning(
    code: str,
    conversion: dict[str, Any] | None,
    steps: list | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The trace records no choice of source and no conversion, so a stored choice, conversion
    gap or projection has no step to point at. The version still gains its record: a code
    routed to a step leaves the steps unknown, taking with them the caveats that point at a
    step, a code routed to a caveat adds none, and one warning names the version and code."""
    store = FakeVersionRecordDatasetStore()
    content_version = "cv_" + "7a" * 16
    body = _family_body()
    body["warnings"] = [{"code": code, "severity": "info", "message": "stored"}]
    if conversion is not None:
        body["trace"]["conversion"] = [conversion]
    _stored_without_record(store, subject_id=_SPEC_ID, content_version=content_version, body=body)

    assert _backfill(store) == 0

    record = read_record(store, subject_id=_SPEC_ID, content_version=content_version)
    assert record is not None
    assert (record.steps, record.caveats) == (steps, [])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        (line["subject_id"], line["content_version"], line["warning_code"])
        for line in lines
        if line["event"] == "datasets_description_backfill_warning_unstated"
    ] == [(_SPEC_ID, content_version, code)]


def _unbackfillable_body(fault: str) -> dict[str, Any]:
    """A body the builders cannot read: one stating no series count with no series statistics
    held, or one whose resolved variable is no object."""
    body = _family_body()
    if fault == "ValueError":
        body["series_count"] = None
    else:
        body["trace"]["resolved_stat_vars"] = [f"dcid:{family_fixture._MEMBER_A}"]
    return body


@pytest.mark.parametrize("fault", ["ValueError", "AttributeError"])
def test_a_version_that_cannot_be_backfilled_is_logged_and_skipped(
    fault: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A body the builders cannot read is a defect of that version alone, whatever it raises:
    it stays without a record, logged with its traceback, while the next one gains its own,
    and the run succeeds."""
    store = FakeVersionRecordDatasetStore()
    unbackfillable, backfillable = "cv_" + "4d" * 16, "cv_" + "5e" * 16
    _stored_without_record(
        store, subject_id=_SPEC_ID, content_version=unbackfillable, body=_unbackfillable_body(fault)
    )
    _stored_without_record(
        store, subject_id=_SPEC_ID, content_version=backfillable, body=_family_body()
    )

    assert _backfill(store) == 0

    assert read_record(store, subject_id=_SPEC_ID, content_version=unbackfillable) is None
    assert read_record(store, subject_id=_SPEC_ID, content_version=backfillable) is not None
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    failed = [line for line in lines if line["event"] == "datasets_description_backfill_failed"]
    assert [(line["subject_id"], line["content_version"]) for line in failed] == [
        (_SPEC_ID, unbackfillable)
    ]
    assert failed[0]["error"].startswith(f"{fault}:")
    assert f"\n{fault}: " in failed[0]["traceback"]
    assert [
        (line["backfilled"], line["failed"])
        for line in lines
        if line["event"] == "datasets_descriptions_backfilled"
    ] == [(1, 1)]


_CONNECTION_LOST = DatasetStoreConnectionError("connection lost")
_OVERWRITTEN = ArtifactIntegrityError(
    digest="sha256:" + "a" * 64, tier=Tier.working, computed_digest="sha256:" + "b" * 64
)


@pytest.mark.parametrize(
    ("broken", "error", "exit_status", "transient"),
    [
        pytest.param(None, None, 0, False, id="a-body-no-run-can-describe"),
        pytest.param(
            (LiveSnapshotReader, "read_version_state"),
            _CONNECTION_LOST,
            1,
            True,
            id="the-version-read-loses-its-connection",
        ),
        pytest.param(
            (maintenance, "insert_description"),
            _CONNECTION_LOST,
            1,
            True,
            id="the-record-insert-loses-its-connection",
        ),
        pytest.param(
            (maintenance, "insert_description"),
            pg8000.exceptions.DatabaseError({"C": "40001", "M": "serialization failure"}),
            1,
            True,
            id="the-record-insert-meets-a-transient-sqlstate",
        ),
        pytest.param(
            (maintenance, "insert_description"),
            pg8000.exceptions.DatabaseError({"C": "23503", "M": "foreign key violation"}),
            0,
            False,
            id="the-record-insert-meets-a-permanent-sqlstate",
        ),
        pytest.param(
            (LiveArtifactInventory, "get_location"),
            ArtifactInventoryError(digest="sha256:" + "a" * 64, tier=Tier.working),
            1,
            True,
            id="the-artifact-location-read-fails-in-the-store",
        ),
        pytest.param(
            (LiveArtifactStore, "get"),
            ServiceUnavailable("bucket unavailable"),
            1,
            True,
            id="the-artifact-read-answers-503",
        ),
        pytest.param(
            (LiveArtifactStore, "get"),
            _OVERWRITTEN,
            0,
            False,
            id="the-artifact-was-overwritten",
        ),
    ],
)
def test_a_run_that_reads_no_version_fails_only_on_a_failure_a_retry_clears(
    broken: tuple[object, str] | None,
    error: Exception | None,
    exit_status: int,
    transient: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Versions that can never be described, a damaged object included, leave every later run
    with nothing it can read, and those runs pass; a store that is unreachable or overloaded
    for every version fails the run."""
    store = FakeVersionRecordDatasetStore()
    artifact_store = _live_artifact_store(store)
    if broken is None:
        _stored_without_record(
            store,
            subject_id=_SPEC_ID,
            content_version="cv_" + "4d" * 16,
            body=_unbackfillable_body("ValueError"),
        )
    else:
        statistics = artifact_store.put(_statistics_artifact(), tier=Tier.working)
        _stored_without_record(
            store,
            subject_id=_SPEC_ID,
            content_version="cv_" + "4d" * 16,
            body=_family_body(),
            artifacts=(
                ArtifactRecord(
                    artifact_digest=statistics.digest,
                    rendering={"statistics": "series"},
                    media_type=STATISTICS_MEDIA_TYPE,
                ),
            ),
        )

        def failing(*args: object, **kwargs: object) -> None:
            assert error is not None
            raise error

        monkeypatch.setattr(*broken, failing)

    assert _backfill(store, artifact_store) == exit_status

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    (failed,) = [line for line in lines if line["event"] == "datasets_description_backfill_failed"]
    assert failed["transient"] is transient
    if error is not None:
        assert failed["error"].startswith(f"{type(error).__name__}: ")
    assert [
        (line["failed"], line["transient"])
        for line in lines
        if line["event"] == "datasets_descriptions_backfilled"
    ] == [(1, int(transient))]


def test_a_run_past_its_budget_leaves_the_rest_to_the_next_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The run starts no version once its budget has passed, the most recently recorded first,
    and exits 0; the next run backfills the versions it left."""
    store = FakeVersionRecordDatasetStore()
    older, newer = "cv_" + "4d" * 16, "cv_" + "5e" * 16
    for content_version in (older, newer):
        _stored_without_record(
            store, subject_id=_SPEC_ID, content_version=content_version, body=_family_body()
        )
    # The deadline, the first version's check and its backfill time, then the second's check.
    ticks = iter([_BACKFILL_AT, _BACKFILL_AT, _BACKFILL_AT, _BACKFILL_AT + timedelta(hours=1)])

    assert (
        run_backfill_descriptions(
            store=store,
            artifact_store=FakeArtifactStore(),
            code_index=family_fixture._code_index(),
            now=lambda: next(ticks),
            budget=timedelta(minutes=45),
        )
        == 0
    )

    assert read_record(store, subject_id=_SPEC_ID, content_version=newer) is not None
    assert read_record(store, subject_id=_SPEC_ID, content_version=older) is None
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        (line["backfilled"], line["remaining"])
        for line in lines
        if line["event"] == "datasets_descriptions_backfilled"
    ] == [(1, 1)]

    assert _backfill(store) == 0
    assert read_record(store, subject_id=_SPEC_ID, content_version=older) is not None


def test_a_collapse_is_backfilled_from_its_base_versions_record() -> None:
    """A multi-contributor collapse the new image wrote without a record, whose manifest no
    longer lists `stat_vars_summed`, gains the sum step the mint gives it from its base's
    record, backfilled first."""
    store = FakeVersionRecordDatasetStore()
    collapsed = mint_family(datasets_client(store, fixture="family"), options=GROUPBY_OPTIONS)
    spec_id, content_version = collapsed["definition"]["spec_id"], collapsed["content_version"]
    minted = read_record(store, subject_id=spec_id, content_version=content_version)
    assert minted is not None
    assert [step.op for step in minted.steps or []] == ["sum_variables"]
    assert "stat_vars_summed" not in {warning["code"] for warning in collapsed["warnings"]}
    for version in version_rows(store, spec_id):
        forget_record(store, subject_id=spec_id, content_version=version)

    assert _backfill(store) == 0
    record = read_record(store, subject_id=spec_id, content_version=content_version)

    assert record is not None
    assert record.basis == "backfill"
    assert record.steps == minted.steps
    assert record.variables == minted.variables
    assert record.variables is not None
    assert record.variables.dimensions == []


def test_a_party_totals_cut_is_backfilled_from_its_base_without_a_sum_step() -> None:
    """A cut version sums nothing, so its backfilled record keeps its base's steps, as the
    record its mint wrote does."""
    client, store, spec_id, sidecar = party_fixture._rig()
    cut = party_fixture._mint(client, spec_id=spec_id, sidecar=sidecar, options=party_fixture._CUT)
    content_version = cut["content_version"]
    (base_version,) = version_rows(store, spec_id) - {content_version}
    minted = read_record(store, subject_id=spec_id, content_version=content_version)
    assert minted is not None
    assert "sum_variables" not in [step.op for step in minted.known_steps]
    for version in version_rows(store, spec_id):
        forget_record(store, subject_id=spec_id, content_version=version)

    assert (
        run_backfill_descriptions(
            store=store,
            artifact_store=FakeArtifactStore(),
            code_index=party_fixture._code_index(),
            now=lambda: _BACKFILL_AT,
        )
        == 0
    )
    record = read_record(store, subject_id=spec_id, content_version=content_version)
    base = read_record(store, subject_id=spec_id, content_version=base_version)

    assert record is not None and base is not None
    assert record.basis == "backfill"
    assert record.known_steps == base.known_steps
    assert "sum_variables" not in [step.op for step in record.known_steps]


def test_a_collapse_whose_base_failed_waits_for_a_run_that_backfills_the_base(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A collapse whose base failed earlier in the run gets no stepless record: it is logged
    with both content versions and counted as failed, so the next run derives its sum step
    from the base's record."""
    store = FakeVersionRecordDatasetStore()
    collapsed = mint_family(datasets_client(store, fixture="family"), options=GROUPBY_OPTIONS)
    spec_id, content_version = collapsed["definition"]["spec_id"], collapsed["content_version"]
    minted = read_record(store, subject_id=spec_id, content_version=content_version)
    assert minted is not None
    (base_version,) = version_rows(store, spec_id) - {content_version}
    for version in version_rows(store, spec_id):
        forget_record(store, subject_id=spec_id, content_version=version)

    def failing_base(version: Any, **kwargs: Any) -> Any:
        if version.content_version == base_version:
            raise ValueError("a defect in the base's body")
        return backfill_record(version, **kwargs)

    monkeypatch.setattr(maintenance, "backfill_record", failing_base)
    assert _backfill(store) == 0

    assert read_record(store, subject_id=spec_id, content_version=content_version) is None
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        (line["subject_id"], line["content_version"], line["base_content_version"])
        for line in lines
        if line["event"] == "datasets_description_backfill_base_unrecorded"
    ] == [(spec_id, content_version, base_version)]
    assert [
        (line["backfilled"], line["failed"])
        for line in lines
        if line["event"] == "datasets_descriptions_backfilled"
    ] == [(0, 2)]

    monkeypatch.setattr(maintenance, "backfill_record", backfill_record)
    assert _backfill(store) == 0

    record = read_record(store, subject_id=spec_id, content_version=content_version)
    assert record is not None
    assert record.steps == minted.steps


def test_the_maintenance_job_dispatches_backfill_descriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []
    monkeypatch.setattr(maintenance, "_backfill_descriptions", lambda: ran.append("backfill") or 0)

    assert maintenance.main(["backfill-descriptions"]) == 0
    assert ran == ["backfill"]
