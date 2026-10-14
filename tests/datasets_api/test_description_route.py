"""`GET /api/datasets/{id}/{content_version}/description` and its `/values` page, for a `spec_`
version with its `cv_` token and an `ops_` frame with its `ocv_` token.

The route renders on every request and answers with an `ETag` over the rendered body and
`Cache-Control: max-age=60, must-revalidate`. It checks revocation before it compares
`If-None-Match`, including a collapsed version's base, so a 304 never outlives a revocation. Its
body signs no artifact URL, and its data link is the manifest route, which keeps `no-store`.
`/description/values?property=` pages one open dimension's values with their variable counts,
read from the record, refuses a property the record does not hold open, and refuses a revoked
version as the description does. A computed version's data link names its frame without a
signature, and without steps each input's `id` is `null` with a gap. A record that fills late moves the tag, so a conditional GET sees it.

Each test mints or publishes through the client, whose writer stores the version's record. Most
then replace it with this module's record of the version (`_seed`), so the facts they read are
fixed here. A refusal renders at the status the code registry gives it on
`datasets.description`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from datasets_api.description_record import role_dimensions
from datasets_api.description_store import insert_description
from dc_graph import GraphCapExceededError, GraphDeadlineExceededError
from description.facts import (
    Builds,
    CaveatFact,
    CodeIndexStamp,
    ComputedFacts,
    DescriptionRecord,
    DimensionFacts,
    DimensionValue,
    Evidence,
    FixedValue,
    InputVersion,
    SeriesFacts,
    SourceEvidence,
    SourceUnitCount,
    StatusCounts,
    VariableFacts,
)
from description.steps import ComposedUnit, SimpleUnit, VariableSide
from dfo_platform.static_response import encode_json_body
from graph_generation import FakeGeneration
from materialise.statistics import EntityFigure, RoleFigure
from origin_catalogue.service import CatalogueService
from tests.origin_catalogue.fake_catalogue import FakeCatalogue, fake_catalogue_service

from . import test_definition_door as definition_fixture
from ._app_factories import RunningClient
from ._description_doubles import (
    DENOMINATOR,
    FAMILY_AXIS,
    GROUPBY_OPTIONS,
    NUMERATOR,
    OCV,
    OPS_ID,
    datasets_client,
    forget_record,
    mint_definition,
    mint_family,
    publish,
    publish_body,
    quotient_steps,
    revoke,
    version_rows,
)
from ._graph_doubles import FakeMintGraphReader
from ._version_record_store import FakeVersionRecordDatasetStore
from .test_definition_door_currency import _RenderableArtifactStore

_KINDS = ["spec", "ops"]
_RECORDED = datetime(2026, 10, 11, 8, 0, tzinfo=UTC)


def _series() -> SeriesFacts:
    return SeriesFacts(
        series=1,
        rows=1,
        periods=1,
        span=None,
        grain=None,
        latest_period=None,
        series_reaching_latest=None,
        entities_by_role=None,
        by_source_unit=None,
        status=StatusCounts(estimated=None, provisional=None),
    )


def _evidence() -> Evidence:
    return Evidence(
        prepared="2026-10-11",
        title="A seeded version",
        sources=[],
        source_release=None,
        labels=[],
        membership=None,
        builds=Builds(origin_catalogue=None, code_index=None, coverage=None, method_status=None),
    )


def _record(
    subject_id: str, content_version: str, *, dimensions: list[DimensionFacts] | None
) -> DescriptionRecord:
    """The record of a `spec_` version holding `dimensions`, or of an `ops_` frame computed
    from `NUMERATOR` and `DENOMINATOR`."""
    is_spec = subject_id.startswith("spec_")
    return DescriptionRecord(
        id=subject_id,
        content_version=content_version,
        schema_version=1,
        basis="mint" if is_spec else "publish",
        index_stamp=CodeIndexStamp(
            content_hash="seeded", exported_at="2026-10-11", graph_endpoint="exemplar"
        )
        if dimensions is not None
        else None,
        variables=VariableFacts(
            shape=[],
            variable=None,
            fixed=[],
            dimensions=dimensions,
            admitted_at_mint=1,
            admitted_from="resolved_variables",
            with_data=1,
        )
        if is_spec
        else None,
        computed=None
        if is_spec
        else ComputedFacts(
            inputs=[
                InputVersion(id=NUMERATOR[0], content_version=NUMERATOR[1]),
                InputVersion(id=DENOMINATOR[0], content_version=DENOMINATOR[1]),
            ],
            composed=None,
        ),
        series=_series(),
        steps=None,
        caveats=[],
        evidence=_evidence(),
        late_fields=[],
    )


_FAMILY_DIMENSIONS = [
    DimensionFacts(
        property=FAMILY_AXIS,
        within=None,
        values=[
            DimensionValue(value="TestAxisA", variables=1),
            DimensionValue(value="TestAxisB", variables=1),
        ],
        undeclared_variables=0,
        partial=False,
    )
]


def _seed(
    store: FakeVersionRecordDatasetStore,
    subject_id: str,
    content_version: str,
    *,
    dimensions: list[DimensionFacts] | None = None,
) -> None:
    """Store this module's record of the version in place of the one its writer stored."""
    forget_record(store, subject_id=subject_id, content_version=content_version)
    with store.transaction() as tx:
        insert_description(
            tx,
            _record(subject_id, content_version, dimensions=dimensions),
            recorded_at=_RECORDED,
        )


class _Described:
    """A client over one store holding one version of `kind`, and that version's address.
    `seeded` replaces the record its writer stored with this module's (`_seed`). `graph_reader`
    and `catalogue` replace the client's graph and catalogue service (`datasets_client`)."""

    def __init__(
        self,
        kind: str,
        *,
        steps: list[dict[str, object]] | None = None,
        seeded: bool = True,
        graph_reader: FakeMintGraphReader | None = None,
        catalogue: CatalogueService | None = None,
    ) -> None:
        from artifact_store import LiveArtifactInventory

        self.store = FakeVersionRecordDatasetStore()
        self.artifact_store = _RenderableArtifactStore(
            inventory=LiveArtifactInventory(store=self.store)
        )
        self.graph_reader = graph_reader
        self.client = self.client_under(catalogue)
        if kind == "spec":
            manifest = mint_definition(self.client)
            self.id, self.content_version = (
                manifest["definition"]["spec_id"],
                manifest["content_version"],
            )
        else:
            response = publish(self.client, publish_body(steps=steps))
            assert response.status_code == 200, response.text
            self.id, self.content_version = OPS_ID, OCV
        if seeded:
            _seed(self.store, self.id, self.content_version)

    def client_under(self, catalogue: CatalogueService | None) -> RunningClient:
        """A client over this version's store and graph, binding `catalogue` to each request."""
        return datasets_client(
            self.store,
            artifact_store=self.artifact_store,
            with_description_store=True,
            graph_reader=self.graph_reader,
            catalogue=catalogue,
        )

    @property
    def path(self) -> str:
        return f"/api/datasets/{self.id}/{self.content_version}/description"

    def get(self, *, etag: str | None = None):
        headers = {"if-none-match": etag} if etag is not None else {}
        return self.client.get(self.path, headers=headers)

    def body(self) -> dict[str, Any]:
        response = self.get()
        assert response.status_code == 200, response.text
        return response.json()


def _catalogue(build_id: str) -> CatalogueService:
    return fake_catalogue_service(FakeCatalogue(build_id=build_id))


@pytest.mark.parametrize(
    ("generation", "build_id", "moves"),
    [
        pytest.param("gen-1", "build-1", False, id="neither-moves"),
        pytest.param("gen-2", "build-1", True, id="graph-generation-moves"),
        pytest.param("gen-1", "build-2", True, id="catalogue-build-moves"),
    ],
)
def test_a_descriptions_vocabulary_revision_moves_with_the_graph_generation_and_catalogue_build(
    generation: str, build_id: str, moves: bool
) -> None:
    provider = FakeGeneration(token="gen-1")
    # An `ops_` version reads no spec registry, which a second client would not share.
    version = _Described(
        "ops",
        graph_reader=definition_fixture._graph(generation=provider),
        catalogue=_catalogue("build-1"),
    )
    before = version.body()["built_from"]["vocabulary"]

    provider.advance(generation)
    response = version.client_under(_catalogue(build_id)).get(version.path)

    assert response.status_code == 200, response.text
    after = response.json()["built_from"]["vocabulary"]
    assert after.startswith("voc_")
    assert (after != before) is moves


def _directives(response) -> set[str]:
    return {part.strip() for part in response.headers["cache-control"].split(",")}


@pytest.mark.parametrize("kind", _KINDS)
def test_a_description_answers_with_an_etag_over_its_rendered_body(kind: str) -> None:
    version = _Described(kind)

    response = version.get()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["describes"]["id"] == version.id
    assert body["describes"]["content_version"] == version.content_version
    encoded, etag = encode_json_body(body)
    assert response.content == encoded
    # The test client accepts gzip, and a gzip body carries the weak form of the validator.
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["etag"] == f"W/{etag}"


@pytest.mark.parametrize("kind", _KINDS)
def test_a_description_is_cached_for_a_minute_and_then_revalidated(kind: str) -> None:
    response = _Described(kind).get()

    assert response.status_code == 200, response.text
    directives = _directives(response)
    assert {"max-age=60", "must-revalidate"} <= directives
    assert "no-store" not in directives


class _LabelDeadlineGraph(FakeMintGraphReader):
    """Records the deadline of each label read; once `expired`, a label read raises as one
    past its deadline does."""

    def __init__(self) -> None:
        super().__init__()
        self.expired = False
        self.label_deadlines: list[float | None] = []

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        if "name" in properties:
            self.label_deadlines.append(deadline)
            if self.expired:
                raise GraphDeadlineExceededError("the label read passed its deadline")
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


@pytest.mark.parametrize("expired", [False, True], ids=["in-time", "expired"])
def test_the_label_read_is_bounded_by_the_descriptions_graph_deadline(expired: bool) -> None:
    """The label read of a place the record names carries the description's 10 s graph
    deadline, and a read past it degrades the description to `no-store`."""
    graph = _LabelDeadlineGraph()
    version = _Described("ops", graph_reader=graph)
    record = _record(version.id, version.content_version, dimensions=None).model_copy(
        update={"caveats": [CaveatFact(kind="coverage_gap", entities=["country/ZZZ"])]}
    )
    forget_record(version.store, subject_id=version.id, content_version=version.content_version)
    with version.store.transaction() as tx:
        insert_description(tx, record, recorded_at=_RECORDED)
    graph.expired, graph.label_deadlines = expired, []

    started = time.monotonic()
    response = version.get()

    assert response.status_code == 200, response.text
    (deadline,) = graph.label_deadlines
    assert deadline is not None
    assert started < deadline <= time.monotonic() + 10
    assert ("no-store" in _directives(response)) is expired


class _CappedGraph(FakeMintGraphReader):
    """Raises on every node read as a response past the client's byte cap does."""

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        raise GraphCapExceededError("over the byte cap", cap_kind="bytes", cap=1, observed=2)


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(
            {"caveats": [CaveatFact(kind="coverage_gap", entities=["country/ZZZ"])]},
            id="label-read",
        ),
        pytest.param(
            {
                "evidence": _evidence().model_copy(
                    update={
                        "sources": [
                            SourceEvidence(
                                ref="dc/unlistedSource",
                                name=None,
                                licence=None,
                                downloaded=None,
                                pipeline_steps=None,
                            )
                        ]
                    }
                )
            },
            id="source-read",
        ),
    ],
)
def test_a_graph_read_over_its_size_cap_degrades_the_description_to_no_store(
    update: dict[str, Any],
) -> None:
    version = _Described("ops", graph_reader=_CappedGraph())
    record = _record(version.id, version.content_version, dimensions=None).model_copy(update=update)
    forget_record(version.store, subject_id=version.id, content_version=version.content_version)
    with version.store.transaction() as tx:
        insert_description(tx, record, recorded_at=_RECORDED)

    response = version.get()

    assert response.status_code == 200, response.text
    assert "no-store" in _directives(response)


class _LabelReadGraph(FakeMintGraphReader):
    """Records the dcids each label read names, and raises as a capped read does on one
    naming `failing`."""

    def __init__(self, *, failing: str | None) -> None:
        super().__init__()
        self.failing = failing
        self.label_dcids: set[str] = set()

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        if "name" in properties:
            self.label_dcids.update(dcids)
            if self.failing in dcids:
                raise GraphCapExceededError(
                    "over the byte cap", cap_kind="bytes", cap=1, observed=2
                )
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


@pytest.mark.parametrize(
    "failing", [None, "country/FRA"], ids=["counted", "an-unused-ref-would-fail-the-read"]
)
def test_an_input_sends_the_label_read_only_the_refs_its_title_names(failing: str | None) -> None:
    """The numerator's title names its variable, its fixed recipient and the group its
    providers narrow to; its providers and the entity its caveat names join no read, so a
    read that would fail on one leaves the description cacheable."""
    graph = _LabelReadGraph(failing=failing)
    version = _Described("ops", graph_reader=graph)
    providers = DimensionFacts(
        property="financingProvider",
        within="dc/group/DAC",
        values=[
            DimensionValue(value="country/FRA", variables=1),
            DimensionValue(value="country/NOR", variables=1),
        ],
        undeclared_variables=0,
        partial=False,
    )
    numerator = _record(*NUMERATOR, dimensions=[providers])
    assert numerator.variables is not None
    _store_record(
        version,
        numerator.model_copy(
            update={
                "variables": numerator.variables.model_copy(
                    update={
                        "variable": "Amount_Oda",
                        "fixed": [FixedValue(property="financingRecipient", value="country/KEN")],
                    }
                ),
                "caveats": [CaveatFact(kind="coverage_gap", entities=["country/ZZZ"])],
            }
        ),
    )
    _store_record(version, _record(*DENOMINATOR, dimensions=None))

    response = version.get()

    assert response.status_code == 200, response.text
    assert "no-store" not in _directives(response)
    assert graph.label_dcids == {"Amount_Oda", "country/KEN", "dc/group/DAC"}


def test_the_manifest_route_beside_a_description_keeps_no_store() -> None:
    version = _Described("spec")

    description = version.get()
    manifest = version.client.get(f"/api/datasets/{version.id}/{version.content_version}")

    assert description.status_code == manifest.status_code == 200
    assert "no-store" not in _directives(description)
    assert manifest.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("kind", _KINDS)
def test_an_unchanged_description_answers_304_to_its_etag(kind: str) -> None:
    version = _Described(kind)
    etag = version.get().headers["etag"]

    response = version.get(etag=etag)

    assert response.status_code == 304
    assert response.headers["etag"] == etag


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("conditional", [True, False], ids=["if-none-match", "unconditional"])
def test_a_revoked_version_refuses_even_to_its_last_etag(kind: str, conditional: bool) -> None:
    version = _Described(kind)
    etag = version.get().headers["etag"]
    revoke(version.store, subject_id=version.id, content_version=version.content_version)

    response = version.get(etag=etag if conditional else None)

    assert response.status_code == 410, response.text
    assert response.json()["code"] == "snapshot_revoked"


def test_a_collapsed_versions_description_refuses_once_its_base_is_revoked() -> None:
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store, fixture="family", with_description_store=True)
    collapsed = mint_family(client, options=GROUPBY_OPTIONS)
    spec_id = collapsed["definition"]["spec_id"]
    (base,) = version_rows(store, spec_id) - {collapsed["content_version"]}
    _seed(store, spec_id, collapsed["content_version"], dimensions=_FAMILY_DIMENSIONS)
    path = f"/api/datasets/{spec_id}/{collapsed['content_version']}/description"
    first = client.get(path)
    assert first.status_code == 200, first.text
    revoke(store, subject_id=spec_id, content_version=base)

    response = client.get(path, headers={"if-none-match": first.headers["etag"]})

    assert response.status_code == 410, response.text
    assert response.json()["code"] == "snapshot_revoked"


@pytest.mark.parametrize("kind", _KINDS)
def test_a_description_signs_no_artifact_url(kind: str) -> None:
    version = _Described(kind)
    signings = len(version.artifact_store.signings)

    response = version.get()

    assert response.status_code == 200, response.text
    assert len(version.artifact_store.signings) == signings


def test_a_computed_descriptions_data_link_names_its_frame_and_carries_no_signature() -> None:
    body = _Described("ops").body()

    assert body["cite"]["data"] == f"/api/presentation/frames/{OPS_ID}/{OCV}"
    assert "signature" not in body["cite"]["data"].lower()


def test_a_computed_version_without_steps_gives_each_input_a_null_id_with_a_gap() -> None:
    body = _Described("ops", seeded=False).body()

    inputs = body["freshness"]["inputs"]
    assert [(entry["id"], entry["content_version"]) for entry in inputs] == [
        (None, NUMERATOR[1]),
        (None, DENOMINATOR[1]),
    ]
    assert {"freshness.inputs.0.id", "freshness.inputs.1.id"} <= set(body["gaps"])


def _store_record(version: _Described, record: DescriptionRecord) -> None:
    """Store `record` in place of any record its version holds."""
    forget_record(version.store, subject_id=record.id, content_version=record.content_version)
    with version.store.transaction() as tx:
        insert_description(tx, record, recorded_at=_RECORDED)


def test_a_computed_description_names_each_input_by_its_title_and_reads_its_units_variable() -> (
    None
):
    """The numerator's title is composed from its record and the place label read with the
    others; the denominator, whose words compose none, takes the title composed at mint. The
    per-capita unit's population is worded from the same read."""
    graph = FakeMintGraphReader(
        arcs={"country/KEN": {"name": ("Kenya",)}, "Count_Person": {"name": ("Total population",)}}
    )
    version = _Described("ops", graph_reader=graph)
    per_capita = ComposedUnit(
        numerator=SimpleUnit(ref="USDollar"), denominator=VariableSide(stat_var="Count_Person")
    )
    _store_record(
        version,
        _record(version.id, version.content_version, dimensions=None).model_copy(
            update={
                "series": _series().model_copy(
                    update={
                        "by_source_unit": [SourceUnitCount(source=None, unit=per_capita, series=1)]
                    }
                )
            }
        ),
    )
    numerator = _record(*NUMERATOR, dimensions=None)
    assert numerator.variables is not None
    _store_record(
        version,
        numerator.model_copy(
            update={
                "variables": numerator.variables.model_copy(
                    update={
                        "shape": [
                            FixedValue(property="populationType", value="FinancialFlow"),
                            FixedValue(property="measuredProperty", value="amount"),
                        ],
                        "fixed": [
                            FixedValue(property="financingRecipient", value="country/KEN"),
                            FixedValue(property="financialFlowType", value="Disbursement"),
                            FixedValue(property="nettingBasis", value="Gross"),
                        ],
                    }
                )
            }
        ),
    )
    denominator = _record(*DENOMINATOR, dimensions=None)
    _store_record(
        version,
        denominator.model_copy(
            update={"evidence": denominator.evidence.model_copy(update={"title": "Population"})}
        ),
    )

    body = version.body()

    titles = ["Gross disbursements to Kenya", "Population"]
    assert [source["label"] for source in body["sources"]] == titles
    assert [entry["label"] for entry in body["freshness"]["inputs"]] == titles
    assert body["about"]["unit"]["denominator"] == {
        "stat_var": "Count_Person",
        "label": "Total population",
    }
    assert not {"sources.0.label", "sources.1.label", "about.unit.denominator.label"} & set(
        body["gaps"]
    )


def _description_transactions(store: FakeVersionRecordDatasetStore) -> list[int]:
    """Count, from now on, each of `store`'s transactions that reads a description record."""
    counted: list[int] = []
    transaction = store.transaction

    @contextmanager
    def counting():
        with transaction() as tx:
            fetch_all = tx.fetch_all
            reads: list[str] = []

            def recording(sql: str, params=()):
                reads.append(sql)
                return fetch_all(sql, params)

            tx.fetch_all = recording  # type: ignore[method-assign]
            yield tx
        if any("datasets.description_record" in sql for sql in reads):
            counted.append(1)

    store.transaction = counting  # type: ignore[method-assign]
    return counted


def test_a_computed_description_reads_its_inputs_records_in_one_transaction() -> None:
    """The version's own record is one transaction, and every input's another, whatever the
    number of inputs. An input without a record is left out of that read and keeps its place."""
    version = _Described("ops")
    inputs = [("spec_" + f"{n:02x}" * 16, "cv_" + f"{n + 16:02x}" * 16) for n in range(8)]
    record = _record(version.id, version.content_version, dimensions=None)
    assert record.computed is not None
    _store_record(
        version,
        record.model_copy(
            update={
                "computed": record.computed.model_copy(
                    update={"inputs": [InputVersion(id=i, content_version=cv) for i, cv in inputs]}
                )
            }
        ),
    )
    for n, held in enumerate(inputs):
        if n == 3:
            continue
        stored = _record(*held, dimensions=None)
        _store_record(
            version,
            stored.model_copy(
                update={"evidence": stored.evidence.model_copy(update={"title": f"Input {n}"})}
            ),
        )
    transactions = _description_transactions(version.store)

    body = version.body()

    assert len(transactions) == 2
    labels = [entry["label"] for entry in body["freshness"]["inputs"]]
    assert [label for n, label in enumerate(labels) if n != 3] == [
        f"Input {n}" for n in range(8) if n != 3
    ]
    assert labels[3] != "Input 3"


def test_an_open_property_no_index_names_takes_its_graph_label() -> None:
    stat_var, recipient = definition_fixture._STAT_VAR, definition_fixture._RECIPIENT
    # The definition door's graph, with the property's name.
    graph = FakeMintGraphReader(
        arcs={
            stat_var: {
                "typeOf": ("StatisticalVariable",),
                "observationProperties": ("recipient",),
                "measuredProperty": ("amount",),
            },
            "dacTypeOfFinance": {"name": ("DAC type of finance",)},
        },
        facets={(stat_var, recipient): [definition_fixture._FACET]},
        labels={"OECD_DAC2A": "OECD DAC2a", recipient: "Kenya"},
    )
    version = _Described("spec", graph_reader=graph)
    _store_record(
        version,
        _record(
            version.id,
            version.content_version,
            dimensions=[
                DimensionFacts(
                    property="dacTypeOfFinance",
                    within=None,
                    values=[DimensionValue(value="DACTypeOfFinance110", variables=1)],
                    undeclared_variables=0,
                    partial=False,
                )
            ],
        ),
    )

    (varies,) = version.body()["about"]["varies"]

    assert varies["property"] == {"ref": "dacTypeOfFinance", "label": "DAC type of finance"}


def test_a_dataset_descriptions_data_link_is_the_manifest_route() -> None:
    version = _Described("spec")

    body = version.body()

    assert body["cite"]["data"] == f"/api/datasets/{version.id}/{version.content_version}"
    assert version.client.get(body["cite"]["data"]).status_code == 200


def _family_version() -> tuple[FakeVersionRecordDatasetStore, Any, str, str]:
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store, fixture="family", with_description_store=True)
    manifest = mint_family(client, options={"bundle": False})
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    _seed(store, spec_id, content_version, dimensions=_FAMILY_DIMENSIONS)
    return store, client, spec_id, content_version


def _values(client, spec_id: str, content_version: str, **params: object):
    return client.get(
        f"/api/datasets/{spec_id}/{content_version}/description/values", params=params
    )


def test_an_open_dimension_links_to_its_values_page() -> None:
    _store, client, spec_id, content_version = _family_version()

    response = client.get(f"/api/datasets/{spec_id}/{content_version}/description")
    assert response.status_code == 200, response.text
    body = response.json()

    (axis,) = [
        entry for entry in body["about"]["varies"] if entry["property"]["ref"] == FAMILY_AXIS
    ]
    assert axis["values"] == (
        f"/api/datasets/{spec_id}/{content_version}/description/values?property={FAMILY_AXIS}"
    )


def test_the_values_page_lists_a_dimensions_values_with_their_variable_counts() -> None:
    _store, client, spec_id, content_version = _family_version()

    response = _values(client, spec_id, content_version, property=FAMILY_AXIS)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["property"]["ref"] == FAMILY_AXIS
    assert sorted((item["ref"], item["variables"]) for item in body["items"]) == [
        ("TestAxisA", 1),
        ("TestAxisB", 1),
    ]
    assert (body["next_cursor"], body["complete"]) == (None, True)


def test_the_values_page_continues_from_its_cursor_without_repeating_a_value() -> None:
    _store, client, spec_id, content_version = _family_version()

    first = _values(client, spec_id, content_version, property=FAMILY_AXIS, limit=1).json()
    assert first["next_cursor"] is not None
    assert first["complete"] is False
    second = _values(
        client,
        spec_id,
        content_version,
        property=FAMILY_AXIS,
        limit=1,
        cursor=first["next_cursor"],
    ).json()

    assert len(first["items"]) == len(second["items"]) == 1
    assert sorted(item["ref"] for item in first["items"] + second["items"]) == [
        "TestAxisA",
        "TestAxisB",
    ]
    assert (second["next_cursor"], second["complete"]) == (None, True)


def test_the_values_page_pages_an_open_roles_entities_by_the_variables_holding_each() -> None:
    """The record's provider dimension is built from the fold's role figures, as a mint builds
    it for a definition that leaves the provider open."""
    store = FakeVersionRecordDatasetStore()
    client = datasets_client(store, fixture="family", with_description_store=True)
    manifest = mint_family(client, options={"bundle": False})
    spec_id, content_version = manifest["definition"]["spec_id"], manifest["content_version"]
    providers = RoleFigure(
        role="financingProvider",
        values=(
            EntityFigure(ref="dcid:dc/o/p3", variables=1),
            EntityFigure(ref="dcid:dc/o/p1", variables=4),
            EntityFigure(ref="dcid:dc/o/p2", variables=4),
        ),
    )
    _seed(store, spec_id, content_version, dimensions=role_dimensions({}, [providers]))

    first = _values(client, spec_id, content_version, property="financingProvider", limit=2)
    assert first.status_code == 200, first.text
    second = _values(
        client,
        spec_id,
        content_version,
        property="financingProvider",
        limit=2,
        cursor=first.json()["next_cursor"],
    ).json()

    pages = [first.json(), second]
    assert [[(item["ref"], item["variables"]) for item in page["items"]] for page in pages] == [
        [("dc/o/p1", 4), ("dc/o/p2", 4)],
        [("dc/o/p3", 1)],
    ]
    assert first.json()["property"] == {"ref": "financingProvider", "label": "provider"}
    assert [page["complete"] for page in pages] == [False, True]
    assert second["next_cursor"] is None


@pytest.mark.parametrize("sent", ["another_property", "undecodable"])
def test_the_values_page_refuses_a_cursor_issued_for_another_list(sent: str) -> None:
    store, client, spec_id, content_version = _family_version()
    providers = RoleFigure(
        role="financingProvider",
        values=(EntityFigure(ref="dc/o/p1", variables=2), EntityFigure(ref="dc/o/p2", variables=1)),
    )
    _seed(
        store,
        spec_id,
        content_version,
        dimensions=[*role_dimensions({}, [providers]), *_FAMILY_DIMENSIONS],
    )
    issued = _values(client, spec_id, content_version, property="financingProvider", limit=1)
    cursor = issued.json()["next_cursor"] if sent == "another_property" else "not-a-cursor"

    response = _values(
        client, spec_id, content_version, property=FAMILY_AXIS, limit=1, cursor=cursor
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert (body["code"], body["pointer"]) == ("request_invalid", "/cursor")


def test_the_values_page_refuses_a_property_the_record_does_not_hold_open() -> None:
    _store, client, spec_id, content_version = _family_version()

    response = _values(client, spec_id, content_version, property="crsPurpose")

    assert response.status_code == 400, response.text
    body = response.json()
    assert (body["code"], body["pointer"]) == ("request_invalid", "/property")


def test_the_values_page_of_a_revoked_version_refuses_as_its_description_does() -> None:
    store, client, spec_id, content_version = _family_version()
    revoke(store, subject_id=spec_id, content_version=content_version)

    response = _values(client, spec_id, content_version, property=FAMILY_AXIS)

    assert response.status_code == 410, response.text
    assert response.json()["code"] == "snapshot_revoked"


def test_a_late_record_shows_through_a_conditional_get() -> None:
    """A frame published without steps describes itself with `steps: null`. A republish whose
    lineage matches fills them, which moves the tag, so a GET conditional on the old tag
    answers 200 with the steps."""
    version = _Described("ops", seeded=False)
    before = version.get()
    assert before.status_code == 200, before.text
    assert before.json()["steps"] is None
    assert {"freshness.inputs.0.id", "freshness.inputs.1.id"} <= set(before.json()["gaps"])
    steps = quotient_steps()
    assert publish(version.client, publish_body(steps=steps)).status_code == 200

    after = version.get(etag=before.headers["etag"])

    assert after.status_code == 200, after.text
    assert after.headers["etag"] != before.headers["etag"]
    assert [step["id"] for step in after.json()["steps"]] == [step["id"] for step in steps]
    assert [version["id"] for version in after.json()["freshness"]["inputs"]] == [
        NUMERATOR[0],
        DENOMINATOR[0],
    ]
    assert not [gap for gap in after.json()["gaps"] if gap.endswith(".id")]


def _damage_record(
    store: FakeVersionRecordDatasetStore, subject_id: str, content_version: str
) -> None:
    """Leave the version's stored record as JSON that fails validation."""
    with store.transaction() as tx:
        tx.execute(
            "UPDATE datasets.description_record SET record = %s "
            "WHERE subject_id = %s AND content_version = %s",
            ('{"id": "damaged"}', subject_id, content_version),
        )


@pytest.mark.parametrize("kind", ["spec", "ops"])
def test_a_record_that_fails_validation_answers_as_a_version_without_one(kind: str) -> None:
    """The door takes the path of a version holding no record, the refusal the description
    door's mappings give it."""
    damaged, absent = _Described(kind), _Described(kind)
    _damage_record(damaged.store, damaged.id, damaged.content_version)
    forget_record(absent.store, subject_id=absent.id, content_version=absent.content_version)

    answered, expected = damaged.get(), absent.get()

    assert answered.status_code == expected.status_code
    assert answered.json()["code"] == expected.json()["code"]


def test_an_input_whose_record_fails_validation_is_described_as_one_without_a_record() -> None:
    """A computed version still describes itself, wording the damaged input as it words one
    holding no record."""
    damaged, absent = _Described("ops"), _Described("ops")
    for version in (damaged, absent):
        _store_record(version, _record(*NUMERATOR, dimensions=None))
        _store_record(version, _record(*DENOMINATOR, dimensions=None))
    _damage_record(damaged.store, *NUMERATOR)
    forget_record(absent.store, subject_id=NUMERATOR[0], content_version=NUMERATOR[1])

    # Each client's graph stamps its own generation, which only the vocabulary revision names.
    described, expected = damaged.body(), absent.body()
    assert described.pop("built_from")["facts_schema"] == expected.pop("built_from")["facts_schema"]
    assert described == expected
