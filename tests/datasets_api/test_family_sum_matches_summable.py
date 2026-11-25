"""Search's `Slot.summable` and `ResolvedStatVars.sums_components` equal the outcome of the
mint's sum over the same family.

Each family is resolved as the mint resolves it, one of its members holds a row and the
rest hold none, and the mint sums the frame down to one series per year. The sum completes
exactly when Search reads the family's summable axes true. Search serves
`sums_components` true on every declared composite, and the mint collapses each one's
components: a stacked composite's by adding them, the spliced headline's by serving each
year from the one component holding it.
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest

import search.engine.retrieval.bound_values as bound_values_module
from artifact_store import LiveArtifactInventory
from code_index import CodeIndex
from code_index.composites import DECLARED_COMPOSITES
from code_index.declarations import COMPOSITES, DAC2A_BILATERAL
from code_index.shared import get_index
from dataset_store.snapshot import LiveSnapshotReader
from datasets_api import render
from datasets_api.adapter import resolve_definition_members
from datasets_api.config import DatasetsBuildSettings
from datasets_api.description_store import LiveDescriptionStore
from datasets_api.groupby import SpliceCheck, adds_stat_vars
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import ReshapeResult, _RenderReads, _reshape
from datasets_api.version_store import LiveVersionRecordWriter
from dc_sdmx import SdmxClient
from materialise.enrichment import EnrichmentLookup
from materialise.frame_build import Frame
from materialise.reshape import RollupNotSummableError, Table
from search.engine.retrieval.bound_values import apply_slot_summability
from search.models import BaseIdentity, ResolvedStatVars
from search.scope.composite import _stat_vars, served_components
from search.scope.tables import ScopeFacts
from spec_registry import LiveSpecRegistry
from spec_registry.reader import LiveSpecReader
from statvar_members import MemberEnumerator
from tests.code_index.aid_identity import (
    AID_TYPE,
    CRS_MEMBER_COUNT,
    CRS_PROVENANCE,
    IMPUTED_MEMBER_COUNT,
    IMPUTED_PROVENANCE,
    education_aid_identity,
    imputed_education_aid_identity,
)
from tests.engine._builders import unbound_slot, value_slot

from . import test_composite_download as download
from . import test_definition_door as definition_fixture
from ._app_factories import make_client
from ._artifact_doubles import FakeArtifactStore
from ._frame_chunks import frame_of_rows
from ._graph_doubles import FakeMintGraphReader
from ._member_bounds import mint_bounds
from ._version_record_store import FakeVersionRecordDatasetStore
from .conftest import FakeDatasetStore
from .test_definition_door import _client
from .test_definition_door_currency import _FakeHeadStore, _RenderableArtifactStore
from .test_mint_composite import (
    _PROVIDER,
    _RECIPIENT,
    _csv_rows,
    _dac2a_oda_identity,
    _identity,
    _mint,
)
from .test_pin_door import FakeArtifactReferenceWriter

_SUMMABLE_AXES = ("crsCooperationModality", "crsPurpose", "crsTypeOfFinance", "crsTypeOfFlow")
# The imputed five-tuple declares two of the CRS code systems.
_IMPUTED_SUMMABLE_AXES = ("crsPurpose", "crsTypeOfFlow")
_GRAMMAR_COLUMNS = (
    "stat_var",
    "financingProvider",
    "financingRecipient",
    "time_period",
    "value",
    "unit",
    "measurementMethod",
    "provenance",
    "status",
)


def _index_where_the_imputed_dataset_declares_no_absence_reading() -> CodeIndex:
    index = get_index()
    return CodeIndex(
        stamp=index.stamp,
        declarations=tuple(index.declarations.values()),
        codes=index.codes,
        statvars=index.statvars,
        datasets=tuple(
            dataclasses.replace(dataset, absent_observation="unknown")
            if dataset.provenance == IMPUTED_PROVENANCE
            else dataset
            for dataset in index.datasets.values()
        ),
    )


def _mint_sum(
    identity: dict, *, code_index: CodeIndex, populated: int = 1
) -> tuple[str, int, ReshapeResult | None]:
    """The elected dataset, the member count, and the mint's sum of a frame in which
    `populated` members of the family hold a row, or `None` when the sum refuses for a
    missing member."""
    graph = FakeMintGraphReader()
    shapes = resolve_definition_members(
        identity,
        spec_id="spec_test",
        graph=graph,
        enumerator=MemberEnumerator(),
        code_index=code_index,
        bounds=mint_bounds(),
    ).shapes
    dataset = shapes[0].dataset
    assert dataset is not None
    assert {shape.dataset.provenance for shape in shapes if shape.dataset} == {dataset.provenance}
    roles = shapes[0].declared_roles
    table = Table(
        columns=("stat_var", *roles, "time_period", "value", "unit", "provenance"),
        rows=tuple(
            (shape.stat_var, "country/FRA", "country/KEN", "2022", 7.0, "USD", dataset.provenance)
            for shape in shapes[:populated]
        ),
    )
    try:
        summed = _reshape(
            table,
            options=MintOptions(groupby=GroupbyOptions(by=["time_period"], op="sum")),
            entity_columns=roles,
            reads=_RenderReads(
                enrichment=EnrichmentLookup(graph),
                parties={},
                known={},
                deadline=None,
            ),
            code_index=code_index,
            resolved_stat_var_refs=tuple(shape.stat_var for shape in shapes),
            provenance_refs=(dataset.provenance,),
            subject_id="spec_test",
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "missing_members"
        return dataset.provenance, len(shapes), None
    return dataset.provenance, len(shapes), summed


def _search_summable(identity: dict) -> dict[str, bool]:
    slots = [
        value_slot("how", slot["property"], slot["property"], slot["binding"]["ref"])
        if slot["binding"]["state"] == "value"
        else unbound_slot("how", slot["property"], slot["property"])
        for slot in identity["slots"]
    ]
    stamped = apply_slot_summability(slots, BaseIdentity.model_validate(identity))
    return {slot.key.property.ref: slot.summable for slot in stamped if slot.key.property}


@pytest.mark.parametrize(
    ("identity", "edit_index", "provenance", "member_count", "sums"),
    [
        (education_aid_identity(), False, CRS_PROVENANCE, CRS_MEMBER_COUNT, True),
        (
            imputed_education_aid_identity(),
            False,
            IMPUTED_PROVENANCE,
            IMPUTED_MEMBER_COUNT,
            True,
        ),
        (
            imputed_education_aid_identity(),
            True,
            IMPUTED_PROVENANCE,
            IMPUTED_MEMBER_COUNT,
            False,
        ),
    ],
    ids=["crs", "imputed", "imputed-with-no-absence-reading"],
)
def test_search_summable_equals_the_mints_sum_outcome(
    monkeypatch: pytest.MonkeyPatch,
    identity: dict,
    edit_index: bool,
    provenance: str,
    member_count: int,
    sums: bool,
):
    code_index = (
        _index_where_the_imputed_dataset_declares_no_absence_reading()
        if edit_index
        else get_index()
    )
    monkeypatch.setattr(bound_values_module, "get_index", lambda: code_index)

    elected, members, summed = _mint_sum(identity, code_index=code_index)
    flags = _search_summable(identity)

    assert (elected, members) == (provenance, member_count)
    assert (summed is not None) is sums
    expected_axes = _IMPUTED_SUMMABLE_AXES if provenance == IMPUTED_PROVENANCE else _SUMMABLE_AXES
    assert {axis: flags[axis] for axis in flags if axis in _SUMMABLE_AXES} == dict.fromkeys(
        expected_axes, sums
    )
    # CRS rows declare no aid type. The imputed rows' one aid type is not summable.
    if provenance == IMPUTED_PROVENANCE:
        assert flags[AID_TYPE] is False
    else:
        assert AID_TYPE not in flags


def test_the_summed_imputed_frame_is_a_grammar_frame_without_the_classifier_column():
    """Every imputed member carries one aid type, so the decoded classifier column is
    constant across the family and the summed frame leaves it out."""
    identity = imputed_education_aid_identity()

    _elected, _members, summed = _mint_sum(identity, code_index=get_index(), populated=2)

    assert summed is not None
    assert isinstance(summed.table, Frame)
    assert summed.table.columns == _GRAMMAR_COLUMNS
    assert summed.collapse is not None
    assert summed.collapse.frame.columns == _GRAMMAR_COLUMNS
    assert summed.table.rows == (
        (
            "dcid:spec_test",
            "country/FRA",
            "country/KEN",
            "2022",
            14.0,
            "USD",
            None,
            IMPUTED_PROVENANCE,
            None,
        ),
    )


def _composite_case(identity: dict, populated: download._Populated):
    """A case minting `identity`, named by the composite provenance it binds."""
    return pytest.param(identity, populated, id=identity["facets"]["provenance"]["ref"])


_COMPOSITE_CASES = [
    _composite_case(_identity(), download._stacked_crs()),
    _composite_case(_dac2a_oda_identity("Net"), download._stacked_dac2a()),
    _composite_case(
        {
            **_dac2a_oda_identity("Net"),
            "facets": {"provenance": {"state": "value", "ref": DAC2A_BILATERAL}},
        },
        download._stacked_dac2a(),
    ),
    _composite_case(download._headline_identity(), download._spliced_headline()),
]


def test_the_composite_cases_cover_every_declared_composite():
    assert sorted(case.id for case in _COMPOSITE_CASES) == sorted(
        composite.provenance for composite in COMPOSITES
    )


@pytest.mark.parametrize(("identity", "populated"), _COMPOSITE_CASES)
def test_search_sums_components_and_the_mints_time_sum_collapses_every_composite(
    monkeypatch: pytest.MonkeyPatch, identity: dict, populated: download._Populated
):
    """The flag Search serves on every declared composite's restated StatVar counts
    (`_stat_vars`) is true, and the mint's sum over time collapses the composite's components
    with no `rollup_not_summable` refusal: a stacked composite adds them, and the spliced
    headline serves each year from the one component holding it."""
    refused: list[str] = []
    refusal_init = RollupNotSummableError.__init__

    def recording_init(self, *args, axis: str, precondition: str) -> None:
        refused.append(precondition)
        refusal_init(self, *args, axis=axis, precondition=precondition)

    coverage = download._coverage(populated)
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        sdmx_transport=httpx.MockTransport(
            lambda request: download._sdmx_response(populated, request)
        ),
        coverage=coverage,
        artifact_store=FakeArtifactStore(),
        settings=DatasetsBuildSettings.from_env(),
    )
    facts = ScopeFacts(index=get_index(), coverage=coverage.snapshot())
    declaration = get_index().composite(identity["facets"]["provenance"]["ref"])
    assert declaration is not None

    served = _stat_vars(
        served_components(identity, facts=facts),
        ResolvedStatVars(member_count=0),
        declaration,
        facts=facts,
    )
    # The client's construction builds exemplar refusals of its own, so the recording
    # starts once it is built.
    monkeypatch.setattr(RollupNotSummableError, "__init__", recording_init)
    minted = _mint(
        client, identity=identity, options={"groupby": {"by": ["time_period"], "op": "sum"}}
    )

    assert served.sums_components is True
    assert minted.status_code == 200, minted.text
    assert refused == []


def test_a_time_sum_of_the_headline_serves_each_year_from_the_component_holding_it():
    """Net flows hold 100 and grant equivalents 90 in every year from 2012 to 2022. The
    headline keeps net flows to 2017 and grant equivalents from 2018, so its sum over time
    is one line per year of the window, each year valued as its own component."""
    populated = download._spliced_headline()
    artifacts = FakeArtifactStore()
    client = _client(
        graph=FakeMintGraphReader(labels={_PROVIDER: "France"}),
        sdmx_transport=httpx.MockTransport(
            lambda request: download._sdmx_response(populated, request)
        ),
        coverage=download._coverage(populated),
        artifact_store=artifacts,
        settings=DatasetsBuildSettings.from_env(),
    )

    minted = _mint(
        client,
        identity=download._headline_identity(),
        options={"groupby": {"by": ["time_period"], "op": "sum"}},
    )

    assert minted.status_code == 200, minted.text
    rows = _csv_rows(artifacts, minted.json())
    assert sorted((row["time_period"], float(row["value"])) for row in rows) == [
        ("2014", 100.0),
        ("2015", 100.0),
        ("2016", 100.0),
        ("2017", 100.0),
        ("2018", 90.0),
        ("2019", 90.0),
        ("2020", 90.0),
    ]


_HEADLINE_COLUMNS = ("stat_var", "financingProvider", "time_period", "value", "unit", "provenance")


def _headline_rows(*years_by_member: tuple[str, int, tuple[int, ...]]) -> tuple[tuple, ...]:
    """Rows of the headline frame: each member's value in each of its years."""
    return tuple(
        (f"dcid:{member}", _PROVIDER, str(year), float(value), "USD", download._HEADLINE)
        for member, value, years in years_by_member
        for year in years
    )


def _headline_time_sum(monkeypatch: pytest.MonkeyPatch, rows: tuple[tuple, ...], *, streamed: bool):
    """The mint's sum over time of the headline frame of `rows`, read as one table or as the
    frame slices of one row each a streamed render folds."""
    frame = frame_of_rows(_HEADLINE_COLUMNS, rows)
    members = (f"dcid:{download._NET}", f"dcid:{download._GRANT_EQUIVALENT}")
    if streamed:
        monkeypatch.setattr(render, "_RESHAPE_CHUNK_ROWS", 1)
        chunks = list(render._reslice_chunks([frame]))
        table, frame_chunks = chunks[0], iter(chunks)
    else:
        table, frame_chunks = Table(columns=frame.columns, rows=frame.rows), None
    return _reshape(
        table,
        options=MintOptions(groupby=GroupbyOptions(by=["time_period"], op="sum")),
        entity_columns=("financingProvider",),
        reads=_RenderReads(
            enrichment=EnrichmentLookup(FakeMintGraphReader()), parties={}, known={}, deadline=None
        ),
        code_index=get_index(),
        resolved_stat_var_refs=members,
        provenance_refs=(download._HEADLINE,),
        subject_id="spec_test",
        frame_chunks=frame_chunks,
        pivot_max_output_cells=5_000_000,
    )


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_a_headline_frame_whose_components_share_a_year_is_refused(
    monkeypatch: pytest.MonkeyPatch, streamed: bool
):
    """The headline's sum over time splices disjoint years into one line, and refuses
    `rollup_not_summable` (`spliced_overlap`) for a frame in which net flows and grant
    equivalents both hold 2018, rather than adding the two."""
    spliced = _headline_time_sum(
        monkeypatch,
        _headline_rows(
            (download._NET, 100, (2016, 2017)), (download._GRANT_EQUIVALENT, 90, (2018,))
        ),
        streamed=streamed,
    )
    year, value = spliced.table.columns.index("time_period"), spliced.table.columns.index("value")
    assert sorted((row[year], row[value]) for row in spliced.table.rows) == [
        ("2016", 100.0),
        ("2017", 100.0),
        ("2018", 90.0),
    ]

    with pytest.raises(RollupNotSummableError) as refused:
        _headline_time_sum(
            monkeypatch,
            _headline_rows(
                (download._NET, 100, (2017, 2018)), (download._GRANT_EQUIVALENT, 90, (2018,))
            ),
            streamed=streamed,
        )

    assert (refused.value.code, refused.value.axis, refused.value.precondition) == (
        "rollup_not_summable",
        "stat_var",
        "spliced_overlap",
    )


def _headline_check(*, missing_reads_zero: bool) -> SpliceCheck:
    """A check of the headline frame over an index where net flows hold two members with the
    same codes, the second a copy of the first under another dcid."""
    index = get_index()
    net = index.statvar_by_dcid(download._NET)
    assert net is not None
    twin_index = CodeIndex(
        stamp=index.stamp,
        declarations=tuple(index.declarations.values()),
        codes=index.codes,
        statvars=(*index.statvars, dataclasses.replace(net, dcid="TwinNet")),
        datasets=tuple(index.datasets.values()),
    )
    members = (f"dcid:{download._NET}", "dcid:TwinNet", f"dcid:{download._GRANT_EQUIVALENT}")
    return SpliceCheck(
        DECLARED_COMPOSITES[download._HEADLINE],
        index=twin_index,
        cell_columns=("financingProvider", "time_period"),
        members=members,
        lookup={member: {} for member in members},
        decoded_code_systems=(),
        missing_reads_zero=missing_reads_zero,
    )


def test_a_spliced_cell_lacking_a_member_of_its_own_component_is_refused():
    """A cell expects every member of the component holding it and none of the others'."""
    complete = _headline_check(missing_reads_zero=False)
    complete.note(
        [f"dcid:{download._NET}", "dcid:TwinNet", f"dcid:{download._GRANT_EQUIVALENT}"],
        [(_PROVIDER, "2017"), (_PROVIDER, "2017"), (_PROVIDER, "2018")],
    )
    complete.refuse()
    lacking = [f"dcid:{download._NET}", f"dcid:{download._GRANT_EQUIVALENT}"]
    cells = [(_PROVIDER, "2017"), (_PROVIDER, "2018")]
    read_as_zero = _headline_check(missing_reads_zero=True)
    read_as_zero.note(lacking, cells)
    read_as_zero.refuse()

    incomplete = _headline_check(missing_reads_zero=False)
    incomplete.note(lacking, cells)
    with pytest.raises(RollupNotSummableError) as refused:
        incomplete.refuse()

    assert refused.value.precondition == "missing_members"
    assert "dcid:TwinNet" in str(refused.value)


def _time_sum_description(identity: dict, populated: download._Populated) -> dict:
    """The description record of the version a sum over time of `identity` mints over the
    rows `populated` holds."""
    store = FakeVersionRecordDatasetStore()
    registry = FakeDatasetStore()
    client = make_client(
        registry=LiveSpecRegistry(store=registry),
        spec_reader=LiveSpecReader(store=registry),
        snapshot_reader=LiveSnapshotReader(store=store),
        graph=FakeMintGraphReader(labels={_PROVIDER: "France", _RECIPIENT: "Kenya"}),
        sdmx=SdmxClient(
            base_url=definition_fixture._BASE_URL,
            transport=httpx.MockTransport(
                lambda request: download._sdmx_response(populated, request)
            ),
        ),
        artifact_store=_RenderableArtifactStore(inventory=LiveArtifactInventory(store=store)),
        version_writer=LiveVersionRecordWriter(store=store),
        head_writer=_FakeHeadStore(),
        reference_writer=FakeArtifactReferenceWriter(),
        coverage=download._coverage(populated),
        description_store=LiveDescriptionStore(store=store),
    )
    minted = _mint(
        client, identity=identity, options={"groupby": {"by": ["time_period"], "op": "sum"}}
    )
    assert minted.status_code == 200, minted.text
    manifest = minted.json()
    described = client.get(
        f"/api/datasets/{manifest['definition']['spec_id']}/{manifest['content_version']}"
        "/description"
    )
    assert described.status_code == 200, described.text
    return described.json()


def test_a_headline_time_sum_records_no_sum_and_a_stacked_one_records_its_sum():
    """The headline's splice adds no two values, so its version records no `sum_variables`
    step; the CRS inclusive composite's sum adds its components and records one."""
    spliced = _time_sum_description(download._headline_identity(), download._spliced_headline())
    stacked = _time_sum_description(_identity(), download._stacked_crs())

    assert [step["op"] for step in spliced["steps"]] == []
    assert [step["op"] for step in stacked["steps"]] == ["sum_variables"]


@pytest.mark.parametrize(
    ("composite", "adds"),
    [(None, True), (DECLARED_COMPOSITES[download._HEADLINE], False)],
    ids=["no-composite", "headline"],
)
def test_the_headline_s_two_members_add_only_outside_the_splice(composite, adds: bool):
    """Net flows and grant equivalents hold one tuple of kept codes, so outside a composite
    their collapse adds them; keyed on their headline components, it adds nothing, which the
    analysis door's summed count reads as well."""
    assert (
        adds_stat_vars(
            [download._NET, download._GRANT_EQUIVALENT],
            index=get_index(),
            kept=(),
            levels={},
            components=False,
            composite=composite,
        )
        is adds
    )
