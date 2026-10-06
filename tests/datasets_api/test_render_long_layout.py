"""Tests that `render_and_store` writes long output in the published layout
(`materialise.projection`) while groupby collapse frames keep their grammar columns and
content versions: names beside every code, the split unit, and failed or missing names left
empty with a warning. Every name read of one render shares one deadline and stops after the
first failure. A frame-layout request skips the long layout and reads no names."""

from __future__ import annotations

import csv
import io
import time
from collections import Counter
from collections.abc import Sequence
from types import SimpleNamespace

import openpyxl
import pyarrow.parquet as pq
import pytest
from dfo_frame_grammar.grammar import UnitRequest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.shared import get_index
from code_index.types import StatVarRow
from datasets_api import render as render_module
from datasets_api.manifest_models import CitationSource
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import RenderedDefinition, render_and_store
from dc_graph import GraphDeadlineExceededError, GraphReadError
from materialise.content_version import MintPreimage
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import Table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-09T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)


def _code(dcid: str, code_system: str, code: str, label: str) -> CodeRow:
    return CodeRow(
        dcid=dcid, code_system=code_system, code=code, label=label, synonyms=(), parent=None
    )


_HEALTH = _code("dcid:CRSPurposeCode122", "crsPurpose", "122", "Basic health care")
_MALARIA = _code("dcid:CRSPurposeCode12250", "crsPurpose", "12250", "Malaria control")
_GRANT = _code("dcid:CRSTypeOfFinanceCode110", "crsTypeOfFinance", "110", "Standard grant")
_LOAN = _code("dcid:CRSTypeOfFinanceCode421", "crsTypeOfFinance", "421", "Standard loan")


def _declaration(code_system: str) -> CodeSystemDeclaration:
    return CodeSystemDeclaration(
        code_system=code_system,
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name=code_system,
    )


def _statvar(dcid: str, **constraints: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties=constraints,
    )


_INDEX = CodeIndex(
    stamp=_STAMP,
    declarations=(_declaration("crsPurpose"), _declaration("crsTypeOfFinance")),
    codes=(_HEALTH, _MALARIA, _GRANT, _LOAN),
    statvars=(
        _statvar("SV_A", crsPurpose=_HEALTH.dcid, crsTypeOfFinance=_GRANT.dcid),
        _statvar("SV_B", crsPurpose=_HEALTH.dcid, crsTypeOfFinance=_LOAN.dcid),
        _statvar("SV_C", crsPurpose=_MALARIA.dcid, crsTypeOfFinance=_GRANT.dcid),
        _statvar("SV_D", crsPurpose=_MALARIA.dcid),
    ),
)

_ENTITY_COLUMNS = ("financingProvider", "financingRecipient")
_FRAME_COLUMNS = (
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
_COLLAPSE_ROWS = (
    ("dcid:SV_A", "country/USA", "country/KEN", "2022", 10.0, "USD", None, "p1", None),
    ("dcid:SV_B", "country/USA", "country/KEN", "2022", 5.0, "USD", None, "p1", None),
    ("dcid:SV_C", "country/USA", "country/KEN", "2022", 7.0, "USD", None, "p1", None),
)
_ROWS = (
    *_COLLAPSE_ROWS,
    ("dcid:SV_D", "country/USA", "country/KEN", "2022", 3.0, "USD", None, "p1", None),
)
_ARCS = {
    "country/USA": {"name": ("United States",)},
    "country/KEN": {"name": ("Kenya",), "countryAlpha3Code": ("KEN",)},
    "p1": {"name": ("Source one",)},
    "WDI_GDP": {"name": ("GDP",)},
}
_GRAPH = FakeMintGraphReader(arcs=_ARCS)
_PREIMAGE = MintPreimage(
    graph_revision="rev-1",
    source_revisions=({"source": "OECD_CRS", "revision": "2026-09-01"},),
    facets=({"axis": "provenance", "ref": "p1", "basis": "bound"},),
    conversion=None,
)
_BY_PURPOSE = GroupbyOptions(by=["financingProvider", "financingRecipient", "crsPurpose"], op="sum")

_SITE_POST = [*_FRAME_COLUMNS, "financingProvider_label"]
"""What the website's `mintColumns` posted once a viewer ticked "Entity names": the output
profile's frame columns, `status` included, then the token."""

_LONG_HEADER = (
    "stat_var",
    "financingProvider",
    "financingProvider_label",
    "financingRecipient",
    "financingRecipient_label",
    "crsPurpose",
    "crsPurpose_label",
    "crsTypeOfFinance",
    "crsTypeOfFinance_label",
    "time_period",
    "value",
    "unit",
    "unit_label",
    "currency",
    "price_basis",
    "base_year",
    "measurementMethod",
    "measurementMethod_label",
    "provenance",
    "provenance_label",
)
_COLLAPSE_HEADER = (*_LONG_HEADER[:7], *_LONG_HEADER[9:])
"""`_LONG_HEADER` after a groupby by purpose sums over the type of finance."""


def _render(
    options: MintOptions,
    *,
    rows: Sequence[tuple[object, ...]] = _ROWS,
    chunks: Sequence[Sequence[tuple[object, ...]]] | None = None,
    graph: FakeMintGraphReader = _GRAPH,
    index: CodeIndex = _INDEX,
    citations: dict[str, CitationSource] | None = None,
    deadline: float | None = None,
    spec: dict[str, object] | None = None,
) -> tuple[RenderedDefinition, bytes]:
    chunks = chunks if chunks is not None else (rows,)
    rows = [row for chunk in chunks for row in chunk]
    store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=iter(Table(columns=_FRAME_COLUMNS, rows=tuple(chunk)) for chunk in chunks),
        options=options,
        spec=spec or {"spec_id": "family"},
        provenance_refs=["p1"],
        citations=citations or {},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        prepared_at=PREPARED_AT,
        site=SITE,
        artifact_store=store,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=EnrichmentLookup(graph),
        pivot_max_output_cells=5_000_000,
        code_index=index,
        resolved_stat_var_refs=tuple(str(row[0]) for row in rows),
        groupby_preimage=_PREIMAGE,
        deadline=deadline,
        metadata_read_seconds=1.0,
    )
    return rendered, store.fetch(rendered.artifact.url)


def _csv(body: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(body.decode("utf-8"))))


def _records(body: bytes) -> list[dict[str, str]]:
    header, *rows = _csv(body)
    return [dict(zip(header, row, strict=True)) for row in rows]


def _header(format_: str, body: bytes) -> tuple[str, ...]:
    if format_ == "csv":
        return tuple(_csv(body)[0])
    if format_ == "parquet":
        return tuple(pq.read_table(io.BytesIO(body)).column_names)
    sheet = openpyxl.load_workbook(io.BytesIO(body), read_only=True).worksheets[0]
    return tuple(next(sheet.iter_rows(max_row=1, values_only=True)))


def _unavailable(rendered: RenderedDefinition) -> list[tuple[object, object]]:
    return [
        (warning.get("column"), warning.get("count"))
        for warning in rendered.warnings
        if warning["code"] == "enrichment_unavailable"
    ]


@pytest.mark.parametrize("format_", ["csv", "xlsx", "parquet"])
def test_every_format_writes_the_long_layout(format_):
    _rendered, body = _render(MintOptions(format=format_, bundle=False))

    assert _header(format_, body) == _LONG_HEADER


@pytest.mark.parametrize(
    ("method", "method_arcs", "method_label"),
    [
        pytest.param(None, {}, "", id="no-method"),
        pytest.param(
            "MOSPIProvisionalEstimates",
            {"name": ("MOSPI Provisional Estimates",)},
            "MOSPI Provisional Estimates",
            id="method-with-a-name",
        ),
        pytest.param(
            "WorldBank_WDI_CSV", {"name": ("WorldBank_WDI_CSV",)}, "", id="method-named-by-its-dcid"
        ),
        pytest.param("GrantEquivalent", {}, "", id="method-without-a-name"),
    ],
)
def test_every_code_gets_its_name_and_the_render_warns_of_nothing(
    method, method_arcs, method_label
):
    rows = [(*row[:6], method, *row[7:]) for row in _ROWS]
    graph = FakeMintGraphReader(arcs={**_ARCS, **({method: method_arcs} if method else {})})

    rendered, body = _render(MintOptions(format="csv", bundle=False), rows=rows, graph=graph)
    first = _records(body)[0]

    assert (first["financingProvider_label"], first["financingRecipient_label"]) == (
        "United States",
        "Kenya",
    )
    assert first["provenance_label"] == "Source one"
    # A method shows only a name that says more than its dcid. An unnamed method is not a miss.
    assert (first["measurementMethod"], first["measurementMethod_label"]) == (
        method or "",
        method_label,
    )
    assert _unavailable(rendered) == []
    assert rendered.labels_complete


def test_a_cited_source_takes_its_cited_name_over_the_graph_name():
    _rendered, body = _render(
        MintOptions(format="csv", bundle=False),
        citations={"p1": CitationSource(ref="p1", label="OECD Creditor Reporting System")},
    )

    assert {record["provenance_label"] for record in _records(body)} == {
        "OECD Creditor Reporting System"
    }


_NOMINAL_SHAPE_SPEC: dict[str, object] = {
    "spec_id": "gdp",
    "identity": {"shape": {"kind": "statvar_schema", "measurement_qualifier": "Nominal"}},
}
"""A described identity whose shape states current prices, over StatVars the code index does
not hold."""


@pytest.mark.parametrize(
    ("unit", "stat_var", "spec", "expected"),
    [
        pytest.param(
            "USD", None, None, ("USD", "Current US dollars", "USD", "current", ""), id="native"
        ),
        pytest.param(
            "undata/UNIT_MEASURE-CR_USD",
            None,
            None,
            ("undata/UNIT_MEASURE-CR_USD", "Current US dollars", "USD", "current", ""),
            id="declared-un-dollars",
        ),
        pytest.param(
            "undata/UNIT_MEASURE-CR_USD_K",
            None,
            None,
            ("undata/UNIT_MEASURE-CR_USD_K", "Current US dollars", "USD", "current", ""),
            id="declared-un-thousand-dollars",
        ),
        pytest.param(
            "ChainedUSDollarBaseYr2017",
            None,
            None,
            ("ChainedUSDollarBaseYr2017", "Constant 2017 US dollars", "USD", "constant", "2017"),
            id="declared-chained-2017",
        ),
        pytest.param(
            "USDollar",
            "dcid:WDI_GDP",
            _NOMINAL_SHAPE_SPEC,
            ("USDollar", "Current US dollars", "USD", "current", ""),
            id="native-outside-the-index",
        ),
        pytest.param(
            "INR",
            "dcid:WDI_GDP",
            _NOMINAL_SHAPE_SPEC,
            ("INR", "Current Indian rupees", "INR", "current", ""),
            id="native-graph-currency-outside-the-index",
        ),
        pytest.param(
            UnitRequest(currency="USD", prices="constant", base_year=2023),
            None,
            None,
            ("CurrencyUSD", "Constant 2023 US dollars", "USD", "constant", "2023"),
            id="converted-constant",
        ),
        pytest.param(
            UnitRequest(currency="KES", prices="current"),
            None,
            None,
            ("CurrencyKES", "Current Kenyan shillings", "KES", "current", ""),
            id="converted-current",
        ),
        pytest.param(
            "Percent", None, None, ("Percent", "Percent", "", "", ""), id="unnamed-non-currency"
        ),
    ],
)
def test_the_unit_splits_into_node_label_currency_basis_and_year(unit, stat_var, spec, expected):
    rows = [(stat_var or row[0], *row[1:5], unit, *row[6:]) for row in _ROWS]

    rendered, body = _render(MintOptions(format="csv", bundle=False), rows=rows, spec=spec)

    columns = ("unit", "unit_label", "currency", "price_basis", "base_year")
    assert {tuple(record[c] for c in columns) for record in _records(body)} == {expected}
    assert _unavailable(rendered) == []


def test_a_statvar_without_a_value_on_an_axis_gets_an_empty_code_and_the_absent_reading():
    _rendered, body = _render(MintOptions(format="csv", bundle=False))
    by_stat_var = {record["stat_var"]: record for record in _records(body)}

    assert by_stat_var["dcid:SV_A"]["crsTypeOfFinance"] == _GRANT.dcid
    assert by_stat_var["dcid:SV_A"]["crsTypeOfFinance_label"] == "Standard grant"
    assert by_stat_var["dcid:SV_D"]["crsTypeOfFinance"] == ""
    assert by_stat_var["dcid:SV_D"]["crsTypeOfFinance_label"] == "not_declared"


@pytest.mark.parametrize(
    ("groupby", "rows", "columns", "header"),
    [
        pytest.param(None, _ROWS, _SITE_POST, _LONG_HEADER, id="frame-columns-and-token"),
        pytest.param(
            _BY_PURPOSE, _COLLAPSE_ROWS, _SITE_POST, _COLLAPSE_HEADER, id="groupby-collapse"
        ),
        pytest.param(None, _ROWS, ["financingProvider_label"], _LONG_HEADER, id="label-token"),
        pytest.param(
            None,
            _ROWS,
            ["financingRecipient_iso3", "financingProvider_label"],
            (*_LONG_HEADER[:5], "financingRecipient_iso3", *_LONG_HEADER[5:]),
            id="iso3-and-label-tokens",
        ),
        # A selection that omits `financingProvider` but names its label keeps both.
        pytest.param(
            None,
            _ROWS,
            [
                "stat_var",
                "financingRecipient",
                "time_period",
                "value",
                "unit",
                "provenance",
                "financingProvider_label",
            ],
            (*_LONG_HEADER[:16], *_LONG_HEADER[18:]),
            id="selection-naming-a-label-keeps-its-entity",
        ),
    ],
)
def test_a_posted_label_token_selects_nothing_beyond_its_entity(groupby, rows, columns, header):
    rendered, body = _render(
        MintOptions(format="csv", bundle=False, groupby=groupby, columns=columns), rows=rows
    )

    assert tuple(_csv(body)[0]) == header
    assert _unavailable(rendered) == []


def test_a_groupby_collapse_keeps_its_frame_and_content_version():
    rendered, body = _render(
        MintOptions(format="csv", bundle=False, groupby=_BY_PURPOSE), rows=_COLLAPSE_ROWS
    )

    assert rendered.collapse is not None
    # Recorded before the long layout existed: the collapse frame and its identity are the
    # grammar's, untouched by the written columns.
    assert rendered.collapse.content_version == "cv_dd0e82d2ca72ded72fc3034df1702c37"
    assert rendered.collapse.frame.columns == (
        "stat_var",
        "financingProvider",
        "financingRecipient",
        "stat_var_crsPurpose",
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    header, *rows = _csv(body)
    assert tuple(header) == _COLLAPSE_HEADER
    assert [row[5:8] for row in rows] == [
        [_HEALTH.dcid, "Basic health care", "2022"],
        [_MALARIA.dcid, "Malaria control", "2022"],
    ]
    # The summed family's own native unit keeps the price basis its members share.
    assert {row[header.index("price_basis")] for row in rows} == {"current"}


class _FailingGraph(FakeMintGraphReader):
    """A graph whose batched arc read fails, recording the deadline each read carried. With a
    `clock`, a read holds until its deadline before failing, as one does in a graph outage."""

    def __init__(self, error: GraphReadError, *, clock: _Clock | None = None) -> None:
        super().__init__()
        self._error = error
        self._clock = clock
        self.deadlines: list[float | None] = []

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self.deadlines.append(deadline)
        if self._clock is not None and deadline is not None:
            self._clock.now = max(self._clock.now, deadline)
        raise self._error


class _SlowGraph(FakeMintGraphReader):
    """A graph whose batched arc read answers after `read_seconds` on `clock`, recording the
    deadline each read carried."""

    def __init__(self, *, clock: _Clock, read_seconds: float) -> None:
        super().__init__(arcs={**_ARCS, "country/GHA": {"name": ("Ghana",)}})
        self._clock = clock
        self._read_seconds = read_seconds
        self.deadlines: list[float | None] = []

    def node_arcs_many(self, dcids, *, properties, individual_probe_budget=None, deadline=None):
        self.deadlines.append(deadline)
        self._clock.now += self._read_seconds
        return super().node_arcs_many(
            dcids,
            properties=properties,
            individual_probe_budget=individual_probe_budget,
            deadline=deadline,
        )


class _Clock:
    """The monotonic clock `datasets_api.render` reads, moved only by the graph doubles."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(render_module, "time", SimpleNamespace(monotonic=fake.monotonic))
    return fake


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(GraphReadError("graph unavailable"), id="read-error"),
        pytest.param(GraphDeadlineExceededError("deadline passed"), id="deadline"),
    ],
)
def test_a_failed_name_read_leaves_name_cells_empty_warns_and_completes(error):
    graph = _FailingGraph(error)

    rendered, body = _render(
        MintOptions(format="csv", bundle=False), graph=graph, deadline=time.monotonic()
    )

    records = _records(body)
    assert {record["financingProvider_label"] for record in records} == {""}
    assert {record["provenance_label"] for record in records} == {""}
    # The unit's label is local, so it survives the failed read.
    assert {record["unit_label"] for record in records} == {"Current US dollars"}
    assert _unavailable(rendered) == [
        ("financingProvider_label", 4),
        ("financingRecipient_label", 4),
        ("provenance_label", 4),
    ]
    # One read, bounded by the metadata deadline, and the artifact is not linked.
    assert len(graph.deadlines) == 1
    assert graph.deadlines[0] is not None
    assert not rendered.labels_complete


def test_a_label_two_codes_of_a_system_share_leaves_the_code_empty_and_warns():
    twin = _code("dcid:CRSPurposeCode99999", "crsPurpose", "99999", "Basic health care")
    index = CodeIndex(
        stamp=_STAMP,
        declarations=_INDEX.declarations.values(),
        codes=(*_INDEX.codes, twin),
        statvars=_INDEX.statvars,
    )

    rendered, body = _render(MintOptions(format="csv", bundle=False), index=index)

    by_stat_var = {record["stat_var"]: record for record in _records(body)}
    assert by_stat_var["dcid:SV_A"]["crsPurpose"] == ""
    assert by_stat_var["dcid:SV_A"]["crsPurpose_label"] == "Basic health care"
    assert by_stat_var["dcid:SV_C"]["crsPurpose"] == _MALARIA.dcid
    assert _unavailable(rendered) == [("crsPurpose", 2)]


def test_no_packaged_code_system_gives_two_codes_one_label():
    """An axis code column reads its code back from the decoded label, so a label two codes
    of one declared system share leaves those rows without a code."""
    index = get_index()
    labels = Counter(
        (row.code_system, row.label) for row in index.codes if row.code_system in index.declarations
    )

    assert [key for key, count in labels.items() if count > 1] == []


def test_a_decode_gap_warning_names_the_label_column():
    gap_index_statvar = "dcid:SV_GAP"
    rows = (
        (gap_index_statvar, "country/USA", "country/KEN", "2022", 1.0, "USD", None, "p1", None),
    )
    index = CodeIndex(
        stamp=_STAMP,
        declarations=(_declaration("crsPurpose"),),
        codes=(_HEALTH,),
        statvars=(_statvar("SV_GAP", crsPurpose="dcid:CRSPurposeCode99999"),),
    )

    rendered, _body = _render(MintOptions(format="csv", bundle=False), rows=rows, index=index)

    assert _unavailable(rendered) == [("crsPurpose_label", 1)]


def test_every_chunk_of_a_render_reads_names_against_one_deadline(clock):
    """A later chunk's new names are read against the deadline the render started with, so a
    slow first read leaves later reads less time."""
    graph = _SlowGraph(clock=clock, read_seconds=5.0)
    later = [(*row[:2], "country/GHA", *row[3:]) for row in _ROWS]

    rendered, _body = _render(
        MintOptions(format="csv", bundle=False),
        chunks=(_ROWS, later),
        graph=graph,
        deadline=clock.now,
    )

    # The render's start plus `metadata_read_seconds`, for both chunks' reads.
    assert graph.deadlines == [1_001.0, 1_001.0]
    assert rendered.labels_complete


@pytest.mark.parametrize("bundle", [False, True], ids=["bare", "bundle"])
def test_a_render_stops_reading_names_after_the_first_failed_read(clock, bundle):
    """A graph outage costs a render one read's wait, whatever its chunk count: no later
    chunk with new names, and no bundle codes table, reads again."""
    graph = _FailingGraph(GraphDeadlineExceededError("deadline passed"), clock=clock)
    started = clock.now
    later = [(*row[:2], "country/GHA", *row[3:]) for row in _ROWS]

    rendered, _body = _render(
        MintOptions(format="csv", bundle=bundle),
        chunks=(_ROWS, later),
        graph=graph,
        deadline=started,
    )

    assert graph.deadlines == [started + 1.0]
    assert clock.now == started + 1.0
    assert not rendered.labels_complete


_FRAME_HEADER = (*_FRAME_COLUMNS, "stat_var_crsPurpose", "stat_var_crsTypeOfFinance")
_AXIS_LABELS = (
    ("Basic health care", "Standard grant"),
    ("Basic health care", "Standard loan"),
    ("Malaria control", "Standard grant"),
    ("Malaria control", "not_declared"),
)
_CONSTANT_2023_USD = UnitRequest(currency="USD", prices="constant", base_year=2023)


@pytest.mark.parametrize(
    ("extra", "rows", "header", "expected"),
    [
        pytest.param(
            {},
            _ROWS,
            _FRAME_HEADER,
            [(*row, *labels) for row, labels in zip(_ROWS, _AXIS_LABELS, strict=True)],
            id="decoded-axes",
        ),
        pytest.param(
            {
                "groupby": {
                    "by": ["financingProvider", "financingRecipient", "crsPurpose"],
                    "op": "sum",
                }
            },
            _COLLAPSE_ROWS,
            (*_FRAME_COLUMNS[:3], "stat_var_crsPurpose", *_FRAME_COLUMNS[3:]),
            [
                (
                    "dcid:family",
                    "country/USA",
                    "country/KEN",
                    "Basic health care",
                    "2022",
                    15.0,
                    "USD",
                    None,
                    "p1",
                    None,
                ),
                (
                    "dcid:family",
                    "country/USA",
                    "country/KEN",
                    "Malaria control",
                    "2022",
                    7.0,
                    "USD",
                    None,
                    "p1",
                    None,
                ),
            ],
            id="summed",
        ),
        pytest.param(
            {},
            [(*row[:5], _CONSTANT_2023_USD, *row[6:]) for row in _ROWS],
            _FRAME_HEADER,
            [
                (
                    *row[:5],
                    '{"base_year":2023,"currency":"USD","prices":"constant"}',
                    *row[6:],
                    *labels,
                )
                for row, labels in zip(_ROWS, _AXIS_LABELS, strict=True)
            ],
            id="converted-unit",
        ),
        pytest.param(
            {"columns": ["financingRecipient_iso3"]},
            _ROWS,
            (*_FRAME_HEADER, "financingRecipient_iso3"),
            [(*row, *labels, "KEN") for row, labels in zip(_ROWS, _AXIS_LABELS, strict=True)],
            id="enrichment-column",
        ),
    ],
)
def test_a_frame_layout_file_holds_the_frame_columns_and_cells(extra, rows, header, expected):
    """The columns and cells a bare parquet file held before the long layout, which the
    pre-layout writer produced byte for byte for the same frames. Its rendering is the request
    options as sent, with no layout version stamped."""
    options = {"format": "parquet", "bundle": False, "layout": "frame", **extra}

    rendered, body = _render(MintOptions.model_validate(options), rows=rows)

    table = pq.read_table(io.BytesIO(body))
    assert tuple(table.column_names) == header
    assert [tuple(row.values()) for row in table.to_pylist()] == expected
    assert rendered.artifact.rendering == options


def test_a_frame_layout_render_reads_no_names():
    graph = _FailingGraph(GraphReadError("graph unavailable"))

    rendered, _body = _render(
        MintOptions(format="parquet", bundle=False, layout="frame"), graph=graph
    )

    assert graph.deadlines == []
    assert rendered.labels_complete
    assert _unavailable(rendered) == []
