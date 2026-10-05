"""A groupby collapse renders to identical CSV bytes from one chunk or three through
`render_and_store`.
"""

from __future__ import annotations

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import render_and_store
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

_PURPOSE_HEALTH = CodeRow(
    dcid="dcid:CRSPurposeCode122",
    code_system="crsPurpose",
    code="122",
    label="Basic health care",
    synonyms=(),
    parent=None,
)
_PURPOSE_MALARIA = CodeRow(
    dcid="dcid:CRSPurposeCode12250",
    code_system="crsPurpose",
    code="12250",
    label="Malaria control",
    synonyms=(),
    parent=None,
)
_FINANCE_GRANT = CodeRow(
    dcid="dcid:CRSTypeOfFinanceCode110",
    code_system="crsTypeOfFinance",
    code="110",
    label="Standard grant",
    synonyms=(),
    parent=None,
)
_FINANCE_LOAN = CodeRow(
    dcid="dcid:CRSTypeOfFinanceCode421",
    code_system="crsTypeOfFinance",
    code="421",
    label="Standard loan",
    synonyms=(),
    parent=None,
)

_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="sector",
    ),
    CodeSystemDeclaration(
        code_system="crsTypeOfFinance",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="instrument",
    ),
)


def _statvar(dcid: str, *, purpose: str, finance: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("recipient",),
        constraint_properties={"crsPurpose": purpose, "crsTypeOfFinance": finance},
    )


_INDEX = CodeIndex(
    stamp=_STAMP,
    declarations=_DECLARATIONS,
    codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA, _FINANCE_GRANT, _FINANCE_LOAN),
    statvars=(
        _statvar("SV_A", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_GRANT.dcid),
        _statvar("SV_B", purpose=_PURPOSE_HEALTH.dcid, finance=_FINANCE_LOAN.dcid),
        _statvar("SV_C", purpose=_PURPOSE_MALARIA.dcid, finance=_FINANCE_GRANT.dcid),
    ),
)

_GRAPH = FakeMintGraphReader()
_TABLE_COLUMNS = ("stat_var", "recipient", "time_period", "value", "unit", "provenance")
_ROWS = (
    ("dcid:SV_A", "country/KEN", "2022", 10.0, "USD", "p1"),
    ("dcid:SV_B", "country/KEN", "2022", 5.0, "USD", "p1"),
    ("dcid:SV_C", "country/KEN", "2022", 7.0, "USD", "p1"),
)
_RESOLVED_STAT_VAR_REFS = ("dcid:SV_A", "dcid:SV_B", "dcid:SV_C")


def _render(chunks: tuple[Table, ...]):
    artifact_store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=iter(chunks),
        options=MintOptions(
            format="csv", bundle=False, groupby=GroupbyOptions(by=["crsPurpose"], op="sum")
        ),
        spec={"spec_id": "groupby-streaming-e2e"},
        provenance_refs=["p1"],
        citations={},
        reproducing_call=None,
        content_version="sha256:" + "0" * 64,
        artifact_store=artifact_store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(_GRAPH),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED_STAT_VAR_REFS,
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered, artifact_store.fetch(rendered.artifact.url)


def test_a_three_way_split_matches_the_single_chunk_artifact_digest():
    one_chunk = (Table(columns=_TABLE_COLUMNS, rows=_ROWS),)
    three_chunks = tuple(Table(columns=_TABLE_COLUMNS, rows=(row,)) for row in _ROWS)

    one_rendered, one_bytes = _render(one_chunk)
    three_rendered, three_bytes = _render(three_chunks)

    assert one_rendered.artifact.artifact_digest == three_rendered.artifact.artifact_digest
    assert one_bytes == three_bytes
    body = one_bytes.decode("utf-8")
    assert "15.0" in body  # SV_A + SV_B merged under crsPurpose=health
    assert "7.0" in body  # SV_C stays apart (malaria)
