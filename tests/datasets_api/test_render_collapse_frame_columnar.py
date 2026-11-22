"""A streamed groupby collapse builds its grammar frame from the fold's Arrow table
(`render.collapse_frame`) and hashes it per column (`materialise.content_version.
stream_frame_sha256`). Each render here checks, at every collapse, that the Arrow build equals
the row build (`Frame.build`) it replaced, cell for cell and byte for byte, on the fold output
and on its empty slice, and that the hash is the grammar's `frame_sha256`. The pinned digests
are what the row build rendered for the same inputs: the frame hash and the written file."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import pyarrow as pa
import pytest
from dfo_frame_grammar import Frame, canonical_frame_bytes, frame_sha256

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.declarations import CRS, DATASETS
from code_index.types import PartyRow, RoleAxis, StatVarRow
from datasets_api import render
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import RenderedDefinition, render_and_store
from materialise.content_version import mint_version, stream_frame_sha256
from materialise.enrichment import EnrichmentLookup
from materialise.formats import canonical_frame_rows
from materialise.groupby_columnar import GroupedTable, rows_table

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE


def _purpose(code: str, parent: str | None) -> CodeRow:
    return CodeRow(
        dcid=f"dcid:CRSPurposeCode{code}",
        code_system="crsPurpose",
        code=code,
        label=f"Purpose {code}",
        synonyms=(),
        parent=None if parent is None else f"dcid:CRSPurposeCode{parent}",
    )


def _finance(code: str) -> CodeRow:
    return CodeRow(
        dcid=f"dcid:CRSTypeOfFinanceCode{code}",
        code_system="crsTypeOfFinance",
        code=code,
        label=f"Finance {code}",
        synonyms=(),
        parent=None,
    )


_MODALITY = CodeRow(
    dcid="dcid:CRSCooperationModalityCodeA01",
    code_system="crsCooperationModality",
    code="A01",
    label="Modality A01",
    synonyms=(),
    parent=None,
)
_PURPOSES = (_purpose("120", None), _purpose("122", "120"), _purpose("111", None))
_FINANCES = (_finance("110"), _finance("421"))
_STAT_VARS = tuple(
    (f"SV_{purpose.code}_{finance.code}", purpose, finance)
    for purpose in _PURPOSES
    for finance in _FINANCES
)
_BOTH_SIDES: tuple[tuple[str, RoleAxis], ...] = ((CRS, "origin"), (CRS, "destination"))
_KENYA, _GHANA, _DEVELOPING = "country/KEN", "country/GHA", "DevelopingCountries"

_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://example.test/core/api",
        exported_at="2026-09-09T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="deadbeef",
    ),
    declarations=tuple(
        CodeSystemDeclaration(
            code_system=code_system,
            hierarchical=True,
            disjoint_from_children=True,
            summable=True,
            absent_reading="zero",
            plain_name=plain_name,
        )
        # No StatVar carries `crsCooperationModality`, so no frame decodes it.
        for code_system, plain_name in (
            ("crsPurpose", "sector"),
            ("crsTypeOfFinance", "type"),
            ("crsCooperationModality", "modality"),
        )
    ),
    datasets=tuple(dataset for dataset in DATASETS if dataset.provenance == CRS),
    # Purpose 130 is a code no StatVar holds, so a selection can list a code no row carries.
    codes=(*_PURPOSES, _purpose("130", None), *_FINANCES, _MODALITY),
    statvars=tuple(
        StatVarRow(
            dcid=dcid,
            population_type="Amount",
            measured_property="amount",
            stat_type=None,
            measurement_qualifier=None,
            measurement_denominator=None,
            observation_properties=("recipient",),
            constraint_properties={"crsPurpose": purpose.dcid, "crsTypeOfFinance": finance.dcid},
        )
        for dcid, purpose, finance in _STAT_VARS
    ),
    parties=(
        PartyRow(dcid=_KENYA, kind="party", published=_BOTH_SIDES),
        PartyRow(dcid=_GHANA, kind="party", published=_BOTH_SIDES),
        PartyRow(dcid=_DEVELOPING, kind="total", published=_BOTH_SIDES),
    ),
)
_RESOLVED = tuple(f"dcid:{dcid}" for dcid, _purpose_row, _finance_row in _STAT_VARS)
_ENTITY_COLUMNS = ("financingProvider", "financingRecipient")

_CONVERTED = {
    "currency": "USD",
    "prices": "constant",
    "base_year": 2023,
    "recipe": "ConversionRecipeDAC",
    "deflation_basis": "donor",
}
_COMPOSED = {"numerator": "USDollar", "denominator": {"stat_var": "dcid:Count_Person"}}
_UNIT_BY_RECIPIENT: Mapping[str, object] = {
    _KENYA: "dcid:USDollar",
    _GHANA: _CONVERTED,
    _DEVELOPING: _COMPOSED,
}
"""Each recipient's rows in another `unit` form: a ref, a converted unit in the recipe
layout, a composed unit. A provider's rows of `country/FRA` have no unit."""


def _frame(recipients: Sequence[str], *, methods: Mapping[str, str] | None = None) -> Frame:
    """Every StatVar's rows for two providers, `recipients` and two years, with distinct
    values, a method on one provider's rows and `CRS` provenance. `methods` gives a StatVar's
    rows of that provider another method, which a sum adding it to another StatVar refuses."""
    rows: list[dict[str, object]] = []
    for position, (dcid, _purpose_row, _finance_row) in enumerate(_STAT_VARS):
        for provider in ("country/USA", "country/FRA"):
            for recipient in recipients:
                for year in ("2021", "2022"):
                    rows.append(
                        {
                            "stat_var": f"dcid:{dcid}",
                            "financingProvider": provider,
                            "financingRecipient": recipient,
                            "time_period": year,
                            "value": 1.5 * (position + 1) + len(rows) / 8,
                            "unit": None
                            if provider == "country/FRA"
                            else _UNIT_BY_RECIPIENT[recipient],
                            "measurementMethod": (methods or {}).get(dcid, "OECD_DAC")
                            if provider == "country/USA"
                            else None,
                            "provenance": CRS,
                        }
                    )
    return Frame.build(role_keys=_ENTITY_COLUMNS, rows=rows)


_ALL = _frame((_KENYA, _GHANA, _DEVELOPING))
_TOTALS_ONLY = _frame((_DEVELOPING,))


def _checked_collapse_frame(monkeypatch: pytest.MonkeyPatch) -> list[pa.Table]:
    """Wraps `render.collapse_frame` so each call also builds the frame from the fold
    output's tuple rows, and from its empty slice both ways, and checks they agree. Returns
    the list of fold outputs it saw."""
    build = render.collapse_frame
    seen: list[pa.Table] = []

    def checked(collapsed: GroupedTable, **kwargs: object) -> Frame:
        assert isinstance(collapsed, pa.Table), "the fold gave tuple rows, not Arrow"
        seen.append(collapsed)
        for table in (collapsed, collapsed.slice(0, 0)):
            from_arrow = build(table, **kwargs)
            from_rows = build(rows_table(table), **kwargs)
            assert from_arrow == from_rows
            assert canonical_frame_bytes(from_arrow) == canonical_frame_bytes(from_rows)
            assert stream_frame_sha256((from_arrow,)) == frame_sha256(from_rows)
        return build(collapsed, **kwargs)

    monkeypatch.setattr(render, "collapse_frame", checked)
    return seen


def _render(frame: Frame, options: MintOptions) -> tuple[RenderedDefinition, bytes]:
    artifact_store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=lambda: iter((frame,)),
        options=options,
        spec={"spec_id": "spec_columnar"},
        provenance_refs=[CRS],
        citations={},
        reproducing_call=None,
        content_version=mint_version(frame),
        artifact_store=artifact_store,
        entity_columns=_ENTITY_COLUMNS,
        enrichment=EnrichmentLookup(FakeMintGraphReader()),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED,
        pivot_max_output_cells=5_000_000,
        groupby_fold_max_rows=2_000_000,
        frame_rows=frame.num_rows,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered, artifact_store.fetch(rendered.artifact.url)


def _options(
    format: str,
    by: list[str],
    *,
    levels: dict[str, int] | None = None,
    cut: bool = False,
    layout: str | None = None,
) -> MintOptions:
    return MintOptions(
        format=format,
        bundle=False,
        layout=layout,
        groupby=GroupbyOptions(by=by, op="sum", levels=levels),
        without_party_totals=cut,
    )


@pytest.mark.parametrize(
    ("frame", "options", "rows", "frame_digest", "file_digest"),
    [
        pytest.param(
            _ALL,
            _options("csv", ["time_period"]),
            12,
            "c2aff9820e15b72f8a8212e17df6ba5ad23fdbacfde28dea4b0159c891351ee9",
            "cae970469b4b6d7451e1353ec6a15fcea8cc7a2378d903fddc556c26a487d17a",
            id="summed-whole-csv",
        ),
        pytest.param(
            _ALL,
            _options("parquet", ["crsPurpose", "time_period"], layout="frame"),
            36,
            "5cfad56debc4042b61a0d657272f7d9aba87cbfb448bbe1546206d42f11cfb6c",
            "6f24e830a87eb40091d27e0c81af9bfe6ab560619fb19aa0aca6b68bf01c4c75",
            id="kept-axis-frame-parquet",
        ),
        pytest.param(
            _ALL,
            _options("xlsx", ["crsPurpose", "crsTypeOfFinance", "time_period"]),
            72,
            "7153f1893378e1dc7b616a4a3fb2b4c17743aea832861ac781828a0f472a2541",
            "432d6c49ed988102575c23457084035e3f504ba8d426c459d2f5197386341cb8",
            id="two-kept-axes-xlsx",
        ),
        pytest.param(
            _ALL,
            _options("csv", ["crsPurpose", "time_period"], levels={"crsPurpose": 0}),
            24,
            "db1fafbd6dc5a52bb97c2c98e8f48c607ff714324f1edf57a9ccadb3971b6fe3",
            "d896a796d8dd2ec809438f43018645bde4a0ad444d4ee54f6623fab3ec412491",
            id="level-roll-up-csv",
        ),
        pytest.param(
            _ALL,
            _options("csv", ["crsTypeOfFinance", "time_period"], cut=True),
            20,
            "80a727f7f30c9dbf01e1b2037b67bb1dd856b1515943bdf48fae87fcfb56a79f",
            "080ec7e74d1fe78b24caf9320e4b4baa8ebbbb978f2a3a03155f0bb80779530d",
            id="party-totals-cut-csv",
        ),
        pytest.param(
            _TOTALS_ONLY,
            _options("parquet", ["time_period"], cut=True),
            4,
            "8e082630e3c19e15a6c33128577a6bb133935cab4c7a79c2b908fa503dad9f36",
            "e6d93d6bf0179b4f75babec2bc65a39c0741ab9b50346640ed1608cf9b8e0314",
            id="totals-only-cut-parquet",
        ),
    ],
)
def test_a_streamed_collapse_renders_the_bytes_the_row_build_rendered(
    monkeypatch: pytest.MonkeyPatch,
    frame: Frame,
    options: MintOptions,
    rows: int,
    frame_digest: str,
    file_digest: str,
):
    seen = _checked_collapse_frame(monkeypatch)

    rendered, body = _render(frame, options)

    assert len(seen) == 1
    collapse = rendered.collapse
    assert collapse is not None
    assert collapse.frame.num_rows == rows
    assert collapse.frame_sha256 == frame_sha256(collapse.frame) == frame_digest
    # The long layout reads the frame's rows through `canonical_frame_rows`.
    assert tuple(canonical_frame_rows(collapse.frame.table)) == collapse.frame.rows
    assert hashlib.sha256(body).hexdigest() == file_digest
