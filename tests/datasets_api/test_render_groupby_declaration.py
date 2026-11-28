"""Tests how `_reshape` uses the injected `CodeIndex` to handle absent members in groupby
sums.

A source declaring absence as zero permits a missing member. An undeclared source or mixed
provenance refuses. A dataset of reporting organisations reads a missing member as zero only
in the years its reporting organisation holds, and a total of two of them warns
`reporters_summed`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, DatasetDeclaration, IndexStamp
from code_index.shared import get_index
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import _RenderReads, _reshape
from datasets_api.warnings import reporters_summed_message
from materialise.enrichment import EnrichmentLookup
from materialise.reshape import RollupNotSummableError, Table
from search.engine.retrieval.bound_values import apply_slot_summability
from search.models import BaseIdentity
from tests.engine._builders import unbound_slot
from tests.graph_release.support import serving_index

from ._graph_doubles import FakeMintGraphReader

_STAMP = IndexStamp(
    graph_endpoint="https://example.test/core/api",
    exported_at="2026-09-15T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="deadbeef",
)

_CRS_PROVENANCE = "provenance/OECDCreditorReportingSystem"
_OTHER_PROVENANCE = "provenance/SomeOtherSource"

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

_DECLARATIONS = (
    CodeSystemDeclaration(
        code_system="crsPurpose",
        hierarchical=True,
        disjoint_from_children=True,
        summable=True,
        absent_reading="not_declared",
        plain_name="sector",
    ),
    # This summable axis is absent from the fixture.
    # Naming it in `by` leaves other axes free to collapse.
    CodeSystemDeclaration(
        code_system="crsCooperationModality",
        hierarchical=False,
        disjoint_from_children=False,
        summable=True,
        absent_reading="not_declared",
        plain_name="modality",
    ),
)


def _statvar(dcid: str, *, purpose: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("financingProvider", "financingRecipient"),
        constraint_properties={"crsPurpose": purpose},
    )


def _index(*, declares_crs_zero: bool) -> CodeIndex:
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA),
        statvars=(
            _statvar("SV_health", purpose=_PURPOSE_HEALTH.dcid),
            _statvar("SV_malaria", purpose=_PURPOSE_MALARIA.dcid),
        ),
        datasets=(
            (
                DatasetDeclaration(
                    id=_CRS_PROVENANCE,
                    label="OECD Creditor Reporting System (CRS)",
                    description="A test table.",
                    opt_in=False,
                    absent_observation="zero",
                    account="provider",
                    rank=0,
                ),
            )
            if declares_crs_zero
            else ()
        ),
    )


_ENTITY_COLUMNS = ("financingProvider", "financingRecipient")
_TABLE_COLUMNS = (
    "stat_var",
    "financingProvider",
    "financingRecipient",
    "time_period",
    "value",
    "unit",
    "provenance",
)

# `SV_malaria` is resolved (named in `resolved_stat_var_refs` below) but
# contributes zero rows to the table at all -- the same "entirely absent
# member" shape `test_render_groupby_population.py`'s own
# `test_zero_row_resolved_member_refuses_and_names_it` refuses on HEAD.
_TABLE = Table(
    columns=_TABLE_COLUMNS,
    rows=(("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", _CRS_PROVENANCE),),
)
_RESOLVED_REFS = ("dcid:SV_health", "dcid:SV_malaria")


def _enrichment() -> EnrichmentLookup:
    return EnrichmentLookup(FakeMintGraphReader())


def _options() -> MintOptions:
    return MintOptions(groupby=GroupbyOptions(by=["crsCooperationModality"], op="sum"))


# --- A declared-zero source sums and publishes -----------------------------


def test_a_zero_row_member_from_a_declared_zero_source_sums_and_publishes():
    reshaped = _reshape(
        _TABLE,
        options=_options(),
        entity_columns=_ENTITY_COLUMNS,
        reads=_RenderReads(enrichment=_enrichment(), parties={}, known={}, deadline=None),
        code_index=_index(declares_crs_zero=True),
        resolved_stat_var_refs=_RESOLVED_REFS,
        provenance_refs=(_CRS_PROVENANCE,),
        pivot_max_output_cells=5_000_000,
    )

    assert reshaped.warnings == ()
    # The declared zero reading permits the absent malaria member. Health is the only
    # contributor, so the full collapse restores its StatVar id.
    assert "stat_var_crsPurpose" not in reshaped.table.columns
    stat_var_pos = reshaped.table.columns.index("stat_var")
    value_pos = reshaped.table.columns.index("value")
    assert [row[stat_var_pos] for row in reshaped.table.rows] == ["dcid:SV_health"]
    assert [row[value_pos] for row in reshaped.table.rows] == [10.0]


# --- The same frame under an undeclared provenance still refuses -----------


def test_the_same_zero_row_member_under_an_undeclared_provenance_still_refuses():
    try:
        _reshape(
            _TABLE,
            options=_options(),
            entity_columns=_ENTITY_COLUMNS,
            reads=_RenderReads(enrichment=_enrichment(), parties={}, known={}, deadline=None),
            code_index=_index(declares_crs_zero=False),
            resolved_stat_var_refs=_RESOLVED_REFS,
            provenance_refs=(_CRS_PROVENANCE,),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.precondition == "missing_members"
        assert _PURPOSE_MALARIA.dcid in str(exc)
    else:
        raise AssertionError("expected RollupNotSummableError")


# --- Two provenance refs on one frame refuse before picking one ------------


def test_two_distinct_provenance_refs_refuse_before_any_declaration_lookup():
    # A complete population isolates the provenance disagreement from missing-member errors.
    complete_table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", _CRS_PROVENANCE),
            ("dcid:SV_malaria", "country/USA", "country/KEN", "2022", 4.0, "USD", _CRS_PROVENANCE),
        ),
    )

    try:
        _reshape(
            complete_table,
            options=_options(),
            entity_columns=_ENTITY_COLUMNS,
            reads=_RenderReads(enrichment=_enrichment(), parties={}, known={}, deadline=None),
            code_index=_index(declares_crs_zero=True),
            resolved_stat_var_refs=(),
            provenance_refs=(_CRS_PROVENANCE, _OTHER_PROVENANCE),
            pivot_max_output_cells=5_000_000,
        )
    except RollupNotSummableError as exc:
        assert exc.axis == "provenance"
        assert exc.precondition == "provenance_disagreement"
    else:
        raise AssertionError("expected RollupNotSummableError")


# --- Two member provenances of one dataset agree ----------------------------

_LEDGER = "dataset/Ledger"
_NORTH = "provenance/LedgerNorth"
_SOUTH = "provenance/LedgerSouth"


def _ledger_index(*, absent_zero: bool = False, reporters: bool = True) -> CodeIndex:
    """`_index`'s members under `dataset/Ledger`, whose series two member provenances hold,
    reading an absent observation as zero when `absent_zero`, and declaring its members
    reporting organisations unless `reporters` is false."""
    base = _index(declares_crs_zero=False)
    return CodeIndex(
        stamp=_STAMP,
        declarations=_DECLARATIONS,
        codes=(_PURPOSE_HEALTH, _PURPOSE_MALARIA),
        statvars=tuple(replace(row, datasets=(_LEDGER,)) for row in base.statvars),
        datasets=(
            DatasetDeclaration(
                id=_LEDGER,
                label="Ledger",
                description="A test register filed by publisher.",
                opt_in=False,
                absent_observation="zero" if absent_zero else "unknown",
                account="provider",
                rank=0,
                members_report_overlapping_accounts=reporters,
            ),
        ),
        dataset_members={_LEDGER: (_NORTH, _SOUTH)},
    )


def test_rows_of_two_member_provenances_of_one_dataset_sum_to_one_total():
    """Each member's row carries the dataset's id, as the mint writes it
    (`materialise.columnar.dataset_cells`), so the sum neither refuses
    `provenance_disagreement` nor `facet_homogeneity`, and the total cites the dataset."""
    table = Table(
        columns=_TABLE_COLUMNS,
        rows=(
            ("dcid:SV_health", "country/USA", "country/KEN", "2022", 10.0, "USD", _LEDGER),
            ("dcid:SV_malaria", "country/USA", "country/KEN", "2022", 4.0, "USD", _LEDGER),
        ),
    )

    reshaped = _reshape(
        table,
        options=_options(),
        entity_columns=_ENTITY_COLUMNS,
        reads=_RenderReads(enrichment=_enrichment(), parties={}, known={}, deadline=None),
        code_index=_ledger_index(),
        resolved_stat_var_refs=_RESOLVED_REFS,
        provenance_refs=(_LEDGER,),
        subject_id="spec_ledger",
        pivot_max_output_cells=5_000_000,
    )

    columns = reshaped.table.columns
    assert [
        (row[columns.index("value")], row[columns.index("provenance")])
        for row in reshaped.table.rows
    ] == [(14.0, _LEDGER)]


def _reported(*rows: tuple[str, str, float, str]) -> Table:
    """A Ledger frame of `(StatVar, year, value, reporting organisation)` rows, each from the
    USA to Kenya, as the mint writes a member's row (`materialise.columnar.dataset_cells`)."""
    return Table(
        columns=(*_TABLE_COLUMNS, "reporting_organisation"),
        rows=tuple(
            (f"dcid:{stat_var}", "country/USA", "country/KEN", year, value, "USD", _LEDGER, by)
            for stat_var, year, value, by in rows
        ),
    )


def _sum_ledger(
    table: Table,
    *,
    index: CodeIndex,
    held_years: Callable[[str], frozenset[int]] | None = None,
):
    return _reshape(
        table,
        options=_options(),
        entity_columns=_ENTITY_COLUMNS,
        reads=_RenderReads(enrichment=_enrichment(), parties={}, known={}, deadline=None),
        code_index=index,
        resolved_stat_var_refs=_RESOLVED_REFS,
        provenance_refs=(_LEDGER,),
        subject_id="spec_ledger",
        pivot_max_output_cells=5_000_000,
        held_years=held_years,
    )


@pytest.mark.parametrize(
    ("reporters", "malaria_by", "warned"),
    [
        pytest.param(True, _SOUTH, True, id="two-reporters"),
        pytest.param(True, _NORTH, False, id="one-reporter"),
        pytest.param(False, _SOUTH, False, id="undeclared"),
    ],
)
def test_a_total_of_two_reporting_organisations_rows_warns_reporters_summed(
    reporters: bool, malaria_by: str, warned: bool
):
    """A total adding North's and South's rows of a dataset declaring its members reporting
    organisations warns, counting both; a total of North's rows alone does not, nor does the
    same total under a dataset that declares nothing of its members."""
    table = _reported(("SV_health", "2022", 10.0, _NORTH), ("SV_malaria", "2022", 4.0, malaria_by))

    reshaped = _sum_ledger(table, index=_ledger_index(reporters=reporters))

    assert [row[reshaped.table.columns.index("value")] for row in reshaped.table.rows] == [14.0]
    summed = [w["message"] for w in reshaped.warnings if w["code"] == "reporters_summed"]
    assert summed == ([reporters_summed_message(2)] if warned else [])


# North holds 2021 to 2023 across its series, South 2021 to 2024.
_HELD = {_NORTH: frozenset({2021, 2022, 2023}), _SOUTH: frozenset({2021, 2022, 2023, 2024})}


@pytest.mark.parametrize(
    ("reporters", "last_year", "totals"),
    [
        pytest.param(
            True, 2023, {"2021": 14.0, "2022": 4.0, "2023": 14.0}, id="held-year-reads-zero"
        ),
        pytest.param(True, 2024, None, id="unheld-year-reads-unknown"),
        pytest.param(
            False,
            2024,
            {"2021": 14.0, "2022": 4.0, "2023": 14.0, "2024": 4.0},
            id="undeclared-reads-zero-every-year",
        ),
    ],
)
def test_a_reporters_missing_row_reads_zero_only_in_years_it_holds(
    reporters: bool, last_year: int, totals: dict[str, float] | None
):
    """North's health row is missing in 2022 and 2024. North holds 2022 in another series, so
    the 2022 total reads the missing row as zero; it holds nothing in 2024, so a total that
    year is unknown and the sum refuses. A dataset that declares nothing of its members keeps
    reading every missing row as zero."""
    table = _reported(
        ("SV_health", "2021", 10.0, _NORTH),
        ("SV_health", "2023", 10.0, _NORTH),
        *(("SV_malaria", str(year), 4.0, _SOUTH) for year in range(2021, last_year + 1)),
    )
    index = _ledger_index(absent_zero=True, reporters=reporters)

    def held_years(provenance: str) -> frozenset[int]:
        return _HELD.get(provenance, frozenset())

    if totals is None:
        with pytest.raises(RollupNotSummableError) as refused:
            _sum_ledger(table, index=index, held_years=held_years)
        assert refused.value.precondition == "missing_members"
        return
    reshaped = _sum_ledger(table, index=index, held_years=held_years)
    columns = reshaped.table.columns
    assert {
        row[columns.index("time_period")]: row[columns.index("value")]
        for row in reshaped.table.rows
    } == totals


# The Ledger family's identity, as Search resolves it from `_ledger_index`.
_LEDGER_IDENTITY = BaseIdentity.model_validate(
    {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "Amount",
            "measured_property": "amount",
        },
        "slots": [{"property": "crsPurpose", "binding": {"state": "unbound"}}],
        "requested_window": {"kind": "all"},
    }
)


@pytest.mark.parametrize("last_year", [2023, 2024], ids=["held-years", "unheld-year"])
def test_search_reads_a_ledger_sum_summable_exactly_where_the_collapse_sums(last_year: int):
    """North files nothing in 2024, so a sum to 2024 reads its missing health row as unknown
    and refuses; Search reads the same window not summable. To 2023 both sum."""
    table = _reported(
        ("SV_health", "2021", 10.0, _NORTH),
        ("SV_health", "2023", 10.0, _NORTH),
        *(("SV_malaria", str(year), 4.0, _SOUTH) for year in range(2021, last_year + 1)),
    )
    index = _ledger_index(absent_zero=True)

    def held_years(provenance: str) -> frozenset[int]:
        return _HELD.get(provenance, frozenset())

    try:
        _sum_ledger(table, index=index, held_years=held_years)
    except RollupNotSummableError as refused:
        assert refused.precondition == "missing_members"
        sums = False
    else:
        sums = True
    with serving_index(index):
        (slot,) = apply_slot_summability(
            [unbound_slot("how", "crsPurpose", "crsPurpose")],
            _LEDGER_IDENTITY,
            years=range(2021, last_year + 1),
            held_years=held_years,
        )

    assert sums is (last_year == 2023)
    assert slot.summable is sums


# --- Declaration lookup uses the injected index -----------------------------


def test_declaration_follows_the_injected_index_even_when_the_singleton_disagrees():
    # The served index carries the opposite declaration. A fallback to it changes the result.
    with serving_index(_index(declares_crs_zero=False)):
        reshaped = _reshape(
            _TABLE,
            options=_options(),
            entity_columns=_ENTITY_COLUMNS,
            reads=_RenderReads(enrichment=_enrichment(), parties={}, known={}, deadline=None),
            code_index=_index(declares_crs_zero=True),
            resolved_stat_var_refs=_RESOLVED_REFS,
            provenance_refs=(_CRS_PROVENANCE,),
            pivot_max_output_cells=5_000_000,
        )
        assert reshaped.warnings == ()
        value_pos = reshaped.table.columns.index("value")
        assert [row[value_pos] for row in reshaped.table.rows] == [10.0]

    with serving_index(_index(declares_crs_zero=True)):
        try:
            _reshape(
                _TABLE,
                options=_options(),
                entity_columns=_ENTITY_COLUMNS,
                reads=_RenderReads(enrichment=_enrichment(), parties={}, known={}, deadline=None),
                code_index=_index(declares_crs_zero=False),
                resolved_stat_var_refs=_RESOLVED_REFS,
                provenance_refs=(_CRS_PROVENANCE,),
                pivot_max_output_cells=5_000_000,
            )
        except RollupNotSummableError as exc:
            assert exc.precondition == "missing_members"
        else:
            raise AssertionError("expected RollupNotSummableError")

        # Confirms both calls ran while the served index held the opposite declaration.
        assert get_index().absent_observation(_CRS_PROVENANCE) == "zero"
