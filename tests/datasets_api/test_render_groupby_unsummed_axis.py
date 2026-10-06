"""Tests how `_reshape` sums a family whose members carry a code system not declared summable.

A sum never collapses such a decoded code system. A family holding one value of it sums to a
grammar frame without the constant column. A family holding several values sums per value and
keeps the column.
"""

from __future__ import annotations

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, DatasetDeclaration, IndexStamp
from code_index.types import StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import _reshape
from materialise.enrichment import EnrichmentLookup
from materialise.frame_build import Frame
from materialise.reshape import Table

from ._graph_doubles import FakeMintGraphReader

_PROVENANCE = "provenance/Lines"
_UNSUMMED = "tableLine"
_ENTITY_COLUMNS = ("financingProvider", "financingRecipient")


def _code(dcid: str, code_system: str, label: str) -> CodeRow:
    return CodeRow(
        dcid=dcid, code_system=code_system, code=dcid, label=label, synonyms=(), parent=None
    )


def _declaration(code_system: str, *, summable: bool) -> CodeSystemDeclaration:
    return CodeSystemDeclaration(
        code_system=code_system,
        hierarchical=False,
        disjoint_from_children=True,
        summable=summable,
        absent_reading="not_declared",
        plain_name=code_system,
    )


def _statvar(dcid: str, *, purpose: str, line: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=_ENTITY_COLUMNS,
        constraint_properties={"crsPurpose": purpose, _UNSUMMED: line},
        provenances=(_PROVENANCE,),
    )


_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://example.test/core/api",
        exported_at="2026-09-15T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="deadbeef",
    ),
    declarations=(
        _declaration("crsPurpose", summable=True),
        _declaration(_UNSUMMED, summable=False),
    ),
    codes=(
        _code("PurposeHealth", "crsPurpose", "Health"),
        _code("PurposeMalaria", "crsPurpose", "Malaria control"),
        _code("LineOne", _UNSUMMED, "Line one"),
        _code("LineTwo", _UNSUMMED, "Line two"),
    ),
    statvars=(
        _statvar("SV_health_one", purpose="PurposeHealth", line="LineOne"),
        _statvar("SV_malaria_one", purpose="PurposeMalaria", line="LineOne"),
        _statvar("SV_health_two", purpose="PurposeHealth", line="LineTwo"),
    ),
    datasets=(
        DatasetDeclaration(
            provenance=_PROVENANCE,
            label="Lines",
            description="A test table.",
            opt_in=False,
            absent_observation="zero",
            rank=0,
        ),
    ),
)


def _row(stat_var: str, value: float) -> tuple[object, ...]:
    return (f"dcid:{stat_var}", "country/USA", "country/KEN", "2022", value, "USD", _PROVENANCE)


def _sum(members: tuple[tuple[str, float], ...]):
    return _reshape(
        Table(
            columns=("stat_var", *_ENTITY_COLUMNS, "time_period", "value", "unit", "provenance"),
            rows=tuple(_row(stat_var, value) for stat_var, value in members),
        ),
        options=MintOptions(groupby=GroupbyOptions(by=["time_period"], op="sum")),
        entity_columns=_ENTITY_COLUMNS,
        enrichment=EnrichmentLookup(FakeMintGraphReader()),
        code_index=_INDEX,
        resolved_stat_var_refs=tuple(f"dcid:{stat_var}" for stat_var, _value in members),
        provenance_refs=(_PROVENANCE,),
        subject_id="spec_family",
        pivot_max_output_cells=5_000_000,
    )


def test_a_family_holding_one_unsummed_value_sums_to_a_frame_without_its_column():
    summed = _sum((("SV_health_one", 10.0), ("SV_malaria_one", 4.0)))

    assert isinstance(summed.table, Frame)
    assert summed.collapse is not None
    assert summed.table.columns == (
        "stat_var",
        *_ENTITY_COLUMNS,
        "time_period",
        "value",
        "unit",
        "measurementMethod",
        "provenance",
        "status",
    )
    assert summed.table.rows == (
        (
            "dcid:spec_family",
            "country/USA",
            "country/KEN",
            "2022",
            14.0,
            "USD",
            None,
            _PROVENANCE,
            None,
        ),
    )


def test_a_family_holding_two_unsummed_values_sums_per_value_and_keeps_the_column():
    summed = _sum((("SV_health_one", 10.0), ("SV_malaria_one", 4.0), ("SV_health_two", 3.0)))

    assert not isinstance(summed.table, Frame)
    assert summed.collapse is None
    unsummed = summed.table.columns.index(f"stat_var_{_UNSUMMED}")
    value = summed.table.columns.index("value")
    assert "stat_var" not in summed.table.columns
    assert "stat_var_crsPurpose" not in summed.table.columns
    assert sorted((row[unsummed], row[value]) for row in summed.table.rows) == [
        ("Line one", 14.0),
        ("Line two", 3.0),
    ]
