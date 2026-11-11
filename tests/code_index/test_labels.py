"""Test `normalize_label` over the fixture index, the codes a series holds for a partition
label's `codes`, and the module's import isolation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from code_index import CodeIndex
from code_index.declarations import MEASURE_PARTITION_LABELS
from code_index.labels import held_codes, normalize_label
from code_index.types import CodeRow, CodeSystemDeclaration, IndexStamp, label_for_pairs

_OBJECTIVE = "climatePolicyObjective"
_TOTAL = "ClimateChangePolicyObjective"
_ADAPTATION = "AdaptationOnlyClimateChangePolicyObjective"
_MITIGATION = "MitigationOnlyClimateChangePolicyObjective"


def _objective_index() -> CodeIndex:
    """An index holding the climate objective's total and two of its parts, with no codes."""
    rows = tuple(
        CodeRow(
            dcid=dcid,
            code_system=_OBJECTIVE,
            code=None,
            label=dcid,
            synonyms=(),
            parent=None if dcid == _TOTAL else _TOTAL,
        )
        for dcid in (_TOTAL, _ADAPTATION, _MITIGATION)
    )
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="synthetic",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={_OBJECTIVE: len(rows)},
            statvar_shape_counts={},
            content_hash="synthetic",
        ),
        declarations=(
            CodeSystemDeclaration(
                code_system=_OBJECTIVE,
                hierarchical=True,
                disjoint_from_children=False,
                summable=False,
                absent_reading="not_declared",
                plain_name="climate objective",
            ),
        ),
        codes=rows,
        statvars=(),
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({_OBJECTIVE: [_ADAPTATION]}, {(_OBJECTIVE, _ADAPTATION), (_OBJECTIVE, _TOTAL)}),
        ({_OBJECTIVE: [_ADAPTATION, _MITIGATION]}, {(_OBJECTIVE, _TOTAL)}),
        ({_OBJECTIVE: [_TOTAL]}, {(_OBJECTIVE, _TOTAL)}),
        ({"nettingBasis": ["Gross"]}, set()),
        ({_OBJECTIVE: []}, set()),
    ],
    ids=["part", "two-parts", "total", "not-a-code", "no-values"],
)
def test_a_series_holds_the_codes_every_value_sits_at_or_below(values, expected):
    assert held_codes(_objective_index(), values) == expected


_ATTRIBUTED_GROSS = (
    ("populationType", "FinancialFlow"),
    ("measuredProperty", "attributedAmount"),
    ("measurementQualifier", "Nominal"),
    ("financialFlowType", "Disbursement"),
    ("nettingBasis", "Gross"),
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({_OBJECTIVE: [_ADAPTATION]}, "attributed climate finance gross disbursements"),
        ({}, "imputed multilateral ODA"),
    ],
    ids=["climate", "imputed"],
)
def test_a_label_keyed_on_a_code_splits_a_partition_two_tables_share(values, expected):
    """ONE Data's attributed climate disbursements and the imputed multilateral ODA rows
    share a partition, and only the climate rows hold an objective under the total."""
    match = label_for_pairs(
        _ATTRIBUTED_GROSS,
        MEASURE_PARTITION_LABELS,
        held_codes=held_codes(_objective_index(), values),
    )
    assert match is not None
    assert (match.label, match.uncovered) == (expected, (("measurementQualifier", "Nominal"),))


def test_indexed_dcid_takes_the_index_label(code_index: CodeIndex):
    row = code_index.code_by_dcid("CRSPurposeCode100")
    assert row is not None
    raw = f"{row.label} (CRS Purpose)"

    assert normalize_label(row.dcid, raw, index=code_index) == row.label


def test_unindexed_dcid_keeps_the_raw_label(code_index: CodeIndex):
    assert normalize_label("country/KEN", "Kenya", index=code_index) == "Kenya"


def test_importing_labels_pulls_in_no_search_module():
    probe = (
        "import sys, code_index.labels; "
        "sys.exit(any(m == 'search' or m.startswith('search.') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0
