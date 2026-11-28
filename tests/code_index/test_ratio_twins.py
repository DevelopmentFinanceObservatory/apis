"""`code_index.ratio_twins`: the ratio families a code index's listed rows hold, validated
against every row of each class and never against recall."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import replace

import pytest

from code_index.query import CodeIndex, row_five_tuple
from code_index.ratio_twins import ratio_twins
from code_index.types import IndexStamp, StatVarRow
from tests.code_index.support import INDEX_FIXTURE, load_index

_STAMP = IndexStamp(
    graph_endpoint="https://fake.example",
    exported_at="2026-11-24T00:00:00Z",
    code_counts={},
    statvar_shape_counts={},
    content_hash="",
)
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_DEBTOR = ("financingRecipient",)
_CREDITOR_DEBTOR = ("financingProvider", "financingRecipient")
_CHANNEL = (*_CREDITOR_DEBTOR, "financingChannel")
_STOCK = {"financialPositionType": "AmountOutstanding", "guaranteeStatus": "PPG"}
_USD = {"currencyOfDenomination": "CurrencyUSD", "financialPositionType": "AmountOutstanding"}


def _stock(dcid: str, roles: tuple[str, ...]) -> StatVarRow:
    """A PPG debt stock declaring `roles`."""
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialPosition",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier="Nominal",
        measurement_denominator=None,
        observation_properties=roles,
        constraint_properties=_STOCK,
    )


def _share(
    dcid: str, denominator: str, roles: tuple[str, ...], constraints: dict | None = None
) -> StatVarRow:
    """A share of the stock `denominator` declaring `roles`."""
    return StatVarRow(
        dcid=dcid,
        population_type="FinancialPosition",
        measured_property="amount",
        stat_type="measuredValue",
        measurement_qualifier=None,
        measurement_denominator=denominator,
        observation_properties=roles,
        constraint_properties=constraints or _USD,
    )


_STOCKS = [_stock("stock", _DEBTOR), _stock("stock_by_creditor", _CREDITOR_DEBTOR)]
_PAIR = [
    _share("share", "stock", _DEBTOR),
    _share("share_by_creditor", "stock_by_creditor", _CREDITOR_DEBTOR),
]
_CHAIN_STOCK = _stock("stock_by_channel", _CHANNEL)
_CHAIN_SHARE = _share("share_by_channel", "stock_by_channel", _CHANNEL)


def _index(listed: Sequence[StatVarRow], statvars: Sequence[StatVarRow] = ()) -> CodeIndex:
    """An index listing `listed` beside the exported `statvars`, built by the constructor so
    every view it derives holds the rows."""
    return CodeIndex(
        stamp=_STAMP, declarations=(), codes=(), statvars=statvars, listed_statvars=listed
    )


@pytest.mark.parametrize(
    ("listed", "statvars", "classes"),
    [
        pytest.param(
            [*_STOCKS, *_PAIR],
            [],
            [("stock", _DEBTOR), ("stock_by_creditor", _CREDITOR_DEBTOR)],
            id="valid-pair",
        ),
        pytest.param(
            [*_STOCKS, _CHAIN_STOCK, _CHAIN_SHARE, *_PAIR],
            [],
            [
                ("stock", _DEBTOR),
                ("stock_by_creditor", _CREDITOR_DEBTOR),
                ("stock_by_channel", _CHANNEL),
            ],
            id="chain-of-three",
        ),
        pytest.param(
            [
                *_STOCKS,
                *_PAIR,
                _share(
                    "share_eur",
                    "stock",
                    _CREDITOR_DEBTOR,
                    {**_USD, "currencyOfDenomination": "EUR"},
                ),
            ],
            [],
            None,
            id="class-holds-two-role-sets",
        ),
        pytest.param(
            [
                *_STOCKS,
                *_PAIR,
                _CHAIN_STOCK,
                _CHAIN_SHARE,
                _share("share_by_channel_debtor", "stock_by_channel", _DEBTOR, {**_USD, "x": "y"}),
            ],
            [],
            None,
            id="one-failing-class-of-three",
        ),
        pytest.param(
            [
                *_STOCKS,
                _share("share", "stock", _CREDITOR_DEBTOR),
                _share("share_by_creditor", "stock_by_creditor", _DEBTOR),
            ],
            [],
            None,
            id="roles-order-unlike-denominators",
        ),
        pytest.param([_STOCKS[0], *_PAIR], [], None, id="denominator-not-in-index"),
        pytest.param(
            [_PAIR[0]],
            [*_STOCKS, _PAIR[1]],
            None,
            id="numerator-not-listed",
        ),
        pytest.param(
            [_stock("stock", ()), _STOCKS[1], *_PAIR],
            [],
            None,
            id="subject-only-denominator",
        ),
        pytest.param(
            [*_STOCKS, *_PAIR],
            [replace(_PAIR[1], dcid="share_by_creditor_published", datasets=(_IDS,))],
            None,
            id="class-has-index-family",
        ),
        pytest.param(
            [*_STOCKS, _PAIR[0], replace(_PAIR[1], datasets=(_IDS,))],
            [],
            None,
            id="classes-held-by-different-provenances",
        ),
    ],
)
def test_a_bucket_maps_every_class_to_its_family_only_when_valid(listed, statvars, classes):
    """Shares of a stock and its role twins form one family, base first, when every row of
    every class agrees: each class declares one role set, the classes order as their
    denominators do, one class's roles sit inside every other's, each denominator is a
    role-bearing index row, no class has an index family, and one set of provenances holds
    every row. A bucket that fails any of
    these, in any one class, maps none of its five-tuples."""
    index = _index(listed, statvars)

    mapping = ratio_twins(index)

    shares = [row for row in (*listed, *statvars) if row.measurement_denominator is not None]
    if classes is None:
        assert not any(row_five_tuple(row) in mapping for row in shares)
        return
    families = {id(mapping[row_five_tuple(row)]) for row in shares}
    assert len(families) == 1
    family = mapping[row_five_tuple(shares[0])]
    assert [(c.denominator, c.roles) for c in family.classes] == classes
    assert family.class_of(tuple(reversed(_CREDITOR_DEBTOR))) is family.classes[1]
    assert family.class_of(("financingChannel",)) is None


def _row_measure(row: StatVarRow) -> tuple[object, ...]:
    """What an index row measures, its denominator and roles aside."""
    return (
        row.population_type,
        row.measured_property,
        row.stat_type,
        row.measurement_qualifier,
        tuple(sorted(row.constraint_properties.items())),
    )


def _differ_by_role_twin_denominators(index: CodeIndex, rows: Sequence[StatVarRow]) -> bool:
    """Whether StatVar `rows` measure one thing over two or more denominators that are
    index rows of one measure and denominator, apart in their roles alone, each role set
    nested in the next."""
    denominators = {row.measurement_denominator for row in rows}
    if len(denominators) < 2 or None in denominators:
        return False
    if len({_row_measure(row) for row in rows}) != 1:
        return False
    bases = [index.statvar_by_dcid(d) for d in denominators if d is not None]
    if any(base is None for base in bases):
        return False
    held = [base for base in bases if base is not None]
    if len({(*_row_measure(b), b.measurement_denominator) for b in held}) != 1:
        return False
    role_sets = sorted((frozenset(b.observation_properties) for b in held), key=len)
    return all(a < b for a, b in itertools.pairwise(role_sets))


def test_every_packaged_code_over_role_twin_denominators_is_one_family_over_its_aggregate():
    """Every source code of the packaged index whose StatVars measure one thing over a
    denominator and its role twins (IDS's currency and creditor-type shares of the PPG and
    total debt stocks) is one ratio family whose base divides by the aggregate stock, the
    denominator declaring the debtor alone."""
    index = load_index(INDEX_FIXTURE)
    codes = {
        source.code: rows
        for source in index.source_codes
        if None not in (rows := [index.statvar_by_dcid(dcid) for dcid in source.dcids])
        and _differ_by_role_twin_denominators(index, rows)
    }
    mapping = ratio_twins(index)

    assert len(codes) >= 13
    for code, rows in codes.items():
        families = {id(mapping.get(row_five_tuple(row))) for row in rows}
        family = mapping.get(row_five_tuple(rows[0]))
        assert family is not None, code
        assert families == {id(family)}, code
        base = index.statvar_by_dcid(family.base.denominator)
        assert base is not None and base.observation_properties == _DEBTOR, code
        assert family.base.roles == _DEBTOR, code
        assert [c.roles for c in family.classes[1:]] == [_CREDITOR_DEBTOR], code
