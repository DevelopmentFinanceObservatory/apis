"""`datasets_api.party_totals`: which party roles are open, and which rows a cut leaves out."""

from __future__ import annotations

import pyarrow as pa
import pytest
from dfo_frame_grammar import Frame, frame_sha256

from code_index.types import PartyRow, WholeMarker
from datasets_api.party_totals import cut_frame, cut_party_totals, open_party_roles

_PROVIDER = "financingProvider"
_RECIPIENT = "financingRecipient"
_ROLES = (_PROVIDER, _RECIPIENT)

_DAC_TOTAL = "DACTotal"
_G7 = "GroupOfSeven"
_FRANCE = "country/FRA"
_GERMANY = "country/DEU"
_OFFICIAL = "DACOfficialDonors"
_GATES = "GatesFoundation"
_DEVELOPING = "DevelopingCountries"
_AFRICA_REGIONAL = "AfricaRegional"
_KENYA = "country/KEN"
_GHANA = "country/GHA"

# `_GERMANY` and `_GHANA` have no row, so the index declares no kind for them.
_PARTIES = {
    party.dcid: party
    for party in (
        PartyRow(dcid=_DAC_TOTAL, kind="total"),
        PartyRow(dcid=_G7, kind="total"),
        PartyRow(
            dcid=_OFFICIAL,
            kind="total",
            sector="official",
            whole=(WholeMarker(dataset="provenance/P", covers=("official",)),),
        ),
        PartyRow(dcid=_GATES, kind="party", sector="private"),
        PartyRow(dcid=_FRANCE, kind="party"),
        PartyRow(dcid=_DEVELOPING, kind="total"),
        PartyRow(dcid=_AFRICA_REGIONAL, kind="residual"),
        PartyRow(dcid=_KENYA, kind="party"),
    )
}


def _frame(pairs: list[tuple[str, str]]) -> Frame:
    return Frame.build(
        role_keys=_ROLES,
        rows=[
            {
                "stat_var": "dcid:SV",
                _PROVIDER: provider,
                _RECIPIENT: recipient,
                "time_period": "2022",
                "value": float(index),
            }
            for index, (provider, recipient) in enumerate(pairs)
        ],
    )


def _kept(frame: Frame, *, identity: dict[str, object]) -> tuple[set[tuple[str, str]], list]:
    """The (provider, recipient) pairs a cut keeps of `frame` under `identity`, and its trace
    entries."""
    cut = cut_party_totals(
        lambda: iter((frame,)),
        roles=open_party_roles(identity, _ROLES),
        parties=_PARTIES,
    )
    kept = {
        (row[1], row[2]) for chunk in cut.chunks(lambda: iter((frame,)))() for row in chunk.rows
    }
    return kept, cut.trace_entries()


_EVERY_PAIR = [
    (provider, recipient)
    for provider in (_DAC_TOTAL, _G7, _OFFICIAL, _FRANCE, _GERMANY, _GATES)
    for recipient in (_DEVELOPING, _AFRICA_REGIONAL, _KENYA, _GHANA)
]


@pytest.mark.parametrize(
    ("identity", "kept_providers", "kept_recipients", "entries"),
    [
        pytest.param(
            {"entities": {}},
            {_FRANCE, _GERMANY, _GATES},
            {_AFRICA_REGIONAL, _KENYA, _GHANA},
            [
                {"role": _PROVIDER, "entities": [_OFFICIAL, _DAC_TOTAL, _G7]},
                {"role": _RECIPIENT, "entities": [_DEVELOPING]},
            ],
            id="both-open",
        ),
        pytest.param(
            {"entities": {"destination": [_DEVELOPING, _KENYA]}},
            {_FRANCE, _GERMANY, _GATES},
            {_DEVELOPING, _AFRICA_REGIONAL, _KENYA, _GHANA},
            [{"role": _PROVIDER, "entities": [_OFFICIAL, _DAC_TOTAL, _G7]}],
            id="recipient-bound-by-axis-label",
        ),
        pytest.param(
            {"entities": {"provider": [_DAC_TOTAL]}},
            {_DAC_TOTAL, _G7, _OFFICIAL, _FRANCE, _GERMANY, _GATES},
            {_AFRICA_REGIONAL, _KENYA, _GHANA},
            [{"role": _RECIPIENT, "entities": [_DEVELOPING]}],
            id="provider-bound-by-caller-role",
        ),
    ],
)
def test_a_cut_leaves_out_the_totals_of_each_open_party_role(
    identity, kept_providers, kept_recipients, entries
):
    """Totals, whole totals included, leave an open role; parties of either sector, residuals
    and undeclared parties stay, and a bound role keeps every party whatever its kind."""
    kept, trace = _kept(_frame(_EVERY_PAIR), identity=identity)

    assert kept == {
        (provider, recipient)
        for provider, recipient in _EVERY_PAIR
        if provider in kept_providers and recipient in kept_recipients
    }
    assert trace == entries


def test_a_role_whose_every_row_is_a_total_keeps_them():
    """The totals are then the only parties the role holds."""
    pairs = [(_DAC_TOTAL, _KENYA), (_G7, _DEVELOPING), (_DAC_TOTAL, _GHANA)]

    kept, trace = _kept(_frame(pairs), identity={"entities": {}})

    assert kept == {(_DAC_TOTAL, _KENYA), (_DAC_TOTAL, _GHANA)}
    assert trace == [{"role": _RECIPIENT, "entities": [_DEVELOPING]}]


def test_each_role_reads_the_rows_the_roles_before_it_kept():
    """Once the provider totals leave, every row left on the recipient is a total, so the
    recipient keeps them."""
    pairs = [(_DAC_TOTAL, _KENYA), (_FRANCE, _DEVELOPING)]

    kept, trace = _kept(_frame(pairs), identity={"entities": {}})

    assert kept == {(_FRANCE, _DEVELOPING)}
    assert trace == [{"role": _PROVIDER, "entities": [_DAC_TOTAL]}]


@pytest.mark.parametrize(
    ("identity", "open_roles"),
    [
        ({}, (_PROVIDER, _RECIPIENT)),
        ({"entities": {"origin": [_FRANCE]}}, (_RECIPIENT,)),
        ({"entities": {_RECIPIENT: [_KENYA]}}, (_PROVIDER,)),
        ({"entities": {"creditor": [_FRANCE], "debtor": [_KENYA]}}, ()),
        ({"entities": {"observationAbout": [_KENYA]}}, (_PROVIDER, _RECIPIENT)),
    ],
    ids=["none-bound", "axis-label", "graph-role", "debt-roles", "subject-role"],
)
def test_a_party_role_is_open_unless_an_entity_binds_its_axis(identity, open_roles):
    assert open_party_roles(identity, ("observationAbout", *_ROLES)) == open_roles


def test_the_cut_frame_hashes_and_counts_the_rows_it_keeps():
    """Its hash is the hash of the kept rows built as a frame of their own, and its counts are
    theirs, split across chunks or not."""
    pairs = [(_DAC_TOTAL, _KENYA), (_FRANCE, _KENYA), (_FRANCE, _GHANA), (_G7, _GHANA)]
    whole = _frame(pairs)
    chunks = (
        Frame(columns=whole.columns, table=whole.table.slice(0, 2)),
        Frame(columns=whole.columns, table=whole.table.slice(2)),
    )
    cut = cut_party_totals(
        lambda: iter(chunks), roles=open_party_roles({}, _ROLES), parties=_PARTIES
    )

    kept = cut_frame(lambda: iter(chunks), cut, role_columns=_ROLES)

    expected = Frame.build(
        role_keys=_ROLES,
        rows=[
            {
                "stat_var": "dcid:SV",
                _PROVIDER: _FRANCE,
                _RECIPIENT: recipient,
                "time_period": "2022",
                "value": value,
            }
            for recipient, value in ((_KENYA, 1.0), (_GHANA, 2.0))
        ],
    )
    assert kept.frame_sha256 == frame_sha256(expected) != frame_sha256(whole)
    assert kept.rows == 2
    assert kept.distinct == {_PROVIDER: 1, _RECIPIENT: 2, "years": 1}


def test_the_cut_frame_counts_no_null_role_cell_as_a_value():
    """A null role cell counts as no entity, as a materialisation's coverage counts it."""
    built = _frame([(_DAC_TOTAL, _KENYA), (_FRANCE, _KENYA), (_FRANCE, _GHANA)])
    recipients = built.table.column(_RECIPIENT).to_pylist()
    # France's Ghana row, so the rows stay in key order: a null sorts first.
    recipients[1] = None
    table = built.table.set_column(
        built.table.schema.get_field_index(_RECIPIENT),
        built.table.schema.field(_RECIPIENT),
        pa.array(recipients, type=built.table.schema.field(_RECIPIENT).type),
    )
    whole = Frame.from_table(columns=built.columns, table=table)
    cut = cut_party_totals(
        lambda: iter((whole,)), roles=open_party_roles({}, _ROLES), parties=_PARTIES
    )

    kept = cut_frame(lambda: iter((whole,)), cut, role_columns=_ROLES)

    assert kept.rows == 2
    assert kept.distinct == {_PROVIDER: 1, _RECIPIENT: 1, "years": 1}
