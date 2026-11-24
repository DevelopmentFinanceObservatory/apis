"""The two counts an open role's split states (`enrichment.party_split`) agree with the frame
the default download holds and with what a ranking counts there, over DAC2a's recipients in
the packaged code index: its member count is the distinct members on the frame's column
that are no totals, and its party count, the member count less the residual members, is
how many entities `rank` ranks, since a ranking leaves the residual lines out
(`party_residual`)."""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq

from code_index.declarations import DAC2A
from code_index.shared import get_index
from datasets_api.analysis_roots import frame_from_frame_layout
from dfo_analysis_engine.engine.rank import run_rank
from materialise.enrichment import PARTY_KIND_FIELD, party_cell, party_split

_ROLE = "financingRecipient"
_COUNTRIES = ("country/ETH", "country/KEN", "country/NGA")
_RESIDUALS = ("DACAfricaUnallocated", "DACAsiaUnallocated")
_TOTAL = "DACDevelopingCountries"


def test_member_count_is_the_frame_s_members_and_its_parties_are_what_rank_counts() -> None:
    parties = get_index().parties
    members = (*_COUNTRIES, *_RESIDUALS, _TOTAL)
    kinds = {
        member: party_cell(
            PARTY_KIND_FIELD, member, axis="destination", parties=parties, provenance=DAC2A
        )
        for member in members
    }
    # The recorded index declares each kind the counts read.
    assert [kinds[m] for m in members] == ["party"] * 3 + ["residual"] * 2 + ["total"]
    table = pa.table(
        {
            "stat_var": ["dcid:ODA"] * len(members),
            _ROLE: list(members),
            f"{_ROLE}_party_kind": [kinds[m] for m in members],
            "time_period": ["2021"] * len(members),
            "value": [float(index + 1) for index in range(len(members))],
            "unit": ["USDollar"] * len(members),
            "status": pa.nulls(len(members), pa.string()),
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)
    frame = frame_from_frame_layout(sink.getvalue())
    split = party_split(members, axis="destination", parties=parties, datasets=(DAC2A,))

    ranked = run_rank(
        frame,
        rank_by=[_ROLE],
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
    )

    held = {member for member in members if kinds[member] != "total"}
    assert split.member_count == len(held) == 5
    assert split.member_count - split.residual_count == len(ranked.entity_ranks) == 3
