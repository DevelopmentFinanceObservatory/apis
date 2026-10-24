"""`render_and_store` with `options.without_party_totals`: the version a cut renders under,
alone and under a groupby, and the rendering it is recorded as."""

from __future__ import annotations

from dfo_frame_grammar import Frame

from code_index import CodeIndex, CodeRow, CodeSystemDeclaration, IndexStamp
from code_index.types import PartyRow, StatVarRow
from datasets_api.mint_models import GroupbyOptions, MintOptions
from datasets_api.render import (
    RenderedDefinition,
    artifact_rendering,
    render_and_store,
    rendering_options,
)
from materialise.content_version import mint_version
from materialise.enrichment import EnrichmentLookup

from ._artifact_doubles import FakeArtifactStore
from ._graph_doubles import FakeMintGraphReader
from ._render_inputs import PREPARED_AT, SITE

_SPEC_ID = "spec_party_totals"
_KENYA = "country/KEN"
_GHANA = "country/GHA"
_DEVELOPING = "DevelopingCountries"

_PURPOSE_HEALTH = CodeRow(
    dcid="dcid:CRSPurposeCode122",
    code_system="crsPurpose",
    code="122",
    label="Basic health care",
    synonyms=(),
    parent=None,
)
_PURPOSE_EDUCATION = CodeRow(
    dcid="dcid:CRSPurposeCode111",
    code_system="crsPurpose",
    code="111",
    label="Education",
    synonyms=(),
    parent=None,
)


def _statvar(dcid: str, *, purpose: str) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type="Amount",
        measured_property="amount",
        stat_type=None,
        measurement_qualifier=None,
        measurement_denominator=None,
        observation_properties=("recipient",),
        constraint_properties={"crsPurpose": purpose},
    )


_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://example.test/core/api",
        exported_at="2026-09-09T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="deadbeef",
    ),
    declarations=(
        CodeSystemDeclaration(
            code_system="crsPurpose",
            hierarchical=True,
            disjoint_from_children=True,
            summable=True,
            absent_reading="not_declared",
            plain_name="sector",
        ),
    ),
    codes=(_PURPOSE_HEALTH, _PURPOSE_EDUCATION),
    statvars=(
        _statvar("SV_A", purpose=_PURPOSE_HEALTH.dcid),
        _statvar("SV_B", purpose=_PURPOSE_EDUCATION.dcid),
    ),
)
_PARTIES = {
    party.dcid: party
    for party in (
        PartyRow(dcid=_KENYA, kind="party"),
        PartyRow(dcid=_GHANA, kind="party"),
        PartyRow(dcid=_DEVELOPING, kind="total"),
    )
}
_RESOLVED = ("dcid:SV_A", "dcid:SV_B")
_IDENTITY = {
    "shape": {
        "kind": "statvar_schema",
        "population_type": "Amount",
        "measured_property": "amount",
    },
    "slots": [],
    "entities": {},
    "open_roles": ["destination"],
}


def _frame(recipients: tuple[str, ...]) -> Frame:
    return Frame.build(
        role_keys=("recipient",),
        rows=[
            {
                "stat_var": stat_var,
                "recipient": recipient,
                "time_period": "2022",
                "value": value,
                "provenance": "p1",
            }
            for recipient in recipients
            for stat_var, value in (("dcid:SV_A", 10.0), ("dcid:SV_B", 5.0))
        ],
    )


_WITH_TOTAL = _frame((_KENYA, _GHANA, _DEVELOPING))
_MEMBERS = _frame((_KENYA, _GHANA))


def _render(frame: Frame, options: MintOptions) -> tuple[RenderedDefinition, str]:
    artifact_store = FakeArtifactStore()
    rendered = render_and_store(
        frame_chunks=lambda: iter((frame,)),
        options=options,
        spec={"spec_id": _SPEC_ID, "identity": _IDENTITY},
        provenance_refs=["p1"],
        citations={},
        reproducing_call=None,
        content_version=mint_version(frame),
        artifact_store=artifact_store,
        entity_columns=("recipient",),
        enrichment=EnrichmentLookup(FakeMintGraphReader(), parties=lambda: _PARTIES),
        code_index=_INDEX,
        resolved_stat_var_refs=_RESOLVED,
        pivot_max_output_cells=5_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
    )
    return rendered, artifact_store.fetch(rendered.artifact.url).decode("utf-8")


_CUT = MintOptions(format="csv", bundle=False, without_party_totals=True)
_SUMMED = GroupbyOptions(by=["recipient"], op="sum")


def test_a_cut_that_leaves_totals_out_renders_a_version_minted_from_the_kept_rows():
    rendered, body = _render(_WITH_TOTAL, _CUT)

    derived = rendered.derived
    assert derived is not None
    assert derived.content_version == mint_version(_MEMBERS) != mint_version(_WITH_TOTAL)
    assert derived.rows == _MEMBERS.num_rows
    assert not derived.summed
    assert derived.party_totals_left_out == (
        {"role": "recipient", "provenance": "p1", "entities": [_DEVELOPING]},
    )
    assert [chunk.table for chunk in derived.frame_chunks()] == [_MEMBERS.table]
    assert _DEVELOPING not in body
    assert _KENYA in body


def test_a_cut_that_leaves_nothing_out_renders_under_the_frame_s_own_version():
    rendered, body = _render(_MEMBERS, _CUT)

    assert rendered.derived is None
    assert rendered.artifact.rendering["without_party_totals"] is True
    assert _GHANA in body


def test_a_groupby_with_a_cut_sums_the_kept_members_under_one_version():
    """The collapse of the cut frame is the collapse of a frame that never held the totals,
    and the one version it mints names the totals the cut left out."""
    cut_and_summed, body = _render(
        _WITH_TOTAL,
        MintOptions(format="csv", bundle=False, groupby=_SUMMED, without_party_totals=True),
    )
    summed_members, _ = _render(_MEMBERS, MintOptions(format="csv", bundle=False, groupby=_SUMMED))
    summed_with_total, _ = _render(
        _WITH_TOTAL, MintOptions(format="csv", bundle=False, groupby=_SUMMED)
    )

    derived = cut_and_summed.derived
    assert derived is not None and summed_members.derived is not None
    assert summed_with_total.derived is not None
    assert derived.summed
    assert derived.content_version == summed_members.derived.content_version
    assert derived.content_version != summed_with_total.derived.content_version
    assert derived.party_totals_left_out == (
        {"role": "recipient", "provenance": "p1", "entities": [_DEVELOPING]},
    )
    assert _DEVELOPING not in body
    assert "15.0" in body


def test_a_rendering_records_the_option_only_when_it_is_set():
    """A rendering without the option records no `without_party_totals` field, so its key is
    that of the same options with no such field, and either reads back as the options it was
    rendered under."""
    plain = MintOptions(format="parquet", bundle=False)
    cut = MintOptions(format="parquet", bundle=False, without_party_totals=True)

    assert "without_party_totals" not in artifact_rendering(plain, site=SITE)
    assert artifact_rendering(cut, site=SITE)["without_party_totals"] is True
    assert rendering_options(artifact_rendering(cut, site=SITE), site=SITE) == cut
    assert rendering_options(artifact_rendering(plain, site=SITE), site=SITE) == plain
