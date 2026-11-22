"""A frame-layout file reads a party's kind over the datasets its version cites, so a frame of
the DAC2a bilateral composite types the IMF trust funds a `total`, as DAC2A declares them,
though the CRS declares them a `party`, and a partitioned rank of its providers answers."""

from __future__ import annotations

import pytest
from dfo_frame_grammar import Frame

from code_index import CodeIndex, IndexStamp
from code_index.declarations import CRS, DAC2A, DAC2A_BILATERAL
from code_index.types import PartyRow, RoleAxis
from datasets_api.analysis_roots import frame_from_frame_layout
from datasets_api.mint_models import MintOptions
from datasets_api.render import render_and_store
from dfo_analysis_engine.engine.common import PlaceLevelsMixedError
from dfo_analysis_engine.engine.rank import run_rank
from materialise.content_version import mint_version

from ._artifact_doubles import FakeArtifactStore
from ._render_inputs import ENRICHMENT, PREPARED_AT, SITE

_TRUST_FUNDS = "IMFConcessionalTrustFunds"
_DAC_TOTAL = "DACCountriesTotal"
_PROVIDERS = ("country/FRA", "country/USA", "InternationalMonetaryFund", _TRUST_FUNDS, _DAC_TOTAL)
_RECIPIENTS = ("country/KEN", "country/GHA")
_DONOR: tuple[tuple[str, RoleAxis], ...] = ((CRS, "origin"), (DAC2A, "origin"))

_INDEX = CodeIndex(
    stamp=IndexStamp(
        graph_endpoint="https://example.test/core/api",
        exported_at="2026-11-22T00:00:00Z",
        code_counts={},
        statvar_shape_counts={},
        content_hash="party-kind-sources",
    ),
    declarations=(),
    codes=(),
    statvars=(),
    parties=(
        PartyRow(dcid="country/FRA", kind="party", party_class="official", published=_DONOR),
        PartyRow(dcid="country/USA", kind="party", party_class="official", published=_DONOR),
        PartyRow(
            dcid="InternationalMonetaryFund",
            kind="party",
            party_class="official",
            published=((DAC2A, "origin"),),
        ),
        # A total in DAC2A, equal to the IMF there, and a party in the CRS.
        PartyRow(
            dcid=_TRUST_FUNDS,
            kind="party",
            party_class="official",
            published=_DONOR,
            dataset_kinds=((DAC2A, "total"),),
        ),
        PartyRow(dcid=_DAC_TOTAL, kind="total", published=((DAC2A, "origin"),)),
        *(
            PartyRow(dcid=recipient, kind="party", published=((DAC2A, "destination"),))
            for recipient in _RECIPIENTS
        ),
    ),
)
_FRAME_LAYOUT = MintOptions(format="parquet", bundle=False, layout="frame", columns=["party_kind"])


def _frame(provenance: str | None) -> Frame:
    """The DAC2a total ODA each provider gave each recipient in 2022, every row carrying
    `provenance`, or no `provenance` column for `None`."""
    return Frame.build(
        role_keys=("financingProvider", "financingRecipient"),
        rows=[
            {
                "stat_var": "dcid:DAC2aTotalODA",
                "financingProvider": provider,
                "financingRecipient": recipient,
                "time_period": "2022",
                "value": 10.0 * (at + 1) + len(recipient),
                **({} if provenance is None else {"provenance": provenance}),
            }
            for at, provider in enumerate(_PROVIDERS)
            for recipient in _RECIPIENTS
        ],
    )


def _frame_layout(frame: Frame, *, cited: list[str]) -> Frame:
    """`frame` rendered to its frame-layout file for a version citing `cited`, read back as the
    analysis runner reads a root."""
    artifact_store = FakeArtifactStore()
    provenance = frame.table.column("provenance") if "provenance" in frame.columns else None
    rendered = render_and_store(
        frame_chunks=lambda: iter((frame,)),
        options=_FRAME_LAYOUT,
        spec={"spec_id": "spec_party_kind_sources"},
        provenance_refs=[] if provenance is None else sorted(set(provenance.to_pylist())),
        citations={},
        reproducing_call=None,
        content_version=mint_version(frame),
        artifact_store=artifact_store,
        entity_columns=("financingProvider", "financingRecipient"),
        enrichment=ENRICHMENT,
        code_index=_INDEX,
        resolved_stat_var_refs=("dcid:DAC2aTotalODA",),
        pivot_max_output_cells=5_000_000,
        groupby_fold_max_rows=2_000_000,
        prepared_at=PREPARED_AT,
        site=SITE,
        citation_refs=cited,
    )
    return frame_from_frame_layout(artifact_store.fetch(rendered.artifact.url))


def _kinds(frame: Frame) -> dict[str, str]:
    table = frame.table
    return dict(
        zip(
            table.column("financingProvider").to_pylist(),
            table.column("financingProvider_party_kind").to_pylist(),
            strict=True,
        )
    )


def _top_two_per_recipient(frame: Frame) -> list[tuple[str, str]]:
    result = run_rank(
        frame,
        rank_by=["financingProvider"],
        partition_by=["financingRecipient"],
        aggregation=None,
        window_start="2022",
        window_end="2022",
        ascending=False,
        limit=2,
    )
    table = result.frame.table
    return sorted(
        zip(
            table.column("financingRecipient").to_pylist(),
            table.column("financingProvider").to_pylist(),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "provenance",
    [
        pytest.param(DAC2A_BILATERAL, id="composite-rows"),
        pytest.param(None, id="no-provenance-column"),
    ],
)
def test_a_dac2a_frame_types_a_party_by_the_datasets_its_version_cites(
    provenance: str | None,
) -> None:
    frame = _frame_layout(_frame(provenance), cited=[DAC2A])

    assert _kinds(frame) == {
        "country/FRA": "party",
        "country/USA": "party",
        "InternationalMonetaryFund": "party",
        _TRUST_FUNDS: "total",
        _DAC_TOTAL: "total",
    }
    assert _top_two_per_recipient(frame) == [
        ("country/GHA", "InternationalMonetaryFund"),
        ("country/GHA", "country/USA"),
        ("country/KEN", "InternationalMonetaryFund"),
        ("country/KEN", "country/USA"),
    ]


def test_a_frame_citing_datasets_that_type_a_party_apart_leaves_it_untyped_and_refuses() -> None:
    """A composite of the CRS and DAC2A finds the trust funds a party in one and a total in the
    other, so no one kind holds and a rank of the providers refuses."""
    frame = _frame_layout(_frame(DAC2A_BILATERAL), cited=[CRS, DAC2A])

    assert _kinds(frame)[_TRUST_FUNDS] == "untyped"
    with pytest.raises(PlaceLevelsMixedError):
        _top_two_per_recipient(frame)
