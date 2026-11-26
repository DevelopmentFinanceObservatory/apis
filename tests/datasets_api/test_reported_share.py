"""How much of a partial year an opted-in frame records as reported
(`IncompleteAfter.reported_share`, `materialise.reported_share`), and the projected and
partial entries a collapse and a composite keep (`manifest.collapsed_incomplete_after`,
`SeriesStatisticsSink`'s `component_of`)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from datasets_api.manifest import collapsed_incomplete_after, incomplete_after
from materialise import MaterialisationResult
from materialise.frame_build import build_frame
from materialise.generation import elected_entities
from materialise.reported_share import YearTotalsFold
from materialise.statistics import (
    SeriesFigures,
    SeriesStatisticsSink,
    StatusFigures,
    compute_series_statistics_table,
)
from materialise.window import RequestedWindow
from origin_catalogue import bound_catalogue
from tests.origin_catalogue.fake_catalogue import dev_provenance_nodes, documenting

from ._version_record_store import FakeVersionRecordDatasetStore
from .test_elected_year import _AID as _ELECTED_AID
from .test_elected_year import _ALL_REPORT_2025, _MALAWI, Held, _donors, _index, _Rig
from .test_incomplete_after import _DAC2A as _DAC2A_NODE
from .test_incomplete_after import _source_as

_DAC2A = "provenance/OECDDAC2A"
_IDS = "provenance/WorldBankInternationalDebtStatistics"
_COMPOSITE = "provenance/DebtComposite"
_AID = "dcid:Amount_Aid"
_DEBT = "dcid:Amount_Debt"
_SHARE_OF_GNI = "dcid:Percent_Aid_GNI"


def _row(
    stat_var: str,
    recipient: str,
    year: int,
    value: float,
    *,
    unit: str = "USDollar",
    provenance: str = _DAC2A,
    method: str | None = None,
) -> dict[str, object]:
    return {
        "stat_var": stat_var,
        "recipient": recipient,
        "time_period": str(year),
        "value": value,
        "unit": unit,
        "measurementMethod": method,
        "provenance": provenance,
    }


def _minted(rows: list[dict[str, object]], *, components: tuple[object, ...] = ()) -> object:
    """The record a mint over `rows`, read whole before any window cut, stores under a
    catalogue marking DAC2A's 2025 partial."""
    frame = build_frame(role_keys=("recipient",), rows=rows)
    component_of = {
        ref.removeprefix("dcid:"): cast(str, component["provenance"])
        for component in cast("tuple[Mapping[str, object], ...]", components)
        for ref in cast("list[str]", component["stat_vars"])
    }
    sink = SeriesStatisticsSink(continuing=(), component_of=component_of)
    sink.write(compute_series_statistics_table(frame))
    fold = YearTotalsFold(component_of=component_of)
    fold.add(frame.table)
    result = cast(
        "MaterialisationResult",
        SimpleNamespace(
            trace=SimpleNamespace(components=components),
            series_statistics=sink.finish(),
            year_totals=fold.totals,
        ),
    )
    arcs = dev_provenance_nodes()[_DAC2A]
    catalogue = documenting({_DAC2A: arcs, _IDS: {}, _COMPOSITE: {}})
    with bound_catalogue(catalogue):
        judged = incomplete_after(result)
    assert judged is not None
    return judged.entries


@pytest.mark.parametrize(
    ("rows", "share"),
    [
        pytest.param(
            [
                _row(_AID, "MWI", 2024, 600.0),
                _row(_AID, "MWI", 2025, 1.0),
                _row(_AID, "KEN", 2024, 400.0),
                # A series new in 2025 adds to the reported value.
                _row(_AID, "TZA", 2025, 50.0),
            ],
            {"basis": "value", "share": 0.051, "against": 2024},
            id="a-currency-measure-shares-by-value",
        ),
        pytest.param(
            [
                _row(_AID, "MWI", 2024, 1_000_000.0),
                _row(_AID, "KEN", 2024, 276_000.0),
                # The only 2025 row is from a series with no 2024 row.
                _row(_AID, "TZA", 2025, 28_754.0),
            ],
            {"basis": "value", "share": 28_754 / 1_276_000, "against": 2024},
            id="a-partial-year-of-new-series-alone-is-not-zero",
        ),
        pytest.param(
            [
                _row(_SHARE_OF_GNI, recipient, 2024, 0.5, unit="Percent")
                for recipient in ("MWI", "KEN", "TZA", "UGA")
            ]
            + [_row(_SHARE_OF_GNI, "MWI", 2025, 0.7, unit="Percent")],
            {"basis": "series", "share": 0.25, "against": 2024},
            id="any-other-measure-shares-by-series",
        ),
    ],
)
def test_a_partial_entry_states_how_much_of_its_year_is_reported(
    rows: list[dict[str, object]], share: dict[str, object]
) -> None:
    (entry,) = cast("list[dict[str, object]]", _minted(rows))

    assert entry["status"] == "partial"
    assert entry["partial_period"] == "2025"
    assert entry["reported_share"] == pytest.approx(share)


@pytest.mark.parametrize(
    ("rows", "share"),
    [
        pytest.param(
            [_row(_AID, "MWI", 2024, -100.0), _row(_AID, "MWI", 2025, 40.0)],
            None,
            id="a-prior-year-summing-below-zero-states-no-share",
        ),
        pytest.param(
            [_row(_AID, "MWI", 2024, 100.0), _row(_AID, "MWI", 2025, -40.0)],
            None,
            id="a-partial-year-summing-below-zero-states-no-share",
        ),
        pytest.param(
            [_row(_AID, "MWI", 2024, 100.0), _row(_AID, "MWI", 2025, 150.0)],
            {"basis": "value", "share": 1.5, "against": 2024},
            id="a-partial-year-above-the-prior-year-states-a-share-above-one",
        ),
    ],
)
def test_a_value_share_states_only_a_meaningful_ratio(
    rows: list[dict[str, object]], share: dict[str, object] | None
) -> None:
    (entry,) = cast("list[dict[str, object]]", _minted(rows))

    assert entry["status"] == "partial"
    if share is None:
        assert "reported_share" not in entry
    else:
        assert entry["reported_share"] == pytest.approx(share)


def test_a_partial_entry_with_no_prior_year_states_no_share() -> None:
    (entry,) = cast("list[dict[str, object]]", _minted([_row(_AID, "MWI", 2025, 1.0)]))

    assert entry["status"] == "partial"
    assert "reported_share" not in entry


def test_an_opted_in_composite_records_its_ids_component_s_projections() -> None:
    """Every row of a composite carries the composite's provenance, so its projected rows are
    judged by the component serving their variable."""
    rows = [_row(_DEBT, "MWI", year, 10.0, provenance=_COMPOSITE) for year in (2024, 2025)] + [
        _row(_DEBT, "MWI", year, 12.0, provenance=_COMPOSITE, method="Estimated")
        for year in (2026, 2027)
    ]
    components = ({"provenance": _IDS, "stat_vars": [_DEBT]},)

    entries = _minted(rows, components=components)

    assert entries == [
        {
            "provenance": _COMPOSITE,
            "stat_vars": [_DEBT],
            "source": _IDS,
            "status": "projected",
            "partial_period": "2026",
        }
    ]


_SUMMED = "dcid:spec_summed"


def _collapsed(raw: list[dict[str, object]], *, provenance: str) -> object:
    """The record of a groupby sum over a version storing `raw`, its rows reaching 2027."""
    figures = SeriesFigures(
        series=1,
        span=None,
        grain=None,
        latest_period=None,
        series_reaching_latest=None,
        entities_by_role=(),
        by_source_unit=(),
        status=StatusFigures(estimated=None, provisional=None),
        latest_by_source={(provenance, _SUMMED): "2027"},
    )
    return collapsed_incomplete_after(raw, figures)


def test_a_groupby_sum_over_an_ids_frame_keeps_its_projected_entry() -> None:
    raw: list[dict[str, object]] = [
        {
            "provenance": _IDS,
            "stat_vars": [_DEBT],
            "source": _IDS,
            "status": "projected",
            "partial_period": "2026",
        }
    ]

    assert _collapsed(raw, provenance=_IDS) == [{**raw[0], "stat_vars": [_SUMMED]}]


@pytest.mark.parametrize(
    ("share", "kept"),
    [
        pytest.param({"basis": "value", "share": 0.001, "against": 2024}, True, id="value"),
        pytest.param({"basis": "series", "share": 0.25, "against": 2024}, False, id="series"),
    ],
)
def test_a_groupby_sum_keeps_a_value_share_and_drops_a_series_share(
    share: dict[str, object], kept: bool
) -> None:
    raw: list[dict[str, object]] = [
        {
            "provenance": _DAC2A,
            "stat_vars": [_AID],
            "source": _DAC2A,
            "status": "partial",
            "partial_period": "2025",
            "reported_share": share,
        }
    ]

    (entry,) = cast("list[dict[str, object]]", _collapsed(raw, provenance=_DAC2A))

    assert entry["status"] == "partial"
    assert entry.get("reported_share") == (share if kept else None)


@pytest.mark.parametrize(
    ("window", "indexed"),
    [
        pytest.param(
            RequestedWindow(
                kind="range", start_year=2025, end_year=2025, include=frozenset({"partial"})
            ),
            _MALAWI,
            id="a-range-of-the-partial-year-alone",
        ),
        pytest.param(
            RequestedWindow(kind="latest", include=frozenset({"partial"})),
            _ALL_REPORT_2025,
            id="a-latest-read-electing-the-partial-year",
        ),
    ],
)
def test_a_mint_of_the_partial_year_counts_it_against_the_year_before(
    version_store: FakeVersionRecordDatasetStore,
    window: RequestedWindow,
    indexed: Held,
) -> None:
    """Two of the eight donors reporting 2024 have reported 2025, each a value of 1, so a
    quarter of 2024's value is in. A `latest` read counts its 2024 rows from a read of that
    year alone, in one currency, so by value too. The frame still holds 2025 alone."""
    rig = _Rig(version_store, indexed={_ELECTED_AID: indexed}, live={_ELECTED_AID: _MALAWI})

    with bound_catalogue(_source_as(_DAC2A_NODE)):
        minted, _ = rig.mint(window)

    assert minted.coverage.window is not None
    assert (minted.coverage.window["start_year"], minted.coverage.window["end_year"]) == (
        2025,
        2025,
    )
    stored = rig.deps.snapshot_reader.read_version(rig.stored.spec_id, minted.content_version)
    assert stored is not None
    (entry,) = cast("list[dict[str, object]]", stored.facts.incomplete_after)
    assert entry["status"] == "partial"
    assert entry["reported_share"] == {"basis": "value", "share": 0.25, "against": 2024}


def test_a_latest_mint_opting_in_records_the_partial_year_it_serves_as_elected(
    version_store: FakeVersionRecordDatasetStore,
) -> None:
    """The year coverage a latest mint stores counts the years its read serves, so a read
    opting in to 2025 states 2025 as its elected year and newest recent year."""
    rig = _Rig(version_store, indexed={_ELECTED_AID: _ALL_REPORT_2025})

    with bound_catalogue(_source_as(_DAC2A_NODE)):
        rig.mint(RequestedWindow(kind="latest", include=frozenset({"partial"})))

    coverage = rig.stored_coverage()
    assert coverage["elected_year"] == 2025
    recent = cast("list[dict[str, object]]", coverage["recent_years"])
    assert recent[0]["year"] == 2025


@pytest.mark.parametrize(
    ("include", "donors"),
    [
        pytest.param(frozenset(), 8, id="held-back"),
        pytest.param(frozenset({"partial"}), 9, id="opted-in"),
    ],
)
def test_the_entities_a_latest_election_counts_follow_its_opt_in(
    include: frozenset[str], donors: int
) -> None:
    """A ninth donor holds 2025 alone, the partial year, so only a read opting in to it
    counts that donor's series."""
    held: Held = {**_donors(8, years=range(2021, 2025)), **_donors(1, years=(2025,), first=9)}
    rig = _Rig(FakeVersionRecordDatasetStore(), indexed={_ELECTED_AID: held})
    request = replace(
        rig.requests[_ELECTED_AID], window=RequestedWindow(kind="latest", include=include)
    )

    with bound_catalogue(_source_as(_DAC2A_NODE)):
        counted = elected_entities(
            request, "financingProvider", coverage=_index({_ELECTED_AID: held}), facet_ids=["usd"]
        )

    assert len(counted) == donors
